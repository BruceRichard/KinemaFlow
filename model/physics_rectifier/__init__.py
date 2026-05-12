# Physical Plausibility Flow Rectification
# Based on KinemaFlow paper Sec 3.3: swept signed-distance energy + predictor-corrector
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from torch import Tensor


class PEBE(nn.Module):
    """Part Energy Boundary Encoder - predicts oriented quadratic SDF primitives.

    Maps a 768-dim geometric latent z_i to K oriented quadratic primitives
    {(mu_k, S_k, epsilon_k)}_{k=1..K}, forming a differentiable SDF proxy.
    """

    def __init__(self, latent_dim: int = 768, num_primitives: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.num_primitives = num_primitives
        self.latent_dim = latent_dim

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.mu_head = nn.Linear(hidden_dim, num_primitives * 3)
        self.scale_head = nn.Linear(hidden_dim, num_primitives * 6)
        self.eps_head = nn.Linear(hidden_dim, num_primitives)

    def forward(self, z: Tensor) -> Dict[str, Tensor]:
        """Predict primitives from latent code.

        Args:
            z: [B, latent_dim] latent codes

        Returns:
            dict with keys:
                mu:      [B, K, 3] primitive centers
                S_tril:  [B, K, 3, 3] lower-triangular scale matrices (PSD by construction)
                epsilon: [B, K] distance thresholds
        """
        B = z.shape[0]
        K = self.num_primitives
        h = self.mlp(z)

        mu = self.mu_head(h).view(B, K, 3)
        epsilon = F.softplus(self.eps_head(h).view(B, K)) + 0.01

        S_params = self.scale_head(h).view(B, K, 6)
        S_tril = torch.zeros(B, K, 3, 3, device=z.device, dtype=z.dtype)
        S_tril[:, :, 0, 0] = F.softplus(S_params[:, :, 0]) + 0.001
        S_tril[:, :, 1, 1] = F.softplus(S_params[:, :, 1]) + 0.001
        S_tril[:, :, 2, 2] = F.softplus(S_params[:, :, 2]) + 0.001
        S_tril[:, :, 1, 0] = S_params[:, :, 3]
        S_tril[:, :, 2, 0] = S_params[:, :, 4]
        S_tril[:, :, 2, 1] = S_params[:, :, 5]

        S_psd = S_tril @ S_tril.transpose(-2, -1)

        return {'mu': mu, 'S': S_psd, 'epsilon': epsilon}


def transform_primitive(mu: Tensor, S: Tensor, T_mat: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply SE(3) transform T to a primitive (mu, S).

    Args:
        mu:   [..., 3] center
        S:    [..., 3, 3] scale PSD matrix
        T_mat: [4, 4] or [..., 4, 4] SE(3) transform

    Returns:
        mu_T: [..., 3] transformed center
        S_T:  [..., 3, 3] transformed scale
    """
    R = T_mat[..., :3, :3]
    t = T_mat[..., :3, 3]
    mu_T = (R @ mu.unsqueeze(-1)).squeeze(-1) + t
    S_T = R @ S @ R.transpose(-2, -1)
    return mu_T, S_T


def primitive_sdf(points: Tensor, mu: Tensor, S: Tensor, epsilon: Tensor) -> Tensor:
    """Compute SDF of a single primitive at query points.

    phi_k(p) = (p - mu)^T S (p - mu) - epsilon

    Args:
        points:  [B, N, 3] query points
        mu:      [B, 3] primitive center
        S:       [B, 3, 3] primitive scale
        epsilon: [B] threshold

    Returns:
        sdf: [B, N] signed distance
    """
    d = points - mu.unsqueeze(1)
    Sd = (S.unsqueeze(1) @ d.unsqueeze(-1)).squeeze(-1)
    quad = (d * Sd).sum(dim=-1)
    return quad - epsilon.unsqueeze(1)


def part_sdf(points: Tensor, mu: Tensor, S: Tensor, epsilon: Tensor) -> Tensor:
    """Compute SDF of a part (minimum over all primitives).

    Args:
        points:  [B, N, 3] query points
        mu:      [B, K, 3] primitive centers
        S:       [B, K, 3, 3] primitive scales
        epsilon: [B, K] thresholds

    Returns:
        sdf: [B, N] part SDF (min over primitives)
    """
    B, K, _ = mu.shape
    N = points.shape[1]
    sdf_all = []
    for k in range(K):
        sdf_k = primitive_sdf(points, mu[:, k], S[:, k], epsilon[:, k])
        sdf_all.append(sdf_k)
    sdf_all = torch.stack(sdf_all, dim=1)
    return sdf_all.min(dim=1).values


def repulsive_potential(sdf_i: Tensor, sdf_j: Tensor, tau: float = 0.05) -> Tensor:
    """Repulsive potential psi for collision detection.

    psi(p, T) = max(0, tau - phi(p, T))

    Args:
        sdf_i: [B, N] SDF of part i at points
        sdf_j: [B, N] SDF of part j at points
        tau:   distance threshold

    Returns:
        energy: [B] per-batch energy
    """
    psi_i = F.relu(tau - sdf_i)
    psi_j = F.relu(tau - sdf_j)
    return (psi_i * psi_j).mean(dim=1)


class SweptCollisionEnergy(nn.Module):
    """Swept signed-distance collision energy over articulation states.

    Computes E_phys(x) = (1/|T|) * sum_{tau in T} sum_{(i,j) in E} E_ij(tau, i, j)
    over a quadrature set of motion states (closed, open, midpoint, intermediates).
    """

    def __init__(self, tau: float = 0.05, num_contact_samples: int = 256,
                 broad_phase_radius: float = 1.0):
        super().__init__()
        self.tau = tau
        self.num_contact_samples = num_contact_samples
        self.broad_phase_radius = broad_phase_radius

    def sample_contact_points(self, mu: Tensor, S: Tensor, num_samples: int) -> Tensor:
        """Sample contact points from primitive surfaces.

        Uses Gaussian sampling around each primitive center with covariance S^{-1}.
        """
        B, K = mu.shape[0], mu.shape[1]
        device = mu.device
        samples_per_prim = max(num_samples // K, 1)
        total = samples_per_prim * K

        L = torch.linalg.cholesky(S)
        noise = torch.randn(B, K, samples_per_prim, 3, device=device, dtype=mu.dtype)
        samples = mu.unsqueeze(2) + (L.unsqueeze(2) @ noise.unsqueeze(-1)).squeeze(-1)

        samples = samples.view(B, total, 3)
        return samples[:, :num_samples]

    def broad_phase_filter(self, mu_i: Tensor, mu_j: Tensor) -> Tensor:
        """Sphere-based broad phase: return True for pairs needing fine check."""
        centers_i = mu_i.mean(dim=1)
        centers_j = mu_j.mean(dim=1)
        dist = (centers_i - centers_j).norm(dim=-1)
        return dist < self.broad_phase_radius

    def pairwise_energy(self, prim_i: Dict[str, Tensor], prim_j: Dict[str, Tensor],
                        T_i: Tensor, T_j: Tensor) -> Tensor:
        """Compute E_ij between two parts at given transform state.

        Args:
            prim_i/j: dict with 'mu', 'S', 'epsilon' from PEBE
            T_i/j:    [B, 4, 4] SE(3) transforms

        Returns:
            E_ij: [B] pairwise collision energy
        """
        B = T_i.shape[0]
        device = T_i.device

        mu_i, S_i, eps_i = prim_i['mu'], prim_i['S'], prim_i['epsilon']
        mu_j, S_j, eps_j = prim_j['mu'], prim_j['S'], prim_j['epsilon']

        mu_i_T, S_i_T = transform_primitive(mu_i, S_i, T_i)
        mu_j_T, S_j_T = transform_primitive(mu_j, S_j, T_j)

        broad_mask = self.broad_phase_filter(mu_i_T, mu_j_T)
        if not broad_mask.any():
            return torch.zeros(B, device=device)

        samples = self.sample_contact_points(mu_i_T, S_i_T, self.num_contact_samples)

        sdf_i = torch.zeros(B, self.num_contact_samples, device=device)
        sdf_j = torch.zeros(B, self.num_contact_samples, device=device)

        idx = broad_mask.nonzero(as_tuple=True)[0]
        for b in idx:
            sdf_i[b] = part_sdf(
                samples[b:b+1], mu_i_T[b:b+1], S_i_T[b:b+1], eps_i[b:b+1]
            )
            sdf_j[b] = part_sdf(
                samples[b:b+1], mu_j_T[b:b+1], S_j_T[b:b+1], eps_j[b:b+1]
            )

        energy = repulsive_potential(sdf_i, sdf_j, self.tau)
        energy[~broad_mask] = 0.0
        return energy

    def forward(self, part_primitives: List[Dict[str, Tensor]],
                transforms_per_state: List[Dict[int, Tensor]],
                parent_indices: List[int]) -> Tensor:
        """Compute total swept collision energy over articulation states.

        E_phys = (1/|T|) * sum_{tau} sum_{i,j not parent-child} E_ij(tau, i, j)

        Args:
            part_primitives: list of PEBE outputs for each part
            transforms_per_state: list of {part_idx: [B, 4, 4]} per state tau
            parent_indices: dfn_fa for each part

        Returns:
            E_phys: [B] total collision energy (scalar per batch element)
        """
        n_states = len(transforms_per_state)
        n_parts = len(part_primitives)
        B = list(transforms_per_state[0].values())[0].shape[0]
        device = list(transforms_per_state[0].values())[0].device

        non_adjacent_pairs = []
        for i in range(n_parts):
            for j in range(i + 1, n_parts):
                if parent_indices[i] != j and parent_indices[j] != i:
                    non_adjacent_pairs.append((i, j))

        if not non_adjacent_pairs:
            return torch.zeros(B, device=device, requires_grad=True)

        total_E = torch.zeros(B, device=device)
        for transforms in transforms_per_state:
            for i, j in non_adjacent_pairs:
                if i in transforms and j in transforms:
                    T_i = transforms[i]
                    T_j = transforms[j]
                    E_ij = self.pairwise_energy(
                        part_primitives[i], part_primitives[j], T_i, T_j
                    )
                    total_E = total_E + E_ij

        return total_E / (n_states * max(len(non_adjacent_pairs), 1))


class PhysicalRectifier:
    """Predictor-Corrector for physical plausibility during flow matching inference.

    Implements Eq. 9: x̂ = x - γ ∇E_phys(x)
    Applied during early/middle solver steps (first 80%), disabled near end.
    """

    def __init__(self, pebe: PEBE, energy: SweptCollisionEnergy,
                 guidance_scale: float = 1.5, active_ratio: float = 0.8):
        self.pebe = pebe
        self.energy = energy
        self.guidance_scale = guidance_scale
        self.active_ratio = active_ratio

    @torch.enable_grad()
    def correct(self, latent_codes: Tensor, primitives: List[Dict[str, Tensor]],
                transforms_per_state: List[Dict[int, Tensor]],
                parent_indices: List[int], step: int, total_steps: int) -> Tensor:
        """Apply corrector step if within active window.

        Returns:
            corrected_latent: Tensor [B, D] after gradient correction
        """
        if step / max(total_steps, 1) >= self.active_ratio:
            return latent_codes

        latent_codes = latent_codes.detach().requires_grad_(True)

        E_phys = self.energy(primitives, transforms_per_state, parent_indices)
        if E_phys.sum() == 0:
            return latent_codes.detach()

        grad = torch.autograd.grad(E_phys.sum(), latent_codes,
                                   retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            corrected = latent_codes - self.guidance_scale * grad
        return corrected
