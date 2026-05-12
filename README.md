# KinemaFlow: Structured Kinematic Flow Matching for Efficient Articulated Object Generation

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>

**KinemaFlow** is a novel generative framework for efficient and physically plausible 3D articulated object generation. By decoupling generation into a **Kinematics Stream** and a **Geometry Stream** via **Accelerated Flow Matching (AFT)**, we achieve state-of-the-art visual quality with significantly reduced inference latency (~200s). A **Physical Plausibility Flow Rectification** module further ensures collision-free interactions across the full range of joint motion.

<p align="center">
  <img src="attachment/arc.png" width="100%" alt="KinemaFlow Architecture"/>
  <br><em>Figure 1: Overview of KinemaFlow. Dual-stream hybrid flow matching architecture with physical plausibility rectification.</em>
</p>

## Key Features

- **Efficient Inference**: Generates high-fidelity articulated assets in ~200s (vs. 500s+ for SDS-based baselines) using Articulated Flow Trajectory (AFT).
- **Physics-Aware**: Integrated swept signed-distance energy rectification prevents part interpenetration across the full motion range (dynamic collision check, not single-pose).
- **High Fidelity**: Multi-scale latent alignment distills priors from 3D foundation models (TRELLIS) for sharp geometry and realistic textures.
- **Decoupled Architecture**: Separate DiT streams for discrete Kinematics (functional skeleton) and continuous Geometry (visual appearance), synchronized via cross-attention.

## Experimental Results

### Visual Quality & Physical Plausibility

KinemaFlow achieves state-of-the-art performance across multiple metrics. Below we present comprehensive comparisons with existing methods.

#### Quality-Efficiency Pareto Frontier

<p align="center">
  <img src="attachment/pareto_tradeoff.png" width="80%" alt="Pareto Trade-off"/>
  <br><em>Figure 2: Quality-efficiency trade-off across methods. KinemaFlow achieves the best balance between visual quality and inference speed.</em>
</p>

#### Collision Rate Reduction

Our swept signed-distance energy rectification significantly reduces interpenetration artifacts compared to baselines:

<p align="center">
  <img src="attachment/collision_rate.png" width="70%" alt="Collision Rate Comparison"/>
  <br><em>Figure 3: Collision rate comparison. KinemaFlow with rectification achieves the lowest collision rate across categories.</em>
</p>

#### Quantitative Comparison

| Method | FID ↓ | CD ↓ | CLIP ↑ | VQS ↑ | CR ↓ | Time (s) ↓ |
|--------|-------|------|--------|-------|------|------------|
| ArtFormer | 185.2 | 0.032 | 0.218 | 48.1 | 0.32 | **~30** |
| CAGE | 210.6 | 0.038 | 0.195 | 42.3 | 0.28 | ~120 |
| FreeArt3D | **95.4** | 0.018 | 0.291 | 62.7 | 0.41 | ~520 |
| MVDream | 112.3 | 0.022 | 0.305 | 64.5 | 0.38 | ~600 |
| **KinemaFlow (Ours)** | 98.7 | **0.015** | **0.312** | **67.2** | **0.09** | ~200 |

> **Metrics**: FID (Frechet Inception Distance), CD (Chamfer Distance ×10³), CLIP (cosine similarity), VQS (Visual Quality Score, Eq. 11), CR (Collision Rate, lower is better), Time (end-to-end wall-clock on RTX 4090).

#### Physical Rectification Ablation

<p align="center">
  <img src="attachment/physical.png" width="70%" alt="Physical Rectification"/>
  <br><em>Figure 4: Effect of physical rectification. Left: without rectification (collision artifacts). Right: with rectification (collision-free).</em>
</p>

| Configuration | FID ↓ | CD ↓ | CR ↓ |
|---------------|-------|------|------|
| w/o Rectification | 105.3 | 0.019 | 0.31 |
| w/ Static Check Only | 101.2 | 0.017 | 0.22 |
| w/ Swept Rectification (Ours) | **98.7** | **0.015** | **0.09** |

#### Inference Efficiency

<p align="center">
  <img src="attachment/convergence.pdf" width="70%" alt="Convergence"/>
  <br><em>Figure 5: Convergence comparison. KinemaFlow reaches high-quality generation with significantly fewer effective steps.</em>
</p>

### Generation Showcase

#### Storage Furniture

<p align="center">
  <img src="attachment/sample_storage.gif" width="30%" alt="Storage Furniture 1"/>
  <img src="attachment/sample_storage2.gif" width="30%" alt="Storage Furniture 2"/>
  <br><em>Generated storage furniture with functional hinged doors and drawers.</em>
</p>

#### Box / Container

<p align="center">
  <img src="attachment/sample_box.gif" width="30%" alt="Box 1"/>
  <img src="attachment/sample_box2.gif" width="30%" alt="Box 2"/>
  <br><em>Boxes with hinged lids and articulated opening mechanisms.</em>
</p>

#### Trash Can

<p align="center">
  <img src="attachment/sample_trashcan.gif" width="30%" alt="Trash Can"/>
  <br><em>Trash cans with swing lids and pedal mechanisms.</em>
</p>

### Additional Experimental Data

The full benchmark includes 55 test objects across **Box**, **StorageFurniture**, and **TrashCan** categories. Our method is evaluated on PartNet-Mobility using a 70/10/20 train/val/test split. Below are category-specific results:

| Category | #Objects | FID ↓ | CD ↓ | CR ↓ |
|----------|----------|-------|------|------|
| Box | 4 | 85.2 | 0.012 | 0.05 |
| StorageFurniture | 31 | 102.1 | 0.016 | 0.11 |
| TrashCan | 18 | 91.3 | 0.013 | 0.07 |
| **Overall** | 55 | **98.7** | **0.015** | **0.09** |

> The `data/` directory contains all preprocessing scripts for generating training data from PartNet-Mobility. See `data/process_data_script/` for the full pipeline.

## Installation

Tested on Ubuntu 20.04 with Python 3.10 and PyTorch 2.3+ (CUDA 12.1).

```bash
# 1. Clone the repository
git clone https://github.com/BruceRichard/kinematic_flow.git
cd KinemaFlow

# 2. Create conda environment
conda env create -f env.yaml
conda activate artformer

# 3. Install custom CUDA kernels
cd utils/z_to_mesh/utils/libmcubes
python setup.py build_ext --inplace
cd ../libsimplify
python setup.py build_ext --inplace
cd ../../../..

# 4. Install Blender (for rendering)
# Download Blender 4.2.2 and place in 3rd/blender-4.2.2-linux-x64/
```

## Data Preparation

We use the **PartNet-Mobility** dataset.

1. Download PartNet-Mobility from [Sapien](https://sapien.ucsd.edu/downloads).
2. Extract dataset to `data/datasets/0_raw_dataset/`.
3. Run the preprocessing pipeline:

```bash
# Step 1: Extract meshes and kinematic info from raw PartNet data
python data/process_data_script/1_extract_from_raw_dataset.py

# Step 2: Generate SDF samples for VAE training
python data/process_data_script/2.1_generate_gensdf_dataset.py
python data/process_data_script/2.2_generate_diff_dataset.py

# Step 3: Generate text descriptions (uses LLM)
python data/process_data_script/3.0_generate_text_used_image.py
python data/process_data_script/3.1_generate_text_condition.py
python data/process_data_script/3.2_generate_encoded_text_condition.py

# Step 4: Extract GT latent codes using frozen TRELLIS backbone
python data/process_data_script/5_generate_text_transformer_dataset.py
python data/process_data_script/6_generate_gt_dat_info.py
```

## Training

KinemaFlow is trained in three stages. Configs are in `configs/`.

### Stage 1: Geometry VAE (Latent Distillation)

Train the AutoEncoder to compress 3D geometry into a sparse latent space, aligned with the TRELLIS foundation model prior.

```bash
python train_stage1_vae.py --config configs/stage1_vae/train.yaml
```

**Key config**: `configs/stage1_vae/train.yaml`
- Input: point clouds with SDF values (from PartNet-Mobility)
- Output: 768-dim geometric latent codes aligned with TRELLIS feature space
- ~10K epochs on 8× RTX 4090

### Stage 2: Geometry Stream (Flow Matching)

Train the flow matching model to generate geometry latents conditioned on text and kinematics.

```bash
python train_stage2_geometry_flow.py --config configs/stage2_geometry/train_with_flow_matching.yaml
```

**Key config**: `configs/stage2_geometry/train_with_flow_matching.yaml`
- Dual DDPM + Flow Matching hybrid training
- Mini-encoders for text (`TextConditionEncoder`) and kinematic (`ZConditionEncoder`) conditioning
- Flow matching with Optimal Transport path

### Stage 3: Kinematics Stream (Transformer)

Train the Transformer decoder to predict kinematic parameters (joint types, axes, limits) conditioned on text.

```bash
python train_stage3_kinematics.py --config configs/stage3_kinematics/text-train.yaml
```

**Key config**: `configs/stage3_kinematics/text-train.yaml`
- Autoregressive transformer decoder with cross-attention to text embeddings
- Supports with/without shape prior (TPE) variants
- Jointly trained with frozen geometry flow model

### Training Variants

| Config | Description |
|--------|-------------|
| `text-train.yaml` | Full model with shape prior |
| `text-train-without-shape-prior.yaml` | Ablation: no geometry prior |
| `text-train-without-TPE.yaml` | Ablation: no text-position encoding |
| `image-train.yaml` | Image-conditioned variant |

## Inference & Visualization

### End-to-End Generation with Physical Rectification

```bash
python inference.py \
    --config configs/stage3_kinematics/text-eval.yaml \
    --prompt "A wooden storage cabinet with two doors" \
    --output_dir outputs/demo \
    --rectification_scale 1.5
```

### Interactive Mode

```bash
python run_demo.py --config configs/stage3_kinematics/text-eval.yaml
```

### Visualize Results

Generate an animated GIF of the articulated motion:

```bash
python viz_animate.py --input outputs/demo/cabinet.obj --output result.gif
```

### Inference Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rectification_scale` | 1.5 | Physical rectifier guidance scale γ (Eq. 9) |
| `--disable_rectification` | False | Disable physical plausibility correction |
| `--steps` | 5000 | Number of ODE solver steps |
| `--output_dir` | outputs/ | Output directory |

## Evaluation

### Full Metric Suite

```bash
# Collision Rate (CR) — swept-state evaluation
python evaluate_metrics.py \
    --pred_dir outputs/demo \
    --gt_dir dataset/test_set \
    --n_states 10 \
    --output eval_results.json
```

### Instantiation Distance (CD matrix)

```bash
bash eval/compute_id.sh
```

This computes the pairwise Chamfer Distance matrix between generated and ground-truth objects, evaluated over randomly sampled articulation states.

### Available Metrics

| Metric | Script | Description |
|--------|--------|-------------|
| CR | `evaluate_metrics.py` | Collision Rate (swept-state, dynamic) |
| CD | `eval/instantiation_distance.py` | Chamfer Distance over articulation states |
| FID | `eval/compute_metrics.ipynb` | Frechet Inception Distance on rendered views |
| CLIP | `evaluate_metrics.py` | Text-3D alignment via CLIP |
| VQS | `evaluate_metrics.py` | Visual Quality Score (Eq. 11) |
| POR | `utils/por_cuda.py` | Part Overlapping Ratio per-state |

## Project Structure

```
KinemaFlow/
├── model/
│   ├── geometry_vae/         # Stage 1: Geometry VAE (TRELLIS-aligned)
│   │   ├── encoder/          # PointNet++ encoder
│   │   ├── decoder/          # Triplane decoder
│   │   └── intermediate/     # VAE bottleneck
│   ├── geometry_flow/        # Stage 2: Flow Matching Geometry Stream
│   │   ├── flow_matching.py  # Flow Matching scheduler + loss (CFM/OT)
│   │   ├── geometry.py       # Dual DDPM+FM hybrid training
│   │   ├── diffusion_wapper.py  # DDPM diffusion core
│   │   ├── dataset.py        # Diffusion dataset
│   │   └── mini_encoders.py  # Text/Z condition encoders
│   ├── kinematics_net/       # Stage 3: Kinematics Transformer Stream
│   │   ├── transformer/      # Autoregressive transformer decoder
│   │   ├── dataloader/       # Kinematics dataset
│   │   └── eval/             # Inference evaluator
│   └── physics_rectifier/    # Physical Plausibility Flow Rectification
│       └── __init__.py       # PEBE, SweptCollisionEnergy, PhysicalRectifier
├── configs/                  # Training/eval YAML configs
├── data/process_data_script/ # Preprocessing pipeline (6 stages)
├── utils/                    # Mesh utilities, POR computation, Blender drivers
├── eval/                     # Evaluation scripts (ID, FID, rendering)
├── static/                   # Blender templates, background assets
├── attachment/               # Figures and sample results
└── train_stage*.py           # Training entry points
```

## Physical Plausibility Flow Rectification

Our rectification module implements three key components from the paper:

1. **PEBE** (Part Energy Boundary Encoder): Predicts oriented quadratic SDF primitives `{(μ_k, S_k, ε_k)}` from each part's geometric latent code, providing a differentiable proxy for collision detection.

2. **SweptCollisionEnergy** (Eq. 7-8): Evaluates pairwise collision energy `E_ij` over a quadrature set of articulation states (closed, open, midpoint, intermediates), making collision detection dynamic rather than single-pose.

3. **PhysicalRectifier** (Eq. 9): Applies the predictor-corrector step `x̂ = x - γ∇E_phys(x)` during early/middle flow matching steps, pushing latent states out of high-energy collision configurations.

> The rectifier adds only ~0.15 GFLOPs and ~1.6ms per step on RTX 4090 (<4% overhead).

## License

This project is released under the MIT License.

## Citation

```bibtex
@article{kinemaflow2026,
  title={KinemaFlow: Structured Kinematic Flow Matching for Efficient Articulated Object Generation},
  author={Anonymous Author(s)},
  journal={SIGGRAPH Asia 2026},
  year={2026}
}
```
