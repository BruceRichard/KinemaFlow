# [ArtFormer]: This file contains the class to make the inference (generation of articulated object) from text or image condition.
import os
import copy
import torch
import json
import time
import random
import pickle

from pathlib import Path

import torch.utils
import torch.nn.functional as F
from tqdm import trange
# from rich import print
from transformers import AutoTokenizer, T5EncoderModel
from ..dataloader import TransDiffusionDataset
from .. import TransDiffusionCombineModel
from models.geometry_vae import GeometryVAE

from utils import untokenize_part_info, generate_gif_toy, fit_into_bounding_box
from utils.por_cuda import POR, get_trans_matrix, apply_transformations
import utils.mesh as MeshUtils
from utils.mylogging import Log
from utils.z_to_mesh import GenSDFLatentCodeEvaluator

from models.physics_rectifier import PEBE, SweptCollisionEnergy, PhysicalRectifier, transform_primitive

import sys
sys.path.append('../../..')
from eval.visualize import visualize_obj_high_q

class Evaluater():
    def __init__(self, eval_config):
        self.eval_config = eval_config
        self.device = eval_config['device']
        self.number_of_trial = self.eval_config['number_of_trial']

        Log.info("Loading model %s", TransDiffusionCombineModel)
        self.model = TransDiffusionCombineModel.load_from_checkpoint(eval_config['checkpoint_path'])
        self.model.eval()
        # self.models.geometry_flow.model.cond_dropout = False
        self.m_config = self.model.config

        self.z_mini_encoder = self.model.z_mini_encoder

        d_configs = self.m_config['dataset_n_dataloader']

        self.dataset = TransDiffusionDataset(dataset_path=d_configs['dataset_path'],
                cut_off=d_configs['cut_off'],
                enc_data_fieldname=d_configs['enc_data_fieldname'],
                cache_data=False)

        self.eval_output_path = Path(self.eval_config['eval_output_path']) / time.strftime("%m-%d-%I%p-%M-%S")
        os.makedirs(self.eval_output_path, exist_ok=True)

        self.start_token = copy.deepcopy(self.dataset.start_token).to(self.device)
        self.end_token = copy.deepcopy(self.dataset.end_token).to(self.device)

        Log.info("Loading model %s", T5EncoderModel)
        self.tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-large', cache_dir='cache/t5_cache')
        self.text_encoder = T5EncoderModel.from_pretrained('google-t5/t5-large', cache_dir='cache/t5_cache').to(self.device)
        #TODO: check need to do self.text_encoder.eval() or not
        self.text_encoder.eval()
        self.t5_max_sentence_length = self.eval_config['t5_max_sentence_length']

        # self.equal_part_threshold = self.eval_config['equal_part_threshold']

        # self.latentcode_evaluator = LatentCodeEvaluator(Path(self.dataset.get_onet_ckpt_path()), 100000, 16, self.device)

        Log.info("Loading model %s", GeometryVAE)
        self.gensdf_config = self.eval_config['gensdf_latentcode_evaluator']
        # self.gensdf_config['gensdf_model_path'] = self.dataset.get_best_sdf_ckpt_path()
        self.ali = self.model.ali # GeometryVAE.load_from_checkpoint(self.gensdf_config['gensdf_model_path'])
        self.ali.eval()
        self.latentcode_evaluator = GenSDFLatentCodeEvaluator(self.ali, eval_mesh_output_path=self.eval_output_path,
                                                             resolution=self.gensdf_config['resolution'],
                                                             max_batch=self.gensdf_config['max_batch'],
                                                             device=self.device)

        # Physical Plausibility Rectification components
        self.pebe = PEBE(latent_dim=768, num_primitives=8).to(self.device)
        pebe_ckpt = self.eval_config.get('pebe_checkpoint_path', None)
        if pebe_ckpt and Path(pebe_ckpt).exists():
            Log.info("Loading PEBE checkpoint from %s", pebe_ckpt)
            self.pebe.load_state_dict(torch.load(pebe_ckpt, map_location=self.device))
        self.pebe.eval()
        self.collision_energy = SweptCollisionEnergy(
            tau=self.eval_config.get('collision_tau', 0.05),
            num_contact_samples=self.eval_config.get('contact_samples', 256)
        )
        self.rectifier = PhysicalRectifier(
            self.pebe, self.collision_energy,
            guidance_scale=self.eval_config.get('rectification_scale', 1.5),
            active_ratio=0.8
        )

    def encode_text(self, text):
        input_ids = self.tokenizer([text], return_tensors="pt", padding='max_length',
                                    max_length=self.t5_max_sentence_length).input_ids
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(input_ids)
        encoded_text = outputs.last_hidden_state.detach()
        return encoded_text

    def generate_non_padding_mask(self, len):
        return torch.ones(1, len).to(self.device)

    def is_end_token(self, token):
        length = token.size(0)
        difference = torch.nn.functional.mse_loss(token[:length], self.end_token[:length])
        Log.info('    - Difference with end token: %s', difference.item())
        return difference < self.equal_part_threshold

    def _build_articulation_transforms(self, processed_nodes: list,
                                        n_states: int = 5) -> list:
        """Build per-state SE(3) transforms for all parts.

        States are sampled from joint limits evenly: closed(0), open(1),
        midpoint(0.5), and intermediates.

        Returns:
            list of dicts: [ {part_idx: [1, 4, 4]}, ... ] per state
        """
        ratios = torch.linspace(0, 1, n_states)
        id_to_fa = {}
        for node in processed_nodes:
            cur_id = node.get('dfn', 0)
            fa = node.get('dfn_fa', 0)
            id_to_fa[cur_id] = fa

        transforms_per_state = []
        for ratio in ratios:
            M_dict = {}
            for part in processed_nodes:
                cur_id = part.get('dfn', 0)
                M = get_trans_matrix(part, ratio.item())
                M_dict[cur_id] = M

            keys = sorted(M_dict.keys())
            for cur_id in keys:
                if cur_id != 0 and id_to_fa.get(cur_id, 0) in M_dict:
                    fa = id_to_fa[cur_id]
                    M_dict[cur_id] = M_dict[cur_id] @ M_dict[fa]

            batched = {k: v.unsqueeze(0) for k, v in M_dict.items()}
            transforms_per_state.append(batched)

        return transforms_per_state

    def _compute_primitives_from_nodes(self, processed_nodes: list) -> list:
        """Compute PEBE primitives for each part from latent codes.

        Returns:
            list of dicts with 'mu', 'S', 'epsilon' per part
        """
        primitives = []
        for part in processed_nodes:
            if 'z' in part:
                z = part['z'].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    prim = self.pebe(z)
                primitives.append({
                    'mu': prim['mu'],
                    'S': prim['S'],
                    'epsilon': prim['epsilon'],
                })
            else:
                primitives.append({
                    'mu': torch.zeros(1, 8, 3, device=self.device),
                    'S': torch.eye(3, device=self.device).view(1, 1, 3, 3).expand(1, 8, 3, 3),
                    'epsilon': torch.ones(1, 8, device=self.device),
                })
        return primitives

    def _rectify_latents(self, latent_codes: torch.Tensor,
                          processed_nodes: list,
                          step: int, total_steps: int) -> torch.Tensor:
        """Apply physical rectification corrector step.

        x̂ = x - γ ∇E_phys(x)  (Eq. 9)
        """
        primitives = self._compute_primitives_from_nodes(processed_nodes)
        parent_indices = [n.get('dfn_fa', 0) for n in processed_nodes]
        transforms_per_state = self._build_articulation_transforms(
            processed_nodes, n_states=5
        )
        return self.rectifier.correct(
            latent_codes, primitives, transforms_per_state,
            parent_indices, step, total_steps
        )

    def inference_from_text(self, text, enc_data=None, need_mesh=True):
        Log.info('[1] Inference text: %s', len(text))
        if enc_data is None:
            encoded_text = self.encode_text(text)
        else:
            encoded_text = enc_data.unsqueeze(0).to(self.device)

        exist_node = {
            'fa': torch.tensor([0]).to(self.device),
            'token': copy.deepcopy((self.start_token[:16])).unsqueeze(0).to(self.device),
            'text_hat': torch.zeros((64)).unsqueeze(0).to(self.device),
            'z_hat': torch.zeros((4, 768)).unsqueeze(0).to(self.device),
            'latent': torch.zeros((768)).unsqueeze(0).to(self.device)
        }
        round = 1
        Log.info('[2] Generate nodes')
        atten_weights_list = []

        use_shape_prior = True
        while True:
            current_length = exist_node['token'].size(0)
            Log.info('   - Generate nodes round: %s, part count: %s', round, exist_node['token'].size(0))
            with torch.no_grad():
                # input: (batch, seq, xxx) ---> (batch|seq, xxx) base on `padding_mask`, the dimension of batch & seq are merged.
                # batch=1 for evaluation.
                output = self.models.kinematics_net({
                                'fa': exist_node['fa'].unsqueeze(0),        # batched.
                                'token': torch.cat((exist_node['token'], exist_node['text_hat']), dim=1).unsqueeze(0),
                            },
                            self.generate_non_padding_mask(current_length),
                            encoded_text) # unbatched.
            atten_weights_list.append(output['cross_attn_weight_list'])
            # Solve End Token.
            # True -> not end token, False -> end token
            end_token_mask = output['is_end_token_logits'] > 0
            Log.info('   - Check end token: %s', output['is_end_token_logits'])
            Log.info('   - Check end token mask: %s', end_token_mask)
            if not torch.any(end_token_mask):
                break

            articulated_info = output['articulated_info'][end_token_mask]

            condition = output['condition']
            if isinstance(condition, dict):
                pred_text_hat = condition['text_hat'][end_token_mask] # torch.Size([1, 64])
                pred_z_logits = condition['z_logits'][end_token_mask] # torch.Size([1, 4, 128])
                q_z, _KL, _perplexity, _logits = self.z_mini_encoder.forward_with_logits_or_x(tau=0.5, logits=pred_z_logits)
                latent_code = None
            else:
                latent_code = condition[end_token_mask] # torch.Size([1, 768])
                pred_text_hat = torch.zeros((64)).unsqueeze(0).to(self.device)
                q_z = None
                use_shape_prior = False

            result = articulated_info

            fa_idx = torch.arange(end_token_mask.shape[0], device=self.device)
            fa_idx = fa_idx[end_token_mask]

            exist_node['fa'] = torch.cat((exist_node['fa'], fa_idx), dim=0)
            exist_node['token'] = torch.cat((exist_node['token'], result), dim=0)

            if pred_text_hat is not None:   exist_node['text_hat'] = torch.cat((exist_node['text_hat'], pred_text_hat), dim=0)
            if q_z is not None:             exist_node['z_hat'] = torch.cat((exist_node['z_hat'], q_z), dim=0)
            if latent_code is not None:     exist_node['latent'] = torch.cat((exist_node['latent'], latent_code), dim=0)


        Log.info('[3] reconstruct latent code with condition')
        if use_shape_prior:
            cond = {
                'z_hat': exist_node['z_hat'],
                'text': exist_node['text_hat'],
            }
            latent = self.models.geometry_flow.model.generate_conditional(cond)

            enable_rectification = self.eval_config.get('enable_rectification', True)
            if enable_rectification:
                temp_process = []
                for idx in range(exist_node['fa'].shape[0]):
                    token_data = exist_node['token'][idx].cpu().tolist()
                    part_info = untokenize_part_info(token_data)
                    part_info['dfn'] = idx
                    part_info['dfn_fa'] = exist_node['fa'][idx].item()
                    temp_process.append(part_info)

                if len(temp_process) > 1:
                    Log.info('   - Applying physical rectification on latents')
                    rectified_latent = self._rectify_latents(
                        latent, temp_process, step=0, total_steps=100
                    )
                    latent = rectified_latent

            exist_node['token'] = torch.cat((exist_node['token'], latent), dim=-1)
        else:
            exist_node['token'] = torch.cat((exist_node['token'], exist_node['latent']), dim=-1)

        processed_nodes = []
        Log.info('[4] Generate mesh')

        for idx in trange(exist_node['fa'].shape[0], desc='   - Generate mesh'):
            dfn_fa = exist_node['fa'][idx].item()
            token  = exist_node['token'][idx].cpu().tolist()
            processed_node = {
                'dfn': idx,
                'dfn_fa': dfn_fa,
            }
            part_info = untokenize_part_info(token)

            z = torch.tensor(part_info['latent_code']).to(self.device)
            if need_mesh: part_info['mesh'] = self.latentcode_evaluator.generate_mesh(z.unsqueeze(0))
            # import pdb; pdb.set_trace()
            part_info['z'] = z
            # raw_points_sdf, rho = self.latentcode_evaluator.generate_uniform_point_cloud_inside_mesh(z.unsqueeze(0))
            # part_info['points'], part_info['rho'] = fit_into_bounding_box(raw_points_sdf, rho, part_info['bbx'])

            processed_node.update(part_info)
            processed_nodes.append(processed_node)

        # import pdb; pdb.set_trace()

        # We do not want start token.
        return processed_nodes[1:], atten_weights_list

    def inference_to_output_path(self, text, output_path, enc_data=None, blender_generated_gif=False):
        output_path.mkdir(exist_ok=True, parents=True)
        processed_nodes, atten_weights_list = self.inference_from_text(text, enc_data)

        # for debug only.
        # processed_nodes, atten_weights_list = pickle.load(open('/ssd1/dengzhidong/.sym/final/ArtFormer/elog/Final_OP1_05-27-01PM-29-48/StorageFurniture_45243_1/0/output.dat', 'rb')), None

        # output_data_path = output_path / "output.dat"
        # with open(output_data_path, 'wb') as f: f.write(pickle.dumps(processed_nodes))
        # Log.info("[Write] %s", output_data_path)

        output_tex_path = output_path / "input.txt"
        output_tex_path.write_text(text)
        Log.info("[Write] %s", output_tex_path)

        output_gif_path = output_path / "gif"
        generate_gif_toy(processed_nodes, output_gif_path, bar_prompt="   - Generate Frames", blender_generated_gif=blender_generated_gif)
        Log.info("[Write] %s", output_gif_path)

        # output_temp_path : Path = output_path / "temp"
        # output_temp_path.mkdir(exist_ok=True, parents=True)
        # Log.info("[Write] %s", output_temp_path)

        # for ratio in [0, 0.5, 1]:
        #     visualize_obj_high_q(processed_nodes, output_temp_path / str(ratio), output_path / str(ratio), ratio)

        # return atten_weights_list

    def inference_dat_file_only(self, text, output_dat_path, enc_data=None):
        processed_nodes, atten_weights_list = self.inference_from_text(text, need_mesh=False, enc_data=enc_data)
        with open(output_dat_path, 'wb') as f:
            f.write(pickle.dumps(processed_nodes))

    def inference(self, text):
        number_of_trial = self.number_of_trial
        list_processed_nodes = [None] * number_of_trial
        for trial in trange(number_of_trial, desc="Doing trial"):
            processed_nodes = self.inference_from_text(text)
            list_processed_nodes[trial] = {
                'data': processed_nodes,
                'rate': POR(processed_nodes, n_sample=8192),
            }
            rate = list_processed_nodes[trial]['rate']
            output_gif_path = (Path(self.eval_output_path) / f'output_{trial}_{rate}.gif')
            Log.info('[4] Generate Gif: %s', output_gif_path.as_posix())

            generate_gif_toy(processed_nodes, output_gif_path,
                            bar_prompt="   - Generate Frames")
            Log.info('[5] Done')

        output_json_path = (Path(self.eval_output_path) / f'output.json')
        output_json_path.write_text('{"text": "' + text + '"}')

        output_data_path = (Path(self.eval_output_path) / f'output.data')
        with open(output_data_path, 'wb') as f:
            f.write(pickle.dumps(list_processed_nodes))
        Log.info("Saved data checkpoint %s.", output_data_path.as_posix())