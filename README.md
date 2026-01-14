# KinemaFlow: Structured Kinematic Flow Matching for Efficient Articulated Object Generation

<!-- <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-202X.XXXXX-b31b1b.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a> -->

<!-- > **Authors**: [Your Name], [Co-author Name], ...
> **Affiliation**: [Your Affiliation] -->

**KinemaFlow** is a novel generative framework for efficient and physically plausible 3D articulated object generation. By decoupling generation into a **Kinematics Stream** and a **Geometry Stream** via **Accelerated Flow Matching (AFT)**, we achieve state-of-the-art visual quality with significantly reduced inference latency (~200s). A **Physical Plausibility Flow Rectification** module further ensures collision-free interactions.

<p align="center">
  <img src="attachment\image.png" width="90%"/>
</p>

## 🌟 Key Features
- **⚡ Efficient Inference**: Generates high-fidelity articulated assets in ~200s (vs. 500s+ for baselines).
- **🔧 Physics-Aware**: Integrated energy-based *Physical Rectification* to prevent part interpenetration.
- **🎨 High Fidelity**: Leverages *Latent Distillation* for sharp texture and geometry details.
- **🧩 Decoupled Architecture**: Separate streams for Kinematics (motion) and Geometry (shape).

## 🛠️ Installation

The code is tested on Ubuntu 20.04 with Python 3.9 and PyTorch 2.0+ (CUDA 11.8).

```bash
# 1. Clone the repository
git clone https://github.com/BruceRichard/kinematic_flow.git
cd KinemaFlow

# 2. Create conda environment
conda env create -f env.yaml
conda activate kinemaflow

# 3. Install custom CUDA kernels (for efficient geometric processing)
cd utils/z_to_mesh/utils/libmcubes
python setup.py build_ext --inplace
cd ../libsimplify
python setup.py build_ext --inplace

```

## 📂 Data Preparation

We use the **PartNet-Mobility** dataset.

1. Download the dataset from [Sapien/PartNet-Mobility](https://sapien.ucsd.edu/downloads).
2. Preprocess the data to extract latents and kinematic chains:

```bash
# Extract ground truth latents using the frozen Latent Distiller (TRELLIS backbone)
python data/preprocessing/1_extract_from_raw_dataset.py --input_dir /path/to/partnet --output_dir ./dataset/processed

```

## 🚀 Training

KinemaFlow is trained in three stages. Configs are located in `configs/`.

### Stage 1: Latent Distillation (Geometry VAE)

Train the AutoEncoder to compress 3D geometry into a sparse latent space.

```bash
python train_stage1_vae.py --config configs/stage1_vae/train.yaml

```

### Stage 2: Geometry Stream (Flow Matching)

Train the diffusion-based flow matching model to generate geometry latents conditioned on text/kinematics.

```bash
python train_stage2_geometry_flow.py --config configs/stage2_geometry/train_with_flow_matching.yaml

```

### Stage 3: Kinematics Stream

Train the Transformer to predict kinematic parameters (joint types, axes, limits).

```bash
python train_stage3_kinematics.py --config configs/stage3_kinematics/text-train.yaml

```

## 🎮 Inference & Visualization

To generate an articulated object from a text prompt (e.g., "A wooden storage cabinet with two doors"):

```bash
# End-to-end generation with Physical Rectification enabled
python inference.py \
    --prompt "A red safe with a digital lock" \
    --output_dir outputs/demo \
    --rectification_scale 1.5 \
    --steps 5000

```

**Visualize Results:**
Generate a GIF of the articulated motion:

```bash
python viz_animate.py --input outputs/demo/safe.obj --output result.gif

```

## 📊 Evaluation

To evaluate Fidelity (FID, CD), Semantics (CLIP), and Physical Plausibility (CR):

```bash
python evaluate_metrics.py --pred_dir outputs/ --gt_dir dataset/test_set/

```
<!-- 
## 🔗 Citation

If you find our work useful, please cite:

```bibtex
@article{kinemaflow2026,
  title={KinemaFlow: Structured Kinematic Flow Matching for Efficient Articulated Object Generation},
  author={...},
  journal={arXiv preprint},
  year={2026}
}

``` -->