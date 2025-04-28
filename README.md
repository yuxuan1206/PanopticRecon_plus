<h1 align="center">PanopticRecon++: Leverage Cross-Attention <br>for End-to-End Open-Vocabulary Panoptic Reconstruction</h1>
<p align="center"><a href="https://arxiv.org/abs/2501.01119"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://yuxuan1206.github.io/panopticrecon_pp/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
</p>

## Installation

```bash
# Create conda environment
conda create -n pr python=3.8
conda activate pr
# Requires Python 3.8, PyTorch 1.11.0, CUDA 11.3

# Install required packages
pip install -e third-party/GrounedingDINO
pip install -e third-party/segment_anything

# Download models
cd third-party/GrounedingDINO
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
cd ../segment_anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Data Processing
(Same as [PanopticRecon: Leverage Open-vocabulary Instance Segmentation for Zero-shot Panoptic Reconstruction](https://yuxuan1206.github.io/PanopticRecon/))

### Generate 2D semantic, instance and panoptic masks using [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)

```bash
python scripts/groundsam_mask.py
```

The script supports command line parameters for flexible configuration:

```bash
python groundsam_mask.py --dataset scannet --scene 0087_02 --box_threshold 0.25
```

Available parameters:
- `--dataset`: Dataset name (e.g., scannet, scannet++, etc.)
- `--scene`: Scene ID
- `--dataset_root`: Dataset root path
- `--box_threshold`: Box threshold for GroundingDINO
- `--text_threshold`: Text threshold for GroundingDINO
- `--instance_threshold`: Instance threshold
- `--dino_config`: GroundingDINO config file path
- `--dino_model`: GroundingDINO model weights path
- `--sam_checkpoint`: SAM model checkpoint path

Example scene configurations are defined in [scene_config.py](data/scene_config.py), which can be modified or extended as needed.

