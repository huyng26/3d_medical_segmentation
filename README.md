# 3D Medical Image Segmentation (IT4343E)

This project reimplements three 3D medical image segmentation architectures from scratch to gain a deeper understanding of how Computer Vision is applied to volumetric medical data.

**Implemented models:**
- 3D U-Net
- SkipDenseNet3D (Attention U-Net variant)
- Swin UNETR (wrapping MONAI's pretrained Swin Transformer encoder)

---

## Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- CUDA-capable GPU (recommended; CPU-only mode also works)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/huyng26/3d_medical_segmentation.git
cd 3d_medical_segmentation
```

### 2. Create a virtual environment with uv

```bash
uv venv .venv --python 3.10
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install the package and all dependencies

```bash
uv pip install -e .
```

This installs `medseg` as an editable package, so any code changes take effect immediately without reinstalling.

To also install development tools (pytest, ruff):

```bash
uv pip install -e ".[dev]"
```

---

## Data Preparation

### BTCV (Multi-Atlas Labeling Beyond the Cranial Vault)

Download from [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) and arrange as:

```
data/BTCV/
├── imagesTr/   # training volumes  (.nii.gz)
├── labelsTr/   # training labels   (.nii.gz)
├── imagesTs/   # test volumes      (.nii.gz)
└── labelsTs/   # test labels       (.nii.gz)
```

### MSD (Medical Segmentation Decathlon)

Download Task02 (Heart) or Task09 (Spleen) from the [MSD website](http://medicalsegmentationdecathlon.com/) and arrange as:

```
data/MSD/
├── Task02_Heart/
│   ├── imagesTr/
│   ├── labelsTr/
│   └── imagesTs/
└── Task09_Spleen/
    ├── imagesTr/
    ├── labelsTr/
    └── imagesTs/
```

---

## Training

```bash
python -m medseg.training.train \
    --dataset       btcv \
    --data_path     data/BTCV \
    --model_name    swin_unetr \
    --num_classes   14 \
    --in_channels   1 \
    --img_size      96 96 96 \
    --batch_size    2 \
    --num_workers   4 \
    --num_epochs    100 \
    --lr            1e-4 \
    --weight_decay  1e-5 \
    --amp
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | `btcv` or `msd` |
| `--data_path` | — | Path to the dataset root |
| `--model_name` | *(required)* | `unet3d`, `attention_unet`, or `swin_unetr` |
| `--num_classes` | *(required)* | Number of segmentation classes (14 for BTCV) |
| `--in_channels` | `1` | Input image channels (1 for CT) |
| `--img_size` | `96 96 96` | Patch size D H W fed to the model |
| `--batch_size` | `2` | Training batch size |
| `--num_workers` | `4` | DataLoader worker count |
| `--num_epochs` | `100` | Number of training epochs |
| `--lr` | `1e-4` | Initial learning rate |
| `--weight_decay` | `1e-5` | AdamW weight decay |
| `--amp` | `False` | Enable automatic mixed precision |
| `--pretrain` | `""` | Path to pretrained Swin encoder weights |
| `--save_dir` | `./checkpoints` | Directory to save checkpoints |
| `--msd_task` | `2` | MSD task: `2` = Heart, `9` = Spleen |

---

## Project Structure

```
3d_medical_segmentation/
├── medseg/
│   ├── cfg.py              # Argument parser (all CLI flags)
│   ├── models/
│   │   ├── __init__.py     # build_model() factory
│   │   ├── unet3d.py
│   │   ├── attention_unet.py
│   │   └── swin_unetr.py
│   ├── data_utils/
│   │   ├── btcv.py         # BTCV dataloader
│   │   ├── msd.py          # MSD dataloader
│   │   └── transforms.py   # MONAI augmentation pipelines
│   ├── training/
│   │   └── train.py        # Training loop entry point
│   ├── inference/          # Sliding-window inference utilities
│   ├── evaluation/         # Metrics and evaluation scripts
│   └── utils/              # Shared helpers
├── configs/                # YAML config files
├── data/                   # Dataset root (not tracked by git)
├── tests/                  # pytest test suite
├── pyproject.toml
└── README.md
```

---

## Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=medseg
```