# iSR: Joint Colorization + Super-Resolution for VR Gaming

This repository provides a minimal reference implementation of **iSR**, a joint pipeline for **stereo colorization + super-resolution** tailored to **VR gaming content**.

---

## Installation

### 1) Create conda environment
```bash
conda create --name iSR python=3.10
conda activate iSR
```

### 2) Install PyTorch (CUDA 12.1)
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3) Install remaining dependencies
From the repo root:
```bash
pip install -r requirements.txt
```

---

## Checkpoints

Model checkpoints are hosted on Google Drive (too large to include in the repository):

📁 **[Download checkpoints](https://drive.google.com/drive/folders/12ruw5OJn9GsTKiljB9fHj_CVRkOfyqZF?usp=sharing)**

Download the folders and place them as follows:

```
iSR/
├── checkpoints/
│   └── BeatSaber/
│       └── scale4Channels32.pth
└── checkpoints_SR/
    └── BeatSaber/
        └── scale4Channels48Block4.ckpt
```

---

## Quick start (Inference)

From the repo root:
```bash
python infer.py \
  --input_dir   Example_images/X4_stitch \
  --out_dir     results/ \
  --sicnet_ckpt checkpoints/BeatSaber/scale4Channels32.pth \
  --sr_ckpt     checkpoints_SR/BeatSaber/scale4Channels48Block4.ckpt
```

CPU only (no GPU required):
```bash
python infer.py \
  --input_dir   Example_images/X4_stitch \
  --out_dir     results/ \
  --sicnet_ckpt checkpoints/BeatSaber/scale4Channels32.pth \
  --sr_ckpt     checkpoints_SR/BeatSaber/scale4Channels48Block4.ckpt \
  --device cpu
```

### Arguments

| Argument | Description |
|---|---|
| `--input_dir` | Directory containing input SBS PNG images (see `Example_images/`) |
| `--out_dir` | Output directory — created automatically if missing |
| `--sicnet_ckpt` | Path to stereo colorization checkpoint (`.pth`) |
| `--sr_ckpt` | Path to super-resolution checkpoint (`.ckpt`) — omit to skip SR |
| `--device` | `cuda` (default) or `cpu` |
| `--overwrite` | Re-process images even if output already exists |

> **Input format:** inputs are stereo SBS (side-by-side) VR frames — left half is the grayscale source, right half is the RGB reference. Sample frames are provided in `Example_images/`.

---

## Repository structure

```
iSR/
├── infer.py                    # Inference entry point (CLI)
├── config.py                   # SICNet model config
├── config_ft.py                # SR model config
├── utils.py                    # Shared utilities (reparameterize, etc.)
├── model/                      # Stereo colorization model (SICNet)
├── model_SR/                   # Super-resolution model
├── Example_images/             # Sample SBS input frames
├── checkpoints/                # Colorization checkpoints (download separately)
├── checkpoints_SR/             # SR checkpoints (download separately)
├── requirements.txt            # Python dependencies
├── trt_build_color_and_sr.py   # Optional: ONNX export + TensorRT engine builder
└── Artifact.ipynb              # Reviewer notebook (environment setup + execution)
```

---

## Citation

If you use this repository in academic work, please cite the accompanying iSR paper.

---

## License

See `LICENSE`.
