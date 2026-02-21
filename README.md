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

## Quick start (Inference)

From the repo root:
```bash
python infer.py   --input_dir Example_images/X4_stitch   --out_dir results/   --sicnet_ckpt checkpoints/BeatSaber/scale4Channels32.pth   --sr_ckpt checkpoints_SR/BeatSaber/scale4Channels48Block4.ckpt
```
```CPU-Only
python infer.py   --input_dir Example_images/X4_stitch   --out_dir results/   --sicnet_ckpt checkpoints/BeatSaber/scale4Channels32.pth   --sr_ckpt checkpoints_SR/BeatSaber/scale4Channels48Block4.ckpt --device cpu
```

### Arguments
- `--input_dir`: directory containing input images (see `Example_images/`)
- `--out_dir`: output directory (created if missing)
- `--sicnet_ckpt`: stereo colorization checkpoint (`.pth`)
- `--sr_ckpt`: super-resolution checkpoint (`.ckpt`)

> Note: Input format depends on `infer.py`. In this project, inputs are typically **stereo/SBS (side-by-side) VR frames**. If your inputs differ, adjust preprocessing in `infer.py` accordingly.

---

## Repository structure (high level)

- `infer.py` : inference entry point
- `model/` : stereo colorization model code (SICNet/SiCNet-based)
- `model_SR/` : super-resolution model code
- `checkpoints/` : colorization checkpoints (avoid committing)
- `checkpoints_SR/` : SR checkpoints (avoid committing)
- `config.py`, `config_ft.py` : model/config settings
- `trt_build_color_and_sr.py` : ONNX export + TensorRT engine builder (optional)


---

## Citation

If you use this repository in academic work, please cite the accompanying iSR paper.

---

## License

See `LICENSE`.
