#!/usr/bin/env python3
"""
infer.py — Single-file PyTorch inference for SBS colorization + super-resolution.

Input:  folder of SBS PNG images (side-by-side, 500×250)
          left  half = grayscale source
          right half = RGB reference

Output: folder of colorized SBS PNG images
          left  half = colorized prediction
          right half = SR-upscaled prediction  (if --sr_ckpt provided)
                       or original right half   (if SR is skipped)

Usage:
  # Colorization only
  python infer.py \\
      --input_dir  frames/ \\
      --out_dir    results/ \\
      --sicnet_ckpt checkpoints/sicnet.pth

  # Colorization + SR
  python infer.py \\
      --input_dir  frames/ \\
      --out_dir    results/ \\
      --sicnet_ckpt checkpoints/sicnet.pth \\
      --sr_ckpt     checkpoints/sr.ckpt

  # CPU-only (slow but no GPU required)
  python infer.py --input_dir frames/ --out_dir results/ \\
      --sicnet_ckpt checkpoints/sicnet.pth --device cpu
"""

import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# ---------- safe globals for torch.load(weights_only=True) ----------
try:
    from torch.serialization import add_safe_globals
except ImportError:
    # Torch < 2.4 doesn't have add_safe_globals
    def add_safe_globals(*args, **kwargs):
        return None
from yacs.config import CfgNode
from collections import defaultdict
add_safe_globals([CfgNode, set, frozenset, defaultdict])

# ---------- repo imports ----------
import model.SICNet as SICNet
import config as sic_config
from model_SR import LitRT4KSR_Rep
import config_ft as sr_config
from utils import reparameterize


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sicnet(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load SICNet colorization model from checkpoint."""
    cfg = sic_config.get_config()
    net = SICNet.SICNet(channels=cfg.MODEL.channels)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    print(f"[ok] Loaded SICNet from {ckpt_path}")
    return net.to(device).eval()


def load_sr(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load SR model (reparameterized) from Lightning checkpoint."""
    lit = LitRT4KSR_Rep.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        config=sr_config,
    )
    lit.model = reparameterize(sr_config, lit.model, device, save_rep_checkpoint=False)
    print(f"[ok] Loaded SR model from {ckpt_path}")
    return lit.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def collect_images(folder: Path):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMG_EXTS)


def load_sbs(img_path: Path):
    """
    Load a side-by-side image and split at the midpoint.

    Returns:
        left_gray  – PIL Image (L mode)
        right_rgb  – PIL Image (RGB mode)
        orig_size  – (width, height) of the full SBS image
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    mid = w // 2
    left_gray = img.crop((0,   0, mid, h)).convert("L")
    right_rgb = img.crop((mid, 0, w,   h)).convert("RGB")
    return left_gray, right_rgb, (w, h)


def pil_to_tensor(pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL Image → float32 tensor on device, shape (1, C, H, W), range [0, 1]."""
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    if arr.ndim == 2:                     # grayscale → (1, H, W)
        arr = arr[None, ...]
    else:                                 # RGB → (3, H, W)
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, C, H, W)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Float32 tensor (1, C, H, W) or (C, H, W) → PIL Image (uint8)."""
    t = t.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    arr = (t.numpy() * 255).astype(np.uint8)
    if arr.shape[0] == 1:
        return Image.fromarray(arr[0], mode="L").convert("RGB")
    return Image.fromarray(arr.transpose(1, 2, 0), mode="RGB")


def make_sbs(left: Image.Image, right: Image.Image) -> Image.Image:
    """Stitch two same-height RGB images into a single SBS image."""
    assert left.height == right.height, \
        f"Height mismatch: {left.height} vs {right.height}"
    canvas = Image.new("RGB", (left.width + right.width, left.height))
    canvas.paste(left,  (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_colorization(sicnet: torch.nn.Module,
                     left_gray: Image.Image,
                     right_rgb: Image.Image,
                     device: torch.device):
    """
    Run SICNet on a single (left_gray, right_rgb) pair.

    Returns:
        left_col   – colorized left  as PIL RGB Image
        right_col  – colorized right as PIL RGB Image
    """
    L = pil_to_tensor(left_gray,  device)   # (1, 1, H, W)
    R = pil_to_tensor(right_rgb,  device)   # (1, 3, H, W)

    left_pre, right_pre, *_ = sicnet(left=L, right=R)

    return tensor_to_pil(left_pre), tensor_to_pil(right_pre)


@torch.inference_mode()
def run_sr(sr_model: torch.nn.Module,
           sbs_rgb: Image.Image,
           device: torch.device) -> Image.Image:
    """
    Run SR on a full SBS RGB image.

    Returns:
        upscaled SBS as PIL RGB Image
    """
    x = pil_to_tensor(sbs_rgb, device)       # (1, 3, H, W)
    y, _ = sr_model.model(x, None)
    return tensor_to_pil(y)


# ─────────────────────────────────────────────────────────────────────────────
# Main folder-level loop
# ─────────────────────────────────────────────────────────────────────────────

def process_folder(args):
    input_dir = Path(args.input_dir).resolve()
    out_dir   = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[config] device={device}  input={input_dir}  output={out_dir}")

    # ── load models ──────────────────────────────────────────────────────────
    sicnet   = load_sicnet(args.sicnet_ckpt, device)
    sr_model = load_sr(args.sr_ckpt, device) if args.sr_ckpt else None

    if sr_model is None:
        print("[info] No SR checkpoint provided — colorized output only.")

    # ── collect images ───────────────────────────────────────────────────────
    images = collect_images(input_dir)
    if not images:
        print(f"[warn] No images found in {input_dir}")
        return

    print(f"[info] Found {len(images)} image(s). Processing ...")

    ok = skipped = errors = 0
    for img_path in images:
        out_path = out_dir / (img_path.stem + "_colorized.png")

        if out_path.exists() and not args.overwrite:
            print(f"  [skip] {img_path.name} (already exists)")
            skipped += 1
            continue

        try:
            # 1) split SBS
            left_gray, right_rgb, orig_size = load_sbs(img_path)

            # 2) colorize
            left_col, right_col = run_colorization(sicnet, left_gray, right_rgb, device)

            # 3) stitch colorized halves back into SBS
            colorized_sbs = make_sbs(left_col, right_col)

            # 4) optional SR pass on the colorized SBS
            if sr_model is not None:
                colorized_sbs = run_sr(sr_model, colorized_sbs, device)

            # 5) save
            colorized_sbs.save(out_path)
            print(f"  [ok] {img_path.name:40s} → {out_path.name}")
            ok += 1

        except Exception as exc:
            print(f"  [error] {img_path.name}: {exc}")
            errors += 1

    print(f"\n[done] processed={ok}  skipped={skipped}  errors={errors}")
    print(f"       results saved to {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="PyTorch inference: colorize (+ optionally SR) a folder of SBS PNG images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input_dir",   required=True,
                    help="Folder containing SBS input PNGs "
                         "(left half = grayscale source, right half = RGB reference)")
    ap.add_argument("--out_dir",     required=True,
                    help="Where colorized output PNGs are saved")
    ap.add_argument("--sicnet_ckpt", required=True,
                    help="Path to SICNet colorization checkpoint (.pth)")
    ap.add_argument("--sr_ckpt",     default=None,
                    help="(optional) Path to SR Lightning checkpoint (.ckpt). "
                         "If omitted, SR is skipped.")
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Torch device to run on (cuda / cpu)")
    ap.add_argument("--overwrite",   action="store_true",
                    help="Re-process images even if the output file already exists")
    args = ap.parse_args()
    process_folder(args)


if __name__ == "__main__":
    main()
