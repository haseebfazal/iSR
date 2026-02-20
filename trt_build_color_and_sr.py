# trt_build_color_and_sr.py  (updated)
import os
import argparse
import numpy as np
from glob import glob
from PIL import Image
import torch
import onnx

# ---------- safe globals for torch.load(weights_only=True) ----------
from torch.serialization import add_safe_globals
from yacs.config import CfgNode
from collections import defaultdict
add_safe_globals([CfgNode, set, frozenset, defaultdict])

# ---------- repo imports ----------
import model.SICNet as SICNet
import config as sic_config
from model_SR import LitRT4KSR_Rep
import config_ft as sr_config
from utils import reparameterize

# ---------- TensorRT / CUDA ----------
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# ----------------- helpers -----------------
def set_trt_workspace_limit(config, bytes_):
    """TRT >= 8.6 / 10.x uses set_memory_pool_limit; older uses max_workspace_size."""
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, bytes_)
    else:
        config.max_workspace_size = bytes_  # legacy


def prefer_cublas_only(config):
    """Favor cuBLAS/cuBLASLt tactics and avoid cuDNN/Cask (stability-first)."""
    if hasattr(config, "set_tactic_sources"):
        try:
            config.set_tactic_sources(trt.TacticSource.CUBLAS | trt.TacticSource.CUBLAS_LT)
        except Exception:
            pass  # ignore on very old TRT


def prefer_fast_sr_tactics(config):
    """Allow fast tactics for SR (cuBLAS/cuBLASLt/cuDNN/CASK)."""
    if hasattr(config, "set_tactic_sources"):
        try:
            config.set_tactic_sources(
                trt.TacticSource.CUBLAS | trt.TacticSource.CUBLAS_LT |
                trt.TacticSource.CUDNN | trt.TacticSource.CASK
            )
        except Exception:
            pass


def load_sicnet(device, sicnet_ckpt_path):
    cfg = sic_config.get_config()
    net = SICNet.SICNet(channels=cfg.MODEL.channels)
    try:
        ckpt = torch.load(sicnet_ckpt_path, map_location='cpu', weights_only=True)
    except Exception:
        ckpt = torch.load(sicnet_ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    return net.to(device).eval()


def load_sr_and_force_reparam(device):
    """
    Load SR Lightning module and **force** re-parameterization before export.
    We do this regardless of sr_config.infer_reparameterize to guarantee
    the exported ONNX is the fused 'Rep' graph.
    """
    lit = LitRT4KSR_Rep.load_from_checkpoint(
        checkpoint_path=sr_config.checkpoint_path_infer,
        config=sr_config
    )
    # force reparam
    lit.model = reparameterize(sr_config, lit.model, device, save_rep_checkpoint=False)
    return lit.to(device).eval()


# ----------------- ONNX export -----------------
def export_sicnet_onnx(model, out_path, h, w_half, device_export):
    """
    Exports SICNet to ONNX with fixed shapes and EVAL training mode.
    Outputs: left_col, right_col
    """
    class Wrap(torch.nn.Module):
        def __init__(self, net): super().__init__(); self.net = net
        def forward(self, L, R):
            left_pre, right_pre, _, _ = self.net(left=L, right=R)
            return left_pre, right_pre

    model = model.to(device_export).eval()
    L = torch.randn(1, 1, h, w_half, device=device_export, dtype=torch.float32)
    R = torch.randn(1, 3, h, w_half, device=device_export, dtype=torch.float32)

    with torch.inference_mode():
        torch.onnx.export(
            Wrap(model), (L, R), out_path,
            input_names=["left_mono", "right_rgb"],
            output_names=["left_col", "right_col"],
            opset_version=17,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,  # important
            dynamic_axes=None  # fixed shapes to match TRT profiles
        )
    onnx.checker.check_model(out_path)
    print(f"Exported SICNet ONNX (device={device_export}): {out_path}")


def export_sr_onnx(lit, out_path, h, w, device_export):
    """
    Exports SR to ONNX with fixed shapes and EVAL training mode.
    Output: sbs_sr
    NOTE: lit is already re-parameterized.
    """
    class Wrap(torch.nn.Module):
        def __init__(self, litm): super().__init__(); self.m = litm.model if hasattr(litm, "model") else litm
        def forward(self, x):
            y, _ = self.m(x, None)
            return y

    lit = lit.to(device_export).eval()
    X = torch.randn(1, 3, h, w, device=device_export, dtype=torch.float32)

    with torch.inference_mode():
        torch.onnx.export(
            Wrap(lit), (X,), out_path,
            input_names=["sbs_rgb"], output_names=["sbs_sr"],
            opset_version=17,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            dynamic_axes=None
        )
    onnx.checker.check_model(out_path)
    print(f"Exported SR ONNX (device={device_export}): {out_path}")


# ----------------- INT8 calibration plumbed (optional) -----------------
class ImageSBSCalib:
    def __init__(self, calib_dir, max_samples=128):
        self.paths = sorted(glob(os.path.join(calib_dir, "*.*")))[:max_samples]
    def __iter__(self):
        for p in self.paths:
            img = Image.open(p).convert("RGB")
            np_img = np.array(img)
            H, W, _ = np_img.shape
            mid = W // 2
            left, right = np_img[:, :mid, :], np_img[:, mid:, :]
            gray = np.round(0.299*left[...,0] + 0.587*left[...,1] + 0.114*left[...,2]).astype(np.uint8)
            L = (gray.astype(np.float32) / 255.0)[None, None, ...]            # 1x1xH x W/2
            R = (right.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]   # 1x3xH x W/2
            sbs = np.concatenate([right, right], axis=1).astype(np.float32) / 255.0
            sbs = sbs.transpose(2, 0, 1)[None]                                 # 1x3xH x W
            yield L, R, sbs


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, stream, cache_file):
        super().__init__()
        self.stream = iter(stream)
        self.cache_file = cache_file
        self.d = []
    def get_batch_size(self): return 1
    def get_batch(self, names):
        try:
            batch = next(self.stream)
        except StopIteration:
            return None
        host = [np.ascontiguousarray(b) for b in batch]
        if not self.d:
            self.d = [cuda.mem_alloc(arr.nbytes) for arr in host]
        for d, arr in zip(self.d, host):
            cuda.memcpy_htod(d, arr)
        return [int(x) for x in self.d]
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            return open(self.cache_file, "rb").read()
        return None
    def write_calibration_cache(self, cache):
        open(self.cache_file, "wb").write(cache)


# ----------------- optimization profiles (FIXED 250×250 and 250×500) -----------------
def add_profiles_sicnet(builder, network, config, h, w_half):
    """Fixed profiles: (1,1,250,250) and (1,3,250,250)"""
    profile = builder.create_optimization_profile()
    in_names = [network.get_input(i).name for i in range(network.num_inputs)]
    assert "left_mono" in in_names and "right_rgb" in in_names, f"Unexpected inputs: {in_names}"

    profile.set_shape("left_mono",  min=(1, 1, 250, 250), opt=(1, 1, 250, 250), max=(1, 1, 250, 250))
    profile.set_shape("right_rgb",  min=(1, 3, 250, 250), opt=(1, 3, 250, 250), max=(1, 3, 250, 250))
    config.add_optimization_profile(profile)


def add_profiles_sr(builder, network, config, h, w):
    """Fixed profile: (1,3,250,500)"""
    profile = builder.create_optimization_profile()
    in_names = [network.get_input(i).name for i in range(network.num_inputs)]
    assert "sbs_rgb" in in_names, f"Unexpected inputs: {in_names}"

    profile.set_shape("sbs_rgb", min=(1, 3, 250, 500), opt=(1, 3, 250, 500), max=(1, 3, 250, 500))
    config.add_optimization_profile(profile)


# ----------------- engine build (TRT 10 serialized) -----------------
def build_serialized(onnx_path, out_plan_path, add_profiles_fn, h, w_or_whalf,
                     use_int8, use_fp16, calibrator=None,
                     workspace_bytes=512<<20, tactic_mode="safe", enable_tf32=False):
    """
    tactic_mode: "safe" => cuBLAS/LT only; "fast" => cuBLAS/LT + cuDNN + CASK
    """
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError(f"ONNX parse failed for {onnx_path}")

        # Builder config
        config = builder.create_builder_config()
        set_trt_workspace_limit(config, workspace_bytes)

        if tactic_mode == "safe":
            prefer_cublas_only(config)
        else:
            prefer_fast_sr_tactics(config)

        if use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if enable_tf32 and hasattr(trt.BuilderFlag, "TF32"):
            try:
                config.set_flag(trt.BuilderFlag.TF32)
            except Exception:
                pass

        if use_int8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if calibrator is not None:
                config.int8_calibrator = calibrator

        # Fixed optimization profiles
        add_profiles_fn(builder, network, config, h, w_or_whalf)

        # Build serialized network (TRT 8.6+/10.x)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError(f"build_serialized_network() returned None for {onnx_path}")

        # Optional: quick deserialize check
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(plan)
        if engine is None:
            raise RuntimeError("deserialize_cuda_engine() failed")

        with open(out_plan_path, "wb") as f:
            f.write(plan)
        print("Saved:", out_plan_path)


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("Build TRT engines for SICNet and SR")
    ap.add_argument("--sicnet_ckpt", required=True)
    ap.add_argument("--calib_dir", required=True, help="Folder with a few SBS images for INT8 calibration")
    ap.add_argument("--out_dir", required=True)
    # Note: h/w_half are still accepted but we’ll export & profile at fixed 250×250 & 250×500
    ap.add_argument("--h", type=int, default=250)
    ap.add_argument("--w_half", type=int, default=250)
    ap.add_argument("--int8", action="store_true", help="Build INT8 engines (uses calibration data)")
    ap.add_argument("--fp16_only", action="store_true", help="FP16 engines only (no INT8)")
    ap.add_argument("--export-device", choices=["cpu", "cuda"], default="cpu",
                    help="Device for ONNX export (cpu avoids OOM)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    export_dev = torch.device(args.export_device)

    # 1) Load models on export device
    sicnet = load_sicnet(export_dev, args.sicnet_ckpt)
    sr_lit = load_sr_and_force_reparam(export_dev)  # <— forced rep

    # 2) Export ONNX (fixed shapes)
    sicnet_onnx = os.path.join(args.out_dir, "sicnet.onnx")
    sr_onnx     = os.path.join(args.out_dir, "sr.onnx")
    export_sicnet_onnx(sicnet, sicnet_onnx, h=250, w_half=250, device_export=export_dev)
    export_sr_onnx(sr_lit, sr_onnx, h=250, w=500, device_export=export_dev)

    # 3) Prepare (optional) INT8 calibration
    calib = None
    if args.int8:
        stream = ImageSBSCalib(args.calib_dir, max_samples=128)
        calib  = EntropyCalibrator(stream, os.path.join(args.out_dir, "calib.cache"))

    # 4) Build engines (use FP16 unless --fp16_only is set; we keep FP16 on)
    use_fp16 = True
    use_int8 = bool(args.int8)

    # --- SICNet: stability-first ---
    build_serialized(
        sicnet_onnx,
        os.path.join(args.out_dir, "sicnet.trt"),
        add_profiles_sicnet,
        h=250,
        w_or_whalf=250,
        use_int8=use_int8,
        use_fp16=use_fp16,
        calibrator=calib,
        workspace_bytes=512 << 20,      # 512 MB
        tactic_mode="safe",             # cuBLAS/LT only
        enable_tf32=False
    )

    # --- SR: speed-first ---
    build_serialized(
        sr_onnx,
        os.path.join(args.out_dir, "sr.trt"),
        add_profiles_sr,
        h=250,
        w_or_whalf=500,
        use_int8=use_int8,
        use_fp16=use_fp16,
        calibrator=calib,
        workspace_bytes=2 << 30,        # 2 GB
        tactic_mode="fast",             # cuBLAS/LT + cuDNN + CASK
        enable_tf32=True                # accelerate any FP32 leftovers
    )


if __name__ == "__main__":
    main()
