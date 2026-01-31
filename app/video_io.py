from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch

# Decord for fast video decoding
from decord import VideoReader, cpu

# -----------------------------
# Config
# -----------------------------
@dataclass
class VideoPreprocessConfig:
    clip_len: int = 16
    size: int = 112
    fps_sample: int | None = None  # keep None for now (uniform over frames)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)   # ImageNet


# -----------------------------
# Helpers
# -----------------------------
def _uniform_indices(num_frames: int, clip_len: int) -> np.ndarray:
    """
    Uniformly sample clip_len indices from [0, num_frames-1].
    If num_frames < clip_len, we repeat indices (pad) to reach clip_len.
    """
    if num_frames <= 0:
        raise ValueError("Video has zero frames.")

    if num_frames >= clip_len:
        return np.linspace(0, num_frames - 1, clip_len).round().astype(np.int64)

    # pad by repeating last frame index
    base = np.arange(num_frames, dtype=np.int64)
    pad = np.full((clip_len - num_frames,), num_frames - 1, dtype=np.int64)
    return np.concatenate([base, pad], axis=0)


def _resize_frames_np(frames: np.ndarray, size: int) -> np.ndarray:
    """
    Resize frames (T,H,W,3) to (T,size,size,3) using OpenCV.
    Keeps dtype uint8.
    """
    import cv2  # local import (keeps import time lower if unused)

    out = []
    for f in frames:
        # cv2 expects (W,H) ordering in resize size tuple
        out.append(cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR))
    return np.stack(out, axis=0)


def _to_tensor_chw(frames: np.ndarray, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    """
    frames: uint8 numpy array (T,H,W,3)
    returns: float32 tensor (1,3,T,H,W) normalized
    """
    # (T,H,W,3) -> float in [0,1]
    x = frames.astype(np.float32) / 255.0

    # normalize
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)
    x = (x - mean_arr) / std_arr

    # (T,H,W,3) -> (3,T,H,W)
    x = np.transpose(x, (3, 0, 1, 2))
    x_t = torch.from_numpy(x).float().unsqueeze(0)  # (1,3,T,H,W)
    return x_t


# -----------------------------
# Main API
# -----------------------------
def load_clip_tensor(
    video_path: Union[str, Path],
    cfg: VideoPreprocessConfig = VideoPreprocessConfig(),
) -> torch.Tensor:
    """
    Load a single uniformly sampled clip from a video and return model-ready tensor.
    Output: (1, 3, cfg.clip_len, cfg.size, cfg.size)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    vr = VideoReader(str(video_path), ctx=cpu(0))
    num_frames = len(vr)

    idx = _uniform_indices(num_frames=num_frames, clip_len=cfg.clip_len)
    frames = vr.get_batch(idx).asnumpy()  # (T,H,W,3), uint8

    frames = _resize_frames_np(frames, cfg.size)  # (T,size,size,3)

    clip = _to_tensor_chw(frames, cfg.mean, cfg.std)  # (1,3,T,H,W)
    return clip


def quick_video_info(video_path: Union[str, Path]) -> dict:
    """
    Utility: returns basic info for debugging.
    """
    video_path = Path(video_path)
    vr = VideoReader(str(video_path), ctx=cpu(0))
    return {
        "path": str(video_path),
        "num_frames": len(vr),
        "height": vr[0].shape[0],
        "width": vr[0].shape[1],
    }
