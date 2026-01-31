from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from decord import VideoReader, cpu
import cv2

from .clip_sampler import sample_indices


def _resize_frame(frame: np.ndarray, size: int = 112) -> np.ndarray:
    """
    frame: HxWxC (uint8)
    returns: size x size x C
    """
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


class RWF2000Clips(Dataset):
    """
    Reads videos from paths in CSV and returns:
      video: FloatTensor [C, T, H, W] in range [0,1]
      label: LongTensor []
    """
    def __init__(
        self,
        csv_path: str,
        clip_len: int = 16,
        frame_size: int = 112,
        sampling: str = "random",   # random for train
        normalize: bool = True,     # optionally ImageNet normalize
        strict: bool = False,       # if True, raise on decode errors
    ):
        self.df = pd.read_csv(csv_path)
        if "video_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain columns: video_path, label")

        self.clip_len = int(clip_len)
        self.frame_size = int(frame_size)
        self.sampling = sampling
        self.normalize = normalize
        self.strict = strict

        # ImageNet mean/std (common for torchvision video models)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.df)

    def _decode_clip(self, video_path: str) -> np.ndarray:
        """
        Returns frames: [T, H, W, C] uint8
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        idx = sample_indices(num_frames=num_frames, clip_len=self.clip_len, mode=self.sampling)

        # Decord supports batch get
        frames = vr.get_batch(idx).asnumpy()  # [T, H, W, C], uint8
        return frames

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        video_path = str(row["video_path"])
        label = int(row["label"])

        try:
            frames = self._decode_clip(video_path)  # [T,H,W,C] uint8
        except Exception as e:
            if self.strict:
                raise
            # fallback: return a black clip (keeps training running)
            frames = np.zeros((self.clip_len, self.frame_size, self.frame_size, 3), dtype=np.uint8)

        # resize each frame to frame_size x frame_size
        resized = np.stack([_resize_frame(f, self.frame_size) for f in frames], axis=0)  # [T, S, S, C]

        # to torch: [T, S, S, C] -> [C, T, S, S]
        x = torch.from_numpy(resized).permute(3, 0, 1, 2).float() / 255.0  # [C,T,H,W]

        if self.normalize:
            x = (x - self.mean) / self.std

        y = torch.tensor(label, dtype=torch.long)
        return x, y
