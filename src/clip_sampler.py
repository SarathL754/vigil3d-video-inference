from __future__ import annotations
import numpy as np

def uniform_indices(num_frames: int, clip_len: int) -> np.ndarray:
    """
    Uniformly sample clip_len indices across [0, num_frames-1].
    If num_frames < clip_len, we pad by repeating last index.
    """
    if num_frames <= 0:
        return np.zeros((clip_len,), dtype=np.int64)

    if num_frames >= clip_len:
        # e.g., num_frames=100, clip_len=16 -> evenly spaced
        return np.linspace(0, num_frames - 1, clip_len).astype(np.int64)

    # pad: [0..num_frames-1] then repeat last
    idx = np.arange(num_frames, dtype=np.int64)
    pad = np.full((clip_len - num_frames,), num_frames - 1, dtype=np.int64)
    return np.concatenate([idx, pad], axis=0)


def random_clip_indices(num_frames: int, clip_len: int) -> np.ndarray:
    """
    Sample a contiguous clip of length clip_len using a random start.
    If num_frames < clip_len, pad by repeating last index.
    """
    if num_frames <= 0:
        return np.zeros((clip_len,), dtype=np.int64)

    if num_frames >= clip_len:
        max_start = num_frames - clip_len
        start = np.random.randint(0, max_start + 1)
        return np.arange(start, start + clip_len, dtype=np.int64)

    idx = np.arange(num_frames, dtype=np.int64)
    pad = np.full((clip_len - num_frames,), num_frames - 1, dtype=np.int64)
    return np.concatenate([idx, pad], axis=0)


def sample_indices(num_frames: int, clip_len: int, mode: str) -> np.ndarray:
    mode = mode.lower().strip()
    if mode in ("uniform", "linspace"):
        return uniform_indices(num_frames, clip_len)
    if mode in ("random", "rand"):
        return random_clip_indices(num_frames, clip_len)
    if mode in ("center", "middle"):
        # center contiguous window (good for val)
        if num_frames <= 0:
            return np.zeros((clip_len,), dtype=np.int64)
        if num_frames >= clip_len:
            start = (num_frames - clip_len) // 2
            return np.arange(start, start + clip_len, dtype=np.int64)
        idx = np.arange(num_frames, dtype=np.int64)
        pad = np.full((clip_len - num_frames,), num_frames - 1, dtype=np.int64)
        return np.concatenate([idx, pad], axis=0)

    raise ValueError(f"Unknown sampling mode: {mode}")
