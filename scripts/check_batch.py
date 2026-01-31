import torch
from torch.utils.data import DataLoader

from src.rwf2000_dataset import RWF2000Clips

def main():
    train_csv = "data/processed/rwf2000_train.csv"
    val_csv   = "data/processed/rwf2000_val.csv"

    train_ds = RWF2000Clips(train_csv, clip_len=16, frame_size=112, sampling="random", normalize=True)
    val_ds   = RWF2000Clips(val_csv,   clip_len=16, frame_size=112, sampling="center", normalize=True)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    x, y = next(iter(train_loader))

    print("Batch video tensor:", x.shape, x.dtype)   # expected [B,3,16,112,112]
    print("Batch labels:", y.tolist())

    # quick min/max to ensure not NaNs
    print("min/max:", float(x.min()), float(x.max()))

if __name__ == "__main__":
    main()
