from __future__ import annotations
import torch
from torch.utils.data import DataLoader

from src.rwf2000_dataset import RWF2000Clips
from src.models.r3d import R3D18Violence


@torch.no_grad()
def eval_full(model, loader, device):
    model.eval()
    cm = torch.zeros((2, 2), dtype=torch.long)  # rows=true, cols=pred
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        pred = logits.argmax(dim=1)

        total += y.numel()
        correct += (pred == y).sum().item()

        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1

    acc = correct / max(1, total)

    # Fight class = 1 F1
    tp = cm[1, 1].item()
    fp = cm[0, 1].item()
    fn = cm[1, 0].item()

    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return acc, f1, cm


def main():
    val_csv = "data/processed/rwf2000_val.csv"
    ckpt_path = "outputs/checkpoints/r3d18_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    ds = RWF2000Clips(val_csv, clip_len=16, frame_size=112, sampling="center", normalize=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    model = R3D18Violence(num_classes=2, pretrained=False).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"[INFO] loaded checkpoint: {ckpt_path} (epoch {ckpt.get('epoch')})")

    acc, f1, cm = eval_full(model, loader, device)
    print(f"\n[VAL] acc={acc:.4f} | f1(Fight)= {f1:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()
