from __future__ import annotations
import os
import time
import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.rwf2000_dataset import RWF2000Clips
from src.models.r3d import R3D18Violence


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    # For binary F1 (Fight=1)
    tp = fp = fn = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)               # [B,2]
        pred = logits.argmax(dim=1)     # [B]

        total += y.numel()
        correct += (pred == y).sum().item()

        # Fight class = 1
        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()

    acc = correct / max(1, total)
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return acc, f1


def main():
    # ---- paths ----
    train_csv = "data/processed/rwf2000_train.csv"
    val_csv   = "data/processed/rwf2000_val.csv"

    # ---- hyperparams (safe defaults for Colab) ----
    seed = 42
    epochs = 8
    batch_size = 8          # try 8 on T4; if OOM, reduce to 4
    lr = 3e-4
    weight_decay = 1e-4
    clip_len = 16
    frame_size = 112
    num_workers = 2

    save_dir = "outputs/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # ---- data ----
    train_ds = RWF2000Clips(train_csv, clip_len=clip_len, frame_size=frame_size,
                            sampling="random", normalize=True, strict=False)
    val_ds   = RWF2000Clips(val_csv, clip_len=clip_len, frame_size=frame_size,
                            sampling="center", normalize=True, strict=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # ---- model ----
    model = R3D18Violence(num_classes=2, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine schedule is nice for finetuning
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f1 = -1.0

    print(f"[INFO] train steps/epoch = {len(train_loader)} | total_steps = {total_steps}")

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            global_step += 1

            running_loss += loss.item()

            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                running_loss = 0.0
                print(f"Epoch {epoch}/{epochs} | step {global_step} | loss {avg_loss:.4f}")

        # ---- eval each epoch ----
        val_acc, val_f1 = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"\n[VAL] epoch={epoch} | acc={val_acc:.4f} | f1={val_f1:.4f} | time={dt:.1f}s")

        # ---- save checkpoint ----
        ckpt_path = os.path.join(save_dir, f"r3d18_epoch{epoch}_acc{val_acc:.4f}_f1{val_f1:.4f}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_f1": val_f1,
            "clip_len": clip_len,
            "frame_size": frame_size,
        }, ckpt_path)

        # ---- track best ----
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_path = os.path.join(save_dir, "r3d18_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "clip_len": clip_len,
                "frame_size": frame_size,
            }, best_path)
            print(f"[INFO] ✅ new best saved: {best_path}")

    print(f"\n✅ Training complete. Best F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
