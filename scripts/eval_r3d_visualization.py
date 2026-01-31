import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from torch.utils.data import DataLoader
from src.rwf2000_dataset import RWF2000Clips
from src.models.r3d import R3D18Violence


@torch.no_grad()
def get_probs_and_labels(model, loader, device):
    model.eval()
    y_true = []
    y_prob = []  # probability of Fight (class 1)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]  # class-1 probability

        y_true.extend(y.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())

    return y_true, y_prob


def main():
    val_csv = "data/processed/rwf2000_val.csv"
    ckpt_path = "outputs/checkpoints/r3d18_best.pt"  # change if needed
    report_dir = "outputs/reports"
    os.makedirs(report_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    ds = RWF2000Clips(val_csv, clip_len=16, frame_size=112, sampling="center", normalize=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    model = R3D18Violence(num_classes=2, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"[INFO] loaded checkpoint: {ckpt_path} (epoch {ckpt.get('epoch')})")

    y_true, y_prob = get_probs_and_labels(model, loader, device)
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    # Save predictions
    out_csv = os.path.join(report_dir, "eval_predictions.csv")
    pd.DataFrame({"y_true": y_true, "y_prob_fight": y_prob, "y_pred": y_pred}).to_csv(out_csv, index=False)
    print(f"[INFO] saved: {out_csv}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["NonFight", "Fight"])

    plt.figure()
    disp.plot(values_format="d")
    plt.title("Confusion Matrix (Val)")
    plt.tight_layout()
    cm_path = os.path.join(report_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.show()
    print(f"[INFO] saved: {cm_path}")

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Fight as positive)")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(report_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=200)
    plt.show()
    print(f"[INFO] saved: {roc_path}")

    # --- Precision-Recall Curve ---
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.plot(rec, prec, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Fight as positive)")
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(report_dir, "pr_curve.png")
    plt.savefig(pr_path, dpi=200)
    plt.show()
    print(f"[INFO] saved: {pr_path}")

    # Print quick metrics
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    print(f"\n[VAL] accuracy={acc:.4f} | ROC-AUC={roc_auc:.4f} | AP={ap:.4f}")


if __name__ == "__main__":
    main()
