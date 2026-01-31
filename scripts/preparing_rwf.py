import os
import csv
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def collect_split(root: Path, split: str):
    """
    Returns list of dicts: {video_path, label, split}
    label: 1 = Fight, 0 = NonFight
    """
    rows = []
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    class_map = {"NonFight": 0, "Fight": 1}

    for cls, label in class_map.items():
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {cls_dir}")

        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                rows.append({
                    "video_path": str(p),
                    "label": int(label),
                    "split": split
                })

    return rows

def write_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

def main():
    # ✅ change this if your drive path differs
    DATA_ROOT = Path("/content/drive/MyDrive/RWF-2000")
    OUT_DIR = Path("VideoProj/data/processed")

    print(f"[INFO] DATA_ROOT = {DATA_ROOT.resolve()}")
    print(f"[INFO] OUT_DIR   = {OUT_DIR.resolve()}")

    train_rows = collect_split(DATA_ROOT, "train")
    val_rows   = collect_split(DATA_ROOT, "val")

    write_csv(train_rows, OUT_DIR / "rwf2000_train.csv")
    write_csv(val_rows,   OUT_DIR / "rwf2000_val.csv")

    print("\n✅ Done.")
    print(f"Train videos: {len(train_rows)}")
    print(f"Val videos:   {len(val_rows)}")
    print(f"Wrote: {OUT_DIR / 'rwf2000_train.csv'}")
    print(f"Wrote: {OUT_DIR / 'rwf2000_val.csv'}")

if __name__ == "__main__":
    main()
