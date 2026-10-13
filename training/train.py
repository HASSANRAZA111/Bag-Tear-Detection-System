"""Train the YOLO model and copy the best weights to models/best.pt.

Usage (Colab GPU recommended):
  python training/train.py --data data/bag_tear/data.yaml --epochs 80
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained checkpoint")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu")
    ap.add_argument("--out", default="models/best.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    # Absolute path: a relative "project" gets nested under runs/detect/ by ultralytics.
    project = str(Path("runs").resolve())
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                patience=args.patience, device=args.device, project=project, name="bag_tear",
                exist_ok=True)

    best = Path(model.trainer.best)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"\nBest weights copied to: {out}")

    try:  # honest final score on photos the model never saw
        model.val(data=args.data, split="test", project=project, name="bag_tear_test",
                  exist_ok=True)
    except Exception as e:  # no test split, etc.
        print(f"(skipped test-set evaluation: {e})")


if __name__ == "__main__":
    main()
