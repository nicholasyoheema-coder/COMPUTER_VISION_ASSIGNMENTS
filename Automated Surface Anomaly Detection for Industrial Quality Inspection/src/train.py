import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="yolo11n_neu_defect")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this configured experiment.")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=2,
        patience=8,
        optimizer="auto",
        seed=42,
        pretrained=True,
        amp=True,
        close_mosaic=8,
        plots=True,
        save=True,
        save_period=5,
        project=args.project,
        name=args.name,
        exist_ok=True
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)

    print("Best checkpoint:", best)


if __name__ == "__main__":
    main()
