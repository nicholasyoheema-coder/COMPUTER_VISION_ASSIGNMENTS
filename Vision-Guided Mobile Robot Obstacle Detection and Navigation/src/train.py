import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fraction", type=float, default=0.20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise FileNotFoundError(data)

    model = YOLO(args.model)

    device = int(args.device) if args.device.isdigit() else args.device

    model.train(
        data=str(data),
        epochs=args.epochs,
        fraction=args.fraction,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=4,
        optimizer="auto",
        patience=3,
        seed=42,
        deterministic=True,
        amp=True,
        cache=False,
        plots=True,
        val=True,
        save=True,
        project="runs",
        name="yolo11n_rod",
        exist_ok=True
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print("Best model:", best)

if __name__ == "__main__":
    main()
