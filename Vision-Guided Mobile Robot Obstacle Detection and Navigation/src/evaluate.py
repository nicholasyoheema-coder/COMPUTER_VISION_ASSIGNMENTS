import argparse
import json
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    weights = Path(args.weights)
    data = Path(args.data)

    if not weights.exists() or not data.exists():
        raise FileNotFoundError("Weights or dataset YAML not found")

    device = int(args.device) if args.device.isdigit() else args.device
    model = YOLO(str(weights))

    metrics = model.val(
        data=str(data),
        split=args.split,
        imgsz=640,
        batch=16,
        device=device,
        plots=True,
        project="runs/evaluation",
        name=args.split,
        exist_ok=True
    )

    output = {
        "split": args.split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map)
    }

    print(json.dumps(output, indent=4))

    Path("evaluation_metrics.json").write_text(
        json.dumps(output, indent=4)
    )

if __name__ == "__main__":
    main()
