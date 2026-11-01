import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="neu_predictions")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(args.weights)

    if not Path(args.source).exists():
        raise FileNotFoundError(args.source)

    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=0,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True
    )

    print("Images processed:", len(results))
    print("Predictions saved under:", Path(args.project) / args.name)


if __name__ == "__main__":
    main()
