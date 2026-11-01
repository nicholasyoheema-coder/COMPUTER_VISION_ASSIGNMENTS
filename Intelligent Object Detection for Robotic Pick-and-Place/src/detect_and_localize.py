import argparse
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--output", default="results/detections/pick_point_detections.csv")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        raise FileNotFoundError(root)

    images = {
        p.stem.lower(): p
        for p in root.rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg"]
    }
    xml_files = list(root.rglob("*.xml"))

    matched = [
        images[p.stem.lower()]
        for p in xml_files
        if p.stem.lower() in images
    ]

    if not matched:
        raise RuntimeError("No annotated images found.")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.weights)

    class_ids = [
        i for i, name in model.names.items()
        if name.lower() in ["bottle", "cup"]
    ]

    rows = []
    results = model.predict(
        source=[str(p) for p in matched],
        imgsz=args.imgsz,
        conf=args.conf,
        classes=class_ids,
        device=device,
        stream=True,
        verbose=False
    )

    for result in results:
        height, width = result.orig_shape

        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            rows.append({
                "image": Path(result.path).name,
                "class": model.names[class_id],
                "confidence": confidence,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "pick_x_pixels": cx,
                "pick_y_pixels": cy,
                "pick_x_normalized": cx / width,
                "pick_y_normalized": cy / height
            })

    columns = [
        "image", "class", "confidence", "x1", "y1", "x2", "y2",
        "pick_x_pixels", "pick_y_pixels",
        "pick_x_normalized", "pick_y_normalized"
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output, index=False)

    print("Images evaluated:", len(matched))
    print("Detections:", len(df))
    print("Saved:", output)


if __name__ == "__main__":
    main()
