import argparse
import json
from pathlib import Path
import pandas as pd
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--output", default="results/metrics")
    args = parser.parse_args()

    model = YOLO(args.weights)
    result = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=16,
        device=0,
        plots=True,
        verbose=True
    )

    box = result.box
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map)
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    (output / "overall_metrics.json").write_text(
        json.dumps(overall, indent=4)
    )

    ap50 = box.ap50
    ap = box.ap
    p = box.p
    r = box.r

    rows = []
    for class_id, name in model.names.items():
        if class_id < len(ap50):
            rows.append({
                "class": name,
                "precision": float(p[class_id]) if class_id < len(p) else None,
                "recall": float(r[class_id]) if class_id < len(r) else None,
                "mAP50": float(ap50[class_id]),
                "mAP50_95": float(ap[class_id].mean()) if class_id < len(ap) else None
            })

    pd.DataFrame(rows).to_csv(
        output / "classwise_metrics.csv", index=False
    )

    print(json.dumps(overall, indent=4))


if __name__ == "__main__":
    main()
