import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["bottle", "cup"]


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - intersection

    return intersection / union if union else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="results/metrics")
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()

    root = Path(args.dataset)
    predictions = pd.read_csv(args.predictions)
    predictions["image_key"] = predictions["image"].apply(
        lambda x: Path(str(x)).name.lower()
    )

    images = {
        p.stem.lower(): p
        for p in root.rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg"]
    }

    ground_truth = {}
    annotated_images = []

    for xml_path in root.rglob("*.xml"):
        image_path = images.get(xml_path.stem.lower())
        if image_path is None:
            continue

        xml_root = ET.parse(xml_path).getroot()
        gt = {name: [] for name in CLASSES}

        for obj in xml_root.findall("object"):
            name = obj.findtext("name", "").strip().lower()
            if name not in CLASSES:
                continue

            bbox = obj.find("bndbox")
            if bbox is None:
                continue

            gt[name].append([
                float(bbox.findtext("xmin")),
                float(bbox.findtext("ymin")),
                float(bbox.findtext("xmax")),
                float(bbox.findtext("ymax"))
            ])

        key = image_path.name.lower()
        ground_truth[key] = gt
        annotated_images.append(image_path)

    rows = []

    # Include images with no predictions to count all false negatives.
    for image_path in annotated_images:
        key = image_path.name.lower()
        image_preds = predictions[predictions["image_key"] == key]

        for name in CLASSES:
            boxes = ground_truth[key][name]
            preds = image_preds[
                image_preds["class"].str.lower() == name
            ].sort_values("confidence", ascending=False)

            matched = set()

            for _, pred in preds.iterrows():
                box = [pred["x1"], pred["y1"], pred["x2"], pred["y2"]]
                best_iou = 0.0
                best_index = None

                for index, gt_box in enumerate(boxes):
                    if index in matched:
                        continue
                    score = iou(box, gt_box)
                    if score > best_iou:
                        best_iou = score
                        best_index = index

                if best_index is not None and best_iou >= args.iou:
                    matched.add(best_index)
                    outcome = "TP"
                else:
                    outcome = "FP"

                rows.append({
                    "image": image_path.name,
                    "class": name,
                    "result": outcome,
                    "IoU": best_iou
                })

            for index in range(len(boxes)):
                if index not in matched:
                    rows.append({
                        "image": image_path.name,
                        "class": name,
                        "result": "FN",
                        "IoU": np.nan
                    })

    details = pd.DataFrame(rows)
    summary = []

    for name in CLASSES:
        group = details[details["class"] == name]
        tp = int((group["result"] == "TP").sum())
        fp = int((group["result"] == "FP").sum())
        fn = int((group["result"] == "FN").sum())

        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2*p*r/(p+r) if p+r else 0

        matched_ious = group.loc[group["result"] == "TP", "IoU"]

        summary.append({
            "Class": name,
            "Ground Truth": tp + fn,
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": p, "Recall": r, "F1-score": f1,
            "Mean IoU (TP)": matched_ious.mean() if len(matched_ious) else 0
        })

    metrics = pd.DataFrame(summary)

    tp = int((details["result"] == "TP").sum())
    fp = int((details["result"] == "FP").sum())
    fn = int((details["result"] == "FN").sum())

    total_gt = sum(
        len(boxes)
        for image in ground_truth.values()
        for boxes in image.values()
    )

    assert tp + fn == total_gt

    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "classwise_metrics.csv", index=False)
    details.to_csv(output / "detection_match_details.csv", index=False)

    overall = {
        "annotated_images": len(annotated_images),
        "ground_truth_objects": total_gt,
        "TP": tp, "FP": fp, "FN": fn,
        "IoU_threshold": args.iou,
        "precision": p, "recall": r, "F1_score": f1
    }

    (output / "overall_metrics.json").write_text(
        json.dumps(overall, indent=4)
    )

    print(metrics.round(4).to_string(index=False))
    print(json.dumps(overall, indent=4))


if __name__ == "__main__":
    main()
