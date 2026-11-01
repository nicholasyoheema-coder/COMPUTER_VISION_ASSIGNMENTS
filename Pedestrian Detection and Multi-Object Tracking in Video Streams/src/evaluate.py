import argparse
import json
from pathlib import Path

import numpy as np

# Compatibility for older motmetrics with NumPy 2.x
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import pandas as pd
import motmetrics as mm


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pedestrian MOT tracking"
    )
    parser.add_argument("--gt", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--output-dir", default="tracking_metrics")
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()

    gt = pd.read_csv(
        args.gt,
        header=None,
        names=[
            "frame", "identity", "x", "y",
            "width", "height", "active",
            "class", "visibility"
        ]
    )

    # Pedestrian class only; exclude inactive/ignored annotations.
    gt = gt[
        (gt["class"] == 1) &
        (gt["active"] == 1) &
        (gt["width"] > 0) &
        (gt["height"] > 0)
    ].copy()

    pred = pd.read_csv(args.tracks)

    accumulator = mm.MOTAccumulator(auto_id=False)

    for frame in range(1, args.frames + 1):
        g = gt[gt["frame"] == frame]
        p = pred[pred["frame"] == frame]

        gt_ids = g["identity"].astype(int).tolist()
        pred_ids = p["track_id"].astype(int).tolist()

        gt_boxes = g[
            ["x", "y", "width", "height"]
        ].to_numpy(dtype=float).reshape(-1, 4)

        pred_boxes = p[
            ["x", "y", "width", "height"]
        ].to_numpy(dtype=float).reshape(-1, 4)

        distances = mm.distances.iou_matrix(
            gt_boxes,
            pred_boxes,
            max_iou=args.iou
        )

        accumulator.update(
            gt_ids, pred_ids, distances, frameid=frame
        )

    handler = mm.metrics.create()

    summary = handler.compute(
        accumulator,
        metrics=mm.metrics.motchallenge_metrics,
        name="tracking"
    )

    row = summary.loc["tracking"]

    keys = [
        "num_frames", "num_objects", "num_predictions",
        "num_matches", "num_false_positives", "num_misses",
        "num_switches", "num_fragmentations",
        "mota", "idf1", "idp", "idr",
        "precision", "recall"
    ]

    output_metrics = {}

    for key in keys:
        if key in row.index and pd.notna(row[key]):
            value = row[key]
            output_metrics[key] = (
                int(value) if key.startswith("num_")
                else round(float(value), 6)
            )

    output_metrics["iou_threshold"] = args.iou
    output_metrics["ground_truth_filter"] = "active pedestrian class 1"

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "tracking_metrics.json").write_text(
        json.dumps(output_metrics, indent=4)
    )

    pd.DataFrame([output_metrics]).to_csv(
        output / "tracking_metrics.csv",
        index=False
    )

    print(mm.io.render_summary(
        summary,
        formatters=handler.formatters,
        namemap=mm.io.motchallenge_metric_names
    ))

    print("Saved:", output)


if __name__ == "__main__":
    main()
