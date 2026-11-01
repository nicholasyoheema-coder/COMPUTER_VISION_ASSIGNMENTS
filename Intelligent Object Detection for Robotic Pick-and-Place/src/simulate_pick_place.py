import argparse
from pathlib import Path
import pandas as pd

DESTINATIONS = {
    "bottle": {"bin": "BOTTLE_BIN", "x": 0.25, "y": 0.50},
    "cup": {"bin": "CUP_BIN", "x": 0.75, "y": 0.50}
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--output",
        default="results/simulation/pick_and_place_plan.csv"
    )
    args = parser.parse_args()

    detections = pd.read_csv(args.predictions)
    if detections.empty:
        raise RuntimeError("No detections found.")

    operations = []

    for image_name, group in detections.groupby("image", sort=True):
        group = group.sort_values("confidence", ascending=False)

        for sequence, (_, row) in enumerate(group.iterrows(), start=1):
            name = str(row["class"]).lower()
            if name not in DESTINATIONS:
                continue

            destination = DESTINATIONS[name]

            operations.append({
                "image": image_name,
                "sequence": sequence,
                "object": name,
                "confidence": float(row["confidence"]),
                "pick_x_pixels": float(row["pick_x_pixels"]),
                "pick_y_pixels": float(row["pick_y_pixels"]),
                "pick_x_normalized": float(row["pick_x_normalized"]),
                "pick_y_normalized": float(row["pick_y_normalized"]),
                "destination_bin": destination["bin"],
                "place_x_virtual": destination["x"],
                "place_y_virtual": destination["y"],
                "operation": "SIMULATED_PICK_AND_PLACE"
            })

    plan = pd.DataFrame(operations)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output, index=False)

    print("Simulated operations:", len(plan))
    print(plan["destination_bin"].value_counts().to_string())
    print("Saved:", output)


if __name__ == "__main__":
    main()
