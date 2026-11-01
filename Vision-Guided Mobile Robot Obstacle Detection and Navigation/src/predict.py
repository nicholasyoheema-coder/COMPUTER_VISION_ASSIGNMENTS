import argparse
import json
from pathlib import Path
from collections import Counter

import cv2
from ultralytics import YOLO

IGNORED_CLASSES = {
    "building", "road", "traffic sign", "pedestrian crosswalk"
}

def analyse_navigation(result, threshold=0.10, tie_margin=0.03):
    height, width = result.orig_shape

    scores = {"LEFT": 0.0, "CENTER": 0.0, "RIGHT": 0.0}
    zones = {
        "LEFT": (0, width/3),
        "CENTER": (width/3, 2*width/3),
        "RIGHT": (2*width/3, width)
    }

    counts = Counter()
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            confidence = float(box.conf.item())
            class_id = int(box.cls.item())
            class_name = result.names[class_id]

            if confidence < 0.25 or class_name.lower() in IGNORED_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            box_width = max(0.0, x2-x1)
            box_height = max(0.0, y2-y1)

            if box_width <= 0 or box_height <= 0:
                continue

            area_ratio = box_width * box_height / (width*height)
            risk = confidence * min(area_ratio*5, 1.0)

            for zone, (left, right) in zones.items():
                overlap = max(0.0, min(x2, right)-max(x1, left))
                fraction = overlap / box_width

                if overlap > 0:
                    scores[zone] += risk*fraction
                    counts[zone] += 1

            detections.append({
                "class": class_name,
                "confidence": confidence
            })

    left = scores["LEFT"]
    center = scores["CENTER"]
    right = scores["RIGHT"]

    if center >= threshold:
        if left >= threshold and right >= threshold:
            decision = "STOP"
        elif left + tie_margin < right:
            decision = "TURN LEFT"
        elif right + tie_margin < left:
            decision = "TURN RIGHT"
        else:
            decision = "STOP"
    else:
        decision = "FORWARD"

    return {
        "decision": decision,
        "zone_risk": scores,
        "zone_counts": dict(counts),
        "detections": detections,
        "total_detections": len(detections)
    }

def main():
    parser = argparse.ArgumentParser(
        description="YOLO11n obstacle detection and navigation demonstration"
    )

    parser.add_argument("--source", required=True, help="Input image path")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--output", default="results/predictions")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")

    args = parser.parse_args()

    source = Path(args.source)
    weights = Path(args.weights)
    output = Path(args.output)

    if not source.exists():
        raise FileNotFoundError(source)

    if not weights.exists():
        raise FileNotFoundError(weights)

    image = cv2.imread(str(source))
    if image is None:
        raise ValueError("Input file is not a readable image")

    device = int(args.device) if args.device.isdigit() else args.device
    model = YOLO(str(weights))

    result = model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        verbose=False
    )[0]

    output.mkdir(parents=True, exist_ok=True)

    annotated = result.plot()
    height, width = annotated.shape[:2]

    # Draw the three spatial zones
    for x in [width//3, 2*width//3]:
        cv2.line(annotated, (x, 0), (x, height), (255,255,255), 2)

    navigation = analyse_navigation(result)

    cv2.rectangle(
        annotated, (0, height-55), (width, height), (35,35,35), -1
    )

    cv2.putText(
        annotated,
        "NAVIGATION: " + navigation["decision"],
        (12, height-18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    output_image = output / f"{source.stem}_navigation.jpg"
    cv2.imwrite(str(output_image), annotated)

    output_json = output / f"{source.stem}_summary.json"
    output_json.write_text(json.dumps(navigation, indent=4))

    print("Image:", source)
    print("Detections:", navigation["total_detections"])

    for name, count in Counter(
        d["class"] for d in navigation["detections"]
    ).items():
        print(f"{name}: {count}")

    print("Zone risk:", navigation["zone_risk"])
    print("Navigation decision:", navigation["decision"])
    print("Annotated output:", output_image)
    print("Summary:", output_json)

if __name__ == "__main__":
    main()
