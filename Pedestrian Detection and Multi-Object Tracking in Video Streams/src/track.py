import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="YOLO11n pedestrian tracking with ByteTrack"
    )
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--output-dir", default="tracking_output")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--track-buffer", type=int, default=60)
    args = parser.parse_args()

    sequence = Path(args.sequence_dir)
    image_dir = sequence / "img1"

    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    frames = sorted(
        image_dir.glob("*.jpg"),
        key=lambda p: int(p.stem)
    )

    if not frames:
        raise RuntimeError("No JPG frames found.")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tracker_config = (
        Path(__file__).resolve().parent.parent /
        "configs" / "bytetrack_tuned.yaml"
    )

    # Create config using requested track buffer.
    config_text = tracker_config.read_text()
    config_text = config_text.replace(
        "track_buffer: 60",
        f"track_buffer: {args.track_buffer}"
    )

    runtime_config = output / "bytetrack_runtime.yaml"
    runtime_config.write_text(config_text)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.weights)

    first = cv2.imread(str(frames[0]))
    height, width = first.shape[:2]

    display_width = 1280
    display_height = int(height * display_width / width)

    video_path = output / "tracked_video.mp4"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (display_width, display_height)
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create output video.")

    records = []

    sx = display_width / width
    sy = display_height / height

    for index, frame_path in enumerate(frames):

        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Cannot read {frame_path}")

        result = model.track(
            source=frame,
            persist=True,
            tracker=str(runtime_config),
            classes=[0],
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            verbose=False
        )[0]

        canvas = cv2.resize(
            frame, (display_width, display_height)
        )

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()

            for box, track_id, score in zip(boxes, ids, scores):
                x1, y1, x2, y2 = map(float, box)

                records.append({
                    "frame": index + 1,
                    "track_id": track_id,
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "confidence": score
                })

                p1 = (int(x1 * sx), int(y1 * sy))
                p2 = (int(x2 * sx), int(y2 * sy))

                cv2.rectangle(canvas, p1, p2, (0, 255, 0), 2)
                cv2.putText(
                    canvas,
                    f"ID {track_id}",
                    (p1[0], max(20, p1[1] - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

        cv2.putText(
            canvas,
            f"Frame: {index+1}/{len(frames)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        writer.write(canvas)

        if (index + 1) % 100 == 0:
            print(f"Processed {index+1}/{len(frames)} frames")

    writer.release()

    columns = [
        "frame", "track_id", "x", "y",
        "width", "height", "confidence"
    ]

    tracks = pd.DataFrame(records, columns=columns)
    csv_path = output / "tracks.csv"
    tracks.to_csv(csv_path, index=False)

    print("Frames:", len(frames))
    print("Tracked detections:", len(tracks))
    print("Unique IDs:", tracks["track_id"].nunique())
    print("Video:", video_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
