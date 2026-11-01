import argparse
import math
import shutil
from pathlib import Path
import yaml

def convert_line(line, number_of_classes):
    values = list(map(float, line.split()))

    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite annotation value")

    cls = values[0]

    if not cls.is_integer() or not 0 <= int(cls) < number_of_classes:
        raise ValueError("Invalid class ID")

    cls = int(cls)

    if len(values) == 5:
        _, xc, yc, w, h = values

        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError("Invalid bounding box")

        xmin, xmax = xc-w/2, xc+w/2
        ymin, ymax = yc-h/2, yc+h/2

        overflow = max(0, -xmin, -ymin, xmax-1, ymax-1)

        if overflow > 1.1e-5:
            raise ValueError("Boundary overflow exceeds documented tolerance")

        xmin = max(0.0, xmin)
        ymin = max(0.0, ymin)
        xmax = min(1.0, xmax)
        ymax = min(1.0, ymax)

    elif len(values) >= 7 and (len(values)-1) % 2 == 0:
        coords = values[1:]

        if not all(0 <= v <= 1 for v in coords):
            raise ValueError("Polygon coordinate outside image")

        xs = coords[0::2]
        ys = coords[1::2]

        if len(xs) < 3:
            raise ValueError("Insufficient polygon points")

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

    else:
        raise ValueError("Unknown annotation format")

    w = xmax-xmin
    h = ymax-ymin

    if w <= 0 or h <= 0:
        raise ValueError("Degenerate bounding box")

    xc = (xmin+xmax)/2
    yc = (ymin+ymax)/2

    return f"{cls} {xc:.10f} {yc:.10f} {w:.10f} {h:.10f}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Extracted source dataset directory containing data.yaml")
    parser.add_argument("--output", required=True, help="Output directory for detection dataset")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()

    config_path = source / "data.yaml"

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = yaml.safe_load(config_path.read_text())
    names = config["names"]
    names = [names[k] for k in sorted(names, key=lambda x: int(x))] if isinstance(names, dict) else names
    nc = len(names)

    if output.exists():
        shutil.rmtree(output)

    output.mkdir(parents=True)

    summary = {"images": 0, "objects": 0, "polygons_converted": 0, "empty_files": 0}

    for split in ["train", "valid", "test"]:
        src_images = source / split / "images"
        src_labels = source / split / "labels"
        dst_images = output / split / "images"
        dst_labels = output / split / "labels"

        if not src_images.exists() or not src_labels.exists():
            raise FileNotFoundError(f"Missing folders for {split}")

        shutil.copytree(src_images, dst_images)
        dst_labels.mkdir(parents=True)

        for label in src_labels.glob("*.txt"):
            converted = []

            for line in label.read_text().splitlines():
                if not line.strip():
                    continue

                values = line.split()
                if len(values) != 5:
                    summary["polygons_converted"] += 1

                converted.append(convert_line(line, nc))

            if not converted:
                summary["empty_files"] += 1

            (dst_labels / label.name).write_text(
                "\n".join(converted) + ("\n" if converted else "")
            )

            summary["objects"] += len(converted)

        summary["images"] += len(list(dst_images.iterdir()))

    new_config = {
        "path": str(output),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": nc,
        "names": names
    }

    (output / "data.yaml").write_text(
        yaml.safe_dump(new_config, sort_keys=False)
    )

    print("Prepared dataset:", output)
    print("Summary:", summary)

if __name__ == "__main__":
    main()
