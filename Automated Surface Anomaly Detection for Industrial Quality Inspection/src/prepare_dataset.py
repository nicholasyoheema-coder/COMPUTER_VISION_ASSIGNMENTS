import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import pandas as pd

CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def image_split(path):
    parts = [x.lower() for x in path.parts]
    if "train" in parts:
        return "train"
    if "validation" in parts or "valid" in parts:
        return "test"
    raise ValueError(f"Unknown image split: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", default="p3_yolo_dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source_root)
    output = Path(args.output)

    images, xmls = {}, {}

    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in EXTENSIONS:
            images[path.stem.lower()] = path
        elif path.suffix.lower() == ".xml":
            xmls[path.stem.lower()] = path

    if set(images) != set(xmls):
        raise RuntimeError("Image/XML filenames do not match.")

    records = []

    for stem, image_path in images.items():
        xml_path = xmls[stem]
        root = ET.parse(xml_path).getroot()

        names = [
            obj.findtext("name", "").strip()
            for obj in root.findall("object")
        ]

        if not names or any(n not in CLASS_TO_ID for n in names):
            raise ValueError(f"Invalid classes in {xml_path}")

        records.append({
            "image": image_path,
            "xml": xml_path,
            "split": image_split(image_path),
            "primary_class": Counter(names).most_common(1)[0][0]
        })

    train_records = [r for r in records if r["split"] == "train"]
    test_records = [r for r in records if r["split"] == "test"]

    if len(train_records) != 1440 or len(test_records) != 360:
        raise RuntimeError("Unexpected original dataset split sizes.")

    groups = defaultdict(list)
    for record in train_records:
        groups[record["primary_class"]].append(record)

    rng = random.Random(args.seed)
    target_val = round(len(train_records) * 0.20)

    allocation = {
        name: int(len(groups[name]) * 0.20)
        for name in CLASSES
    }

    remaining = target_val - sum(allocation.values())
    ranked = sorted(
        CLASSES,
        key=lambda n: len(groups[n]) * 0.20 - allocation[n],
        reverse=True
    )

    for name in ranked[:remaining]:
        allocation[name] += 1

    train, val = [], []

    for name in CLASSES:
        group = groups[name][:]
        rng.shuffle(group)
        n = allocation[name]
        val.extend(group[:n])
        train.extend(group[n:])

    rng.shuffle(train)
    rng.shuffle(val)

    splits = {"train": train, "val": val, "test": test_records}

    if output.exists():
        shutil.rmtree(output)

    for split in splits:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    manifest = []
    total_boxes = 0

    for split, split_records in splits.items():
        for record in split_records:
            image_path, xml_path = record["image"], record["xml"]

            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Unreadable image: {image_path}")

            height, width = image.shape[:2]
            root = ET.parse(xml_path).getroot()
            size = root.find("size")

            if size is not None:
                if (int(size.findtext("width")), int(size.findtext("height"))) != (width, height):
                    raise ValueError(f"Dimension mismatch: {image_path.name}")

            lines = []
            for obj in root.findall("object"):
                name = obj.findtext("name").strip()
                class_id = CLASS_TO_ID[name]
                box = obj.find("bndbox")

                x1 = float(box.findtext("xmin"))
                y1 = float(box.findtext("ymin"))
                x2 = float(box.findtext("xmax"))
                y2 = float(box.findtext("ymax"))

                xc = ((x1 + x2) / 2) / width
                yc = ((y1 + y2) / 2) / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height

                coords = [xc, yc, bw, bh]
                if not all(0 <= x <= 1 for x in coords):
                    raise ValueError(f"Invalid box: {xml_path.name}")

                lines.append(
                    f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                )

            shutil.copy2(
                image_path,
                output / "images" / split / image_path.name
            )

            (output / "labels" / split / f"{image_path.stem}.txt").write_text(
                "\n".join(lines) + "\n"
            )

            total_boxes += len(lines)
            manifest.append({
                "image": image_path.name,
                "split": split,
                "primary_class": record["primary_class"]
            })

    yaml = f"path: {output.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    for i, name in enumerate(CLASSES):
        yaml += f"  {i}: '{name}'\n"

    (output / "data.yaml").write_text(yaml)
    pd.DataFrame(manifest).to_csv(output / "split_manifest.csv", index=False)

    if (len(train), len(val), len(test_records), total_boxes) != (1152, 288, 360, 4189):
        raise RuntimeError("Dataset verification failed.")

    print("Dataset conversion completed.")
    print("Train: 1152 | Validation: 288 | Test: 360")
    print("Boxes:", total_boxes)
    print("Output:", output)


if __name__ == "__main__":
    main()
