"""
End-to-end training pipeline for Indian Sign Language (ISL) gesture detection using YOLOv8.

Pipeline steps:
1) Convert Pascal VOC XML annotations to YOLO labels
2) Build YOLO dataset structure and split into train/val
3) Generate data.yaml
4) Train YOLOv8n for configurable epochs
5) Export/copy best weights to best.pt
6) Run webcam inference for real-time testing
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml
from ultralytics import YOLO

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class AnnotationRecord:
    """A parsed annotation with matching image and YOLO-formatted lines."""

    xml_path: Path
    image_path: Path
    image_filename: str
    image_width: float
    image_height: float
    labels: List[Tuple[str, Tuple[float, float, float, float]]]


@dataclass
class ConversionSummary:
    """Summary stats from conversion and split preparation."""

    total_xml: int
    valid_records: int
    missing_images: int
    invalid_xml: int
    empty_annotations: int
    classes: List[str]
    class_distribution: Dict[str, int]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for ISL gesture detection")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("archive (1)"),
        help="Root directory containing Pascal VOC XML and images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Output dataset directory in YOLO layout",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train split ratio between 0 and 1 (default: 0.8)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs") / "isl",
        help="Ultralytics project directory",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="yolov8n_isl",
        help="Ultralytics run name",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete output-dir before rebuilding dataset",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only prepare dataset",
    )
    parser.add_argument(
        "--skip-webcam",
        action="store_true",
        help="Skip webcam inference test",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLOv8 weights to start training from",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for webcam prediction",
    )
    return parser.parse_args()


def ensure_dirs(base_output: Path, overwrite: bool = False) -> Dict[str, Path]:
    if overwrite and base_output.exists():
        logging.info("Removing old output directory: %s", base_output)
        shutil.rmtree(base_output)

    paths = {
        "images_train": base_output / "images" / "train",
        "images_val": base_output / "images" / "val",
        "labels_train": base_output / "labels" / "train",
        "labels_val": base_output / "labels" / "val",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def voc_to_yolo_line(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: float,
    img_h: float,
    class_id: int,
) -> str:
    # Clamp coordinates to image bounds before normalization.
    xmin = max(0.0, min(xmin, img_w))
    xmax = max(0.0, min(xmax, img_w))
    ymin = max(0.0, min(ymin, img_h))
    ymax = max(0.0, min(ymax, img_h))

    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Invalid bounding box after clamping")

    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def find_matching_image(xml_path: Path, xml_filename_hint: Optional[str]) -> Optional[Path]:
    if xml_filename_hint:
        hinted = xml_path.parent / xml_filename_hint
        if hinted.exists() and hinted.suffix.lower() in IMAGE_EXTENSIONS:
            return hinted

    for ext in IMAGE_EXTENSIONS:
        candidate = xml_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    # Fallback for case-insensitive or unusual names.
    stem = xml_path.stem.lower()
    for candidate in xml_path.parent.iterdir():
        if candidate.is_file() and candidate.stem.lower() == stem and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate

    return None


def parse_voc_xml(xml_path: Path) -> Tuple[Optional[AnnotationRecord], Optional[str]]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None, "invalid_xml"

    size = root.find("size")
    if size is None:
        return None, "invalid_xml"

    try:
        img_w = float(size.findtext("width", default="0"))
        img_h = float(size.findtext("height", default="0"))
    except ValueError:
        return None, "invalid_xml"

    if img_w <= 0 or img_h <= 0:
        return None, "invalid_xml"

    xml_filename_hint = root.findtext("filename")
    image_path = find_matching_image(xml_path, xml_filename_hint)
    if image_path is None:
        return None, "missing_image"

    labels: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip().lower()
        bndbox = obj.find("bndbox")
        if not class_name or bndbox is None:
            continue

        try:
            xmin = float(bndbox.findtext("xmin", default="0"))
            ymin = float(bndbox.findtext("ymin", default="0"))
            xmax = float(bndbox.findtext("xmax", default="0"))
            ymax = float(bndbox.findtext("ymax", default="0"))
        except ValueError:
            continue

        labels.append((class_name, (xmin, ymin, xmax, ymax)))

    if not labels:
        return None, "empty_annotation"

    return (
        AnnotationRecord(
            xml_path=xml_path,
            image_path=image_path,
            image_filename=image_path.name,
            image_width=img_w,
            image_height=img_h,
            labels=labels,
        ),
        None,
    )


def collect_records(xml_files: Sequence[Path]) -> Tuple[List[AnnotationRecord], ConversionSummary]:
    valid_records: List[AnnotationRecord] = []
    missing_images = 0
    invalid_xml = 0
    empty_annotations = 0
    class_counts: Counter[str] = Counter()

    logging.info("Scanning %d XML files...", len(xml_files))
    for index, xml_file in enumerate(xml_files, start=1):
        record, error = parse_voc_xml(xml_file)
        if error == "missing_image":
            missing_images += 1
            continue
        if error == "invalid_xml":
            invalid_xml += 1
            continue
        if error == "empty_annotation":
            empty_annotations += 1
            continue
        if record is None:
            continue

        valid_records.append(record)
        class_counts.update([name for name, _ in record.labels])

        if index % 1000 == 0:
            logging.info("Parsed %d/%d XML files", index, len(xml_files))

    classes = sorted(class_counts.keys())
    summary = ConversionSummary(
        total_xml=len(xml_files),
        valid_records=len(valid_records),
        missing_images=missing_images,
        invalid_xml=invalid_xml,
        empty_annotations=empty_annotations,
        classes=classes,
        class_distribution=dict(class_counts),
    )
    return valid_records, summary


def write_yolo_labels(
    record: AnnotationRecord,
    class_to_id: Dict[str, int],
    target_label_file: Path,
) -> int:
    lines: List[str] = []
    for class_name, bbox in record.labels:
        class_id = class_to_id[class_name]
        xmin, ymin, xmax, ymax = bbox

        try:
            line = voc_to_yolo_line(
                xmin,
                ymin,
                xmax,
                ymax,
                record.image_width,
                record.image_height,
                class_id,
            )
            lines.append(line)
        except ValueError:
            continue

    target_label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def build_dataset(
    records: Sequence[AnnotationRecord],
    output_dir: Path,
    split_ratio: float,
    seed: int,
    overwrite_output: bool,
) -> Path:
    class_names = sorted({name for record in records for name, _ in record.labels})
    if not class_names:
        raise RuntimeError("No valid classes found. Check XML annotations.")

    class_to_id = {class_name: idx for idx, class_name in enumerate(class_names)}

    logging.info("Creating YOLO folder structure...")
    paths = ensure_dirs(output_dir, overwrite=overwrite_output)

    mutable_records = list(records)
    random.Random(seed).shuffle(mutable_records)
    split_index = int(len(mutable_records) * split_ratio)
    train_records = mutable_records[:split_index]
    val_records = mutable_records[split_index:]

    if not train_records or not val_records:
        raise RuntimeError(
            "Train/val split produced an empty subset. Adjust --split-ratio or dataset size."
        )

    logging.info("Writing train/val images and YOLO labels...")
    train_boxes = 0
    val_boxes = 0

    for split_name, split_records in (("train", train_records), ("val", val_records)):
        for idx, record in enumerate(split_records, start=1):
            if split_name == "train":
                image_target = paths["images_train"] / record.image_filename
                label_target = paths["labels_train"] / f"{record.image_path.stem}.txt"
            else:
                image_target = paths["images_val"] / record.image_filename
                label_target = paths["labels_val"] / f"{record.image_path.stem}.txt"

            shutil.copy2(record.image_path, image_target)
            boxes = write_yolo_labels(record, class_to_id, label_target)

            if split_name == "train":
                train_boxes += boxes
            else:
                val_boxes += boxes

            if idx % 1000 == 0:
                logging.info("Prepared %s samples: %d", split_name, idx)

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    logging.info("Dataset build complete")
    logging.info("Train images: %d | Val images: %d", len(train_records), len(val_records))
    logging.info("Train boxes: %d | Val boxes: %d", train_boxes, val_boxes)

    return yaml_path


def summarize_dataset(summary: ConversionSummary) -> None:
    logging.info("=" * 72)
    logging.info("DATASET SUMMARY")
    logging.info("Total XML files: %d", summary.total_xml)
    logging.info("Valid annotated samples: %d", summary.valid_records)
    logging.info("Missing image pairs: %d", summary.missing_images)
    logging.info("Invalid XML files: %d", summary.invalid_xml)
    logging.info("Empty annotations: %d", summary.empty_annotations)
    logging.info("Number of classes: %d", len(summary.classes))
    logging.info("Classes: %s", ", ".join(summary.classes))
    logging.info("Class distribution:")
    for class_name, count in sorted(summary.class_distribution.items()):
        logging.info("  - %s: %d", class_name, count)
    logging.info("=" * 72)


def train_yolov8(
    yaml_path: Path,
    weights: str,
    epochs: int,
    imgsz: int,
    project: Path,
    run_name: str,
) -> Tuple[Path, Optional[Path]]:
    logging.info("Starting YOLOv8 training with weights: %s", weights)
    model = YOLO(weights)

    train_results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        project=str(project),
        name=run_name,
        verbose=True,
    )

    save_dir = None
    if hasattr(train_results, "save_dir"):
        save_dir = Path(train_results.save_dir)
    elif getattr(model, "trainer", None) is not None and hasattr(model.trainer, "save_dir"):
        save_dir = Path(model.trainer.save_dir)

    if save_dir is None:
        raise RuntimeError("Training finished but could not locate run directory")

    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise RuntimeError(f"best.pt not found at expected location: {best_weights}")

    root_best = Path("best.pt")
    shutil.copy2(best_weights, root_best)
    logging.info("Best weights copied to: %s", root_best.resolve())

    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        print_training_metrics(results_csv)
    else:
        logging.warning("Training metrics CSV not found at: %s", results_csv)

    return best_weights, save_dir


def print_training_metrics(results_csv: Path) -> None:
    try:
        with results_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            logging.warning("No rows found in training metrics CSV")
            return

        last = rows[-1]
        logging.info("FINAL TRAINING METRICS (last epoch):")
        for metric_key in (
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
        ):
            if metric_key in last and last[metric_key] != "":
                logging.info("  %s: %s", metric_key, last[metric_key])
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("Failed to parse training metrics: %s", exc)


def run_webcam_inference(weights_path: Path, conf: float) -> None:
    logging.info("Launching webcam inference using %s", weights_path)
    logging.info("Press 'q' in the prediction window to exit webcam test.")
    model = YOLO(str(weights_path))
    model.predict(source=0, show=True, conf=conf)


def discover_xml_files(source_dir: Path) -> List[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    xml_files = list(source_dir.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found under: {source_dir}")
    return xml_files


def convert_all_annotations(xml_files: Sequence[Path]) -> Tuple[List[AnnotationRecord], ConversionSummary]:
    records, summary = collect_records(xml_files)
    if not records:
        raise RuntimeError("No valid image/XML annotation pairs found.")
    return records, summary


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.split_ratio < 1.0:
        raise ValueError("--split-ratio must be between 0 and 1")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be > 0")


def main() -> None:
    setup_logging()
    args = parse_args()
    validate_args(args)

    logging.info("ISL YOLOv8 pipeline started")
    logging.info("Source dir: %s", args.source_dir.resolve())
    logging.info("Output dir: %s", args.output_dir.resolve())

    xml_files = discover_xml_files(args.source_dir)
    records, conversion_summary = convert_all_annotations(xml_files)

    # Reuse conversion summary from full dataset and print before training.
    summarize_dataset(conversion_summary)

    yaml_path = build_dataset(
        records=records,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
        seed=args.seed,
        overwrite_output=args.overwrite_output,
    )
    logging.info("data.yaml created at: %s", yaml_path.resolve())

    best_weights = Path("best.pt")
    if not args.skip_train:
        best_weights, run_dir = train_yolov8(
            yaml_path=yaml_path,
            weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            project=args.project,
            run_name=args.run_name,
        )
        logging.info("Training run directory: %s", run_dir)
    elif not args.skip_webcam and not best_weights.exists():
        raise FileNotFoundError(
            "--skip-train was used but best.pt was not found in workspace root"
        )

    if not args.skip_webcam:
        run_webcam_inference(best_weights, conf=args.conf)

    logging.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
