"""
YOLOv8 Training Script for Indian Sign Language (ISL) Gesture Detection

This script provides a clean interface to train YOLOv8 models on the ISL dataset.
It includes:
- GPU/CPU auto-detection
- Early stopping
- Resumable training
- Configurable hyperparameters
- Training metrics logging
- Best model checkpoint saving

Usage:
    Basic training (50 epochs):
    $ py train_model.py

    Custom epochs and batch size:
    $ py train_model.py --epochs 100 --batch-size 32

    Resume from checkpoint:
    $ py train_model.py --resume runs/detect/train/weights/last.pt

    Large model (yolov8l instead of yolov8n):
    $ py train_model.py --model yolov8l --epochs 100

    With specific device (GPU or CPU):
    $ py train_model.py --device 0  # GPU index 0
    $ py train_model.py --device cpu
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def setup_logging(level=logging.INFO) -> None:
    """Configure logging to terminal with timestamps and levels."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def detect_device() -> str:
    """
    Auto-detect available device for training.

    Returns:
        str: '0' for CUDA GPU, 'cpu' for CPU, 'mps' for Apple Silicon
    """
    if torch.cuda.is_available():
        device = "0"
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logging.info("✓ GPU detected: %s (Device: cuda:%s)", gpu_name, device)
        if gpu_count > 1:
            logging.info("  (Total GPUs available: %d)", gpu_count)
    elif torch.backends.mps.is_available():
        device = "mps"
        logging.info("✓ Apple Metal Performance Shaders (MPS) available")
    else:
        device = "cpu"
        logging.warning("⚠ No GPU detected. Training will use CPU (slower)")

    return device


def validate_dataset_config(data_yaml_path: Path) -> bool:
    """
    Validate that data.yaml exists and has required fields.

    Args:
        data_yaml_path: Path to data.yaml file

    Returns:
        bool: True if valid, False otherwise

    Raises:
        FileNotFoundError: If data.yaml doesn't exist
    """
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    with data_yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_keys = {"path", "train", "val", "nc", "names"}
    # nc is optional if explicitly counting names
    if "nc" not in config:
        config["nc"] = len(config.get("names", []))

    missing = required_keys - set(config.keys())
    if missing:
        logging.error("data.yaml missing required fields: %s", missing)
        return False

    logging.info("✓ data.yaml validated")
    logging.info("  Classes: %d", config["nc"])
    logging.info("  Class names: %s", ", ".join(config["names"][:5]))
    if len(config["names"]) > 5:
        logging.info("  ... and %d more classes", len(config["names"]) - 5)

    return True


def load_pretrained_model(model_name: str) -> YOLO:
    """
    Load a pretrained YOLOv8 model.

    Args:
        model_name: Model size ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')

    Returns:
        YOLO: Loaded model instance

    Raises:
        ValueError: If model name is invalid
    """
    valid_models = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
    if model_name not in valid_models:
        raise ValueError(
            f"Invalid model: {model_name}. Choose from: {', '.join(valid_models)}"
        )

    weights_name = f"{model_name}.pt"
    logging.info("Loading pretrained model: %s", weights_name)

    try:
        model = YOLO(weights_name)
        logging.info("✓ Model loaded successfully")
        return model
    except Exception as exc:
        logging.error("Failed to load model: %s", exc)
        raise


def train_model(
    model: YOLO,
    data_yaml: Path,
    epochs: int,
    batch_size: int,
    imgsz: int,
    device: str,
    patience: int = 30,
    project: str = "runs/detect",
    name: str = "train",
) -> Path:
    """
    Train the YOLOv8 model on custom dataset.

    Args:
        model: YOLO model instance
        data_yaml: Path to data.yaml file
        epochs: Number of training epochs
        batch_size: Batch size for training (-1 for auto)
        imgsz: Training image size (default 640)
        device: Device to use ('0' for GPU, 'cpu', 'mps')
        patience: Early stopping patience (epochs without improvement)
        project: Output project directory
        name: Run name for outputs

    Returns:
        Path: Directory containing training results

    Raises:
        RuntimeError: If training fails
    """
    logging.info("=" * 72)
    logging.info("TRAINING CONFIGURATION")
    logging.info("=" * 72)
    logging.info("Epochs: %d", epochs)
    logging.info("Batch size: %s", batch_size if batch_size > 0 else "auto")
    logging.info("Image size: %d", imgsz)
    logging.info("Device: %s", device)
    logging.info("Early stopping patience: %d epochs", patience)
    logging.info("Project directory: %s", project)
    logging.info("Run name: %s", name)
    logging.info("=" * 72)

    try:
        logging.info("Starting training...")
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            patience=patience,
            project=project,
            name=name,
            verbose=True,
            # Additional useful training parameters
            save=True,  # Save model checkpoints
            save_period=-1,  # Save every epoch if positive
            cache=True,  # Cache images for faster training
            plots=True,  # Generate training plots
            hsv_h=0.015,  # HSV-Hue augmentation
            hsv_s=0.7,  # HSV-Saturation augmentation
            hsv_v=0.4,  # HSV-Value augmentation
            degrees=10.0,  # Rotation augmentation
            translate=0.1,  # Translation augmentation
            scale=0.5,  # Scale augmentation
            flipud=0.0,  # Flip upside-down
            fliplr=0.5,  # Flip left-right
            mosaic=1.0,  # Mosaic augmentation
            mixup=0.0,  # Mixup augmentation
            copy_paste=0.0,  # Copy-Paste augmentation
        )

        logging.info("✓ Training completed successfully")
        return Path(results.save_dir)

    except Exception as exc:
        logging.error("Training failed: %s", exc)
        raise RuntimeError(f"Training error: {exc}") from exc


def save_best_model(run_dir: Path, workspace_root: Path = Path(".")) -> Path:
    """
    Copy best model from run directory to workspace root.

    Args:
        run_dir: Training run directory containing weights/
        workspace_root: Root workspace directory

    Returns:
        Path: Path to copied best.pt file

    Raises:
        FileNotFoundError: If best.pt not found in run directory
    """
    best_weights = run_dir / "weights" / "best.pt"

    if not best_weights.exists():
        raise FileNotFoundError(f"best.pt not found at: {best_weights}")

    output_path = workspace_root / "best.pt"
    logging.info("Copying best model to: %s", output_path)

    import shutil

    shutil.copy2(best_weights, output_path)
    logging.info("✓ Best model saved to: %s", output_path.resolve())

    return output_path


def print_training_summary(run_dir: Path) -> None:
    """
    Print summary of training results including metrics.

    Args:
        run_dir: Training run directory
    """
    metrics_file = run_dir / "results.csv"

    if not metrics_file.exists():
        logging.warning("Training metrics file not found at: %s", metrics_file)
        return

    import csv

    try:
        with metrics_file.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            logging.warning("No rows in training metrics CSV")
            return

        last_epoch = rows[-1]
        first_epoch = rows[0]

        logging.info("=" * 72)
        logging.info("TRAINING METRICS SUMMARY")
        logging.info("=" * 72)

        # Get improvement from first to last epoch
        metrics_to_print = [
            ("Train Box Loss", "train/box_loss"),
            ("Train Class Loss", "train/cls_loss"),
            ("Val Box Loss", "val/box_loss"),
            ("Val Class Loss", "val/cls_loss"),
            ("mAP50 (B)", "metrics/mAP50(B)"),
            ("mAP50-95 (B)", "metrics/mAP50-95(B)"),
        ]

        for label, key in metrics_to_print:
            if key in last_epoch and last_epoch[key]:
                try:
                    last_val = float(last_epoch[key])
                    first_val = float(first_epoch.get(key, last_val))

                    # Calculate improvement
                    if label.startswith(("Train", "Val")) and "Loss" in label:
                        # Loss should decrease
                        improvement = first_val - last_val
                        direction = "↓" if improvement > 0 else "↑"
                    else:
                        # mAP should increase
                        improvement = last_val - first_val
                        direction = "↑" if improvement > 0 else "↓"

                    logging.info(
                        "  %s: %.4f (change: %s%.4f)",
                        label,
                        last_val,
                        direction,
                        abs(improvement),
                    )
                except (ValueError, KeyError):
                    pass

        logging.info("=" * 72)
        logging.info("Full results available at: %s", run_dir)

    except Exception as exc:
        logging.warning("Failed to parse training metrics: %s", exc)


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for ISL gesture detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py train_model.py                                    # Train with defaults (50 epochs)
  py train_model.py --epochs 100 --batch-size 32     # Custom batch size
  py train_model.py --model yolov8l --epochs 150     # Larger model
  py train_model.py --resume runs/detect/train2/weights/last.pt  # Resume training
  py train_model.py --device cpu --epochs 10         # CPU training
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLOv8 model size (default: yolov8n - nano, fastest)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=-1,
        help="Batch size for training. -1 for auto (default: -1)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: 640)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience in epochs (default: 30)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: '0' for GPU:0, 'cpu', 'mps', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Output project directory (default: runs/detect)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Run name (output saves to project/name/) (default: train)",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("dataset") / "data.yaml",
        help="Path to data.yaml file (default: dataset/data.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from checkpoint weights file",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for training script."""
    setup_logging()

    logging.info("=" * 72)
    logging.info("YOLOv8 ISL Gesture Detection Training")
    logging.info("=" * 72)

    args = parse_arguments()

    # Detect device if auto
    if args.device.lower() == "auto":
        args.device = detect_device()
    else:
        if args.device != "cpu":
            logging.info("Using specified device: %s", args.device)

    # Validate dataset configuration
    try:
        validate_dataset_config(args.data_yaml)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1

    # Load model
    try:
        if args.resume:
            logging.info("Resuming training from checkpoint: %s", args.resume)
            if not args.resume.exists():
                logging.error("Checkpoint not found: %s", args.resume)
                return 1
            model = YOLO(str(args.resume))
        else:
            model = load_pretrained_model(args.model)
    except (ValueError, Exception) as exc:
        logging.error("Model loading failed: %s", exc)
        return 1

    # Train model
    try:
        run_dir = train_model(
            model=model,
            data_yaml=args.data_yaml,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            device=args.device,
            patience=args.patience,
            project=args.project,
            name=args.name,
        )
    except RuntimeError as exc:
        logging.error("Training failed: %s", exc)
        return 1

    # Save best model
    try:
        save_best_model(run_dir)
    except FileNotFoundError as exc:
        logging.warning("Could not save best.pt: %s", exc)
    except Exception as exc:
        logging.warning("Error saving best model: %s", exc)

    # Print summary
    print_training_summary(run_dir)

    logging.info("")
    logging.info("Training pipeline completed successfully!")
    logging.info("Results saved to: %s", run_dir.resolve())
    logging.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
