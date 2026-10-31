"""Score a trained model on the training or validation images."""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model.")
    parser.add_argument("--weights", default="weights/best.pt", help="trained model file")
    parser.add_argument("--data", default="data.yaml", help="dataset config file")
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="which images to score")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split)

    # Ultralytics also prints a per-class table and saves plots to runs/detect/val*.
    print(f"mAP@0.5:      {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
