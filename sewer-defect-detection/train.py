"""Train a YOLOv8 model to detect sewer pipe defects.

The defaults reproduce the original 300-epoch run (see results/args.yaml).
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the sewer defect dataset.")
    parser.add_argument("--data", default="data.yaml", help="dataset config file")
    parser.add_argument("--epochs", type=int, default=300, help="passes through the training images")
    parser.add_argument("--batch", type=int, default=16, help="images per training step")
    parser.add_argument("--imgsz", type=int, default=640, help="image size in pixels")
    parser.add_argument("--patience", type=int, default=100,
                        help="stop early if the validation score hasn't improved in this many epochs")
    parser.add_argument("--pretrained", action="store_true",
                        help="start from COCO-pretrained weights instead of random weights")
    args = parser.parse_args()

    # yolov8n.yaml builds the network with random weights (how the original run trained).
    # yolov8n.pt starts from weights already trained on the COCO dataset.
    model = YOLO("yolov8n.pt" if args.pretrained else "yolov8n.yaml")
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        seed=0,
    )


if __name__ == "__main__":
    main()
