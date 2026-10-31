"""Detect sewer pipe defects in an image, a folder, a video, or a live stream."""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run the trained model on new footage.")
    parser.add_argument("--weights", default="weights/best.pt", help="trained model file")
    parser.add_argument("--source", required=True,
                        help="image, folder, video file, stream URL, or 0 for a webcam")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="minimum confidence (0 to 1) for a detection to be kept")
    args = parser.parse_args()

    model = YOLO(args.weights)

    # stream=True handles one frame at a time, so long videos don't fill up memory.
    # Annotated images and label files are saved to runs/detect/predict*.
    for _ in model.predict(source=args.source, conf=args.conf, stream=True,
                           save=True, save_txt=True, save_conf=True):
        pass


if __name__ == "__main__":
    main()
