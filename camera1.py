"""Standalone camera-1 object-detection entry point."""

import cv2

from object_detector import ObjectDetector


def main() -> None:
    detector = ObjectDetector(model_path="camera1_segmodel.pt", conf_threshold=0.50)
    if not detector.load_model():
        print("Failed to load model. Exiting.")
        return

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while True:
            received, frame = camera.read()
            if not received:
                print("Error: Can't receive frame.")
                break
            annotated_frame, _ = detector.detect(frame, draw_boxes=True)
            cv2.imshow("Camera 1", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()