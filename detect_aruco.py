"""Standalone ArUco marker detector entry point."""

import cv2
import numpy as np


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
WINDOW_TITLE = 'ArUco Detector - DICT_5X5_50'


def main() -> None:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    camera = cv2.VideoCapture(1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    print("ArUco Marker Detector - DICT_5X5_50")
    print("Press 'q' to quit")
    print("-" * 40)

    try:
        while True:
            received, frame = camera.read()
            if not received:
                print("Error: Can't receive frame.")
                break

            corners, ids, _ = aruco_detector.detectMarkers(frame)
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                _draw_marker_sizes(frame, corners)
            else:
                cv2.putText(
                    frame, "No ArUco markers detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            cv2.putText(
                frame, "DICT_5X5_50", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print("\nDetection stopped.")


def _draw_marker_sizes(frame, corners) -> None:
    for marker in corners:
        marker_corners = marker[0]
        center_x = int(np.mean(marker_corners[:, 0]))
        center_y = int(np.mean(marker_corners[:, 1]))
        top_width = np.linalg.norm(marker_corners[0] - marker_corners[1])
        bottom_width = np.linalg.norm(marker_corners[3] - marker_corners[2])
        average_width = (top_width + bottom_width) / 2
        cv2.putText(
            frame, f"{average_width:.0f}px", (center_x - 25, center_y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )


if __name__ == "__main__":
    main()