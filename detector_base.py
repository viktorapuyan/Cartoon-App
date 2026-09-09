"""Shared YOLO object-detector implementation."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


Detection = Dict[str, Any]
Color = Tuple[int, int, int]


class BaseObjectDetector:
    """Load a YOLO model, detect objects, and render detection boxes."""

    _PALETTE: Tuple[Color, ...] = (
        (255, 200, 0), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 255, 0), (128, 0, 255), (0, 165, 255), (255, 0, 255),
        (42, 255, 165), (255, 128, 0),
    )

    def __init__(self, model_path: str, conf_threshold: float = 0.50):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None

    def load_model(self) -> bool:
        """Load the configured YOLO model, reporting failures to the console."""
        try:
            from ultralytics import YOLO

            if not Path(self.model_path).exists():
                print(f"Error: Model file not found at {self.model_path}")
                return False
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except ImportError:
            print("Error: ultralytics package not installed. Install with: pip install ultralytics")
        except Exception as exc:
            print(f"Error loading model: {exc}")
        return False

    def detect(self, frame: np.ndarray, draw_boxes: bool = True) -> Tuple[np.ndarray, List[Detection]]:
        """Run inference and optionally annotate the returned frame."""
        if self.model is None:
            print("Error: Model not loaded. Call load_model() first.")
            return frame, []
        if frame is None or frame.size == 0:
            return frame, []

        try:
            result = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
            detections = self._extract_detections(result)
            annotated_frame = self.draw_bounding_boxes(frame.copy(), detections) if draw_boxes else frame
            return annotated_frame, detections
        except Exception as exc:
            print(f"Error during detection: {exc}")
            return frame, []

    def _extract_detections(self, result: Any) -> List[Detection]:
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        return [
            {
                'bbox': boxes[index],
                'confidence': float(confidences[index]),
                'class_id': int(class_ids[index]),
                'class_name': self.model.names[int(class_ids[index])],
            }
            for index in range(len(boxes))
        ]

    def draw_bounding_boxes(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw all detection boxes and labels onto ``frame``."""
        for detection in detections:
            self._draw_detection(frame, detection)
        return frame

    def _draw_detection(self, frame: np.ndarray, detection: Detection) -> None:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        color = self._get_color_for_detection(detection)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = detection['class_name']
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            frame, (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1), color, -1,
        )
        cv2.putText(
            frame, label, (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
        )

    def _get_color_for_detection(self, detection: Detection) -> Color:
        return self._get_color_for_class(detection['class_id'])

    def _get_color_for_class(self, class_id: int) -> Color:
        return self._PALETTE[class_id % len(self._PALETTE)]

    @staticmethod
    def get_detections_count(detections: List[Detection]) -> int:
        return len(detections)

    @staticmethod
    def filter_detections_by_class(detections: List[Detection], class_name: str) -> List[Detection]:
        return [detection for detection in detections if detection['class_name'] == class_name]