"""Camera-1 detector compatibility module."""

from detector_base import BaseObjectDetector, Color, Detection


class ObjectDetector(BaseObjectDetector):
    """Object detector using camera 1's palette and warning color policy."""

    def __init__(self, model_path: str = "camera1_segmodel.pt", conf_threshold: float = 0.50):
        super().__init__(model_path, conf_threshold)

    def _get_color_for_detection(self, detection: Detection) -> Color:
        if detection['class_name'] == "Not allowed":
            return (0, 0, 255)
        return super()._get_color_for_detection(detection)