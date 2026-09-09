"""Camera-2 detector compatibility module."""

import numpy as np

from detector_base import BaseObjectDetector


class ObjectDetector(BaseObjectDetector):
    """Object detector preserving camera 2's seeded color generation."""

    def __init__(self, model_path: str = "camera2_segmodel.pt", conf_threshold: float = 0.50):
        super().__init__(model_path, conf_threshold)

    def _get_color_for_class(self, class_id: int):
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))