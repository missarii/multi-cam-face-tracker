import cv2
import numpy as np
from typing import Tuple, Optional
from loguru import logger
import time
from PyQt5.QtGui import QPixmap

def draw_face_info(image: np.ndarray, 
                  face_bbox: Tuple[int, int, int, int],
                  name: Optional[str] = None,
                  confidence: Optional[float] = None,
                  age: Optional[int] = None,
                  gender: Optional[str] = None,
                  camera_name: Optional[str] = None,
                  timestamp: Optional[float] = None) -> np.ndarray:
    """
    Draw face bounding box and information on the image
    """
    try:
        img = image.copy()
        x1, y1, x2, y2 = map(int, face_bbox)
        
        # Draw bounding box
        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
