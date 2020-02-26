import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from loguru import logger
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
from pathlib import Path
import time

@dataclass
class Face:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    kps: np.ndarray   # 5 key points
    det_score: float  # detection score
    embedding: np.ndarray
    age: Optional[int] = None
    gender: Optional[str] = None  # 'Male' or 'Female'
    face_img: Optional[np.ndarray] = None

@dataclass
class KnownFace:
    name: str
    embedding: np.ndarray
    image_path: str

class FaceDetector:
    def __init__(self, config: dict):
        self.config = config
        self.recognition_threshold = config['recognition']['recognition_threshold']
        self.detection_threshold = config['recognition']['detection_threshold']
        self.max_batch_size = config['recognition']['max_batch_size']
        self.device = config['recognition']['device']
        self.analysis_enabled = config['recognition'].get('analysis_enabled', True)
        self.model = self._load_model()
        self.known_faces: List[KnownFace] = []
