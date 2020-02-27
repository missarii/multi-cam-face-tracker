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
        
    def _load_model(self) -> FaceAnalysis:
        """Load Model insightface"""
        try:
            model = FaceAnalysis(
                name='buffalo_l',
                root='./models',
                allowed_modules=['detection', 'recognition', 'genderage']
            )
            model.prepare(
                ctx_id=0 if self.device == 'cuda' else -1,
                det_thresh=self.detection_threshold,
                det_size=(640, 640)
            )
            logger.success("Face detection model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load face detection model: {e}")
            raise

    def load_known_faces(self, known_faces_dir: str) -> None:
        """Load known faces from directory"""
        try:
            self.known_faces.clear()
            known_faces_dir = Path(known_faces_dir)
            
            if not known_faces_dir.exists():
                logger.warning(f"Known faces directory {known_faces_dir} does not exist")
                return
                
            for face_file in known_faces_dir.glob('*.*'):
                if face_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                    
                try:
                    img = cv2.imread(str(face_file))
                    if img is None:
                        logger.warning(f"Could not read image {face_file}")
                        continue
                        
                    faces = self.model.get(img)
                    if len(faces) == 0:
                        logger.warning(f"No faces found in {face_file}")
                        continue
                        
                    # Use the first face found in the image
                    face = faces[0]
                    name = face_file.stem
                    self.known_faces.append(KnownFace(
                        name=name,
                        embedding=face.embedding,
                        image_path=str(face_file)
                    ))
                    logger.info(f"Loaded known face: {name}")
