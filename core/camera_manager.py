import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import time
import threading
import queue
import yaml
from pathlib import Path

@dataclass
class CameraConfig:
    id: int
    name: str
    source: str
    enabled: bool
    width: int
    height: int
    fps: int
    rotate: int

class CameraManager:
    def __init__(self, config_path: str):
        self.cameras: Dict[int, CameraConfig] = {}
        self.capture_threads: Dict[int, threading.Thread] = {}
        self.stop_event = threading.Event()
        self.frame_queues: Dict[int, queue.Queue] = {}
        self.load_config(config_path)

    def _cleanup_camera_thread(self, cam_id: int):
        """Clean up camera thread resources"""
        if cam_id in self.capture_threads:
            # Signal thread to stop
            self.stop_event.set()
