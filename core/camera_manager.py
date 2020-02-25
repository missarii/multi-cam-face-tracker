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
            
            # Wait for thread to finish
            thread = self.capture_threads.pop(cam_id)
            thread.join(timeout=2.0)
            
            if thread.is_alive():
                logger.warning(f"Camera ID {cam_id} thread did not stop gracefully")
            
            # Clean up queue
            if cam_id in self.frame_queues:
                try:
                    while True:
                        self.frame_queues[cam_id].get_nowait()
                except queue.Empty:
                    pass
                del self.frame_queues[cam_id]
                
            logger.debug(f"Cleaned up resources for camera ID {cam_id}")

        
    def load_config(self, config_path: str) -> None:
        """Load camera configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            self.cameras.clear()
            for cam_config in config.get('cameras', []):
                cam_id = cam_config['id']
                self.cameras[cam_id] = CameraConfig(
                    id=cam_id,
                    name=cam_config.get('name', f'Camera {cam_id}'),
                    source=cam_config['source'],
                    enabled=cam_config.get('enabled', True),
                    width=cam_config['resolution']['width'],
                    height=cam_config['resolution']['height'],
                    fps=cam_config.get('fps', 30),
                    rotate=cam_config.get('rotate', 0)
                )
