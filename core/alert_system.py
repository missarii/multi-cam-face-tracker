import os
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
from loguru import logger
# from playsound import playsound
import cv2
import numpy as np
from pathlib import Path
from pygame import mixer

from .telegram_manager import TelegramManager
from .face_detection import Face

# The `@dataclass` decorator in Python is used to automatically generate special methods such as
# `__init__`, `__repr__`, `__eq__`, and `__hash__` for a class. In this specific case, the
# `AlertEvent` class is a data class that represents an alert event with the following attributes:
@dataclass
class AlertEvent:
    camera_id: int
    camera_name: str
    face_name: str
    confidence: float
    timestamp: float
    age: Optional[int] = None
    gender: Optional[str] = None  # 'Male' or 'Female'
    screenshot_path: Optional[str] = None

class AlertSystem:
    def __init__(self, config: dict):
        self.config = config
        self.alert_sound = config['app']['alert_sound']
        self.screenshot_dir = Path(config['app']['screenshot_dir'])
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.alert_history: List[AlertEvent] = []
        self.alert_enabled = True
        self.screenshot_enabled = True
        self.telegram = None
        if config.get('telegram', {}).get('enabled', False):
            self.telegram = TelegramManager(
                config['telegram']['bot_token'],
                config['telegram']['chat_id'],
                config['telegram']['rate_limit']
            )
        mixer.init()  # Add this at the start of your application
