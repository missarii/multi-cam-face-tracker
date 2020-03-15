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

        
    def trigger_alert(self, camera_id: int, camera_name: str, face_name: str, face: Face, confidence: float, frame: np.ndarray) -> AlertEvent:
        """Trigger an alert for a recognized face"""
        timestamp = time.time()
        screenshot_path = None
        
        if self.screenshot_enabled:
            screenshot_path = self._capture_screenshot(frame, camera_id, face_name, timestamp)
         # Ensure the path is converted to string and is not None when empty
        
        event = AlertEvent(
            camera_id=camera_id,
            camera_name=camera_name,
            face_name=face_name,
            age=face.age,
            gender=face.gender,
            confidence=confidence,
            timestamp=timestamp,
            screenshot_path=str(screenshot_path) if screenshot_path is not None else None
        )
        logger.debug(f"Event created with screenshot path: {event.screenshot_path}")  # Debug log
        self.alert_history.append(event)

        if self.alert_enabled:
            self._play_alert_sound()
        
        if self.telegram:
            message_lines = [
                "🚨 Face detected!",
                f"👤 Name: {face_name}"
            ]

            if face.age:
                message_lines.append(f"👶 Age: ~{face.age} years")
            if face.gender:
                message_lines.append(f"🚻 Gender: {face.gender}")
                
            message_lines.extend([
                f"📷 Camera: {camera_name}",
                f"🎯 Confidence: {confidence:.2%}",
                f"⏰ Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            ])

            message = "\n".join(message_lines)
            
            self.telegram.send_alert(
                message=message,
                image_path=screenshot_path
            )
            
        logger.info(f"Alert triggered: {face_name} detected on {camera_name} with confidence {confidence:.2f}")
        return event
        
    def _play_alert_sound(self) -> None:
        """
        The function `_play_alert_sound` attempts to play an alert sound file using the `mixer.music`
        module in Python, logging an error if there is an exception.
        """
        try:
            if os.path.exists(self.alert_sound):
                # playsound(self.alert_sound, block=False)
                  mixer.music.load(self.alert_sound)
                  mixer.music.play()
        except Exception as e:
            logger.error(f"Error playing alert sound: {e}")

    def _capture_screenshot(self, frame: np.ndarray, camera_id: int, face_name: str, timestamp: float) -> Optional[Path]:
        """Capture and save a screenshot of the alert"""
        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
            filename = f"{timestamp_str}_cam{camera_id}_{face_name}.jpg"
            filepath = self.screenshot_dir / filename
