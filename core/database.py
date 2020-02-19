import sqlite3
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from loguru import logger
import time

@dataclass
class FaceLogEntry:
    id: int
    timestamp: float
    camera_id: int
    camera_name: str
    face_name: str
    age: Optional[int]
    gender: Optional[str]
    confidence: float
    screenshot_path: Optional[str]

    def __post_init__(self):
        # Convert timestamp to float if it comes as bytes
        if isinstance(self.timestamp, bytes):
            self.timestamp = float(self.timestamp.decode('utf-8'))
        elif isinstance(self.timestamp, str):
            self.timestamp = float(self.timestamp)

class FaceDatabase:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database with required tables"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
