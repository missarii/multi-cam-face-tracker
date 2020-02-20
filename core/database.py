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
                
                # Create face_logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS face_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        camera_id INTEGER NOT NULL,
                        camera_name TEXT NOT NULL,
                        face_name TEXT NOT NULL,
                        age INTEGER,
                        gender TEXT,
                        confidence REAL NOT NULL,
                        screenshot_path TEXT
                    )
                ''')
                
                # Create known_faces table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS known_faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        embedding BLOB NOT NULL,
                        image_path TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                ''')
                
                conn.commit()
                logger.success("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def log_face_event(self, event) -> int:
        """Log a face recognition event to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO face_logs (
                        timestamp, camera_id, camera_name, face_name,
                        age, gender, confidence, screenshot_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    float(event.timestamp),
                    int(event.camera_id),
