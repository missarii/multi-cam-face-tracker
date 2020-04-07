"""
Multi-Camera Face Tracker System
--------------------------------
This is the main entry point for launching the facial recognition and tracking application.
It initializes the GUI, camera streams, face detection, recognition, alert system, and database.

Features:
- Real-time face detection and recognition across multiple camera feeds
- Target face matching with alert notification
- Event logging with timestamp and screenshot
- User-friendly desktop UI built with PyQt5

Author: Darshan Vichhi (Aarambh Dev Hub)
Created: 2025-05-21
"""



import sys
import yaml
from pathlib import Path
from loguru import logger
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer

from ui.main_window import MainWindow

def load_config(config_path: str) -> dict:
    """
    Load the application's configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration settings as a dictionary.

    Side Effects:
        - Creates necessary directories for screenshots, known faces, and logs if they do not exist.

    Raises:
        Exception: If the configuration file cannot be read or parsed.

    Usage:
        config = load_config('config/config.yaml')
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
