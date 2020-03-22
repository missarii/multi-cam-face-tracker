from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton,
                            QLabel, QDateEdit, QComboBox, QSpacerItem, QSizePolicy,
                            QSplitter, QFrame, QMessageBox, QDialog)
from PyQt5.QtCore import Qt, QDate, QDateTime
from PyQt5.QtGui import QPixmap
from loguru import logger
import time
from datetime import datetime, timedelta
from typing import List, Optional
import cv2

from core.database import FaceDatabase, FaceLogEntry
from core.utils import numpy_to_pixmap

class HistoryViewer(QWidget):
    def __init__(self, database, config):
        """Initialize the HistoryViewer with database and configuration, set up UI and load initial data."""
        super().__init__()
        self.database = database
        self.config = config
        self.current_entry = None
        
        self.setup_ui()
        self.load_camera_list()
        self.load_face_list()
        self.refresh_history()
        
    def setup_ui(self):
        """Set up all UI components including filters, list view, and detail view for history entries."""
        main_layout = QVBoxLayout()
        
        # Filter controls
        filter_layout = QHBoxLayout()
        
        # Date range filters
        date_layout = QVBoxLayout()
        date_layout.addWidget(QLabel("Date Range:"))
        
        date_range_layout = QHBoxLayout()
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addDays(-7))
        self.start_date.setCalendarPopup(True)
        date_range_layout.addWidget(self.start_date)
        
        date_range_layout.addWidget(QLabel("to"))
        
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        date_range_layout.addWidget(self.end_date)
        
        date_layout.addLayout(date_range_layout)
        filter_layout.addLayout(date_layout)
