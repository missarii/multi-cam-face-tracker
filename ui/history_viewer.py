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
        
        # Camera filter
        camera_layout = QVBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("All Cameras", None)
        camera_layout.addWidget(self.camera_combo)
        filter_layout.addLayout(camera_layout)
        
        # Face filter
        face_layout = QVBoxLayout()
        face_layout.addWidget(QLabel("Face:"))
        self.face_combo = QComboBox()
        self.face_combo.addItem("All Faces", None)
        face_layout.addWidget(self.face_combo)
        filter_layout.addLayout(face_layout)
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_history)
        filter_layout.addWidget(self.refresh_btn)
        
        filter_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        main_layout.addLayout(filter_layout)
        
        # Splitter for history list and details
        splitter = QSplitter(Qt.Horizontal)
        
        # History list
        self.history_list = QListWidget()
        self.history_list.currentItemChanged.connect(self.on_history_item_selected)
        splitter.addWidget(self.history_list)
        
        # Details panel
        details_frame = QFrame()
        details_frame.setFrameShape(QFrame.StyledPanel)
        details_layout = QVBoxLayout()
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        details_layout.addWidget(self.image_label)
        
        # Details text
        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        details_layout.addWidget(self.details_label)
        
        # Screenshot button
        self.view_screenshot_btn = QPushButton("View Screenshot")
        self.view_screenshot_btn.clicked.connect(self.view_screenshot)
        details_layout.addWidget(self.view_screenshot_btn)
        
        details_frame.setLayout(details_layout)
        splitter.addWidget(details_frame)
