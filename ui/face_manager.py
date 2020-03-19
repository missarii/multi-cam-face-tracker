import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton,
                            QLabel, QFileDialog, QMessageBox, QLineEdit, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from loguru import logger
import cv2
import numpy as np
from pathlib import Path

from core.utils import numpy_to_pixmap, resize_image

class FaceManagerDialog(QDialog):
    def __init__(self, face_detector, known_faces_dir):
        """
        Initialize the FaceManagerDialog.

        Args:
            face_detector: An instance responsible for face detection and management.
            known_faces_dir (str or Path): Directory path where known face images are stored.
        """
        super().__init__()
        self.face_detector = face_detector
        self.known_faces_dir = known_faces_dir
        self.current_image = None
        
        self.setWindowTitle("Face Manager")
        self.setGeometry(200, 200, 800, 600)
        
        self.init_ui()
        self.load_face_list()
        
    def init_ui(self):
        """
        Set up the UI components of the dialog, including face list, preview,
        input fields, and control buttons.
        """
        layout = QVBoxLayout()
        
        # Top section - face list and controls
        top_layout = QHBoxLayout()
        
        # Face list
        self.face_list = QListWidget()
        self.face_list.currentItemChanged.connect(self.on_face_selected)
        top_layout.addWidget(self.face_list, 3)
        
        # Face preview
        self.face_preview = QLabel()
        self.face_preview.setAlignment(Qt.AlignCenter)
        self.face_preview.setMinimumSize(300, 300)
        top_layout.addWidget(self.face_preview, 2)
        
        layout.addLayout(top_layout)
        
        # Middle section - face details
        middle_layout = QHBoxLayout()
        
        # Name input
        name_layout = QVBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_input = QLineEdit()
        name_layout.addWidget(self.name_input)
        middle_layout.addLayout(name_layout)
