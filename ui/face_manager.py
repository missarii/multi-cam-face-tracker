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
        
        layout.addLayout(middle_layout)
        
        # Bottom section - buttons
        button_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("Add Face")
        self.add_btn.clicked.connect(self.add_face)
        button_layout.addWidget(self.add_btn)
        
        self.update_btn = QPushButton("Update Face")
        self.update_btn.clicked.connect(self.update_face)
        button_layout.addWidget(self.update_btn)
        
        self.delete_btn = QPushButton("Delete Face")
        self.delete_btn.clicked.connect(self.delete_face)
        button_layout.addWidget(self.delete_btn)
        
        self.import_btn = QPushButton("Import Image")
        self.import_btn.clicked.connect(self.import_image)
        button_layout.addWidget(self.import_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def load_face_list(self):
        """
        Load and display all face images from the known faces directory into the list widget.
        Only image files with extensions .jpg, .jpeg, .png are considered.
        """
        self.face_list.clear()
        known_faces_dir = Path(self.known_faces_dir)
        
        if not known_faces_dir.exists():
            logger.warning(f"Known faces directory {known_faces_dir} does not exist")
            return
            
        for face_file in known_faces_dir.glob('*.*'):
            if face_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                self.face_list.addItem(face_file.stem)
                
    def on_face_selected(self, current, previous):
        """
        Triggered when a face is selected from the list.

        Loads and displays the corresponding face image in the preview area
        and updates the name input field.

        Args:
            current: The currently selected QListWidgetItem.
            previous: The previously selected QListWidgetItem.
        """
        if current is None:
            self.face_preview.clear()
            self.name_input.clear()
            return
