import sys
import time
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QTabWidget, QScrollArea, QGridLayout,
                            QMessageBox, QFileDialog, QComboBox, QSlider, QSpinBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon
from loguru import logger
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path

from core.face_detection import FaceDetector
from core.camera_manager import CameraManager
from core.alert_system import AlertEvent, AlertSystem
from core.database import FaceDatabase
from core.utils import numpy_to_pixmap, resize_image, draw_face_info
from .face_manager import FaceManagerDialog
from .alert_panel import AlertPanel
from .history_viewer import HistoryViewer

class MainWindow(QMainWindow):
    def __init__(self, config):
        """
        Initialize the main window and all core components.
        
        Args:
            config (dict): The application configuration dictionary.
        """
        super().__init__()
        self.config = config
        self.setWindowTitle(f"{config['app']['name']} v{config['app']['version']}")
        self.setWindowIcon(QIcon(config['app']['logo']))
        self.setGeometry(100, 100, 1200, 800)
        
        self.processing_interval = 0.5  # seconds
        # Initialize core components
        self.face_detector = FaceDetector(config)
        self.camera_manager = CameraManager('config/camera_config.yaml')
        self.alert_system = AlertSystem(config)
        self.database = FaceDatabase(config['app']['database_path'])
        
        # Load known faces
        self.face_detector.load_known_faces(config['app']['known_faces_dir'])
        
        # UI Components
        self.init_ui()
        
        # Start camera threads
        self.camera_manager.start_all_cameras()
        
        # Setup update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(30)  # ~30 FPS
        
        # Track last processed time per camera to limit processing
        self.last_processed: Dict[int, float] = {}
        
    def init_ui(self):
        """
        Set up the main user interface, including tabs, layouts, and status/menu bars.
        """
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Add tabs
        self.setup_monitor_tab()
        self.setup_controls_tab()
        self.setup_history_tab()
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Ready")
        self.status_bar.addPermanentWidget(self.status_label)
        
        # Menu bar
        self.setup_menu_bar()
        
    def setup_menu_bar(self):
        """
        Set up the application's menu bar with actions like exiting, opening tools, and toggling fullscreen.
        """
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        face_manager_action = tools_menu.addAction('Face Manager')
        face_manager_action.triggered.connect(self.open_face_manager)
        
        alert_panel_action = tools_menu.addAction('Alert Panel')
        alert_panel_action.triggered.connect(self.open_alert_panel)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        fullscreen_action = view_menu.addAction('Toggle Fullscreen')
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        
    def setup_monitor_tab(self):
        """
        Set up the 'Monitor' tab which displays live camera feeds in a scrollable grid layout.
        """

        monitor_tab = QWidget()
        self.tab_widget.addTab(monitor_tab, "Monitor")
        
        # Scroll area for camera feeds
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        # Container for camera feeds
        self.camera_container = QWidget()
        self.camera_grid = QGridLayout(self.camera_container)
        self.camera_grid.setSpacing(10)
        
        scroll.setWidget(self.camera_container)
        
        # Layout for monitor tab
        layout = QVBoxLayout(monitor_tab)
        layout.addWidget(scroll)
        
        # Add camera labels
        self.camera_labels = {}
        for cam_id in self.camera_manager.cameras:
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(400, 300)
            self.camera_labels[cam_id] = label
            self.camera_grid.addWidget(label, (cam_id // 2), (cam_id % 2))
            
    def setup_controls_tab(self):
        """
        Set up the 'Controls' tab, which allows users to start/stop cameras,
        adjust recognition threshold, and modify processing intervals.
        """
        controls_tab = QWidget()
        self.tab_widget.addTab(controls_tab, "Controls")
        
        layout = QVBoxLayout(controls_tab)
        
        # Camera controls
        camera_group = QWidget()
        camera_layout = QVBoxLayout(camera_group)
        
        camera_title = QLabel("Camera Controls")
        camera_title.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(camera_title)
