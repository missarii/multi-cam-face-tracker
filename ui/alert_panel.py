from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton,
                            QLabel, QCheckBox, QMessageBox)
from PyQt5.QtCore import Qt
from loguru import logger
import time

class AlertPanel(QDialog):
    def __init__(self, alert_system):
        super().__init__()
        self.alert_system = alert_system
        self.setWindowTitle("Alert Panel")
        self.setGeometry(300, 300, 600, 400)
        
        self.init_ui()
        self.load_alerts()
        
    def init_ui(self):
        """Set up and arrange all the UI widgets including the alert list, checkboxes, and control buttons."""
        layout = QVBoxLayout()
        
        # Alert list
        self.alert_list = QListWidget()
        layout.addWidget(self.alert_list)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.enable_alerts_check = QCheckBox("Enable Alerts")
        self.enable_alerts_check.setChecked(self.alert_system.alert_enabled)
        self.enable_alerts_check.stateChanged.connect(self.toggle_alerts)
        controls_layout.addWidget(self.enable_alerts_check)
        
        self.enable_screenshots_check = QCheckBox("Enable Screenshots")
        self.enable_screenshots_check.setChecked(self.alert_system.screenshot_enabled)
        self.enable_screenshots_check.stateChanged.connect(self.toggle_screenshots)
        controls_layout.addWidget(self.enable_screenshots_check)
