import os
import time
from google import genai
import pyautogui
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import json
import re
from typing import List, Dict, Any
from pynput import keyboard, mouse
import threading
import pyperclip
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QLabel, QTextEdit, QFrame, QPushButton, QHBoxLayout)
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QObject
from PyQt5.QtGui import QFont
import sys

# Load environment variables
load_dotenv()

# Configure Gemini AI with new client
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

client = genai.Client(api_key=GOOGLE_API_KEY)

class SignalHandler(QObject):
    """Class to handle signals for thread-safe UI updates"""
    update_insights_signal = pyqtSignal(str)
    update_actions_signal = pyqtSignal(list)
    update_status_signal = pyqtSignal(str)
    show_confirmation_signal = pyqtSignal(list)
    process_actions_signal = pyqtSignal(str)  # New signal for processing actions

class FloatingWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Job Assistant Insights")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setGeometry(50, 50, 400, 700)  # Made taller to accommodate new section
        
        # Create signal handler
        self.signals = SignalHandler()
        self.signals.update_insights_signal.connect(self._update_insights)
        self.signals.update_actions_signal.connect(self._update_actions)
        self.signals.update_status_signal.connect(self._update_status)
        self.signals.show_confirmation_signal.connect(self._show_confirmation)
        self.signals.process_actions_signal.connect(self._process_actions)
        
        # Create main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # Add title bar
        self.title_bar = QLabel("Job Application Assistant")
        self.title_bar.setFont(QFont('Arial', 12, QFont.Bold))
        self.title_bar.setAlignment(Qt.AlignCenter)
        self.title_bar.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")
        self.title_bar.setFixedHeight(40)
        self.layout.addWidget(self.title_bar)
        
        # Add insights section
        insights_label = QLabel("Current Insights")
        insights_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.layout.addWidget(insights_label)
        
        self.insights_text = QTextEdit()
        self.insights_text.setReadOnly(True)
        self.insights_text.setMinimumHeight(200)
        self.insights_text.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.layout.addWidget(self.insights_text)
        
        # Add actions section
        actions_label = QLabel("Pending Actions")
        actions_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.layout.addWidget(actions_label)
        
        self.actions_text = QTextEdit()
        self.actions_text.setReadOnly(True)
        self.actions_text.setMinimumHeight(200)
        self.actions_text.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.layout.addWidget(self.actions_text)
        
        # Add status bar
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;")
        self.status_bar.setFixedHeight(30)
        self.layout.addWidget(self.status_bar)
        
        # Add hotkey reminder
        hotkey_text = "Hotkeys:\nCtrl+Shift+A: Analyze Form\nCtrl+Shift+J: Analyze Job\nCtrl+Q: Quit"
        self.hotkey_label = QLabel(hotkey_text)
        self.hotkey_label.setStyleSheet("padding: 5px; background-color: #f8f9fa; border: 1px solid #ccc;")
        self.layout.addWidget(self.hotkey_label)
        
        # Add confirmation section
        confirmation_label = QLabel("Action Confirmation")
        confirmation_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.layout.addWidget(confirmation_label)
        
        self.confirmation_text = QTextEdit()
        self.confirmation_text.setReadOnly(True)
        self.confirmation_text.setMinimumHeight(60)
        self.confirmation_text.setStyleSheet("""
            background-color: #f8f9fa;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 5px;
        """)
        self.layout.addWidget(self.confirmation_text)
        
        # Add confirmation buttons
        button_layout = QHBoxLayout()
        
        self.yes_button = QPushButton("Execute (Y)")
        self.yes_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        
        self.no_button = QPushButton("Skip (N)")
        self.no_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        
        self.quit_button = QPushButton("Quit")
        self.quit_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        
        button_layout.addWidget(self.yes_button)
        button_layout.addWidget(self.no_button)
        button_layout.addWidget(self.quit_button)
        
        self.layout.addLayout(button_layout)
        
        # Hide confirmation section initially
        self.confirmation_text.hide()
        self.yes_button.hide()
        self.no_button.hide()
        self.quit_button.hide()
        
        # Store current actions
        self.current_actions = None
        self.current_action_index = 0
        
        # Connect button signals
        self.yes_button.clicked.connect(self.on_yes_clicked)
        self.no_button.clicked.connect(self.on_no_clicked)
        self.quit_button.clicked.connect(self.on_quit_clicked)
        
        # Window dragging
        self.dragging = False
        self.offset = QPoint()
        self.title_bar.mousePressEvent = self.mousePressEvent
        self.title_bar.mouseMoveEvent = self.mouseMoveEvent
        self.title_bar.mouseReleaseEvent = self.mouseReleaseEvent
        
        # Set window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
                border: 1px solid #cccccc;
            }
            QLabel {
                color: #333333;
            }
            QTextEdit {
                font-family: Arial;
                font-size: 10pt;
                color: #333333;
            }
        """)
        
        # Show and raise window
        self.show()
        self.raise_()
        self.activateWindow()
        
        # Test content
        self.insights_text.setText("Widget initialized and ready.\nWaiting for analysis...")
        self.actions_text.setText("No actions pending.\nUse hotkeys to analyze forms or job descriptions.")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.offset = event.pos()
    
    def mouseMoveEvent(self, event):
        if self.dragging:
            self.move(self.mapToGlobal(event.pos() - self.offset))
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
    
    def update_insights(self, text: str):
        """Thread-safe update of insights text area"""
        self.signals.update_insights_signal.emit(text)
    
    def _update_insights(self, text: str):
        """Internal method to update insights in main thread"""
        try:
            self.insights_text.setText(text)
        except Exception as e:
            print(f"Error updating insights: {e}")
            self.status_bar.setText("Error updating insights display")
    
    def update_actions(self, actions: List[tuple]):
        """Thread-safe update of actions text area"""
        self.signals.update_actions_signal.emit(actions)
    
    def _update_actions(self, actions: List[tuple]):
        """Internal method to update actions in main thread"""
        try:
            if not actions:
                self.actions_text.setText("No actionable elements found")
                return
                
            text = ""
            for action in actions:
                if action[0] == 'type':
                    text += f"TYPE at {action[-1]}: {action[3]} - {action[4]}\n"
                else:
                    text += f"{action[0].upper()} at {action[-1]}: {action[3]}\n"
            self.actions_text.setText(text)
        except Exception as e:
            print(f"Error updating actions: {e}")
            self.status_bar.setText("Error updating actions display")
    
    def update_status(self, text: str):
        """Thread-safe update of status bar"""
        self.signals.update_status_signal.emit(text)
    
    def _update_status(self, text: str):
        """Internal method to update status in main thread"""
        self.status_bar.setText(text)

    def _show_confirmation(self, actions):
        """Show confirmation for the next action"""
        self.current_actions = actions
        self.current_action_index = 0
        self._show_next_action()
    
    def _show_next_action(self):
        """Show the next action for confirmation"""
        if not self.current_actions or self.current_action_index >= len(self.current_actions):
            self._hide_confirmation()
            return
        
        action = self.current_actions[self.current_action_index]
        if action[0] == 'type':
            text = f"Type '{action[3]}' at {action[-1]} - {action[4]}"
        else:
            text = f"{action[0].upper()} at {action[-1]}: {action[3]}"
        
        self.confirmation_text.setText(f"Proposed action {self.current_action_index + 1}/{len(self.current_actions)}:\n{text}")
        
        # Show confirmation section
        self.confirmation_text.show()
        self.yes_button.show()
        self.no_button.show()
        self.quit_button.show()
    
    def _hide_confirmation(self):
        """Hide the confirmation section"""
        self.confirmation_text.hide()
        self.yes_button.hide()
        self.no_button.hide()
        self.quit_button.hide()
        self.current_actions = None
        self.current_action_index = 0
    
    def on_yes_clicked(self):
        """Handle Yes button click"""
        if self.current_actions and self.current_action_index < len(self.current_actions):
            action = self.current_actions[self.current_action_index]
            # Execute the action
            self.assistant.execute_single_action(action)
            self.current_action_index += 1
            self._show_next_action()
    
    def on_no_clicked(self):
        """Handle No button click"""
        self.current_action_index += 1
        self._show_next_action()
    
    def on_quit_clicked(self):
        """Handle Quit button click"""
        self._hide_confirmation()

    def _process_actions(self, ai_response: str):
        """Process AI response and update UI in main thread"""
        try:
            actions = self.assistant.parse_ai_response(ai_response)
            if actions:
                self._update_actions(actions)
                self._update_status("Analysis complete")
                self._show_confirmation(actions)
            else:
                self._update_actions([])
                self._update_status("No actionable elements found")
        except Exception as e:
            self._update_status(f"Error processing actions: {str(e)}")
            print(f"Error processing actions: {e}")

class JobApplicationAssistant:
    def __init__(self):
        # Initialize Qt Application
        self.app = QApplication(sys.argv)
        
        # Create floating widget
        self.widget = FloatingWidget()
        self.widget.setParent(None)  # Ensure widget has no parent
        
        # Store widget reference in the widget itself for action execution
        self.widget.assistant = self
        
        # Initialize variables
        self.running = True
        
        # Load user profile
        self.load_user_profile()
        
        # Initialize screen dimensions with proper scaling for Retina display
        screen = self.app.primaryScreen()
        geometry = screen.geometry()
        scale_factor = screen.devicePixelRatio()
        self.screen_width = int(geometry.width() * scale_factor)
        self.screen_height = int(geometry.height() * scale_factor)
        print(f"Detected screen dimensions: {self.screen_width}x{self.screen_height} (Scale factor: {scale_factor})")
        
        # Initialize grid data
        self.initialize_grid_data()
        
        # Set up PyAutoGUI safety features
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1  # Reduced from 1.0 to 0.1 for faster movements
        
        # Create screenshots directory
        self.screenshots_dir = "application_screenshots"
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
        # Application state tracking
        self.current_application = {
            "job_title": "",
            "company": "",
            "job_type": "",
            "skills_matched": [],
            "experience_matched": [],
            "grid_positions": {}
        }
        
        # Set up keyboard listener in a separate thread
        self.keyboard_thread = threading.Thread(target=self.start_keyboard_listener)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
        # Current keys being pressed
        self.current_keys = set()

    def initialize_grid_data(self):
        """Initialize or reset grid data for the entire screen"""
        # Calculate cell dimensions for 10x10 grid
        cell_width = self.screen_width // 10
        cell_height = self.screen_height // 10
        
        # Initialize grid points dictionary
        points = {}
        
        # Generate grid points
        for i in range(10):
            for j in range(10):
                x = j * cell_width + (cell_width // 2)  # Center of cell
                y = i * cell_height + (cell_height // 2)  # Center of cell
                coord = f"{chr(65 + j)}{i + 1}"  # A1, B1, etc.
                points[coord] = {'x': x, 'y': y}
        
        # Create grid data structure
        self.grid_data = {
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'cell_width': cell_width,
            'cell_height': cell_height,
            'points': points
        }
        
        # Save grid reference
        with open('grid_reference.json', 'w') as f:
            json.dump(self.grid_data, f, indent=2)

    def start_keyboard_listener(self):
        """Start keyboard listener in a separate thread"""
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release) as listener:
            listener.join()

    def run(self):
        """Main loop for the job application assistant"""
        if not self.grid_data:
            print("\nERROR: Grid reference not found!")
            print("Please run screen_mapper.py first to create the grid reference.")
            return
            
        print("\n=== Job Application Assistant with Grid System ===")
        print("Commands:")
        print("- Press 'Ctrl+Shift+A' to analyze the current form using the grid")
        print("- Press 'Ctrl+Shift+J' to analyze job description")
        print("- Press 'Ctrl+Q' to quit")
        print("\nGrid system is active (A1-J10)")
        
        try:
            # Run Qt main loop
            sys.exit(self.app.exec_())
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.running = False
            print("\nJob Application Assistant stopped.")
            if os.path.exists("temp_screenshot.png"):
                os.remove("temp_screenshot.png")

    def load_user_profile(self):
        """Load or create user profile with job application details"""
        if os.path.exists('user_profile.json'):
            with open('user_profile.json', 'r') as f:
                self.user_profile = json.load(f)
        else:
            raise ValueError("Please ensure user_profile.json exists with your resume information")

    def get_nearest_grid_point(self, x: int, y: int) -> str:
        """Find the nearest grid point in the dense grid system"""
        if not self.grid_data:
            return None
            
        min_dist = float('inf')
        nearest_point = None
        
        # First find the nearest main cell
        main_cell_width = self.grid_data['cell_width']
        main_cell_height = self.grid_data['cell_height']
        
        main_col = min(max(0, x // main_cell_width), 9)
        main_row = min(max(0, y // main_cell_height), 9)
        main_coord = f"{chr(65 + main_col)}{main_row + 1}"
        
        # Then find the nearest sub-cell
        sub_cell_width = main_cell_width // 10
        sub_cell_height = main_cell_height // 10
        
        sub_x = (x % main_cell_width) // sub_cell_width
        sub_y = (y % main_cell_height) // sub_cell_height
        
        return f"{main_coord}.{sub_x}{sub_y}"

    def annotate_screenshot(self, image_path: str, actions: List[tuple], timestamp: str) -> str:
        """Annotate screenshot with planned actions"""
        # Read the input image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        height, width = img.shape[:2]
        
        # Get scale factor for display
        scale_factor = self.app.primaryScreen().devicePixelRatio()
        
        # Calculate scaling ratios more precisely
        screen = self.app.primaryScreen()
        screen_width = screen.geometry().width()
        screen_height = screen.geometry().height()
        
        # Calculate precise scaling ratios accounting for scale factor
        width_ratio = width / (screen_width * scale_factor)
        height_ratio = height / (screen_height * scale_factor)
        
        # Create a copy of the image for annotation
        annotated_img = img.copy()
        
        # Add title and timestamp
        cv2.putText(annotated_img, "Detected Actions:", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 0, 255), 2)
        
        # Add each action to the image with improved coordinate transformation
        for idx, action in enumerate(actions, 1):
            action_type = action[0]
            
            # Transform coordinates with improved precision
            x = int(action[1] * scale_factor * width_ratio)
            y = int(action[2] * scale_factor * height_ratio)
            
            # Get annotation y position (last element in action tuple)
            annotation_y = int(action[-1] * scale_factor * height_ratio)
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            annotation_y = max(0, min(annotation_y, height - 1))
            
            grid_coord = action[-2]  # Grid coordinate is now second to last element
            
            # Draw more precise markers
            # First draw a thin crosshair for precise center marking
            cv2.line(annotated_img, (x-10, y), (x+10, y), (0, 0, 255), 1)
            cv2.line(annotated_img, (x, y-10), (x, y+10), (0, 0, 255), 1)
            
            # Then draw the circle
            cv2.circle(annotated_img, (x, y), 15, (0, 0, 255), 2)
            
            # Add label with grid coordinate (improved text placement)
            text = f"{idx}. {grid_coord}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Add white background for better text visibility
            cv2.rectangle(annotated_img,
                        (x + 25, annotation_y - text_size[1] - 5),
                        (x + 25 + text_size[0] + 10, annotation_y + 5),
                        (255, 255, 255),
                        -1)
            
            cv2.putText(annotated_img, text, 
                      (x + 30, annotation_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, (0, 0, 255), 2)
            
            # Add description with improved formatting
            desc_text = f"{idx}. {action_type.upper()} at {grid_coord}: {action[3]}"
            desc_size = cv2.getTextSize(desc_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Add white background for description
            cv2.rectangle(annotated_img,
                        (10, 50 + 30 * idx - desc_size[1] - 5),
                        (10 + desc_size[0] + 10, 50 + 30 * idx + 5),
                        (255, 255, 255),
                        -1)
            
            cv2.putText(annotated_img, desc_text,
                      (10, 50 + 30 * idx), 
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.7, (0, 0, 255), 2)
        
        # Save the annotated image
        output_path = os.path.join(
            self.screenshots_dir,
            f"analysis_{timestamp}_annotated.png"
        )
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the image
        success = cv2.imwrite(output_path, annotated_img)
        if not success:
            raise ValueError(f"Failed to save annotated image to {output_path}")
            
        # Verify the file was written
        if not os.path.exists(output_path):
            raise ValueError(f"Annotated image file was not created at {output_path}")
            
        return output_path

    def capture_screenshot(self):
        """Capture a screenshot and overlay the dense grid"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean up any existing temporary files
        temp_files = ['temp_screenshot.png', 'temp_gridded.png', 'temp_annotated.png']
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Take screenshot and save as temporary file
        screenshot = pyautogui.screenshot()
        temp_screenshot_path = 'temp_screenshot.png'
        screenshot.save(temp_screenshot_path)
        
        # Convert screenshot to numpy array for OpenCV processing
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        height, width = img.shape[:2]
        
        # Get screen dimensions and scale factor
        screen = self.app.primaryScreen()
        scale_factor = screen.devicePixelRatio()
        
        # Calculate dimensions in actual pixels
        screen_width_pixels = int(screen.geometry().width() * scale_factor)
        screen_height_pixels = int(screen.geometry().height() * scale_factor)
        
        # Create the gridded overlay
        gridded_img = self._create_grid_overlay(img, screen_width_pixels, screen_height_pixels)
        
        # Save only the final gridded image
        gridded_path = os.path.join(
            self.screenshots_dir,
            f"analysis_{timestamp}.png"
        )
        cv2.imwrite(gridded_path, gridded_img)
        
        # Clean up temporary screenshot
        os.remove(temp_screenshot_path)
        
        return gridded_path, timestamp

    def analyze_application_form(self, image_path):
        """Analyze the job application form using Gemini AI with dense grid reference"""
        try:
            self.widget.update_status("Analyzing form with dense grid...")
            
            # First create the gridded screenshot
            screenshot = pyautogui.screenshot()
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Get screen dimensions and scale factor
            screen = self.app.primaryScreen()
            scale_factor = screen.devicePixelRatio()
            
            # Calculate dimensions in actual pixels
            screen_width_pixels = int(screen.geometry().width() * scale_factor)
            screen_height_pixels = int(screen.geometry().height() * scale_factor)
            
            # Create the gridded overlay and merge it with the screenshot
            gridded_img = self._create_grid_overlay(img, screen_width_pixels, screen_height_pixels)
            
            # Save the gridded image first (for debugging)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gridded_path = os.path.join(
                self.screenshots_dir,
                f"gridded_screenshot_{timestamp}.png"
            )
            cv2.imwrite(gridded_path, gridded_img)
            
            # Now read back the saved image to ensure we're using exactly what we saved
            gridded_img = cv2.imread(gridded_path)
            
            # Create a copy for annotation that includes the grid
            annotated_img = gridded_img.copy()
            height, width = annotated_img.shape[:2]
            
            # Add a white background section at the top for better text visibility
            header_height = 150
            header = np.ones((header_height, width, 3), dtype=np.uint8) * 255
            annotated_img = np.vstack([header, annotated_img])
            
            # Add clear instructions on the white header
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated_img, "Grid Reference System - Please verify coordinates carefully",
                      (10, 30), font, 1.0, (0, 0, 255), 2)
            cv2.putText(annotated_img, "Red lines: Grid boundaries | Blue dots: Intersection points | Green text: Coordinates",
                      (10, 70), font, 0.7, (0, 0, 0), 2)
            cv2.putText(annotated_img, f"Screen dimensions: {screen_width_pixels}x{screen_height_pixels} px | Scale factor: {scale_factor}",
                      (10, 110), font, 0.7, (0, 0, 0), 2)
            
            # Save the annotated version
            annotated_path = os.path.join(
                self.screenshots_dir,
                f"annotated_screenshot_{timestamp}.png"
            )
            cv2.imwrite(annotated_path, annotated_img)
            
            # Convert to PIL Image for Gemini AI - use the annotated version
            annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            
            prompt = f"""
            CRITICAL RESPONSE FORMAT REQUIREMENTS:
            You MUST follow this EXACT format:

            1. Start with "###JSON_START###"
            2. Then provide a PURE JSON array (no markdown formatting, no ```json tags)
            3. End with "%%%JSON_END%%%"

            Example of EXACT format required:
            ###JSON_START###
            [
              {{
                "element_type": "button",
                "grid_coord": "AA1.44",
                "description": "Submit button",
                "recommended_action": "CLICK AA1.44"
              }}
            ]
            %%%JSON_END%%%

            COORDINATE SELECTION STRATEGY:
            For each UI element, you MUST:
            1. Identify the four closest grid points (blue dots) that form a quad around the element
            2. These points should form the smallest possible square/rectangle containing the element
            3. Calculate the relative position of the element within this quad:
               - If element is centered between points: use the average of surrounding sub-coordinates
               - If element is closer to one point: use that point's sub-coordinate
               - If element spans multiple points: use the most central point for interaction

            QUAD SELECTION RULES:
            1. ALWAYS select four visible grid points that:
               - Form a clear boundary around the target element
               - Are all clearly visible with readable sub-coordinates
               - Create the smallest possible quad containing the element
            
            2. For each quad, analyze:
               - Which point is closest to the actual interaction target
               - How the element is positioned relative to the quad corners
               - Whether the element is centered or aligned to a specific point

            3. Coordinate Selection:
               - For centered elements: Use sub-coordinates that represent the center of the quad
               - For aligned elements: Use sub-coordinates closer to the alignment point
               - For form fields: Ensure the click point is in the input area
               - For buttons/checkboxes: Ensure the click point is on the control

            COORDINATE INSTRUCTIONS:
            You MUST:
            1. ONLY use coordinates where you can see actual blue dots in the grid
            2. Look for the closest four blue dots that form a quad around each element
            3. Use the exact sub-grid numbers shown in green (00-99)
            4. Never make up or interpolate coordinates - if you can't see all four quad points, find a different quad
            5. Double-check that each chosen quad has:
               - Four visible blue dots
               - Readable sub-grid numbers for all points
               - Clear line of sight to the element

            Grid System Reference:
            - Main Grid: AA1-ZZ40 (marked with large labels)
            - Sub-grid: Each main cell has numbered points
            - Format: <main_cell>.<sub_position>
              Example: If element is centered in quad with point "AA1.45", use that coordinate

            Look for these elements:
            1. Interactive Elements:
               - Buttons (Apply, Submit, Next, etc.)
               - Checkboxes and radio buttons
               - Text input fields
               - Dropdown menus
               - Navigation elements

            2. Form Fields:
               - Required skills questions
               - Experience questions
               - Education verification
               - Contact information fields

            3. Match with my profile:
            {json.dumps(self.user_profile, indent=2)}

            Current application context:
            {json.dumps(self.current_application, indent=2)}

            FINAL VERIFICATION:
            Before providing coordinates, verify that:
            1. Each element has a clear quad of four points around it
            2. All quad points have visible sub-grid numbers
            3. The chosen coordinate represents the best interaction point
            4. The quad selection follows the minimum size rule

            QUAD ANALYSIS FORMAT:
            For each element, think through:
            1. "I see the element [description]"
            2. "The closest quad points are [TL], [TR], [BL], [BR]"
            3. "The element is [centered/aligned/positioned] within this quad"
            4. "Therefore, I will use coordinate [coord] for interaction"

            REMEMBER: Your response MUST start with ###JSON_START### and end with %%%JSON_END%%%
            The JSON must be a pure array without any markdown formatting or code block tags.
            """
            
            # Generate content using Gemini AI with the annotated image
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, annotated_pil]
            )
            
            if response and response.text:
                self.widget.update_insights(response.text)
                return response.text, annotated_img  # Return both the response and the annotated image
            return None, None
            
        except Exception as e:
            self.widget.update_status(f"Error: {str(e)}")
            print(f"Error analyzing form: {e}")
            return None, None

    def parse_ai_response(self, ai_response):
        """Parse AI response into actionable commands using dense grid coordinates"""
        actions = []
        if not ai_response:
            print("No AI response to parse")
            return actions
            
        try:
            # Clean up the response text
            ai_response = ai_response.strip()
            print("\nProcessing AI Response:")
            print("-------------------")
            print(ai_response)
            print("-------------------")
            
            # Extract JSON between delimiters
            json_match = re.search(r'###JSON_START###\s*(.*?)\s*%%%JSON_END%%%', 
                                 ai_response, re.DOTALL)
            if not json_match:
                print("No valid JSON found between delimiters ###JSON_START### and %%%JSON_END%%%")
                return actions
                
            # Get the JSON string and clean it
            json_str = json_match.group(1).strip()
            
            # Remove any potential markdown or code block formatting
            json_str = re.sub(r'```[^\n]*\n?', '', json_str)
            json_str = json_str.replace('`', '')
            
            print("\nExtracted JSON string:")
            print(json_str)
            
            # Validate JSON structure before parsing
            if not json_str.startswith('[') or not json_str.endswith(']'):
                print("Invalid JSON structure - must be an array")
                return actions
                
            try:
                # Parse the JSON
                action_list = json.loads(json_str)
                
                # Validate action list structure
                if not isinstance(action_list, list):
                    print("Invalid action list structure - must be an array")
                    return actions
                    
                # Get screen dimensions and scale factor
                screen = self.app.primaryScreen()
                scale_factor = screen.devicePixelRatio()
                
                # Get actual screen dimensions in logical pixels (unscaled)
                screen_width = screen.geometry().width()
                screen_height = screen.geometry().height()
                
                # Calculate cell dimensions in logical pixels
                cell_width = screen_width // 40  # 40x40 grid
                cell_height = screen_height // 40
                
                print(f"\nScreen dimensions (logical): {screen_width}x{screen_height}")
                print(f"Cell dimensions (logical): {cell_width}x{cell_height}")
                
                for action_data in action_list:
                    # Validate action data structure
                    required_fields = ['grid_coord', 'recommended_action', 'description']
                    if not all(field in action_data for field in required_fields):
                        print(f"Skipping invalid action data - missing required fields: {action_data}")
                        continue
                    
                    coord = action_data['grid_coord']
                    action_type = action_data['recommended_action'].split()[0].lower()
                    
                    try:
                        # Parse coordinates for two-letter system with sub-positions
                        match = re.match(r'([A-Z]{2})(\d+)(?:\.(\d+))?', coord)
                        if not match:
                            print(f"Invalid coordinate format: {coord}")
                            continue
                            
                        col_letters = match.group(1)  # Two letters (e.g., "BI")
                        row_num = int(match.group(2))  # Main row number
                        sub_pos = match.group(3) or "50"  # Sub-position (default to center if not provided)
                        
                        # Calculate column index (corrected calculation)
                        first_letter = ord(col_letters[0]) - ord('A')
                        second_letter = ord(col_letters[1]) - ord('A')
                        col_index = (first_letter * 26) + second_letter
                        
                        # Ensure column index doesn't exceed grid width
                        col_index = min(col_index, 39)  # 40x40 grid (0-39)
                        
                        # Calculate row index (1-based to 0-based)
                        main_row = min(row_num - 1, 39)  # Ensure row stays within bounds
                        
                        # Calculate sub-position offsets (0-99 range)
                        sub_x = int(sub_pos) // 10
                        sub_y = int(sub_pos) % 10
                        
                        # Calculate final coordinates with sub-position offsets and bounds checking
                        x = min(int((col_index * cell_width) + (cell_width * sub_x / 10) + (cell_width / 20)), screen_width - 1)
                        # Adjust y-coordinate calculation to shift upward
                        base_y = (main_row * cell_height)
                        sub_cell_height = cell_height / 10
                        vertical_offset = cell_height / 2  # Increased from 1/4 to 1/2 cell height to move points higher
                        y = min(int(base_y + (sub_cell_height * sub_y) - vertical_offset), screen_height - 1)
                        
                        # Ensure y coordinate never goes negative
                        y = max(0, y)
                        
                        # Log coordinate calculation
                        print(f"\nCoordinate calculation for {coord}:")
                        print(f"Main cell: {col_letters}{row_num} -> col={col_index} (first={first_letter}, second={second_letter}), row={main_row}")
                        print(f"Sub-position: {sub_pos} -> ({sub_x}, {sub_y})")
                        print(f"Cell dimensions: {cell_width}x{cell_height}")
                        print(f"Final logical coordinates: ({x}, {y})")
                        
                        # Store the action with exact coordinates and adjusted annotation positions
                        annotation_y_offset = vertical_offset  # Use same offset for annotations
                        annotation_y = y - annotation_y_offset  # Move annotation text up with the click point
                        
                        if action_type == 'type':
                            text_match = re.search(r'"([^"]*)"', action_data['recommended_action'])
                            if text_match:
                                text = text_match.group(1)
                                # Include adjusted annotation position
                                actions.append(('type', x, y, text, action_data['description'], coord, annotation_y))
                        elif action_type in ['click', 'select']:
                            # Include adjusted annotation position
                            actions.append((action_type, x, y, action_data['description'], coord, annotation_y))
                            
                    except ValueError as ve:
                        print(f"Error parsing coordinate {coord}: {ve}")
                        continue
                    except Exception as e:
                        print(f"Error processing coordinate {coord}: {e}")
                        continue
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {str(e)}")
                print("Invalid JSON string:")
                print(json_str)
                return actions
                
        except Exception as e:
            print(f"Error parsing AI response: {e}")
            import traceback
            traceback.print_exc()
        
        return actions

    def verify_click(self, x: int, y: int, retries: int = 3, double_click: bool = True) -> bool:
        """Verify and ensure a click is performed at the specified coordinates"""
        mouse_controller = mouse.Controller()
        
        for attempt in range(retries):
            try:
                # Move to position
                mouse_controller.position = (x, y)
                time.sleep(0.1)
                
                # Get current position to verify
                current_x, current_y = mouse_controller.position
                
                # Check if we're within 2 pixels of target (accounting for minor OS variations)
                if abs(current_x - x) <= 2 and abs(current_y - y) <= 2:
                    # First click for focus
                    mouse_controller.click(mouse.Button.left)
                    time.sleep(0.2)  # Increased pause between clicks
                    
                    # Second click for activation if needed
                    if double_click:
                        mouse_controller.click(mouse.Button.left)
                        time.sleep(0.1)
                    
                    return True
                else:
                    print(f"Position mismatch on attempt {attempt + 1}. Target: ({x}, {y}), Actual: ({current_x}, {current_y})")
                    time.sleep(0.2)  # Wait before retry
            
            except Exception as e:
                print(f"Click verification failed on attempt {attempt + 1}: {e}")
                time.sleep(0.2)  # Wait before retry
        
        return False

    def execute_single_action(self, action):
        """Execute a single action with precise coordinates and verified real clicks"""
        try:
            x, y = action[1], action[2]
            coord = action[-1]
            
            # Log the action with precise coordinates
            print(f"Executing {action[0]} at {coord} ({x}, {y})")
            
            if action[0] in ['click', 'select']:
                # Move to position smoothly first
                pyautogui.moveTo(x, y, duration=0.2, tween=pyautogui.easeOutQuad)
                time.sleep(0.3)  # Wait for movement to complete
                
                # First click for focus
                mouse_controller = mouse.Controller()
                mouse_controller.position = (x, y)
                time.sleep(0.2)  # Wait for position
                mouse_controller.click(mouse.Button.left)
                time.sleep(0.3)  # Wait for focus
                
                # Verify position and click again if needed
                current_x, current_y = mouse_controller.position
                if abs(current_x - x) <= 2 and abs(current_y - y) <= 2:
                    mouse_controller.click(mouse.Button.left)
                else:
                    print(f"Position verification failed, retrying with PyAutoGUI")
                    pyautogui.click(x, y)
                
                time.sleep(0.3)  # Wait after click
                
            elif action[0] == 'type':
                # Move smoothly and ensure focus first
                pyautogui.moveTo(x, y, duration=0.2, tween=pyautogui.easeOutQuad)
                time.sleep(0.3)  # Wait for movement
                
                # Initial focus click
                mouse_controller = mouse.Controller()
                mouse_controller.position = (x, y)
                time.sleep(0.2)
                mouse_controller.click(mouse.Button.left)
                time.sleep(0.3)  # Wait for focus
                
                # Double click for text selection
                mouse_controller.click(mouse.Button.left)
                time.sleep(0.1)
                mouse_controller.click(mouse.Button.left)
                time.sleep(0.3)  # Wait for double-click to register
                
                # Select all existing text
                if sys.platform == 'darwin':
                    pyautogui.hotkey('command', 'a')
                else:
                    pyautogui.hotkey('ctrl', 'a')
                time.sleep(0.2)  # Wait for selection
                
                # Type the new text with slower interval
                pyautogui.typewrite(action[3], interval=0.08)  # Slower typing
                time.sleep(0.3)  # Wait after typing
            
            # Store successful action with detailed coordinates
            self.current_application['grid_positions'][coord] = {
                'type': action[0],
                'description': action[3] if action[0] == 'type' else action[-2],
                'coordinates': {'x': x, 'y': y}
            }
            
            # Additional wait between actions
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error executing action at {coord}: {e}")
            self.widget.update_status(f"Error executing action: {e}")
            
            # Attempt recovery with longer delays if action failed
            try:
                if action[0] in ['click', 'select']:
                    print("Attempting recovery with longer delays...")
                    mouse_controller = mouse.Controller()
                    pyautogui.moveTo(x, y, duration=0.3)
                    time.sleep(0.5)  # Longer wait
                    mouse_controller.position = (x, y)
                    time.sleep(0.3)
            except Exception as recovery_error:
                print(f"Recovery attempt failed: {recovery_error}")

    def analyze_job_description(self, job_text: str) -> Dict[str, Any]:
        """Analyze job description to match with user profile"""
        try:
            self.widget.update_status("Analyzing job description...")
            prompt = f"""
            Analyze this job description and match it with my profile:
            
            Job Description:
            {job_text}
            
            My Profile:
            {json.dumps(self.user_profile, indent=2)}
            
            Provide analysis in this format:
            1. Job Title:
            2. Required Skills (that I have):
            3. Experience Matches:
            4. Education Matches:
            5. Certification Matches:
            6. Suggested Talking Points:
            """
            
            # Generate content using the correct model
            response = client.models.generate_content(
                model="gemini-pro",
                contents=prompt
            )
            
            if response and response.text:
                self.widget.update_insights(response.text)
                return response.text
            return None
            
        except Exception as e:
            self.widget.update_status(f"Error: {str(e)}")
            print(f"Error analyzing job description: {e}")
            return None

    def on_analyze_form(self):
        """Handle form analysis hotkey"""
        if not self.running:
            return
            
        print("\nAnalyzing application form...")
        self.widget.update_status("Taking screenshot...")
        
        try:
            # Get timestamp for file naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create screenshots directory if it doesn't exist
            os.makedirs(self.screenshots_dir, exist_ok=True)
            
            # First create the gridded screenshot
            screenshot = pyautogui.screenshot()
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Get screen dimensions and scale factor
            screen = self.app.primaryScreen()
            scale_factor = screen.devicePixelRatio()
            
            # Calculate dimensions in actual pixels
            screen_width_pixels = int(screen.geometry().width() * scale_factor)
            screen_height_pixels = int(screen.geometry().height() * scale_factor)
            
            # Create the gridded overlay
            gridded_img = self._create_grid_overlay(img, screen_width_pixels, screen_height_pixels)
            
            # Save the gridded image
            gridded_path = os.path.join(
                self.screenshots_dir,
                f"analysis_{timestamp}_grid.png"
            )
            
            # Save gridded image
            success = cv2.imwrite(gridded_path, gridded_img)
            if not success:
                raise ValueError(f"Failed to save gridded image to {gridded_path}")
            
            # Verify the file was written
            if not os.path.exists(gridded_path):
                raise ValueError(f"Gridded image file was not created at {gridded_path}")
            
            # Convert to PIL Image for Gemini AI
            annotated_pil = Image.fromarray(cv2.cvtColor(gridded_img, cv2.COLOR_BGR2RGB))
            
            # Rest of the analysis process...
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[self._create_analysis_prompt(), annotated_pil]
            )
            
            if response and response.text:
                print("\nReceived AI response")
                self.widget.update_insights(response.text)
                
                # Parse actions and create initial annotation
                current_actions = self.parse_ai_response(response.text)
                if current_actions:
                    # Initialize verification iteration counter
                    verification_iteration = 0
                    min_iterations = 2
                    max_iterations = 4
                    is_satisfied = False
                    previous_actions = []
                    
                    while (verification_iteration < max_iterations and 
                           (verification_iteration < min_iterations or not is_satisfied)):
                        verification_iteration += 1
                        print(f"\n{'='*50}")
                        print(f"Verification Iteration {verification_iteration}/{max_iterations}")
                        print(f"{'='*50}")
                        
                        # Store current actions for comparison
                        previous_actions.append(current_actions.copy())
                        
                        try:
                            # Generate annotated image with current actions
                            current_annotated_path = self.annotate_screenshot(
                                gridded_path,
                                current_actions,
                                f"{timestamp}_iter{verification_iteration}"
                            )
                            
                            # Verify the annotated image exists
                            if not os.path.exists(current_annotated_path):
                                raise ValueError(f"Annotated image not found at {current_annotated_path}")
                            
                            # Read back the annotated image
                            current_annotated_img = cv2.imread(current_annotated_path)
                            if current_annotated_img is None:
                                raise ValueError(f"Could not read annotated image from {current_annotated_path}")
                            
                            # Convert to PIL for Gemini
                            current_annotated_pil = Image.fromarray(cv2.cvtColor(current_annotated_img, cv2.COLOR_BGR2RGB))
                            
                            # Create verification prompt
                            verification_prompt = self._create_verification_prompt(
                                verification_iteration,
                                max_iterations,
                                min_iterations,
                                current_actions,
                                previous_actions[-1] if len(previous_actions) > 1 else None
                            )
                            
                            # Generate verification using Gemini AI
                            verification_response = client.models.generate_content(
                                model="gemini-2.0-flash",
                                contents=[verification_prompt, current_annotated_pil]
                            )
                            
                            if verification_response and verification_response.text:
                                print(f"\nVerification Analysis (Iteration {verification_iteration}):")
                                print("="*50)
                                print(verification_response.text)
                                print("="*50)
                                
                                # Update insights with iteration status
                                self.widget.update_insights(verification_response.text)
                                
                                # Process verification response and update actions
                                current_actions = self._process_verification_response(
                                    verification_response.text,
                                    current_actions
                                )
                                
                                # Check if AI is satisfied
                                is_satisfied = "SATISFIED" in verification_response.text.upper()
                                if is_satisfied and verification_iteration >= min_iterations:
                                    print(f"\nAI is satisfied with the results after {verification_iteration} iterations")
                                    break
                            else:
                                raise ValueError("No response from AI for verification")
                                
                        except Exception as e:
                            print(f"Error during iteration {verification_iteration}: {str(e)}")
                            break
                    
                    # Update UI with final verified actions
                    if current_actions:
                        self.widget.signals.update_actions_signal.emit(current_actions)
                        self.widget.signals.show_confirmation_signal.emit(current_actions)
                    else:
                        self.widget.update_status("No valid actions after verification")
                        self.widget.signals.update_actions_signal.emit([])
                else:
                    self.widget.update_status("No actionable elements found")
                    self.widget.signals.update_actions_signal.emit([])
            else:
                self.widget.update_status("No response from AI")
                self.widget.signals.update_actions_signal.emit([])
                
        except Exception as e:
            print(f"Error during form analysis: {str(e)}")
            self.widget.update_status(f"Analysis failed: {str(e)}")
            self.widget.signals.update_actions_signal.emit([])

    def _create_analysis_prompt(self):
        """Create the initial analysis prompt"""
        return f"""
        CRITICAL RESPONSE FORMAT REQUIREMENTS:
        You MUST follow this EXACT format:

        1. Start with "###JSON_START###"
        2. Then provide a PURE JSON array (no markdown formatting, no ```json tags)
        3. End with "%%%JSON_END%%%"

        Example of EXACT format required:
        ###JSON_START###
        [
          {{
            "element_type": "button",
            "grid_coord": "AA1.44",
            "description": "Submit button",
            "recommended_action": "CLICK AA1.44"
          }}
        ]
        %%%JSON_END%%%

        PRECISE CENTER CALCULATION REQUIREMENTS:
        For EVERY element, you MUST:
        1. Find the exact boundaries of the element:
           - Identify the top, bottom, left, and right edges
           - Note any padding or margins that affect the true interaction area
           - Consider the actual clickable/interactive region

        2. Calculate the TRUE CENTER:
           - Horizontal center = (left_edge + right_edge) / 2
           - Vertical center = (top_edge + bottom_edge) / 2
           - For text fields: Account for text baseline and input area
           - For buttons: Consider both text and button boundaries
           - For checkboxes/radio: Target the exact center of the control

        3. Quad Point Selection:
           a. Find the four grid points that form the smallest possible quad containing the element
           b. Measure the distances from each quad point to the true center
           c. Calculate the relative position within the quad:
              - Horizontal position = (center_x - left_points) / (right_points - left_points)
              - Vertical position = (center_y - top_points) / (bottom_points - top_points)
           d. Use these ratios to select or interpolate the most precise sub-coordinate

        COORDINATE PRECISION RULES:
        1. NEVER just pick the closest point. Instead:
           - Use the quad points to triangulate the exact center
           - Calculate precise ratios between the quad points
           - Select sub-coordinates that match these ratios

        2. For each element type:
           - Text Fields: Center of the input area, not the label
           - Buttons: True center of the clickable area
           - Checkboxes/Radio: Exact center of the control circle/box
           - Dropdowns: Center of the trigger area
           - Links: Center of the text, accounting for underlines

        3. Sub-coordinate Selection:
           - Use the calculated ratios to determine exact sub-coordinates
           - Example: If center is 40% between points, use corresponding sub-numbers
           - Round to nearest available sub-coordinate while maintaining center alignment

        QUAD ANALYSIS STEPS:
        For each element:
        1. "Element Boundaries:"
           - Top: [pixel_y]
           - Bottom: [pixel_y]
           - Left: [pixel_x]
           - Right: [pixel_x]
           - True Center: ([center_x], [center_y])

        2. "Quad Points:"
           - Top-Left: [coord] at ([x], [y])
           - Top-Right: [coord] at ([x], [y])
           - Bottom-Left: [coord] at ([x], [y])
           - Bottom-Right: [coord] at ([x], [y])

        3. "Center Calculation:"
           - Horizontal Ratio: [ratio]% between [left_coord] and [right_coord]
           - Vertical Ratio: [ratio]% between [top_coord] and [bottom_coord]
           - Selected Sub-coordinates: [sub_x][sub_y] based on ratios

        4. "Final Coordinate:"
           - Main Cell: [cell] (containing true center)
           - Sub-position: [sub] (matching calculated ratios)
           - Full Coordinate: [cell].[sub]

        VERIFICATION REQUIREMENTS:
        Before finalizing each coordinate:
        1. Verify the selected point is EXACTLY at the element's true center
        2. Confirm the sub-coordinates reflect the calculated ratios
        3. Test if the point falls precisely on the intended interaction area
        4. Double-check that no edge cases or rounding errors affect accuracy

        Grid System Reference:
        - Main Grid: AA1-ZZ40 (marked with large labels)
        - Sub-grid: Each main cell has numbered points (00-99)
        - Format: <main_cell>.<sub_position>
        - Example: AA1.45 = Cell AA1, sub-pos (4,5)

        Look for these elements:
        1. Interactive Elements:
           - Buttons (Apply, Submit, Next, etc.)
           - Checkboxes and radio buttons
           - Text input fields
           - Dropdown menus
           - Navigation elements

        2. Form Fields:
           - Required skills questions
           - Experience questions
           - Education verification
           - Contact information fields

        3. Match with my profile:
        {json.dumps(self.user_profile, indent=2)}

        Current application context:
        {json.dumps(self.current_application, indent=2)}

        FINAL VERIFICATION:
        Before providing coordinates, verify that:
        1. Each coordinate represents the TRUE CENTER of its element
        2. The quad calculation and ratios are mathematically correct
        3. Sub-coordinates accurately reflect the center position
        4. No element has been approximated or estimated

        REMEMBER: Your response MUST start with ###JSON_START### and end with %%%JSON_END%%%
        The JSON must be a pure array without any markdown formatting or code block tags.
        """

    def _create_verification_prompt(self, iteration: int, max_iterations: int, min_iterations: int, current_actions: List[tuple], previous_actions: List[tuple] = None) -> str:
        """Create the verification prompt for each iteration"""
        if previous_actions is None:
            previous_actions = []
            
        # Create detailed coordinate-by-coordinate analysis instructions
        coordinate_list = "\n".join([
            f"Coordinate {idx + 1}: {action[-2]} - {action[0].upper()} - {action[3]}"
            for idx, action in enumerate(current_actions)
        ])
        
        return f"""
        COORDINATE VERIFICATION TASK (Iteration {iteration}/{max_iterations})

        You are looking at an annotated screenshot that shows the current form with:
        - Red circles and crosshairs: Click/type target positions
        - White boxes with red text: Coordinate labels and descriptions
        - Grid overlay: Shows the coordinate system

        Please verify each coordinate ONE BY ONE with extreme precision:

        Coordinates to verify:
        {coordinate_list}

        For EACH coordinate, analyze:
        1. VERTICAL ALIGNMENT:
           - Is the click point EXACTLY aligned with the target element?
           - Should it be moved up or down? By how much?
           - Check if text is centered within input fields
           - For radio buttons/checkboxes, ensure click is on the control

        2. HORIZONTAL ALIGNMENT:
           - Is the click point EXACTLY on the target element?
           - Should it be moved left or right? By how much?
           - Check alignment with text fields and buttons
           - Verify click points hit interactive elements

        3. ANNOTATION PLACEMENT:
           - Are labels clearly visible and not overlapping?
           - Do they point to the correct elements?
           - Is text readable and properly positioned?

        RESPONSE FORMAT:
        For each coordinate, provide:
        ###COORD_START###
        Coordinate: [coord]
        Current Position: [description of current position]
        Vertical Alignment: [CORRECT/TOO HIGH/TOO LOW] by [amount] pixels
        Horizontal Alignment: [CORRECT/TOO LEFT/TOO RIGHT] by [amount] pixels
        Annotation: [GOOD/NEEDS ADJUSTMENT]
        Recommended Action: [KEEP/ADJUST/REMOVE]
        New Coordinate (if adjustment needed): [new coordinate]
        ###COORD_END###

        After analyzing all coordinates:
        ###SUMMARY###
        Total Coordinates Checked: [number]
        Coordinates Needing Adjustment: [number]
        Overall Assessment: [SATISFIED/UNSATISFIED]
        Explanation: [why satisfied/unsatisfied]
        ###END###

        CRITICAL REQUIREMENTS:
        - You MUST check each coordinate individually
        - You MUST look at the actual positions in the image
        - You MUST verify both click points and annotations
        - You MUST provide specific adjustment amounts if needed
        - You MUST complete at least {min_iterations} iterations
        - Current iteration: {iteration}

        Remember:
        - Be extremely precise in your analysis
        - Don't assume previous coordinates were correct
        - Check both the click point and its annotation
        - Consider the actual form element positions
        """

    def _process_verification_response(self, verification_text: str, current_actions: List[tuple]) -> List[tuple]:
        """Process the verification response and update actions"""
        adjusted_actions = []
        
        # Extract individual coordinate analyses
        coord_analyses = re.findall(
            r'###COORD_START###(.*?)###COORD_END###',
            verification_text,
            re.DOTALL
        )
        
        # Process each coordinate analysis
        for analysis in coord_analyses:
            try:
                # Extract coordinate being analyzed
                coord_match = re.search(r'Coordinate: ([A-Z]{2}\d+(?:\.\d+)?)', analysis)
                if not coord_match:
                    continue
                coord = coord_match.group(1)
                
                # Find corresponding action
                current_action = None
                for action in current_actions:
                    if action[-2] == coord:  # Grid coordinate is second to last element
                        current_action = action
                        break
                
                if not current_action:
                    continue
                
                # Check if adjustment is needed
                if 'Recommended Action: ADJUST' in analysis:
                    # Extract new coordinate if provided
                    new_coord_match = re.search(r'New Coordinate.*?: ([A-Z]{2}\d+(?:\.\d+)?)', analysis)
                    if new_coord_match:
                        new_coord = new_coord_match.group(1)
                        # Create adjusted action with new coordinate
                        adjusted_action = list(current_action)
                        adjusted_action[-2] = new_coord  # Update grid coordinate
                        # Recalculate x,y positions for new coordinate
                        # (This will be done in next parse_ai_response call)
                        adjusted_actions.append(tuple(adjusted_action))
                        print(f"Adjusting {coord} to {new_coord}")
                    else:
                        # Keep original if no new coordinate provided
                        adjusted_actions.append(current_action)
                        print(f"No new coordinate provided for {coord}, keeping original")
                elif 'Recommended Action: KEEP' in analysis:
                    # Keep the action unchanged
                    adjusted_actions.append(current_action)
                    print(f"Keeping {coord} unchanged")
                # Note: REMOVE actions are simply not added to adjusted_actions
                
            except Exception as e:
                print(f"Error processing coordinate analysis: {e}")
                # Keep original action if there's an error
                if current_action:
                    adjusted_actions.append(current_action)
        
        # Extract summary
        summary_match = re.search(
            r'###SUMMARY###(.*?)###END###',
            verification_text,
            re.DOTALL
        )
        if summary_match:
            summary = summary_match.group(1)
            print("\nVerification Summary:")
            print(summary)
        
        return adjusted_actions

    def on_analyze_job(self):
        """Handle job analysis hotkey"""
        if not self.running:
            return
            
        try:
            # Simulate Ctrl+C to copy selected text
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.5)  # Wait for clipboard
            
            job_text = pyperclip.paste()
            
            if job_text:
                print("\nAnalyzing job description...")
                analysis = self.analyze_job_description(job_text)
                if analysis:
                    print("\nJob Analysis:")
                    print(analysis)
                    
                    save = input("\nSave this analysis for form filling? (y/n): ").lower()
                    if save == 'y':
                        self.current_application["analysis"] = analysis
                else:
                    print("Could not analyze job description")
            else:
                print("No text selected. Please select job description text first.")
        except Exception as e:
            print(f"Error analyzing job description: {e}")

    def map_grid_points(self):
        """Create a dense reference grid map with 100x100 subcells within A1-J10 grid"""
        try:
            self.widget.update_status("Creating dense reference grid map...")
            
            # Take initial screenshot for reference
            screenshot = pyautogui.screenshot()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get dimensions and scale factor
            height, width = screenshot.height, screenshot.width
            scale_factor = self.app.primaryScreen().devicePixelRatio()
            
            # Create mapping results
            mapping_results = {
                'screen_info': {
                    'width': width,
                    'height': height,
                    'scale_factor': scale_factor,
                    'timestamp': timestamp,
                    'main_grid_size': 10,
                    'sub_grid_size': 10  # 10 subcells per main cell
                },
                'points': {}
            }
            
            # Create black canvas for grid map
            grid_map = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Calculate main cell dimensions
            main_cell_width = width // 10
            main_cell_height = height // 10
            
            # Calculate sub-cell dimensions
            sub_cell_width = main_cell_width // 10
            sub_cell_height = main_cell_height // 10
            
            # Draw and label all grid points
            for main_row in range(10):  # Main grid rows (1-10)
                for main_col in range(10):  # Main grid columns (A-J)
                    # Main cell coordinate
                    main_coord = f"{chr(65 + main_col)}{main_row + 1}"
                    
                    # Calculate main cell boundaries
                    main_x_start = main_col * main_cell_width
                    main_y_start = main_row * main_cell_height
                    
                    # Draw main cell boundary
                    cv2.rectangle(grid_map,
                                (main_x_start, main_y_start),
                                (main_x_start + main_cell_width, main_y_start + main_cell_height),
                                (60, 60, 60), 2)
                    
                    # Add main coordinate label
                    cv2.putText(grid_map, main_coord,
                              (main_x_start + 10, main_y_start + 20),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (0, 255, 0), 2)
                    
                    # Create sub-grid within main cell
                    for sub_row in range(10):
                        for sub_col in range(10):
                            # Calculate exact position
                            x = main_x_start + (sub_col * sub_cell_width)
                            y = main_y_start + (sub_row * sub_cell_height)
                            
                            # Generate detailed coordinate
                            sub_coord = f"{main_coord}.{sub_col}{sub_row}"
                            
                            # Store point data
                            mapping_results['points'][sub_coord] = {
                                'position': {'x': x, 'y': y},
                                'main_cell': main_coord,
                                'sub_position': {'col': sub_col, 'row': sub_row}
                            }
                            
                            # Draw reference point
                            cv2.circle(grid_map, (x, y), 1, (0, 255, 0), -1)
                            
                            # Add sub-coordinate label (every 2nd point to avoid overcrowding)
                            if sub_col % 2 == 0 and sub_row % 2 == 0:
                                cv2.putText(grid_map, f"{sub_col}{sub_row}",
                                          (x + 2, y + 2),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.3, (0, 255, 0), 1)
            
            # Add grid lines and labels
            enhanced_map = self._enhance_dense_grid_map_with_labels(grid_map, width, height, 
                                                                      main_cell_width, main_cell_height,
                                                                      sub_cell_width, sub_cell_height)
            
            # Save mapping results
            mapping_file = os.path.join(
                self.screenshots_dir,
                f"dense_grid_mapper_{timestamp}.json"
            )
            with open(mapping_file, 'w') as f:
                json.dump(mapping_results, f, indent=2)
            
            # Save the grid map
            map_file = os.path.join(
                self.screenshots_dir,
                f"dense_grid_mapper_visualization_{timestamp}.png"
            )
            cv2.imwrite(map_file, enhanced_map)
            
            # Generate summary
            summary = f"""
            Dense Reference Grid Map Complete:
            - Screen Resolution: {width}x{height}
            - Scale Factor: {scale_factor}
            - Main Grid: A1-J10
            - Sub-grid: 10x10 within each main cell
            - Total Cells: 100x100
            - Main Cell Size: {main_cell_width}x{main_cell_height} pixels
            - Sub-cell Size: {sub_cell_width}x{sub_cell_height} pixels
            - Results saved to: {mapping_file}
            - Visualization: {map_file}
            
            Coordinate Format:
            - Main Grid: A1-J10
            - Sub-grid: A1.45 = Cell A1, sub-pos (4,5)
            """
            
            self.widget.update_insights(summary)
            self.widget.update_status("Dense reference grid map complete!")
            
            return mapping_results
            
        except Exception as e:
            self.widget.update_status(f"Error during mapping: {str(e)}")
            print(f"Error during grid mapping: {e}")
            return None

    def _enhance_dense_grid_map_with_labels(self, grid_map, width, height, 
                                          main_cell_width, main_cell_height,
                                          sub_cell_width, sub_cell_height):
        """Add clear labels and legend to the dense grid map"""
        # Create a larger canvas to accommodate labels
        margin = 50
        canvas = np.zeros((height + 2*margin, width + 2*margin, 3), dtype=np.uint8)
        
        # Copy the grid map to the center of the canvas
        canvas[margin:margin+height, margin:margin+width] = grid_map
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        
        # Column labels (A-J)
        for i in range(10):
            label = chr(65 + i)
            x = margin + i * main_cell_width + (main_cell_width // 2)
            # Top labels
            cv2.putText(canvas, label, (x-10, margin-10), font, font_scale, (255, 255, 255), 2)
            # Bottom labels
            cv2.putText(canvas, label, (x-10, margin+height+30), font, font_scale, (255, 255, 255), 2)
        
        # Row labels (1-10)
        for i in range(10):
            label = str(i + 1)
            y = margin + i * main_cell_height + (main_cell_height // 2)
            # Left labels
            cv2.putText(canvas, label, (margin-30, y+10), font, font_scale, (255, 255, 255), 2)
            # Right labels
            cv2.putText(canvas, label, (margin+width+10, y+10), font, font_scale, (255, 255, 255), 2)
        
        # Add legend
        legend_y = margin + height + 60
        cv2.putText(canvas, "Dense Reference Grid Map (100x100)", (margin, legend_y), font, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, "Main Grid: A1-J10", (margin, legend_y + 30), font, font_scale, (255, 255, 255), 1)
        cv2.putText(canvas, "Sub-grid: 00-99 per cell", (margin + 300, legend_y + 30), font, font_scale, (255, 255, 255), 1)
        cv2.putText(canvas, "Format: A1.45 = Cell A1, sub-pos (4,5)", (margin + 600, legend_y + 30), font, font_scale, (0, 255, 0), 1)
        
        # Add visual examples in legend
        cv2.circle(canvas, (width - 150, 30), 3, (255, 0, 0), -1)  # Example intersection point
        cv2.line(canvas, (width - 100, 30), (width - 50, 30), (0, 255, 0), 1)  # Example grid line
        
        return canvas

    def _create_grid_overlay(self, img, screen_width_pixels, screen_height_pixels):
        """Create a consistent grid overlay on the image"""
        height, width = img.shape[:2]
        
        # Calculate cell dimensions - 40x40 grid
        main_cell_width = width // 40
        main_cell_height = height // 40
        
        # Create overlay
        overlay = img.copy()
        
        # Define colors (BGR format)
        GRID_COLOR = (0, 0, 255)  # Pure red - best for AI detection
        MARKER_COLOR = (255, 0, 0)  # Blue markers for intersection points
        
        # Draw main grid with intersection markers
        for i in range(41):  # 41 lines for 40 cells
            x = i * main_cell_width
            cv2.line(overlay, (x, 0), (x, height), GRID_COLOR, 1)
            
            for j in range(41):
                y = j * main_cell_height
                if i == 0:  # Draw horizontal lines once
                    cv2.line(overlay, (0, y), (width, y), GRID_COLOR, 1)
                
                # Draw intersection markers
                intersection_x = x
                intersection_y = y
                
                # Draw a more visible intersection marker
                cv2.circle(overlay, (intersection_x, intersection_y), 3, MARKER_COLOR, -1)
                
                # Add coordinate text at each intersection
                if i < 40 and j < 40:  # Don't add text for the last lines
                    # Calculate column letters using base-26 system
                    first_letter = chr(65 + (i // 26))
                    second_letter = chr(65 + (i % 26))
                    coord = f"{first_letter}{second_letter}{j+1}"
                    
                    # Draw coordinate background for better visibility
                    text_size = cv2.getTextSize(coord, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    text_x = intersection_x + 5
                    text_y = intersection_y - 5
                    
                    # Draw white background rectangle
                    cv2.rectangle(overlay, 
                                (text_x - 2, text_y - text_size[1] - 2),
                                (text_x + text_size[0] + 2, text_y + 2),
                                (255, 255, 255),
                                -1)
                    
                    # Draw coordinate text
                    cv2.putText(overlay, coord,
                              (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.4,
                              GRID_COLOR,
                              1)
        
        # Add matrix indicators along edges
        for i in range(40):
            # Column indicators (AA-ZZ)
            first_letter = chr(65 + (i // 26))
            second_letter = chr(65 + (i % 26))
            col_label = f"{first_letter}{second_letter}"
            
            # Draw column label at top
            cv2.putText(overlay, col_label,
                      (i * main_cell_width + 5, 20),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.5,
                      GRID_COLOR,
                      2)
            
            # Row indicators (1-40)
            row_label = str(i + 1)
            # Draw row label on left
            cv2.putText(overlay, row_label,
                      (5, i * main_cell_height + 20),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.5,
                      GRID_COLOR,
                      2)
        
        # Blend overlay with original image
        alpha = 0.5  # Balanced transparency
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, overlay)
        
        # Add legend with larger text
        legend_height = 100
        legend = np.zeros((legend_height, width, 3), dtype=np.uint8)
        legend.fill(255)  # White background
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Add legend text with clear formatting
        def add_legend_text(text, pos_x, pos_y):
            cv2.putText(legend, text, (pos_x, pos_y), font, 1.0, (0, 0, 0), 2)  # Black text
        
        add_legend_text("Matrix Reference System:", 10, 30)
        add_legend_text(f"• Grid Size: 40x40 (AA1-ZZ40) - Screen: {screen_width_pixels}x{screen_height_pixels} px", 10, 60)
        add_legend_text("• Blue Dots: Click/Type Targets   • Red Lines: Grid Reference", 10, 90)
        
        # Add visual examples in legend
        cv2.circle(legend, (width - 150, 30), 3, MARKER_COLOR, -1)  # Example intersection point
        cv2.line(legend, (width - 100, 30), (width - 50, 30), GRID_COLOR, 1)  # Example grid line
        
        # Combine image with legend
        img_with_legend = np.vstack([overlay, legend])
        
        return img_with_legend

    def on_press(self, key):
        """Handle key press"""
        try:
            # Add the pressed key to current keys
            if hasattr(key, 'char'):  # Regular keys
                self.current_keys.add(key.char.lower())
            else:  # Special keys
                self.current_keys.add(key)
            
            # Check for Ctrl+Shift+A
            if (keyboard.Key.ctrl_l in self.current_keys and 
                keyboard.Key.shift in self.current_keys and 
                'a' in self.current_keys):
                self.on_analyze_form()
            
            # Check for Ctrl+Shift+J
            elif (keyboard.Key.ctrl_l in self.current_keys and 
                  keyboard.Key.shift in self.current_keys and 
                  'j' in self.current_keys):
                self.on_analyze_job()
            
            # Check for Ctrl+Shift+M (New mapping hotkey)
            elif (keyboard.Key.ctrl_l in self.current_keys and 
                  keyboard.Key.shift in self.current_keys and 
                  'm' in self.current_keys):
                self.map_grid_points()
            
            # Check for Ctrl+Q
            elif (keyboard.Key.ctrl_l in self.current_keys and 
                  'q' in self.current_keys):
                self.running = False
                self.app.quit()
            
            # Check for Escape
            elif key == keyboard.Key.esc:
                self.running = False
                self.app.quit()
                
        except AttributeError:
            pass  # Ignore attribute errors from special keys

    def on_release(self, key):
        """Handle key release"""
        try:
            if hasattr(key, 'char'):  # Regular keys
                self.current_keys.discard(key.char.lower())
            else:  # Special keys
                self.current_keys.discard(key)
        except AttributeError:
            pass  # Ignore attribute errors from special keys

if __name__ == "__main__":
    assistant = JobApplicationAssistant()
    assistant.run() 