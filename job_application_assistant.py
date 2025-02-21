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
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Get scale factor for display
        scale_factor = self.app.primaryScreen().devicePixelRatio()
        
        # Add title and timestamp
        cv2.putText(img, "Detected Actions:", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 0, 255), 2)
        
        # Add each action to the image
        for idx, action in enumerate(actions, 1):
            action_type = action[0]
            
            # Convert logical coordinates to screen coordinates for display
            x = int(action[1] * scale_factor)
            y = int(action[2] * scale_factor)
            grid_coord = action[-1]
            
            # Draw circle at action point
            cv2.circle(img, (x, y), 15, (0, 0, 255), 2)
            
            # Add label with grid coordinate (offset to not overlap with circle)
            cv2.putText(img, f"{idx}. {grid_coord}", 
                      (x + 25, y - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, (0, 0, 255), 2)
            
            # Add description in the top margin
            text = f"{idx}. {action_type.upper()} at {grid_coord}: {action[3]}"
            cv2.putText(img, text,
                      (10, 50 + 30 * idx), 
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.7, (0, 0, 255), 2)
        
        # Save annotated image
        annotated_path = os.path.join(
            self.screenshots_dir, 
            f"annotated_screenshot_{timestamp}.png"
        )
        cv2.imwrite(annotated_path, img)
        return annotated_path

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
            
            # Convert to PIL Image for Gemini AI
            gridded_pil = Image.fromarray(cv2.cvtColor(gridded_img, cv2.COLOR_BGR2RGB))
            
            prompt = f"""
            CRITICAL COORDINATE INSTRUCTIONS:
            You are looking at a screenshot with a green grid overlay. You MUST:
            1. ONLY use coordinates where you can see actual green dots in the grid
            2. Look for the closest green dot to each UI element
            3. Use the exact sub-grid numbers shown in green (00-99)
            4. Never make up or interpolate coordinates - if you can't see a green dot near the element, use the nearest visible one
            5. Double-check that your chosen coordinate has both:
               - A visible green dot
               - A readable sub-grid number nearby

            Grid System Reference:
            - Main Grid: A1-J10 (marked with large green labels)
            - Sub-grid: Each main cell has numbered points (shown in small green numbers)
            - Format: <main_cell>.<sub_position>
              Example: If you see main cell "A1" and sub-grid point "45", use "A1.45"

            COORDINATE SELECTION RULES:
            1. For each element, find the EXACT green dot closest to it
            2. Look at the nearest sub-grid number (small green numbers)
            3. Combine the main cell label with that exact sub-grid number
            4. Do NOT guess or interpolate positions - use only visible dots and numbers

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

            RESPONSE FORMAT:
            1. Start with "###GRID_ACTIONS_START###"
            2. JSON array of actions, each with:
               - element_type: Type of element
               - grid_coord: ONLY use coordinates you can see in the grid
               - description: What this element is
               - recommended_action: Format as "TYPE <coord> \"<text>\"" or "CLICK <coord>"
            3. End with "###GRID_ACTIONS_END###"

            Example:
            ###GRID_ACTIONS_START###
            [
              {{
                "element_type": "Text input",
                "grid_coord": "A1.44",  // Only if you see the "44" sub-grid number near a green dot in cell A1
                "description": "Name field",
                "recommended_action": "TYPE A1.44 \"John Doe\": Enter full name"
              }}
            ]
            ###GRID_ACTIONS_END###

            Current application context:
            {json.dumps(self.current_application, indent=2)}

            FINAL VERIFICATION:
            Before providing coordinates, verify that:
            1. Each coordinate corresponds to a visible green dot
            2. You can see the sub-grid number clearly
            3. The dot is the closest one to the UI element
            4. You are not guessing or interpolating positions
            """
            
            # Generate content using Gemini AI with the exact saved and reloaded image
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, gridded_pil]
            )
            
            if response and response.text:
                self.widget.update_insights(response.text)
                return response.text, gridded_img  # Return both the response and the exact image used
            return None, None
            
        except Exception as e:
            self.widget.update_status(f"Error: {str(e)}")
            print(f"Error analyzing form: {e}")
            return None, None

    def parse_ai_response(self, ai_response):
        """Parse AI response into actionable commands using dense grid coordinates"""
        actions = []
        if not ai_response:
            return actions
            
        try:
            # Extract JSON between delimiters
            json_match = re.search(r'###GRID_ACTIONS_START###\s*(.*?)\s*###GRID_ACTIONS_END###', 
                                 ai_response, re.DOTALL)
            if not json_match:
                print("No valid JSON found between delimiters")
                return actions
                
            json_str = json_match.group(1)
            action_list = json.loads(json_str)
            
            # Get screen dimensions and scale factor
            screen = self.app.primaryScreen()
            scale_factor = screen.devicePixelRatio()
            
            # Get actual screen dimensions in logical pixels (unscaled)
            screen_width = screen.geometry().width()
            screen_height = screen.geometry().height()
            
            # Calculate cell dimensions in logical pixels
            cell_width = screen_width // 40  # 40x40 grid
            cell_height = screen_height // 40
            
            print(f"Screen dimensions (logical): {screen_width}x{screen_height}")
            print(f"Cell dimensions (logical): {cell_width}x{cell_height}")
            
            for action_data in action_list:
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
                    
                    # Calculate column index (AA=0, AB=1, ..., ZZ=675)
                    first_letter = ord(col_letters[0]) - ord('A')  # First letter (A-Z) = 0-25
                    second_letter = ord(col_letters[1]) - ord('A')  # Second letter (A-Z) = 0-25
                    main_col = (first_letter * 26) + second_letter  # Use base-26 for letters
                    
                    # Calculate row index (1-based to 0-based)
                    main_row = row_num - 1
                    
                    # Calculate sub-position offsets (0-99 range)
                    sub_x = int(sub_pos) // 10
                    sub_y = int(sub_pos) % 10
                    
                    # Calculate final coordinates with sub-position offsets
                    x = int((main_col * cell_width) + (cell_width * sub_x / 10) + (cell_width / 20))
                    y = int((main_row * cell_height) + (cell_height * sub_y / 10) + (cell_height / 20))
                    
                    # Log coordinate calculation
                    print(f"\nCoordinate calculation for {coord}:")
                    print(f"Main cell: {col_letters}{row_num} -> col={main_col} (first={first_letter}, second={second_letter}), row={main_row}")
                    print(f"Sub-position: {sub_pos} -> ({sub_x}, {sub_y})")
                    print(f"Cell dimensions: {cell_width}x{cell_height}")
                    print(f"Final logical coordinates: ({x}, {y})")
                    
                    # Store the action with exact coordinates
                    if action_type == 'type':
                        text_match = re.search(r'"([^"]*)"', action_data['recommended_action'])
                        if text_match:
                            text = text_match.group(1)
                            actions.append(('type', x, y, text, action_data['description'], coord))
                    elif action_type in ['click', 'select']:
                        actions.append((action_type, x, y, action_data['description'], coord))
                        
                except ValueError as ve:
                    print(f"Error parsing coordinate {coord}: {ve}")
                    continue
                except Exception as e:
                    print(f"Error processing coordinate {coord}: {e}")
                    continue
        
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
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
                    mouse_controller.click(mouse.Button.left)
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
        
        # Get timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Initial Analysis
        # Analyze the form and get both the AI response and the gridded image
        ai_analysis, gridded_img = self.analyze_application_form(None)
        
        if ai_analysis and gridded_img is not None:
            print("\nInitial AI Analysis:", ai_analysis)
            
            # Parse actions and create initial annotation
            current_actions = self.parse_ai_response(ai_analysis)
            if current_actions:
                # Initialize verification iteration counter
                verification_iteration = 0
                max_iterations = 4
                is_satisfied = False
                
                # Keep track of the latest annotated image
                latest_annotated_path = None
                
                # Get screen dimensions and scale factor once
                screen = self.app.primaryScreen()
                scale_factor = screen.devicePixelRatio()
                screen_width = screen.geometry().width()
                screen_height = screen.geometry().height()
                
                # Get image dimensions
                height, width = gridded_img.shape[:2]
                
                # Calculate scaling ratios
                width_ratio = width / screen_width
                height_ratio = height / screen_height
                
                while verification_iteration < max_iterations and not is_satisfied:
                    verification_iteration += 1
                    print(f"\nVerification Iteration {verification_iteration}/{max_iterations}")
                    
                    # Create annotated version using the gridded image
                    current_annotated_img = gridded_img.copy()
                    
                    # Add annotations for current actions
                    for idx, action in enumerate(current_actions, 1):
                        action_type = action[0]
                        
                        # Convert logical coordinates to image coordinates using scaling ratios
                        x = int(action[1] * width_ratio)
                        y = int(action[2] * height_ratio)
                        grid_coord = action[-1]
                        
                        # Log coordinate transformation
                        print(f"\nCoordinate transformation for {grid_coord}:")
                        print(f"Original coordinates: ({action[1]}, {action[2]})")
                        print(f"Screen dimensions: {screen_width}x{screen_height}")
                        print(f"Image dimensions: {width}x{height}")
                        print(f"Scaling ratios: width={width_ratio}, height={height_ratio}")
                        print(f"Transformed coordinates: ({x}, {y})")
                        
                        # Use different colors based on iteration
                        if verification_iteration == 1:
                            circle_color = (0, 0, 255)  # Red for initial
                        elif verification_iteration == max_iterations:
                            circle_color = (0, 255, 0)  # Green for final
                        else:
                            circle_color = (255, 165, 0)  # Orange for intermediate
                        
                        # Draw circle at action point with larger radius and thicker outline
                        cv2.circle(current_annotated_img, (x, y), 20, circle_color, 3)
                        
                        # Add white background for better text visibility
                        text = f"{idx}. {grid_coord}"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(current_annotated_img,
                                    (x + 25, y - text_size[1] - 5),
                                    (x + 25 + text_size[0], y + 5),
                                    (255, 255, 255),
                                    -1)
                        cv2.putText(current_annotated_img, text,
                                  (x + 25, y),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7, circle_color, 2)
                        
                        # Add description with white background
                        desc_text = f"{idx}. {action_type.upper()} at {grid_coord}: {action[3]}"
                        desc_size = cv2.getTextSize(desc_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(current_annotated_img,
                                    (10, 50 + 30 * idx - desc_size[1] - 5),
                                    (10 + desc_size[0], 50 + 30 * idx + 5),
                                    (255, 255, 255),
                                    -1)
                        cv2.putText(current_annotated_img, desc_text,
                                  (10, 50 + 30 * idx),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7, circle_color, 2)
                    
                    # Add iteration status with white background
                    status_text = f"Verification Iteration {verification_iteration}/{max_iterations}"
                    status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                    cv2.rectangle(current_annotated_img,
                                (10, 10),
                                (10 + status_size[0], 40),
                                (255, 255, 255),
                                -1)
                    cv2.putText(current_annotated_img, status_text,
                              (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              1.0, (0, 0, 255), 2)
                    
                    # Remove previous annotated image if it exists
                    if latest_annotated_path and os.path.exists(latest_annotated_path):
                        os.remove(latest_annotated_path)
                    
                    # Save current iteration's annotated image
                    latest_annotated_path = os.path.join(
                        self.screenshots_dir,
                        f"analysis_{timestamp}_iter{verification_iteration}.png"
                    )
                    cv2.imwrite(latest_annotated_path, current_annotated_img)
                    print(f"\nIteration {verification_iteration} analysis saved to: {latest_annotated_path}")
                    
                    # Convert to PIL for Gemini
                    annotated_pil = Image.fromarray(cv2.cvtColor(current_annotated_img, cv2.COLOR_BGR2RGB))
                    
                    verification_prompt = f"""
                    COORDINATE VERIFICATION TASK (Iteration {verification_iteration}/{max_iterations}):
                    
                    You are looking at an annotated screenshot that shows proposed click/type locations.
                    Each action point is marked with a colored circle and numbered.
                    
                    Current Actions:
                    {json.dumps([(a[0], a[-1], a[3]) for a in current_actions], indent=2)}
                    
                    Please verify with extra attention to accuracy:
                    1. Are ALL marked positions precisely aligned with their intended targets?
                    2. Do ANY coordinates need even slight adjustments?
                    3. Are there ANY misaligned or incorrect positions?
                    4. Should ANY actions be added or removed?
                    5. Is this iteration's result COMPLETELY satisfactory?
                    
                    RESPONSE FORMAT:
                    1. Start with "###VERIFICATION_RESULT###"
                    2. For each action, indicate:
                       - CORRECT: Position is perfectly accurate
                       - ADJUST: Needs adjustment (provide new coordinate)
                       - REMOVE: Action should be removed
                    3. End with "###SATISFACTION_STATUS###"
                    4. State if you are SATISFIED or UNSATISFIED with this iteration
                    5. End with "###VERIFICATION_END###"
                    
                    Example:
                    ###VERIFICATION_RESULT###
                    1. CORRECT: Click at AA1 for Submit button
                    2. ADJUST: Type at BB2 should be BB3 for email field
                    3. REMOVE: Click at CC3 is not a valid target
                    ###SATISFACTION_STATUS###
                    UNSATISFIED: Email field coordinate needs adjustment
                    ###VERIFICATION_END###
                    """
                    
                    # Generate verification using Gemini AI
                    verification_response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[verification_prompt, annotated_pil]
                    )
                    
                    if verification_response and verification_response.text:
                        print(f"\nVerification Analysis (Iteration {verification_iteration}):", verification_response.text)
                        
                        # Update insights with iteration status
                        self.widget.update_insights(
                            f"Verification Iteration {verification_iteration}/{max_iterations}\n\n" +
                            verification_response.text
                        )
                        
                        # Parse verification results
                        verification_text = verification_response.text
                        if "###VERIFICATION_RESULT###" in verification_text and "###VERIFICATION_END###" in verification_text:
                            # Extract satisfaction status
                            satisfaction_match = re.search(r'###SATISFACTION_STATUS###\s*(.*?)\s*###VERIFICATION_END###', 
                                                        verification_text, re.DOTALL)
                            is_satisfied = satisfaction_match and "SATISFIED" in satisfaction_match.group(1).upper()
                            
                            if is_satisfied:
                                print(f"\nAI is satisfied with the results after {verification_iteration} iterations")
                                break
                            
                            # Parse verification lines
                            verification_lines = verification_text.split("###VERIFICATION_RESULT###")[1].split("###SATISFACTION_STATUS###")[0].strip().split("\n")
                            
                            # Create adjusted actions list
                            adjusted_actions = []
                            for idx, action in enumerate(current_actions):
                                # Find corresponding verification line
                                for vline in verification_lines:
                                    if str(idx + 1) in vline:
                                        if "CORRECT" in vline:
                                            adjusted_actions.append(action)
                                        elif "ADJUST" in vline:
                                            # Try to extract new coordinate
                                            new_coord_match = re.search(r'should be ([A-Z]{2}\d+)', vline)
                                            if new_coord_match:
                                                new_coord = new_coord_match.group(1)
                                                # Create adjusted action with new coordinate
                                                adjusted_action = list(action)
                                                adjusted_action[-1] = new_coord
                                                adjusted_actions.append(tuple(adjusted_action))
                                            else:
                                                adjusted_actions.append(action)
                                        # Skip if REMOVE
                                        break
                            
                            # Update current actions for next iteration
                            current_actions = adjusted_actions
                    else:
                        print(f"\nVerification failed on iteration {verification_iteration}")
                        break
                
                # Create final annotated image with verified actions
                if current_actions:
                    final_annotated_img = gridded_img.copy()
                    
                    # Add annotations for final verified actions
                    for idx, action in enumerate(current_actions, 1):
                        action_type = action[0]
                        x = int((action[1] / screen.geometry().width()) * width)
                        y = int((action[2] / screen.geometry().height()) * height)
                        grid_coord = action[-1]
                        
                        # Draw circle at action point (green for final verified)
                        cv2.circle(final_annotated_img, (x, y), 15, (0, 255, 0), 2)
                        
                        # Add label with grid coordinate
                        cv2.putText(final_annotated_img, f"{idx}. {grid_coord}", 
                                  (x + 25, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 255, 0), 2)
                        
                        # Add description in the top margin
                        text = f"{idx}. {action_type.upper()} at {grid_coord}: {action[3]}"
                        cv2.putText(final_annotated_img, text,
                                  (10, 50 + 30 * idx), 
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 255, 0), 2)
                    
                    # Add final verification status
                    status = "VERIFIED AND SATISFIED" if is_satisfied else "VERIFIED (MAX ITERATIONS)"
                    cv2.putText(final_annotated_img, f"FINAL ACTIONS - {status}", 
                              (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              1.0, (0, 255, 0), 2)
                    
                    # Save final annotated image
                    final_annotated_path = os.path.join(
                        self.screenshots_dir,
                        f"final_annotated_{timestamp}.png"
                    )
                    cv2.imwrite(final_annotated_path, final_annotated_img)
                    print(f"\nFinal verified screenshot saved to: {final_annotated_path}")
                    
                    # Update UI with final verified actions
                    self.widget.signals.update_actions_signal.emit(current_actions)
                    self.widget.signals.show_confirmation_signal.emit(current_actions)
                else:
                    self.widget.update_status("No valid actions after verification")
                    self.widget.signals.update_actions_signal.emit([])
            else:
                self.widget.update_status("No actionable elements found")
                self.widget.signals.update_actions_signal.emit([])
        else:
            self.widget.update_status("Analysis failed")
            self.widget.signals.update_actions_signal.emit([])

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