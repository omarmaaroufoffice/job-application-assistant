import os
import time
from google import genai
import pyautogui
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import json
import re
from typing import List, Dict, Any
from pynput import keyboard
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
                self.update_status("Analysis complete")
                self._show_confirmation(actions)
            else:
                self._update_actions([])
                self.update_status("No actionable elements found")
        except Exception as e:
            self.update_status(f"Error processing actions: {str(e)}")
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
        
        # Load grid reference
        self.load_grid_reference()
        
        # Set up PyAutoGUI safety features
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 1.0
        
        # Initialize screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
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

    def load_grid_reference(self):
        """Load grid reference data"""
        try:
            with open('grid_reference.json', 'r') as f:
                self.grid_data = json.load(f)
            print("Grid reference loaded successfully")
        except FileNotFoundError:
            print("Grid reference not found. Please run screen_mapper.py first")
            self.grid_data = None

    def get_nearest_grid_point(self, x: int, y: int) -> str:
        """Find the nearest grid point to given coordinates"""
        if not self.grid_data:
            return None
            
        min_dist = float('inf')
        nearest_point = None
        
        for coord_label, point in self.grid_data['points'].items():
            dist = ((x - point['x'])**2 + (y - point['y'])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                nearest_point = coord_label
        
        return nearest_point

    def annotate_screenshot(self, image_path: str, actions: List[tuple], timestamp: str) -> str:
        """Annotate screenshot with planned actions and grid coordinates"""
        img = cv2.imread(image_path)
        
        # Colors (BGR format)
        colors = {
            'click': (0, 255, 0),    # Green
            'type': (255, 0, 0),     # Blue
            'select': (0, 0, 255)    # Red
        }
        
        annotated = img.copy()
        
        # Add title and timestamp
        cv2.putText(annotated, "Detected Actions (with Grid Coordinates):", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                  1.0, (0, 0, 0), 2)
        
        # Draw grid lines (light gray)
        for i in range(11):  # 10x10 grid
            x = int(i * self.grid_data['cell_width'])
            y = int(i * self.grid_data['cell_height'])
            
            # Vertical lines
            cv2.line(annotated, (x, 0), (x, self.screen_height), (200, 200, 200), 1)
            # Horizontal lines
            cv2.line(annotated, (0, y), (self.screen_width, y), (200, 200, 200), 1)
        
        # Add each action to the image
        for idx, action in enumerate(actions, 1):
            action_type = action[0]
            x, y = action[1], action[2]
            color = colors[action_type]
            grid_coord = action[-1]  # Last element is grid coordinate
            
            # Draw circle at action point
            cv2.circle(annotated, (x, y), 15, color, 2)
            
            # Draw arrow
            cv2.arrowedLine(annotated, 
                         (x + 50, y + 50), 
                         (x, y), 
                         color, 
                         2, 
                         tipLength=0.3)
            
            # Add label with grid coordinate
            cv2.putText(annotated, f"{idx}. {grid_coord}", 
                      (x-5, y+25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, color, 2)
            
            # Add description box
            if action_type == 'type':
                text = f"{idx}. {action_type.upper()} at {grid_coord}: {action[3]}"
            else:
                text = f"{idx}. {action_type.upper()} at {grid_coord}: {action[-2]}"  # -2 is description
                
            (text_width, text_height), _ = cv2.getTextSize(text, 
                                                         cv2.FONT_HERSHEY_SIMPLEX,
                                                         0.7, 2)
            
            # Draw text background
            cv2.rectangle(annotated,
                        (10, 50 + 30 * idx),
                        (10 + text_width, 50 + 30 * idx + text_height + 10),
                        color,
                        -1)
            
            # Draw text
            cv2.putText(annotated, text,
                      (10, 50 + 30 * idx + text_height), 
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.7, (255, 255, 255), 2)
        
        # Add timestamp
        cv2.putText(annotated, f"Analyzed: {timestamp}", 
                  (10, annotated.shape[0] - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 
                  0.6, (0, 0, 0), 1)
        
        # Save annotated image
        annotated_path = os.path.join(
            self.screenshots_dir, 
            f"annotated_screenshot_{timestamp}.png"
        )
        cv2.imwrite(annotated_path, annotated)
        
        return annotated_path

    def capture_screenshot(self):
        """Capture a screenshot and save it temporarily"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot = pyautogui.screenshot()
        
        # Save original screenshot
        original_path = os.path.join(
            self.screenshots_dir, 
            f"original_screenshot_{timestamp}.png"
        )
        screenshot.save(original_path)
        
        return original_path, timestamp

    def analyze_application_form(self, image_path):
        """Analyze the job application form using Gemini AI with grid reference"""
        try:
            self.widget.update_status("Analyzing form...")
            # Open images
            image = Image.open(image_path)
            grid_img = Image.open('grid_reference.png')
            
            prompt = f"""
            Analyze this job application form screenshot using the grid reference system.
            The screen is divided into a 10x10 grid (A1-J10), where:
            - A1 is top-left corner
            - J10 is bottom-right corner
            - Each intersection has a red dot and coordinate label
            
            Look for these elements and map them to the nearest grid coordinates:

            1. Interactive Elements:
               - Blue "Apply now" buttons
               - "Yes/No/Skip" buttons
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

            For each element found, provide:
            - Element type
            - Nearest grid coordinate (e.g., A1, B5, J10)
            - Description
            - Recommended action based on my profile
            
            Format response EXACTLY as:
            - CLICK <grid_coord>: <element_type> - <description>
            - TYPE <grid_coord> "<text>": <field_type> - <content from profile>
            - SELECT <grid_coord>: <option> - <reason>

            Current application context:
            {json.dumps(self.current_application, indent=2)}
            """
            
            # Generate content using the correct model and format
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image, grid_img]
            )
            
            if response and response.text:
                self.widget.update_insights(response.text)
                return response.text
            return None
            
        except Exception as e:
            self.widget.update_status(f"Error: {str(e)}")
            print(f"Error analyzing form: {e}")
            return None

    def parse_ai_response(self, ai_response):
        """Parse AI response into actionable commands using grid coordinates"""
        actions = []
        if ai_response:
            lines = ai_response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('- CLICK'):
                    # Extract grid coordinate (e.g., A1, B5, H4) - handle both <H4> and H4: formats
                    coord_match = re.search(r'[<\s]([A-J][1-9][0]?)[>:]', line)
                    if coord_match:
                        coord = coord_match.group(1)
                        if coord in self.grid_data['points']:
                            point = self.grid_data['points'][coord]
                            description = re.search(r'(?::|>)\s*(.*?)(?:\s*$|-\s*|$)', line)
                            desc = description.group(1) if description else ""
                            actions.append(('click', point['x'], point['y'], desc, coord))
                
                elif line.startswith('- TYPE'):
                    coord_match = re.search(r'[<\s]([A-J][1-9][0]?)[>:]', line)
                    text_match = re.search(r'"([^"]*)"', line)
                    if coord_match and text_match and coord_match.group(1) in self.grid_data['points']:
                        coord = coord_match.group(1)
                        point = self.grid_data['points'][coord]
                        text = text_match.group(1)
                        description = re.search(r'(?::|>)\s*(.*?)(?:\s*$|-\s*|$)', line)
                        desc = description.group(1) if description else ""
                        actions.append(('type', point['x'], point['y'], text, desc, coord))
                
                elif line.startswith('- SELECT'):
                    coord_match = re.search(r'[<\s]([A-J][1-9][0]?)[>:]', line)
                    if coord_match and coord_match.group(1) in self.grid_data['points']:
                        coord = coord_match.group(1)
                        point = self.grid_data['points'][coord]
                        description = re.search(r'(?::|>)\s*(.*?)(?:\s*$|-\s*|$)', line)
                        desc = description.group(1) if description else ""
                        actions.append(('select', point['x'], point['y'], desc, coord))
        
        return actions

    def execute_single_action(self, action):
        """Execute a single action"""
        try:
            x, y = action[1], action[2]
            if action[0] in ['click', 'select']:
                pyautogui.moveTo(x, y, duration=0.5, tween=pyautogui.easeInOutQuad)
                pyautogui.click()
            elif action[0] == 'type':
                pyautogui.moveTo(x, y, duration=0.5)
                pyautogui.click()
                pyautogui.typewrite(action[3], interval=0.1)
            
            # Store successful action in current application state
            self.current_application['grid_positions'][action[-1]] = {
                'type': action[0],
                'description': action[3] if action[0] == 'type' else action[-2]
            }
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error executing action at {action[-1]}: {e}")
            self.widget.update_status(f"Error executing action: {e}")

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
        screenshot_path, timestamp = self.capture_screenshot()
        ai_analysis = self.analyze_application_form(screenshot_path)
        
        if ai_analysis:
            print("\nAI Analysis:", ai_analysis)
            # Process actions in main thread via signal
            self.widget.signals.process_actions_signal.emit(ai_analysis)
            
            # Annotate screenshot with actions (this is done in the background thread)
            actions = self.parse_ai_response(ai_analysis)
            if actions:
                annotated_path = self.annotate_screenshot(screenshot_path, actions, timestamp)
                print(f"\nAnnotated screenshot saved to: {annotated_path}")
        else:
            self.widget.update_status("No actionable elements found")
            self.widget.signals.update_actions_signal.emit([])
            print("No actionable elements found in form")

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