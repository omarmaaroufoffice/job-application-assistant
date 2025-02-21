import os
import time
from google import genai
import pyautogui
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pynput import keyboard, mouse
import threading
import pyperclip
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QLabel, QTextEdit, QFrame, QPushButton, QHBoxLayout, QScrollArea)
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPixmap
import sys

# Load environment variables
load_dotenv()

# Configure Gemini AI with new client
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

client = genai.Client(api_key=GOOGLE_API_KEY)

class GoalManager:
    """Manages high-level goals and determines next logical steps"""
    
    def __init__(self):
        self.current_goal: Optional[str] = None
        self.goal_state: Dict[str, Any] = {}
        self.next_steps: List[Dict[str, Any]] = []
        
    def set_goal(self, goal: str, initial_state: Optional[Dict[str, Any]] = None):
        """Set a new goal and initialize its state"""
        self.current_goal = goal
        self.goal_state = initial_state or {}
        self.next_steps = []
        self._analyze_goal()
    
    def _analyze_goal(self):
        """Analyze the current goal and break it down into steps"""
        if not self.current_goal:
            return
            
        # Create analysis prompt for Gemini
        prompt = f"""
        GOAL ANALYSIS TASK
        
        Current Goal: {self.current_goal}
        Current State: {json.dumps(self.goal_state, indent=2)}
        
        Please analyze this goal and:
        1. Break it down into logical steps
        2. Identify the immediate next action
        3. Consider dependencies and prerequisites
        
        Respond in this exact format:
        ###STEPS_START###
        [
          {{
            "step": "Description of step",
            "type": "action_type",
            "requires": ["prerequisite1", "prerequisite2"],
            "produces": ["output1", "output2"]
          }}
        ]
        ###STEPS_END###
        """
        
        try:
            # Generate step analysis using Gemini
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            if response and response.text:
                # Extract steps JSON
                steps_match = re.search(
                    r'###STEPS_START###\s*(.*?)###STEPS_END###',
                    response.text,
                    re.DOTALL
                )
                
                if steps_match:
                    steps_json = steps_match.group(1).strip()
                    self.next_steps = json.loads(steps_json)
                    print("\nGoal Analysis Complete:")
                    print(f"Goal: {self.current_goal}")
                    print(f"Next Steps: {len(self.next_steps)} steps identified")
        except Exception as e:
            print(f"Error analyzing goal: {e}")
    
    def get_next_step(self) -> Optional[Dict[str, Any]]:
        """Get the next actionable step based on current state"""
        if not self.next_steps:
            return None
            
        # Find first step whose prerequisites are met
        for step in self.next_steps:
            if self._are_prerequisites_met(step):
                return step
        return None
    
    def _are_prerequisites_met(self, step: Dict[str, Any]) -> bool:
        """Check if all prerequisites for a step are met"""
        if 'requires' not in step:
            return True
            
        for req in step['requires']:
            if req not in self.goal_state.get('completed_steps', []):
                return False
        return True
    
    def update_state(self, completed_step: Dict[str, Any], results: Dict[str, Any]):
        """Update goal state after completing a step"""
        if 'completed_steps' not in self.goal_state:
            self.goal_state['completed_steps'] = []
            
        self.goal_state['completed_steps'].append(completed_step['step'])
        self.goal_state.update(results)
        
        # Remove completed step from next steps
        self.next_steps = [s for s in self.next_steps if s['step'] != completed_step['step']]
        
        # Re-analyze if needed
        if not self.next_steps:
            self._analyze_goal()
    
    def suggest_action(self, screenshot: Optional[np.ndarray] = None) -> Tuple[str, Dict[str, Any]]:
        """Suggest the next action based on current state and context"""
        next_step = self.get_next_step()
        if not next_step:
            return "No action needed", {}
            
        # If we have a screenshot, analyze it for context
        if screenshot is not None:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
            
            # Create context-aware prompt
            prompt = f"""
            NEXT ACTION ANALYSIS
            
            Current Goal: {self.current_goal}
            Current Step: {next_step['step']}
            Step Type: {next_step['type']}
            
            Based on the screenshot and current step:
            1. What specific action should be taken?
            2. Where should the action be performed?
            3. What is the expected outcome?
            
            Respond in this format:
            ###ACTION_START###
            {{
              "action": "specific_action",
              "location": "where_to_act",
              "expected_outcome": "what_should_happen"
            }}
            ###ACTION_END###
            """
            
            try:
                # Generate action analysis
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[prompt, pil_image]
                )
                
                if response and response.text:
                    # Extract action JSON
                    action_match = re.search(
                        r'###ACTION_START###\s*(.*?)###ACTION_END###',
                        response.text,
                        re.DOTALL
                    )
                    
                    if action_match:
                        action_json = action_match.group(1).strip()
                        action_details = json.loads(action_json)
                        return next_step['step'], action_details
            except Exception as e:
                print(f"Error analyzing action: {e}")
        
        return next_step['step'], {}

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
        self.setWindowTitle("AI Assistant")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setGeometry(50, 50, 400, 600)  # Increased height to accommodate confirmation
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Create signal handler
        self.signals = SignalHandler()
        self.signals.update_insights_signal.connect(self._update_insights)
        self.signals.update_actions_signal.connect(self._update_actions)
        self.signals.update_status_signal.connect(self._update_status)
        self.signals.show_confirmation_signal.connect(self._show_confirmation)
        self.signals.process_actions_signal.connect(self._process_actions)
        
        # Create a scroll area for the content
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.content_widget = QWidget()
        self.layout = QVBoxLayout(self.content_widget)
        self.scroll_area.setWidget(self.content_widget)
        self.setCentralWidget(self.scroll_area)
        
        # Add title bar
        self.title_bar = QLabel("AI Assistant")
        self.title_bar.setFont(QFont('Arial', 12, QFont.Bold))
        self.title_bar.setAlignment(Qt.AlignCenter)
        self.title_bar.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")
        self.title_bar.setFixedHeight(40)
        self.layout.addWidget(self.title_bar)
        
        # Add confirmation section at the top for better visibility
        confirmation_label = QLabel("Action Confirmation")
        confirmation_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.layout.addWidget(confirmation_label)
        
        self.confirmation_text = QTextEdit()
        self.confirmation_text.setReadOnly(True)
        self.confirmation_text.setMinimumHeight(100)
        self.confirmation_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 2px solid #007bff;
                border-radius: 5px;
                padding: 10px;
                font-size: 12pt;
            }
        """)
        self.layout.addWidget(self.confirmation_text)
        
        # Add confirmation buttons in a horizontal layout
        button_layout = QHBoxLayout()
        
        self.yes_button = QPushButton("Execute (Y)")
        self.yes_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        
        self.no_button = QPushButton("Skip (N)")
        self.no_button.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: black;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
        """)
        
        self.quit_button = QPushButton("Quit")
        self.quit_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        
        button_layout.addWidget(self.yes_button)
        button_layout.addWidget(self.no_button)
        button_layout.addWidget(self.quit_button)
        self.layout.addLayout(button_layout)
        
        # Add goal section
        goal_label = QLabel("Current Goal")
        goal_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.layout.addWidget(goal_label)
        
        self.goal_text = QTextEdit()
        self.goal_text.setReadOnly(True)
        self.goal_text.setMaximumHeight(60)
        self.goal_text.setStyleSheet("background-color: #e3f2fd; border: 1px solid #90caf9;")
        self.layout.addWidget(self.goal_text)
        
        # Add next step section
        next_step_label = QLabel("Next Step")
        next_step_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.layout.addWidget(next_step_label)
        
        self.next_step_text = QTextEdit()
        self.next_step_text.setReadOnly(True)
        self.next_step_text.setMaximumHeight(60)
        self.next_step_text.setStyleSheet("background-color: #e8f5e9; border: 1px solid #a5d6a7;")
        self.layout.addWidget(self.next_step_text)
        
        # Add user input section
        input_label = QLabel("What would you like me to help you with?")
        input_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.layout.addWidget(input_label)
        
        self.user_input = QTextEdit()
        self.user_input.setPlaceholderText("Enter your request here (e.g., 'Help me organize my desktop', 'Fill out this form', etc.)")
        self.user_input.setMaximumHeight(60)
        self.layout.addWidget(self.user_input)
        
        # Add send button
        self.send_button = QPushButton("Send Request")
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.send_button.clicked.connect(self.on_send_request)
        self.layout.addWidget(self.send_button)
        
        # Add conversation history section
        history_label = QLabel("Conversation History")
        history_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.layout.addWidget(history_label)
        
        self.conversation_text = QTextEdit()
        self.conversation_text.setReadOnly(True)
        self.conversation_text.setMinimumHeight(80)  # Reduced from 150
        self.conversation_text.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.layout.addWidget(self.conversation_text)
        
        # Add insights section
        insights_label = QLabel("Current Insights")
        insights_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.layout.addWidget(insights_label)
        
        self.insights_text = QTextEdit()
        self.insights_text.setReadOnly(True)
        self.insights_text.setMinimumHeight(80)  # Reduced from 150
        self.insights_text.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.layout.addWidget(self.insights_text)
        
        # Add actions section
        actions_label = QLabel("Pending Actions")
        actions_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.layout.addWidget(actions_label)
        
        self.actions_text = QTextEdit()
        self.actions_text.setReadOnly(True)
        self.actions_text.setMinimumHeight(80)  # Reduced from 150
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
        
        # Add annotated image display label to show the annotated screenshot
        self.annotated_img_label = QLabel()
        self.annotated_img_label.setAlignment(Qt.AlignCenter)
        self.annotated_img_label.setFixedHeight(200)  # Adjust as needed
        self.layout.addWidget(self.annotated_img_label)
        
        # Connect button signals
        self.yes_button.clicked.connect(self.on_yes_clicked)
        self.no_button.clicked.connect(self.on_no_clicked)
        self.quit_button.clicked.connect(self.on_quit_clicked)
        
        # Initially hide confirmation section
        self.confirmation_text.hide()
        self.yes_button.hide()
        self.no_button.hide()
        self.quit_button.hide()
        
        # Store current actions
        self.current_actions = None
        self.current_action_index = 0
        
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
        
        # Initialize with welcome message
        self.conversation_text.setText("AI Assistant: Hello! I'm here to help you with any task. What would you like me to do?")
        self.insights_text.setText("Ready to assist you.\nUse the input box above to make a request, or use hotkeys for specific tasks.")
        self.actions_text.setText("No actions pending.\nWaiting for your request...")
        self.goal_text.setText("No active goal")
        self.next_step_text.setText("Waiting for your request...")

    def update_goal(self, goal: str):
        """Update the current goal display"""
        self.goal_text.setText(goal)
        
    def update_next_step(self, step: str, details: Dict[str, Any]):
        """Update the next step display"""
        text = f"Step: {step}\n"
        if details:
            text += f"Details: {json.dumps(details, indent=2)}"
        self.next_step_text.setText(text)
        
    def on_send_request(self):
        """Handle user request submission"""
        user_request = self.user_input.toPlainText().strip()
        if not user_request:
            return
            
        # Update goal display
        self.update_goal(user_request)
        
        # Add request to conversation
        self.conversation_history.append({"role": "user", "content": user_request})
        self._update_conversation_display()
        
        # Clear input
        self.user_input.clear()
        
        # Let the assistant handle the request
        self.assistant.handle_user_request(user_request)
        
    def _update_insights(self, text: str):
        """Update insights with goal-oriented information"""
        try:
            # Parse potential goal/step information from text
            goal_match = re.search(r'Goal:\s*(.+?)(?:\n|$)', text)
            step_match = re.search(r'Next Step:\s*(.+?)(?:\n|$)', text)
            
            if goal_match:
                self.update_goal(goal_match.group(1))
            if step_match:
                # Extract action details if present
                details_start = text.find('Action Details:')
                if details_start != -1:
                    try:
                        details_text = text[details_start:].split('Action Details:', 1)[1].strip()
                        details = json.loads(details_text)
                        self.update_next_step(step_match.group(1), details)
                    except:
                        self.update_next_step(step_match.group(1), {})
                else:
                    self.update_next_step(step_match.group(1), {})
            
            # Update insights text
            self.insights_text.setText(text)
            
        except Exception as e:
            print(f"Error updating insights: {e}")
            self.status_bar.setText("Error updating insights display")
            
    def _update_actions(self, actions: List[tuple]):
        """Internal method to update actions in main thread"""
        try:
            if not actions:
                self.actions_text.setText("No actionable elements found")
                return
                
            # Only take the first (most relevant) action
            action = actions[0]
            
            text = "Next Action:\n\n"
            if len(action) >= 6:  # Make sure we have all needed elements
                action_type = action[0]
                grid_coord = action[4]  # Grid coordinate is now in position 4
                description = action[3]  # Description is in position 3
                
                if action_type == 'type':
                    text += f"TYPE at {grid_coord}:\n   Text: \"{action[3]}\"\n   Purpose: {description}"
                else:
                    text += f"{action_type.upper()} at {grid_coord}:\n   Purpose: {description}"
            
            self.actions_text.setText(text)
            
            # Show the confirmation section with only the most relevant action
            self._show_confirmation([action])
            
        except Exception as e:
            print(f"Error updating actions: {e}")
            self.status_bar.setText("Error updating actions display")
            
    def _show_confirmation(self, actions: List[tuple]):
        """Show confirmation with grid context"""
        try:
            if not actions:
                self._hide_confirmation()
                return
                
            self.current_actions = actions
            self.current_action_index = 0
            
            # Create detailed confirmation message for the single action
            action = actions[0]
            confirmation = "Please confirm this action:\n\n"
            
            if len(action) >= 6:
                action_type = action[0]
                grid_coord = action[4]
                description = action[3]
                
                if action_type == 'type':
                    confirmation += f"TYPE at {grid_coord}:\n"
                    confirmation += f"   Text: \"{action[3]}\"\n"
                    confirmation += f"   Purpose: {description}\n"
                else:
                    confirmation += f"{action_type.upper()} at {grid_coord}:\n"
                    confirmation += f"   Purpose: {description}\n"
            
            confirmation += "\nPress 'Execute' to perform this action, 'Skip' to try another action, or 'Quit' to cancel."
            
            # Show confirmation section
            self.confirmation_text.setText(confirmation)
            self.confirmation_text.show()
            self.yes_button.show()
            self.no_button.show()
            self.quit_button.show()
            
        except Exception as e:
            print(f"Error showing confirmation: {e}")
            self.status_bar.setText("Error showing confirmation dialog")
    
    def on_yes_clicked(self):
        """Handle Yes button click with goal tracking"""
        if self.current_actions and self.current_action_index < len(self.current_actions):
            action = self.current_actions[self.current_action_index]
            
            # Execute the action
            success = self.assistant.execute_single_action(action)
            
            if success:
                # Get next step from goal manager
                screenshot = pyautogui.screenshot()
                img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                next_step, action_details = self.assistant.goal_manager.suggest_action(img)
                
                # Update UI with next step
                self.update_next_step(next_step, action_details)
                
                # Move to next action if available
                self.current_action_index += 1
                if self.current_action_index < len(self.current_actions):
                    current_goal = self.assistant.goal_manager.current_goal
                    current_step = self.assistant.goal_manager.get_next_step()
                    if current_goal and current_step:
                        self._show_next_action_with_context(current_goal, current_step)
                    else:
                        self._show_next_action()
                else:
                    self._hide_confirmation()
                    self._update_status("All actions completed")
            else:
                self._update_status("Action failed")
                self._hide_confirmation()
        else:
            self._hide_confirmation()
            self._update_status("All actions completed")
            
    def on_no_clicked(self):
        """Handle No button click with goal adjustment"""
        if self.current_actions and self.current_action_index < len(self.current_actions):
            # Skip current action
            self.current_action_index += 1
            
            # Get current goal context
            current_goal = self.assistant.goal_manager.current_goal
            current_step = self.assistant.goal_manager.get_next_step()
            
            if self.current_action_index < len(self.current_actions):
                if current_goal and current_step:
                    self._show_next_action_with_context(current_goal, current_step)
                else:
                    self._show_next_action()
            else:
                self._hide_confirmation()
                
                # Get next step from goal manager
                screenshot = pyautogui.screenshot()
                img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                next_step, action_details = self.assistant.goal_manager.suggest_action(img)
                
                # Update UI with next step
                self.update_next_step(next_step, action_details)
        else:
            self._hide_confirmation()
            
    def on_quit_clicked(self):
        """Handle Quit button click with goal cleanup"""
        self._hide_confirmation()
        self.assistant.goal_manager.set_goal(None)  # Clear current goal
        self.update_goal("No active goal")
        self.update_next_step("Waiting for your request...", {})
    
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
    
    def update_status(self, text: str):
        """Thread-safe update of status bar"""
        self.signals.update_status_signal.emit(text)
    
    def _update_status(self, text: str):
        """Internal method to update status in main thread"""
        self.status_bar.setText(text)

    def _update_conversation_display(self):
        """Update the conversation history display"""
        try:
            formatted = []
            for msg in self.conversation_history:
                role = "You" if msg["role"] == "user" else "AI Assistant"
                formatted.append(f"{role}: {msg['content']}\n")
            
            # Set text and scroll to bottom
            self.conversation_text.setText("\n".join(formatted))
            scrollbar = self.conversation_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
        except Exception as e:
            print(f"Error updating conversation display: {e}")
            self.status_bar.setText("Error updating conversation display")

    def _hide_confirmation(self):
        """Hide the confirmation section"""
        self.confirmation_text.hide()
        self.yes_button.hide()
        self.no_button.hide()
        self.quit_button.hide()
        self.current_actions = None
        self.current_action_index = 0
    
    def _process_actions(self, ai_response: str):
        """Process AI response and update UI in main thread"""
        try:
            actions = self.assistant.parse_ai_response(ai_response)
            if actions:
                # Only take the first action
                first_action = actions[0]
                self._update_actions([first_action])
                self._update_status("Analyzing single action")
                self._show_confirmation([first_action])
            else:
                self._update_actions([])
                self._update_status("No actionable elements found")
        except Exception as e:
            self._update_status(f"Error processing actions: {str(e)}")
            print(f"Error processing actions: {e}")

    def _show_next_action(self):
        """Show the next action in the confirmation dialog"""
        try:
            if not self.current_actions or self.current_action_index >= len(self.current_actions):
                self._hide_confirmation()
                return
                
            # Get the next action
            action = self.current_actions[self.current_action_index]
            
            # Create confirmation message for single action
            confirmation = "Please confirm the following action:\n\n"
            
            action_type = action[0]
            grid_coord = action[4]
            description = action[3]
            
            if action_type == 'type':
                confirmation += f"TYPE at {grid_coord}:\n"
                confirmation += f"   Text: \"{description}\"\n"
                confirmation += f"   Purpose: {description}\n"
            else:
                confirmation += f"{action_type.upper()} at {grid_coord}:\n"
                confirmation += f"   Purpose: {description}\n"
            
            confirmation += "\nPress 'Execute' to perform this action, or 'Skip' to cancel."
            
            # Show confirmation section
            self.confirmation_text.setText(confirmation)
            self.confirmation_text.show()
            self.yes_button.show()
            self.no_button.show()
            self.quit_button.show()
            
        except Exception as e:
            print(f"Error showing next action: {e}")
            self.status_bar.setText("Error showing next action")
            self._hide_confirmation()

    def _show_next_action_with_context(self, current_goal: str, current_step: Dict[str, Any]):
        """Show the next action with goal context"""
        try:
            if not self.current_actions or self.current_action_index >= len(self.current_actions):
                self._hide_confirmation()
                return
                
            # Get the next action
            action = self.current_actions[self.current_action_index]
            
            # Create confirmation message with context
            confirmation = f"Current Goal: {current_goal}\n"
            confirmation += f"Current Step: {current_step.get('step', 'Unknown')}\n\n"
            confirmation += "Please confirm the following action:\n\n"
            
            action_type = action[0]
            grid_coord = action[4]
            description = action[3]
            
            if action_type == 'type':
                confirmation += f"TYPE at {grid_coord}:\n"
                confirmation += f"   Text: \"{description}\"\n"
                confirmation += f"   Purpose: {description}\n"
            else:
                confirmation += f"{action_type.upper()} at {grid_coord}:\n"
                confirmation += f"   Purpose: {description}\n"
            
            confirmation += "\nPress 'Execute' to perform this action, or 'Skip' to cancel."
            
            # Show confirmation section
            self.confirmation_text.setText(confirmation)
            self.confirmation_text.show()
            self.yes_button.show()
            self.no_button.show()
            self.quit_button.show()
            
        except Exception as e:
            print(f"Error showing next action with context: {e}")
            self.status_bar.setText("Error showing next action")
            self._hide_confirmation()

    def update_annotated_image(self, image_path: str):
        from PyQt5.QtGui import QPixmap  # Ensure QPixmap is imported
        pixmap = QPixmap(image_path)
        # Scale the pixmap to the label's size while keeping aspect ratio
        self.annotated_img_label.setPixmap(pixmap.scaled(self.annotated_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.annotated_img_label.show()

class JobApplicationAssistant:
    def __init__(self):
        # Initialize Qt Application
        self.app = QApplication(sys.argv)
        
        # Add emergency stop flags
        self.emergency_stop = False
        self.last_mouse_pos = (0, 0)
        self.corner_stop_threshold = 10  # pixels from corner to trigger stop
        
        # Add mouse listener for corner detection
        self.mouse_listener = mouse.Listener(on_move=self.on_mouse_move)
        self.mouse_listener.start()
        
        # Initialize screen dimensions with proper scaling for Retina display
        screen = self.app.primaryScreen()
        geometry = screen.geometry()
        scale_factor = screen.devicePixelRatio()
        self.screen_width = int(geometry.width() * scale_factor)
        self.screen_height = int(geometry.height() * scale_factor)
        print(f"Detected screen dimensions: {self.screen_width}x{self.screen_height} (Scale factor: {scale_factor})")
        
        # Initialize current application state
        self.current_application = {
            "status": "ready",
            "analysis": None,
            "form_data": {},
            "last_action": None,
            "progress": 0,
            "errors": []
        }
        
        # Initialize grid data
        self.grid_data = self._initialize_grid_data()
        
        # Create floating widget
        self.widget = FloatingWidget()
        self.widget.setParent(None)  # Ensure widget has no parent
        
        # Store widget reference in the widget itself for action execution
        self.widget.assistant = self
        
        # Initialize goal manager
        self.goal_manager = GoalManager()
        
        # Initialize variables
        self.running = True
        self.current_actions = []  # Add this line to track current actions
        
        # Initialize tool specifications
        self.toolspec = []
        self._initialize_tools()
        
        # Load user profile
        self.load_user_profile()
        
        # Create screenshots directory
        self.screenshots_dir = "application_screenshots"
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
        # Set up PyAutoGUI safety features
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1  # Reduced from 1.0 to 0.1 for faster movements
        
        # Set up keyboard listener in a separate thread
        self.keyboard_thread = threading.Thread(target=self.start_keyboard_listener)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
        # Current keys being pressed
        self.current_keys = set()

    def _initialize_grid_data(self):
        """Initialize the grid data structure with all possible coordinates"""
        grid_data = {'points': {}}
        
        # Get screen scale factor
        screen = self.app.primaryScreen()
        scale_factor = screen.devicePixelRatio()
        
        # Get logical and actual dimensions
        screen_geometry = screen.geometry()
        logical_width = screen_geometry.width()
        logical_height = screen_geometry.height()
        
        # Calculate actual pixel dimensions
        actual_width = int(logical_width * scale_factor)
        actual_height = int(logical_height * scale_factor)
        
        # Calculate cell dimensions for 40x40 grid using actual pixels
        cell_width = actual_width // 40
        cell_height = actual_height // 40
        
        # Store grid dimensions for reference
        grid_data['cell_width'] = cell_width
        grid_data['cell_height'] = cell_height
        grid_data['scale_factor'] = scale_factor
        grid_data['logical_dimensions'] = {'width': logical_width, 'height': logical_height}
        grid_data['actual_dimensions'] = {'width': actual_width, 'height': actual_height}
        
        print("\nInitializing grid with dimensions:")
        print(f"Scale Factor: {scale_factor}")
        print(f"Logical Screen: {logical_width}x{logical_height}")
        print(f"Actual Screen: {actual_width}x{actual_height}")
        print(f"Cell Size: {cell_width}x{cell_height} (actual pixels)")
        
        # Generate all possible grid coordinates
        for row in range(40):  # 0-39
            for col in range(40):  # 0-39
                # Calculate column letters (AA-ZZ)
                first_letter = chr(65 + (col // 26))
                second_letter = chr(65 + (col % 26))
                main_cell = f"{first_letter}{second_letter}{row + 1}"
                
                # Calculate exact pixel position (center of cell) in actual pixels
                x = col * cell_width + (cell_width // 2)
                y = row * cell_height + (cell_height // 2)
                
                # Store in grid data with both actual and logical coordinates
                grid_data['points'][main_cell] = {
                    'x': x,  # Actual pixels
                    'y': y,  # Actual pixels
                    'logical_x': int(x / scale_factor),  # Logical pixels
                    'logical_y': int(y / scale_factor),  # Logical pixels
                    'position': {  # Keep position for compatibility
                        'x': x,
                        'y': y
                    },
                    'row': row + 1,
                    'col': col + 1
                }
        
        # Verify grid initialization
        num_points = len(grid_data['points'])
        print(f"\nInitialized grid with {num_points} points")
        print("Sample coordinates:", list(grid_data['points'].keys())[:5])
        
        # Print sample coordinate conversion
        sample_coord = list(grid_data['points'].keys())[0]
        sample_point = grid_data['points'][sample_coord]
        print(f"\nSample coordinate {sample_coord}:")
        print(f"Actual pixels: ({sample_point['x']}, {sample_point['y']})")
        print(f"Logical pixels: ({sample_point['logical_x']}, {sample_point['logical_y']})")
        
        return grid_data

    def _initialize_tools(self):
        """Initialize available tools following function specification format"""
        self.toolspec.extend([{
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Sends an email using the system default email client",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Email recipient address"
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject line"
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body content"
                        }
                    },
                    "required": ["to", "subject", "body"]
                }
            }
        }])

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
        
        main_col = min(max(0, x // main_cell_width), 39)
        main_row = min(max(0, y // main_cell_height), 39)
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
        
        # Get screen info
        screen = self.app.primaryScreen()
        scale_factor = screen.devicePixelRatio()
        
        # Create a copy of the image for annotation
        annotated_img = img.copy()
        
        # Add title and timestamp
        cv2.putText(annotated_img, "Detected Actions:", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 0, 255), 2)
        
        # Add each action to the image with correct coordinate transformation
        for idx, action in enumerate(actions, 1):
            action_type = action[0]
            
            # Get the raw coordinates - these are already in logical coordinates
            x, y = int(action[1]), int(action[2])
            
            # Convert logical coordinates to actual screen coordinates
            screen_x = int(x * scale_factor)
            screen_y = int(y * scale_factor)
            
            # Ensure coordinates are within bounds
            screen_x = max(0, min(screen_x, width - 1))
            screen_y = max(0, min(screen_y, height - 1))
            
            grid_coord = action[4]  # Grid coordinate
            description = action[3]  # Description
            
            # Draw more precise markers
            # First draw a thin crosshair for precise center marking
            cv2.line(annotated_img, (screen_x-10, screen_y), (screen_x+10, screen_y), (0, 0, 255), 1)
            cv2.line(annotated_img, (screen_x, screen_y-10), (screen_x, screen_y+10), (0, 0, 255), 1)
            
            # Then draw the circle
            cv2.circle(annotated_img, (screen_x, screen_y), 15, (0, 0, 255), 2)
            cv2.circle(annotated_img, (screen_x, screen_y), 3, (0, 0, 255), -1)  # Center dot
            
            # Calculate label position with proper scaling
            label_x = screen_x + 35
            label_y = screen_y - 20 + (idx * 45)
            
            # Draw connecting line from target to label
            cv2.line(annotated_img, 
                    (screen_x + 25, screen_y),
                    (label_x - 5, label_y + 10),
                    (0, 0, 255), 2)
            
            # Split label into lines for better readability
            lines = []
            lines.append(f"{idx}. {action_type.upper()} at {grid_coord}")
            if action_type == 'type':
                lines.append(f"Text: \"{description}\"")
            lines.append(f"Purpose: {description}")
            
            # Calculate background size
            max_line_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for line in lines)
            line_height = cv2.getTextSize(lines[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][1]
            total_height = len(lines) * (line_height + 5)
            
            # Draw white background with black border
            padding = 10
            bg_top = label_y - line_height - padding
            bg_bottom = bg_top + total_height + padding * 2
            bg_right = label_x + max_line_width + padding
            
            # White background
            cv2.rectangle(annotated_img,
                        (label_x - padding, bg_top),
                        (bg_right, bg_bottom),
                        (255, 255, 255),
                        -1)
            
            # Black border
            cv2.rectangle(annotated_img,
                        (label_x - padding, bg_top),
                        (bg_right, bg_bottom),
                        (0, 0, 0),
                        1)
            
            # Draw each line with proper spacing
            for i, line in enumerate(lines):
                y_offset = i * (line_height + 5)
                cv2.putText(annotated_img,
                          line,
                          (label_x, label_y + y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.7,
                          (0, 0, 255),
                          2)
        
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
            
            # Take screenshot for goal manager
            screenshot = pyautogui.screenshot()
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Get next suggested action from goal manager
            next_step, action_details = self.goal_manager.suggest_action(img)
            
            # Update UI with next step
            self.widget.update_insights(f"Next Step: {next_step}\nAction Details: {json.dumps(action_details, indent=2)}")
            
            # Rest of existing analysis code...
            
        except Exception as e:
            self.widget.update_status(f"Error: {str(e)}")
            print(f"Error analyzing form: {e}")
            return None, None
            
    def execute_single_action(self, action):
        """Execute a single action automatically without confirmation"""
        if self.emergency_stop:
            print("Action execution blocked: Emergency stop active")
            return False
            
        try:
            action_type = action[0]
            x, y = action[1], action[2]
            description = action[3]
            grid_coord = action[4]
            
            print(f"\nAutomatically executing action: {action_type} at {grid_coord} ({x}, {y})")
            self.widget.update_status(f"Executing: {description}")
            
            # Move mouse with increased precision
            current_x, current_y = pyautogui.position()
            distance = ((x - current_x) ** 2 + (y - current_y) ** 2) ** 0.5
            duration = min(max(0.3, distance / 1000), 1.0)
            
            # Check for emergency stop before each major action
            if self.emergency_stop:
                return False
            
            # Move in two steps for more accuracy
            pyautogui.moveTo(x + (x - current_x) // 2, y + (y - current_y) // 2, duration=duration/2)
            if self.emergency_stop:
                return False
            time.sleep(0.1)
            
            pyautogui.moveTo(x, y, duration=duration/2)
            if self.emergency_stop:
                return False
            time.sleep(0.2)
            
            # Execute the action
            if action_type == 'click':
                pyautogui.click(x, y)
                if self.emergency_stop:
                    return False
                time.sleep(0.3)
            elif action_type == 'type':
                pyautogui.click(x, y)
                if self.emergency_stop:
                    return False
                time.sleep(0.5)
                text = action[3]
                pyautogui.typewrite(text, interval=0.1)
                if self.emergency_stop:
                    return False
                time.sleep(0.5)
                pyautogui.press('enter')
            
            # Take new screenshot and analyze next state
            if self.emergency_stop:
                return False
                
            self.widget.hide()
            self.app.processEvents()
            time.sleep(0.5)
            
            screenshot = pyautogui.screenshot()
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            self.widget.show()
            
            # Get screen dimensions for grid overlay
            screen = self.app.primaryScreen()
            scale_factor = screen.devicePixelRatio()
            screen_width_pixels = int(screen.geometry().width() * scale_factor)
            screen_height_pixels = int(screen.geometry().height() * scale_factor)
            
            # Create gridded screenshot
            gridded_img = self._create_grid_overlay(img, screen_width_pixels, screen_height_pixels)
            
            # Save screenshot with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(
                self.screenshots_dir,
                f"post_action_{timestamp}.png"
            )
            cv2.imwrite(screenshot_path, gridded_img)
            
            # Convert to PIL for AI analysis
            pil_image = Image.fromarray(cv2.cvtColor(gridded_img, cv2.COLOR_BGR2RGB))
            
            # Create analysis prompt for next action
            prompt = self._create_analysis_prompt()
            
            # Generate next action analysis
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, pil_image]
            )
            
            if response and response.text:
                # Parse next actions
                next_actions = self._parse_ai_response(response.text)
                if next_actions:
                    # Update goal state with results
                    results = {
                        'last_action': action_type,
                        'last_coordinates': f"{x},{y}",
                        'last_description': description,
                        'next_actions_found': len(next_actions)
                    }
                    
                    # Get current step from goal manager
                    current_step = self.goal_manager.get_next_step()
                    if current_step:
                        self.goal_manager.update_state(current_step, results)
                        
                        # Update UI with next step
                        self.widget.update_insights(
                            f"Action completed successfully!\n\n"
                            f"Next Step: {current_step['step']}\n"
                            f"Found {len(next_actions)} potential next actions"
                        )
            
            return True
            
        except Exception as e:
            print(f"Error executing action: {e}")
            self.widget.update_status(f"Error executing action: {e}")
            return False

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
                    
                    # Store only the first action for processing
                    self.current_actions = [current_actions[0]]
                    current_actions = self.current_actions
                    
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
                            
                            # Take a new screenshot to compare with
                            verification_screenshot = pyautogui.screenshot()
                            verification_img = cv2.cvtColor(np.array(verification_screenshot), cv2.COLOR_RGB2BGR)
                            
                            # Create verification image with grid overlay
                            verification_gridded = self._create_grid_overlay(verification_img, screen_width_pixels, screen_height_pixels)
                            
                            # Save verification image
                            verification_path = os.path.join(
                                self.screenshots_dir,
                                f"verification_{timestamp}_iter{verification_iteration}.png"
                            )
                            cv2.imwrite(verification_path, verification_gridded)
                            
                            # Create verification prompt
                            verification_prompt = self._create_verification_prompt(
                                verification_iteration,
                                max_iterations,
                                min_iterations,
                                current_actions,
                                previous_actions[-1] if len(previous_actions) > 1 else None
                            )
                            
                            # Generate verification using Gemini AI - use both annotated and verification images
                            verification_response = client.models.generate_content(
                                model="gemini-2.0-flash",
                                contents=[
                                    verification_prompt,
                                    current_annotated_pil,  # Show the annotated plan
                                    Image.fromarray(cv2.cvtColor(verification_gridded, cv2.COLOR_BGR2RGB))  # Show current state
                                ]
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
                                
                                # Update current_actions with verified coordinates
                                if current_actions:
                                    self.current_actions = [current_actions[0]]
                                    current_actions = self.current_actions
                                
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
                    
                    # Update UI with final verified action
                    if self.current_actions:
                        self.widget.signals.update_actions_signal.emit(self.current_actions)
                        self.widget.signals.show_confirmation_signal.emit(self.current_actions)
                    else:
                        self.widget.update_status("No valid actions after verification")
                        self.widget.signals.update_actions_signal.emit([])
                else:
                    self.current_actions = []
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

        You are looking at TWO images:
        1. ANNOTATED PLAN (First Image):
           - Shows the proposed click points with red circles and crosshairs
           - Includes white boxes with coordinate labels and descriptions
           - Has the grid overlay for reference

        2. CURRENT STATE (Second Image):
           - Shows the actual form's current state
           - Has the grid overlay for reference
           - Use this to verify element positions and alignment

        Please verify each coordinate ONE BY ONE with extreme precision by comparing both images:

        Coordinates to verify:
        {coordinate_list}

        For EACH coordinate, analyze:
        1. VERTICAL ALIGNMENT:
           - Compare the click point in the annotated plan with the actual element position
           - Is the click point EXACTLY aligned with the target element?
           - Should it be moved up or down? By how much?
           - Check if text is centered within input fields
           - For radio buttons/checkboxes, ensure click is on the control

        2. HORIZONTAL ALIGNMENT:
           - Compare the click point in the annotated plan with the actual element position
           - Is the click point EXACTLY on the target element?
           - Should it be moved left or right? By how much?
           - Check alignment with text fields and buttons
           - Verify click points hit interactive elements

        3. ELEMENT STATE VERIFICATION:
           - Check if the element is still in the same position
           - Verify the element hasn't changed state or moved
           - Ensure no overlapping elements have appeared
           - Confirm the element is still interactive

        4. ANNOTATION PLACEMENT:
           - Are labels clearly visible and not overlapping?
           - Do they point to the correct elements?
           - Is text readable and properly positioned?

        RESPONSE FORMAT:
        For each coordinate, provide:
        ###COORD_START###
        Coordinate: [coord]
        Current Position: [description of current position]
        Element State: [UNCHANGED/CHANGED/HIDDEN]
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
        - You MUST compare both the annotated plan and current state
        - You MUST verify both click points and annotations
        - You MUST provide specific adjustment amounts if needed
        - You MUST complete at least {min_iterations} iterations
        - Current iteration: {iteration}

        Remember:
        - Be extremely precise in your analysis
        - Don't assume previous coordinates were correct
        - Check both the click point and its annotation
        - Consider the actual form element positions
        - Compare the annotated plan with the current state
        """

    def _process_verification_response(self, verification_response: str, current_actions: List[tuple]) -> List[tuple]:
        """Process the verification response and update coordinates if needed"""
        if not verification_response or not current_actions:
            return current_actions

        # Extract coordinate analysis sections
        coord_sections = re.finditer(
            r'###COORD_START###(.*?)###COORD_END###',
            verification_response,
            re.DOTALL
        )

        updated_actions = []
        for action in current_actions:
            action_updated = False
            coord = action[-2]  # Get the coordinate from the action tuple

            # Find matching coordinate section
            for section in coord_sections:
                section_text = section.group(1)
                if f"Coordinate: {coord}" in section_text:
                    # Check element state first
                    state_match = re.search(
                        r'Element State: (UNCHANGED|CHANGED|HIDDEN)',
                        section_text
                    )
                    if state_match and state_match.group(1) in ['CHANGED', 'HIDDEN']:
                        print(f"Element state changed or hidden for {coord}, forcing readjustment")
                        # Force readjustment by setting alignment to incorrect
                        action_updated = True
                        continue

                    # Parse vertical alignment
                    vertical_match = re.search(
                        r'Vertical Alignment: (TOO HIGH|TOO LOW|CORRECT)(?:\s+by\s+(\d+)\s+pixels)?',
                        section_text
                    )
                    if vertical_match:
                        alignment, pixels = vertical_match.groups()
                        if alignment != "CORRECT" and pixels:
                            # Calculate new y-coordinate
                            y_adjustment = int(pixels)
                            if alignment == "TOO HIGH":
                                y_adjustment = y_adjustment  # Move down
                            else:  # TOO LOW
                                y_adjustment = -y_adjustment  # Move up
                            
                            # Update the y-coordinate in the action tuple
                            action = list(action)
                            action[2] = action[2] + y_adjustment
                            action = tuple(action)
                            action_updated = True
                            print(f"Adjusted y-coordinate for {coord} by {y_adjustment} pixels")

                    # Parse horizontal alignment
                    horizontal_match = re.search(
                        r'Horizontal Alignment: (TOO LEFT|TOO RIGHT|CORRECT)(?:\s+by\s+(\d+)\s+pixels)?',
                        section_text
                    )
                    if horizontal_match:
                        alignment, pixels = horizontal_match.groups()
                        if alignment != "CORRECT" and pixels:
                            # Calculate new x-coordinate
                            x_adjustment = int(pixels)
                            if alignment == "TOO LEFT":
                                x_adjustment = x_adjustment  # Move right
                            else:  # TOO RIGHT
                                x_adjustment = -x_adjustment  # Move left
                            
                            # Update the x-coordinate in the action tuple
                            action = list(action)
                            action[1] = action[1] + x_adjustment
                            action = tuple(action)
                            action_updated = True
                            print(f"Adjusted x-coordinate for {coord} by {x_adjustment} pixels")

                    # Check for new coordinate recommendation
                    new_coord_match = re.search(r'New Coordinate.*?:\s*([A-Z]{2}\d+\.\d+)', section_text)
                    if new_coord_match:
                        new_coord = new_coord_match.group(1)
                        # Update the coordinate in the action tuple
                        action = list(action)
                        action[-2] = new_coord
                        action = tuple(action)
                        action_updated = True
                        print(f"Updated coordinate from {coord} to {new_coord}")

                    break  # Found and processed the matching section

            if action_updated:
                print(f"Updated action: {action}")
            updated_actions.append(action)

        return updated_actions

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
        cell_width = width // 40
        cell_height = height // 40
        
        # Create overlay
        overlay = img.copy()
        
        # Define colors (BGR format)
        GRID_COLOR = (0, 0, 255)  # Pure red - best for AI detection
        MARKER_COLOR = (255, 0, 0)  # Blue markers for intersection points
        
        # Draw grid with intersection markers
        for i in range(41):  # 41 lines for 40 cells
            x = i * cell_width
            cv2.line(overlay, (x, 0), (x, height), GRID_COLOR, 1)
            
            for j in range(41):
                y = j * cell_height
                if i == 0:  # Draw horizontal lines once
                    cv2.line(overlay, (0, y), (width, y), GRID_COLOR, 1)
                
                # Draw intersection markers
                if i < 40 and j < 40:  # Don't add markers for the last lines
                    intersection_x = x + (cell_width // 2)
                    intersection_y = y + (cell_height // 2)
                    
                    # Draw a more visible intersection marker
                    cv2.circle(overlay, (intersection_x, intersection_y), 3, MARKER_COLOR, -1)
                    
                    # Calculate coordinate text
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
                      (i * cell_width + 5, 20),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.5,
                      GRID_COLOR,
                      2)
            
            # Row indicators (1-40)
            row_label = str(i + 1)
            # Draw row label on left
            cv2.putText(overlay, row_label,
                      (5, i * cell_height + 20),
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
        
        # Combine image with legend
        img_with_legend = np.vstack([overlay, legend])
        
        return img_with_legend

    def _find_closest_valid_coordinate(self, invalid_coord: str) -> Optional[str]:
        """Find the closest valid grid coordinate to an invalid one"""
        try:
            # Parse the invalid coordinate
            if len(invalid_coord) < 3:
                print(f"Invalid coordinate format: {invalid_coord}")
                return None
                
            # Extract letters and numbers
            letters = ''.join(c for c in invalid_coord if c.isalpha()).upper()
            numbers = ''.join(c for c in invalid_coord if c.isdigit())
            
            if not letters or not numbers:
                print(f"Could not extract letters and numbers from: {invalid_coord}")
                return None
                
            # Ensure we have two letters
            if len(letters) == 1:
                letters = 'A' + letters
            elif len(letters) > 2:
                letters = letters[:2]
                
            # Convert to column index
            col_index = (ord(letters[0]) - 65) * 26 + (ord(letters[1]) - 65)
            col_index = min(max(0, col_index), 39)  # Ensure 0-39
            
            # Convert row number
            try:
                row_num = int(numbers)
                row_num = min(max(1, row_num), 40)  # Ensure 1-40
            except ValueError:
                print(f"Could not parse row number: {numbers}")
                return None
            
            # Construct valid coordinate
            first_letter = chr(65 + (col_index // 26))
            second_letter = chr(65 + (col_index % 26))
            valid_coord = f"{first_letter}{second_letter}{row_num}"
            
            # Verify it exists in grid data
            if valid_coord in self.grid_data['points']:
                print(f"Found valid coordinate: {valid_coord}")
                return valid_coord
            else:
                print(f"Constructed coordinate not found in grid: {valid_coord}")
                return None
                
        except Exception as e:
            print(f"Error finding closest coordinate: {e}")
            return None

    def handle_user_request(self, request: str):
        """Handle user request and update application state"""
        try:
            print(f"\nProcessing request: {request}")
            self.widget.update_status("Processing request...")
            
            # Update current application state
            self.current_application["status"] = "processing"
            self.current_application["last_action"] = request
            
            # Create action details for analysis
            action_details = {
                "action": "analyze",
                "request": request,
                "location": "screen",
                "expected_outcome": "identify UI elements and map to grid"
            }
            
            # Set the goal in the goal manager
            self.goal_manager.set_goal(request)
            
            # Analyze current state with action details
            self.analyze_current_state(action_details)
            
            # Update application status
            self.current_application["status"] = "ready"
            self.widget.update_status("Ready for next action")
            
        except Exception as e:
            print(f"Error handling request: {e}")
            self.current_application["status"] = "error"
            self.current_application["errors"].append(str(e))
            self.widget.update_status(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def start_keyboard_listener(self):
        """Start keyboard listener in a separate thread"""
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release) as listener:
            listener.join()

    def on_press(self, key):
        """Handle key press events with emergency stop"""
        try:
            # Add the pressed key to current keys
            if hasattr(key, 'char'):
                self.current_keys.add(key.char.lower())
            else:
                self.current_keys.add(key)
            
            # Check for Emergency Stop (Ctrl+X)
            if (keyboard.Key.ctrl_l in self.current_keys and 
                'x' in self.current_keys):
                self.trigger_emergency_stop("Ctrl+X pressed")
                return
            
            # Rest of existing key handling
            if (keyboard.Key.ctrl_l in self.current_keys and 
                keyboard.Key.shift in self.current_keys and 
                'a' in self.current_keys):
                self.on_analyze_form()
            elif (keyboard.Key.ctrl_l in self.current_keys and 
                  keyboard.Key.shift in self.current_keys and 
                  'j' in self.current_keys):
                self.on_analyze_job()
            elif (keyboard.Key.ctrl_l in self.current_keys and 
                  'q' in self.current_keys):
                self.running = False
                self.app.quit()
            elif key == keyboard.Key.esc:
                self.running = False
                self.app.quit()
        except AttributeError:
            pass

    def on_release(self, key):
        """Handle key release events"""
        try:
            if hasattr(key, 'char'):  # Regular keys
                self.current_keys.discard(key.char.lower())
            else:  # Special keys
                self.current_keys.discard(key)
        except AttributeError:
            pass  # Ignore attribute errors from special keys

    def analyze_current_state(self, action_details: Dict[str, Any]):
        """Analyze and execute actions automatically"""
        try:
            # Create timestamped directory for this analysis session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(self.screenshots_dir, f"analysis_session_{timestamp}")
            os.makedirs(session_dir, exist_ok=True)
            
            # Reset emergency stop flag at start of new analysis
            self.emergency_stop = False
            
            # Create initial plan
            initial_plan = self._create_and_analyze_plan(session_dir, action_details)
            if not initial_plan:
                self.widget.update_status("Could not create initial plan")
                return
            
            # Execute actions automatically
            for action in initial_plan:
                if self.emergency_stop:
                    print("Automation stopped by emergency trigger")
                    break
                    
                # Execute action
                success = self.execute_single_action(action)
                if not success:
                    print(f"Action failed: {action}")
                    break
                    
                # Wait between actions
                time.sleep(1.0)
                
                # Take new screenshot and analyze next action if needed
                self._verify_and_update_plan()
            
            if not self.emergency_stop:
                self.widget.update_status("Automated execution completed")
            
        except Exception as e:
            print(f"Error in automated execution: {e}")
            self.widget.update_status(f"Automation failed: {str(e)}")

    def _create_and_analyze_plan(self, session_dir: str, action_details: Dict[str, Any], 
                                failed_action: Optional[tuple] = None) -> List[tuple]:
        """Create and analyze a new plan based on current screen state"""
        try:
            # Hide the assistant widget and wait
            self.widget.hide()
            self.app.processEvents()
            time.sleep(0.5)
            
            # Take screenshot without the assistant UI
            screenshot = pyautogui.screenshot()
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Show the widget again
            self.widget.show()
            
            # Get screen dimensions for grid overlay
            screen = self.app.primaryScreen()
            scale_factor = screen.devicePixelRatio()
            screen_width_pixels = int(screen.geometry().width() * scale_factor)
            screen_height_pixels = int(screen.geometry().height() * scale_factor)
            
            # Create and save gridded screenshot
            gridded_img = self._create_grid_overlay(img, screen_width_pixels, screen_height_pixels)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gridded_path = os.path.join(session_dir, f"gridded_screenshot_{timestamp}.png")
            cv2.imwrite(gridded_path, gridded_img)
            
            # Convert to PIL for AI analysis
            pil_image = Image.fromarray(cv2.cvtColor(gridded_img, cv2.COLOR_BGR2RGB))
            
            # Create analysis prompt
            prompt = self._create_planning_prompt(action_details, failed_action)
            
            # Generate analysis
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, pil_image]
            )
            
            if response and response.text:
                # Parse actions from response
                actions = self._parse_ai_response(response.text)
                
                if actions:
                    # Create and save annotated screenshot
                    annotated_path = self.annotate_screenshot(gridded_path, actions, timestamp)
                    self.widget.update_annotated_image(annotated_path)
                    
                    return actions
            
            return []
            
        except Exception as e:
            print(f"Error creating plan: {e}")
            return []

    def _verify_and_update_plan(self) -> Dict[str, Any]:
        """Verify the last executed action and determine if plan needs updating"""
        try:
            # Take verification screenshot
            self.widget.hide()
            self.app.processEvents()
            time.sleep(0.5)
            
            screenshot = pyautogui.screenshot()
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            self.widget.show()
            
            # Create gridded verification screenshot
            screen = self.app.primaryScreen()
            scale_factor = screen.devicePixelRatio()
            screen_width_pixels = int(screen.geometry().width() * scale_factor)
            screen_height_pixels = int(screen.geometry().height() * scale_factor)
            
            gridded_img = self._create_grid_overlay(img, screen_width_pixels, screen_height_pixels)
            
            # Save verification screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            verification_path = os.path.join(
                self.current_plan['session_dir'],
                f"verification_{timestamp}.png"
            )
            cv2.imwrite(verification_path, gridded_img)
            
            # Convert to PIL for AI analysis
            pil_image = Image.fromarray(cv2.cvtColor(gridded_img, cv2.COLOR_BGR2RGB))
            
            # Create verification prompt
            current_action = self.current_plan['actions'][self.current_plan['current_index']]
            prompt = self._create_verification_prompt(current_action)
            
            # Generate verification analysis
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, pil_image]
            )
            
            if response and response.text:
                # Parse verification result
                result = self._parse_verification_response(response.text, current_action)
                
                # Save verification result
                result_path = os.path.join(
                    self.current_plan['session_dir'],
                    f"verification_result_{timestamp}.json"
                )
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                return result
            
            return {'status': 'failed', 'reason': 'No verification response'}
            
        except Exception as e:
            print(f"Error in verification: {e}")
            return {'status': 'failed', 'reason': str(e)}

    def _create_planning_prompt(self, action_details: Dict[str, Any], 
                              failed_action: Optional[tuple] = None) -> str:
        """Create prompt for planning analysis"""
        base_prompt = self._create_analysis_prompt()  # Use existing analysis prompt as base
        
        # Add planning-specific instructions
        planning_context = f"""
        PLANNING REQUIREMENTS:
        1. Create a sequential plan of actions
        2. Each action must be achievable and verifiable
        3. Consider dependencies between actions
        4. Account for possible state changes
        
        Current Goal: {self.goal_manager.current_goal}
        Action Required: {action_details.get('action')}
        Expected Outcome: {action_details.get('expected_outcome')}
        """
        
        if failed_action:
            planning_context += f"""
            Previous Action Failed:
            - Type: {failed_action[0]}
            - Location: {failed_action[4]}
            - Purpose: {failed_action[3]}
            
            Please analyze why the action might have failed and provide an alternative approach.
            """
        
        return base_prompt + "\n" + planning_context

    def _create_verification_prompt(self, action: tuple) -> str:
        """Create prompt for verification analysis"""
        return f"""
        VERIFY ACTION RESULT
        
        Action Executed:
        - Type: {action[0]}
        - Location: {action[4]}
        - Purpose: {action[3]}
        
        Please analyze the current screen state and verify:
        1. Was the action successful?
        2. Did it achieve its intended purpose?
        3. Are there any unexpected changes?
        4. Does the plan need to be adjusted?
        
        Respond in this format:
        ###VERIFICATION_START###
        {{
            "status": "success|failed|needs_adjustment",
            "details": "Detailed explanation",
            "next_action": {{
                "type": "continue|retry|adjust|replan",
                "reason": "Explanation"
            }},
            "adjusted_action": {{
                // Only if status is needs_adjustment
                "action": "click|type",
                "grid_coord": "AA1",
                "description": "New description",
                "text": "Text if type action"
            }}
        }}
        ###VERIFICATION_END###
        """

    def _parse_verification_response(self, response_text: str, current_action: tuple) -> Dict[str, Any]:
        """Parse the verification response"""
        try:
            # Extract JSON from response
            match = re.search(r'###VERIFICATION_START###\s*(.*?)###VERIFICATION_END###', 
                            response_text, re.DOTALL)
            if not match:
                return {'status': 'failed', 'reason': 'Could not parse verification response'}
            
            verification = json.loads(match.group(1))
            
            # If adjustment needed, parse new action
            if verification['status'] == 'needs_adjustment' and 'adjusted_action' in verification:
                adj = verification['adjusted_action']
                # Convert to action tuple format
                if adj['action'] == 'type':
                    adjusted_action = ('type', current_action[1], current_action[2], 
                                     adj['text'], adj['grid_coord'], adj['description'])
                else:
                    adjusted_action = (adj['action'], current_action[1], current_action[2],
                                     adj['description'], adj['grid_coord'])
                verification['adjusted_action'] = adjusted_action
            
            return verification
            
        except Exception as e:
            print(f"Error parsing verification: {e}")
            return {'status': 'failed', 'reason': str(e)}

    def _display_current_plan(self, status: str, actions: List[tuple]):
        """Display current plan status and remaining actions"""
        try:
            # Update status
            self.widget.update_status(status)
            
            # Create plan display text
            plan_text = f"Current Plan Status: {status}\n\n"
            plan_text += "Remaining Actions:\n"
            
            for idx, action in enumerate(actions, 1):
                action_type = action[0]
                grid_coord = action[4]
                description = action[3]
                
                plan_text += f"{idx}. {action_type.upper()} at {grid_coord}\n"
                plan_text += f"   Purpose: {description}\n\n"
            
            # Update insights with plan
            self.widget.update_insights(plan_text)
            
        except Exception as e:
            print(f"Error displaying plan: {e}")
            self.widget.update_status("Error displaying plan")

    def _parse_ai_response(self, response_text: str) -> List[tuple]:
        """Parse the AI response and extract actions with proper scaling"""
        try:
            # Get screen scale factor
            screen = self.app.primaryScreen()
            scale_factor = screen.devicePixelRatio()
            print(f"\nScreen scale factor: {scale_factor}")
            
            # Get screen dimensions
            screen_geometry = screen.geometry()
            logical_width = screen_geometry.width()
            logical_height = screen_geometry.height()
            
            # Calculate actual pixel dimensions
            actual_width = int(logical_width * scale_factor)
            actual_height = int(logical_height * scale_factor)
            
            print(f"Screen dimensions:")
            print(f"Logical: {logical_width}x{logical_height}")
            print(f"Actual: {actual_width}x{actual_height}")
            
            # Clean up the response text
            response_text = response_text.strip()
            print("\nRaw AI Response:")
            print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
            
            # Try different patterns to extract JSON
            json_patterns = [
                r'###ACTIONS_START###\s*(.*?)###ACTIONS_END###',
                r'###JSON_START###\s*(.*?)%%%JSON_END%%%',
                r'\{\s*"grid_actions"\s*:\s*\[.*?\]\s*\}',
                r'\[\s*\{.*?\}\s*\]'
            ]
            
            analysis_json = None
            for pattern in json_patterns:
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    try:
                        potential_json = match.group(1) if pattern.startswith(r'###') else match.group(0)
                        json.loads(potential_json)
                        analysis_json = potential_json
                        print(f"\nSuccessfully extracted JSON using pattern: {pattern[:30]}...")
                        break
                    except json.JSONDecodeError:
                        continue
            
            if not analysis_json:
                print("Could not find valid JSON in response")
                return []
            
            try:
                analysis = json.loads(analysis_json)
                if isinstance(analysis, list):
                    analysis = {"grid_actions": analysis}
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                return []
            
            actions = []
            grid_actions = analysis.get('grid_actions', [])
            
            print("\nProcessing grid actions:", json.dumps(grid_actions, indent=2))
            
            if isinstance(grid_actions, list):
                for action in grid_actions:
                    if isinstance(action, dict):
                        # Extract action type from either recommended_action or element_type
                        action_type = None
                        text = None
                        
                        # Parse recommended_action if present
                        recommended_action = action.get('recommended_action', '')
                        if recommended_action:
                            # Extract action type and text from recommended_action
                            action_parts = recommended_action.split()
                            if action_parts:
                                action_type = action_parts[0].lower()
                                if action_type == 'type':
                                    # Try multiple patterns to extract text
                                    text_patterns = [
                                        r'type\s+"([^"]+)"',  # Text in quotes
                                        r'type\s+([^\s]+)\s+(?:in|at|into)',  # Text before in/at/into
                                        r'type\s+(.+?)(?:\s+in|\s+at|\s+into|$)'  # Any text until in/at/into or end
                                    ]
                                    for pattern in text_patterns:
                                        text_match = re.search(pattern, recommended_action, re.IGNORECASE)
                                        if text_match:
                                            text = text_match.group(1)
                                            break
                                    
                                    # If no text found, take everything after TYPE
                                    if not text and len(action_parts) > 1:
                                        text = ' '.join(action_parts[1:]).strip()
                                        # Remove trailing "in", "at", "into" if present
                                        text = re.sub(r'\s+(?:in|at|into)\s+.*$', '', text, flags=re.IGNORECASE)
                        
                        # If no action type found in recommended_action, use element_type
                        if not action_type:
                            element_type = action.get('element_type', '').lower()
                            if element_type == 'text_field':
                                action_type = 'type'
                            elif element_type in ['button', 'link', 'checkbox', 'radio']:
                                action_type = 'click'
                        
                        # Get text from action if not already set
                        if action_type == 'type' and not text:
                            text = action.get('text', '')  # Try to get text from action directly
                        
                        grid_coord = action.get('grid_coord', '')
                        description = action.get('description', '')
                        
                        print(f"\nProcessing action: {action_type} at {grid_coord}")
                        print(f"Text (if type action): {text}")
                        
                        if not action_type or not grid_coord:
                            print(f"Warning: Skipping action due to missing required fields: {action}")
                            continue
                        
                        # Handle grid coordinates with proper validation
                        if grid_coord in self.grid_data['points']:
                            point_data = self.grid_data['points'][grid_coord]
                        else:
                            print(f"Warning: Grid coordinate {grid_coord} not found in grid data")
                            # Parse the invalid coordinate
                            match = re.match(r'([A-Z]{2})(\d+)(?:\.(\d{2}))?', grid_coord)
                            if match:
                                letters, number, subgrid = match.groups()
                                # Convert letters to column index (AA=0, AB=1, etc.)
                                col = (ord(letters[0]) - 65) * 26 + (ord(letters[1]) - 65)
                                # Ensure row is within bounds
                                row = min(max(1, int(number)), 40)
                                # Create valid coordinate
                                valid_coord = f"{letters}{row}"
                                if valid_coord in self.grid_data['points']:
                                    point_data = self.grid_data['points'][valid_coord]
                                    print(f"Using closest valid coordinate: {valid_coord}")
                                else:
                                    print(f"Warning: Could not find valid coordinate for {grid_coord}")
                                    continue
                            else:
                                print(f"Warning: Invalid coordinate format: {grid_coord}")
                                continue
                        
                        # Get raw coordinates
                        raw_x = point_data['x'] if 'x' in point_data else point_data.get('position', {}).get('x')
                        raw_y = point_data['y'] if 'y' in point_data else point_data.get('position', {}).get('y')
                        
                        if raw_x is not None and raw_y is not None:
                            # Convert raw coordinates to logical coordinates for MacBook Pro Retina
                            x = int(raw_x / scale_factor)
                            y = int(raw_y / scale_factor)
                            
                            print(f"Coordinate conversion:")
                            print(f"Raw: ({raw_x}, {raw_y})")
                            print(f"Scaled for Retina: ({x}, {y})")
                            
                            if action_type == 'type':
                                if text:  # Only add type action if we have text
                                    actions.append(('type', x, y, text, grid_coord, description))
                                else:
                                    print(f"Warning: Skipping type action due to missing text: {action}")
                            else:  # click or other actions
                                actions.append((action_type, x, y, description, grid_coord))
                        else:
                            print(f"Warning: Could not extract x,y coordinates from point_data: {point_data}")
            
            print(f"\nCreated {len(actions)} actions")
            return actions
            
        except Exception as e:
            print(f"Error parsing AI response: {e}")
            import traceback
            traceback.print_exc()
            return []

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

    def _save_analysis_screenshots(self, img, actions, timestamp, session_dir):
        """Save original and annotated screenshots with proper organization"""
        try:
            # Create annotated version if there are actions
            if actions:
                # Use the gridded image that was already saved
                gridded_path = os.path.join(session_dir, "gridded_screenshot.png")
                if not os.path.exists(gridded_path):
                    print("Warning: Gridded screenshot not found, using original image")
                    annotated_img = img.copy()
                else:
                    annotated_img = cv2.imread(gridded_path)
                
                # Add each action to the annotated image
                for idx, action in enumerate(actions, 1):
                    action_type = action[0]
                    # For Retina displays, we need to use the raw coordinates
                    x, y = int(action[1]), int(action[2])
                    grid_coord = action[4]
                    description = action[3] if action_type == 'type' else action[3]
                    
                    # Calculate annotation offset to avoid overlapping
                    y_offset = idx * 30  # Space out annotations vertically
                    
                    # Draw action marker with precise targeting
                    cv2.circle(annotated_img, (x, y), 15, (0, 0, 255), 2)  # Red circle
                    cv2.line(annotated_img, (x-10, y), (x+10, y), (0, 0, 255), 1)  # Crosshair
                    cv2.line(annotated_img, (x, y-10), (x, y+10), (0, 0, 255), 1)
                    
                    # Add label with white background and improved positioning
                    label = f"{idx}. {action_type.upper()} at {grid_coord}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Position label to the right of the point with offset
                    label_x = x + 25
                    label_y = y + y_offset
                    
                    # Draw white background for text with padding
                    padding = 5
                    cv2.rectangle(annotated_img,
                                (label_x - padding, label_y - text_size[1] - padding),
                                (label_x + text_size[0] + padding, label_y + padding),
                                (255, 255, 255),
                                -1)
                    
                    # Draw text with improved visibility
                    cv2.putText(annotated_img, label,
                              (label_x, label_y),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 255), 2)
                    
                    # Add description below label
                    desc_y = label_y + 20
                    desc_text = f"Purpose: {description}"
                    desc_size = cv2.getTextSize(desc_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    
                    # Draw white background for description
                    cv2.rectangle(annotated_img,
                                (label_x - padding, desc_y - text_size[1] - padding),
                                (label_x + desc_size[0] + padding, desc_y + padding),
                                (255, 255, 255),
                                -1)
                    
                    # Draw description text
                    cv2.putText(annotated_img, desc_text,
                              (label_x, desc_y),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.6, (0, 0, 255), 1)
                
                # Save annotated screenshot
                annotated_path = os.path.join(session_dir, "annotated_screenshot.png")
                cv2.imwrite(annotated_path, annotated_img)
                print(f"Saved annotated screenshot to: {annotated_path}")
                
            return True
        except Exception as e:
            print(f"Error saving screenshots: {e}")
            return False

    def on_mouse_move(self, x, y):
        """Handle mouse movement for emergency stop"""
        try:
            # Get screen dimensions
            screen = self.app.primaryScreen()
            width = screen.geometry().width()
            height = screen.geometry().height()
            
            # Check if mouse is in top-right corner
            if x >= width - self.corner_stop_threshold and y <= self.corner_stop_threshold:
                print("\nEmergency Stop Triggered: Mouse moved to top-right corner")
                self.trigger_emergency_stop("Mouse moved to top-right corner")
        except Exception as e:
            print(f"Error in mouse move handler: {e}")

    def trigger_emergency_stop(self, reason: str):
        """Emergency stop all automation"""
        self.emergency_stop = True
        self.widget.update_status(f"EMERGENCY STOP: {reason}")
        print(f"\nEMERGENCY STOP TRIGGERED: {reason}")
        print("Stopping all automated actions...")
        
        # Clear any pending actions
        self.current_actions = []
        if hasattr(self, 'current_plan'):
            self.current_plan = {'actions': [], 'current_index': 0}
        
        # Update UI to show stopped state
        self.widget.update_insights("Automation stopped by emergency trigger.\nReason: " + reason)
        self.widget.signals.update_actions_signal.emit([])

if __name__ == "__main__":
    assistant = JobApplicationAssistant()
    assistant.run() 