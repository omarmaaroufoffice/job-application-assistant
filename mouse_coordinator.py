import pyautogui
import keyboard
import json
import os
from time import sleep
from pynput import mouse

class MouseCoordinator:
    def __init__(self):
        # Initialize PyAutoGUI safety features
        pyautogui.FAILSAFE = True  # Move mouse to corner to stop
        pyautogui.PAUSE = 0.5  # Add delay between actions
        
        # Load or create saved positions
        self.positions = self.load_positions()
        
        # Track current mouse position
        self.current_pos = (0, 0)
        
        # Flag for position display
        self.showing_position = False
        
    def load_positions(self):
        """Load saved positions from file"""
        if os.path.exists('saved_positions.json'):
            with open('saved_positions.json', 'r') as f:
                return json.load(f)
        return {}
    
    def save_positions(self):
        """Save positions to file"""
        with open('saved_positions.json', 'w') as f:
            json.dump(self.positions, f, indent=2)
    
    def on_move(self, x, y):
        """Track mouse movement"""
        self.current_pos = (x, y)
        if self.showing_position:
            print(f"\rCurrent position: ({x}, {y})    ", end='')
    
    def save_current_position(self):
        """Save current mouse position"""
        x, y = self.current_pos
        name = input("\nEnter name for this position: ")
        if name:
            self.positions[name] = {'x': x, 'y': y}
            self.save_positions()
            print(f"Saved position '{name}' at ({x}, {y})")
    
    def list_positions(self):
        """List all saved positions"""
        print("\nSaved Positions:")
        for name, pos in self.positions.items():
            print(f"- {name}: ({pos['x']}, {pos['y']})")
    
    def move_to_position(self):
        """Move to a saved position"""
        self.list_positions()
        name = input("\nEnter position name to move to: ")
        if name in self.positions:
            pos = self.positions[name]
            print(f"Moving to {name}...")
            pyautogui.moveTo(pos['x'], pos['y'], duration=0.5, tween=pyautogui.easeInOutQuad)
        else:
            print("Position not found!")
    
    def delete_position(self):
        """Delete a saved position"""
        self.list_positions()
        name = input("\nEnter position name to delete: ")
        if name in self.positions:
            del self.positions[name]
            self.save_positions()
            print(f"Deleted position '{name}'")
        else:
            print("Position not found!")
    
    def toggle_position_display(self):
        """Toggle showing current mouse position"""
        self.showing_position = not self.showing_position
        if not self.showing_position:
            print()  # New line to clear position display
    
    def run(self):
        """Main loop"""
        print("\n=== Mouse Coordinator ===")
        print("Commands:")
        print("- Press 'Ctrl+Shift+P' to toggle position display")
        print("- Press 'Ctrl+Shift+S' to save current position")
        print("- Press 'Ctrl+Shift+M' to move to a saved position")
        print("- Press 'Ctrl+Shift+L' to list saved positions")
        print("- Press 'Ctrl+Shift+D' to delete a position")
        print("- Press 'Ctrl+Q' to quit")
        print("\nMove mouse to corner of screen to stop auto-movement")
        
        # Set up keyboard shortcuts
        keyboard.add_hotkey('ctrl+shift+p', self.toggle_position_display)
        keyboard.add_hotkey('ctrl+shift+s', self.save_current_position)
        keyboard.add_hotkey('ctrl+shift+m', self.move_to_position)
        keyboard.add_hotkey('ctrl+shift+l', self.list_positions)
        keyboard.add_hotkey('ctrl+shift+d', self.delete_position)
        
        # Start mouse listener
        with mouse.Listener(on_move=self.on_move) as listener:
            keyboard.wait('ctrl+q')
        
        print("\nMouse Coordinator stopped.")

if __name__ == "__main__":
    coordinator = MouseCoordinator()
    coordinator.run() 