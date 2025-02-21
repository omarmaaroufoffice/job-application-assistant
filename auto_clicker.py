import os
import time
from google import genai
import pyautogui
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini AI with new client
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

client = genai.Client(api_key=GOOGLE_API_KEY)

class AIAutoClicker:
    def __init__(self):
        # Set up PyAutoGUI safety features
        pyautogui.FAILSAFE = True  # Move mouse to corner to stop
        pyautogui.PAUSE = 0.5  # Add delay between actions
        
        # Initialize screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
    def capture_screenshot(self):
        """Capture a screenshot and save it temporarily"""
        screenshot = pyautogui.screenshot()
        temp_path = "temp_screenshot.png"
        screenshot.save(temp_path)
        return temp_path

    def analyze_image_with_ai(self, image_path):
        """Analyze the screenshot using Gemini AI"""
        try:
            # Open image using PIL
            image = Image.open(image_path)
            
            # Create the prompt for analysis
            prompt = """
            Analyze this screenshot and:
            1. Identify clickable elements (buttons, links, text fields)
            2. Provide exact coordinates for each element as (x, y)
            3. For text fields, specify what text should be typed
            Format response as:
            - CLICK x,y: <description>
            - TYPE "text": <description>
            """
            
            # Generate content using Gemini 2.0 Flash
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image]
            )
            
            if response.candidates and response.candidates[0].content:
                return response.candidates[0].content.text
            return None
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None

    def parse_ai_response(self, ai_response):
        """Parse AI response into actionable commands"""
        actions = []
        if ai_response:
            # Split response into lines and process each instruction
            lines = ai_response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('- CLICK'):
                    # Extract coordinates if present (x, y)
                    import re
                    coords = re.findall(r'(\d+),\s*(\d+)', line)
                    if coords:
                        x, y = map(int, coords[0])
                        actions.append(('click', x, y))
                elif line.startswith('- TYPE'):
                    # Extract text between quotes
                    text_match = re.search(r'"([^"]*)"', line)
                    if text_match:
                        text = text_match.group(1)
                        actions.append(('type', text))
        return actions

    def execute_actions(self, actions):
        """Execute the parsed actions"""
        for action in actions:
            try:
                if action[0] == 'click':
                    x, y = action[1], action[2]
                    # Ensure coordinates are within screen bounds
                    x = min(max(0, x), self.screen_width)
                    y = min(max(0, y), self.screen_height)
                    # Add smooth movement
                    pyautogui.moveTo(x, y, duration=0.5, tween=pyautogui.easeInOutQuad)
                    pyautogui.click()
                elif action[0] == 'type':
                    text = action[1]
                    pyautogui.typewrite(text, interval=0.1)
            except Exception as e:
                print(f"Error executing action {action}: {e}")

    def run(self, interval=5):
        """Main loop for the auto clicker"""
        print("AI Auto Clicker started. Move mouse to any corner to stop.")
        try:
            while True:
                # Capture and analyze screenshot
                screenshot_path = self.capture_screenshot()
                print("Analyzing screenshot...")
                ai_analysis = self.analyze_image_with_ai(screenshot_path)
                
                if ai_analysis:
                    print("AI Analysis:", ai_analysis)
                    # Parse and execute actions
                    actions = self.parse_ai_response(ai_analysis)
                    print("Executing actions:", actions)
                    self.execute_actions(actions)
                else:
                    print("No actionable elements found in screenshot")
                
                # Clean up screenshot
                if os.path.exists(screenshot_path):
                    os.remove(screenshot_path)
                
                # Wait before next iteration
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nAI Auto Clicker stopped by user.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Clean up any remaining temporary files
            if os.path.exists("temp_screenshot.png"):
                os.remove("temp_screenshot.png")

if __name__ == "__main__":
    auto_clicker = AIAutoClicker()
    auto_clicker.run() 