# AI Auto Clicker

An intelligent auto clicker that uses Google's Gemini AI to analyze screenshots and perform automated clicking and typing actions based on the screen content.

## Features

- Takes screenshots automatically at regular intervals
- Analyzes screenshots using Google's Gemini AI Vision model
- Automatically moves mouse and clicks based on AI analysis
- Can type text based on AI recommendations
- Built-in safety features (move mouse to corner to stop)
- Configurable delay between actions

## Prerequisites

- Python 3.7 or higher
- Google Cloud API key with Gemini AI access

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

1. Make sure your Google API key is set in the `.env` file
2. Run the script:
   ```bash
   python auto_clicker.py
   ```
3. The program will start taking screenshots and analyzing them
4. To stop the program, move your mouse to any corner of the screen

## Safety Features

- Move mouse to any corner of the screen to stop the program (failsafe)
- 0.5-second delay between actions to prevent too rapid clicking
- Coordinates are validated to ensure they're within screen bounds
- Temporary files are automatically cleaned up

## How It Works

1. The script captures a screenshot of your screen
2. The screenshot is sent to Gemini AI for analysis
3. The AI returns a description of clickable elements and text fields
4. The script parses the AI's response into actionable commands
5. PyAutoGUI executes the mouse movements, clicks, and typing
6. The process repeats after the configured interval

## Customization

You can modify the following parameters in the script:
- `interval`: Time between screenshots (default: 5 seconds)
- `pyautogui.PAUSE`: Delay between actions (default: 0.5 seconds)

## Error Handling

The script includes comprehensive error handling:
- Invalid API key detection
- Screenshot capture errors
- AI analysis errors
- Action execution errors
- Automatic cleanup of temporary files

## Contributing

Feel free to submit issues and enhancement requests! 