# Job Application Assistant

An intelligent assistant that helps automate and streamline the job application process using AI-powered form analysis and automated input.

## Features

- **Intelligent Form Analysis**: Uses Google's Gemini AI to analyze job application forms
- **Automated Input**: Automatically fills in form fields based on your profile
- **Grid-Based Navigation**: Precise coordinate system for accurate form interaction
- **Real-time Insights**: Provides instant feedback and suggestions
- **Hotkey Support**: Quick access to key features
  - `Ctrl+Shift+A`: Analyze current form
  - `Ctrl+Shift+J`: Analyze job description
  - `Ctrl+Q`: Quit application

## Requirements

- Python 3.8+
- PyQt5
- Google Gemini AI API
- OpenCV
- Other dependencies listed in requirements.txt

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
6. Create a `user_profile.json` with your profile information

## Usage

1. Run the application: `python job_application_assistant.py`
2. Use hotkeys to analyze forms and job descriptions
3. Follow the on-screen instructions for form filling

## Grid System

The application uses a sophisticated grid system for precise form interaction:
- Main grid: 40x40 cells (AA1-ZZ40)
- Each cell contains reference points for accurate positioning
- Visual overlay helps verify correct element targeting

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

MIT License - see LICENSE file for details 