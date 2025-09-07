"""
AI Configuration for Gradio Data Analysis Platform
Manages API keys and AI service settings
"""

import os
from typing import Optional

class AIConfig:
    """Configuration for AI services"""
    
    def __init__(self):
        self.gemini_api_key = self._get_gemini_api_key()
    
    def _get_gemini_api_key(self) -> Optional[str]:
        """Get Gemini AI API key from environment or config"""
        # Try environment variable first
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            return api_key
        
        # Try reading from config file
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if line.startswith('GEMINI_API_KEY='):
                        return line.split('=', 1)[1].strip()
        except FileNotFoundError:
            pass
        
        return None
    
    def is_configured(self) -> bool:
        """Check if AI services are properly configured"""
        return self.gemini_api_key is not None
    
    def get_setup_instructions(self) -> str:
        """Get instructions for setting up AI services"""
        return """
To enable AI-powered data cleaning, you need to set up Gemini AI:

1. Get a Gemini AI API key from Google AI Studio:
   https://aistudio.google.com/app/apikey

2. Set the API key in one of these ways:
   
   Option A - Environment Variable:
   export GEMINI_API_KEY="your-api-key-here"
   
   Option B - Create .env file:
   echo "GEMINI_API_KEY=your-api-key-here" > .env

3. Restart the application

Note: The application will automatically try different Gemini models:
- gemini-2.5-pro (most powerful reasoning model)
- gemini-2.5-flash (hybrid reasoning, 1M token context)
- gemini-2.5-flash-lite (most cost effective)

Without API key, you can still use manual data cleaning operations.
"""

# Global configuration instance
ai_config = AIConfig()