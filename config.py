"""
Configuration settings for the Gradio Data Analysis Platform
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class AppConfig:
    """Application configuration settings"""
    
    # Server settings
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 7860
    DEBUG_MODE: bool = True
    SHARE_GRADIO: bool = False
    
    # Data processing limits
    MAX_FILE_SIZE_MB: int = 100
    MAX_DATAFRAME_ROWS: int = 100000
    MAX_DATAFRAME_COLS: int = 1000
    
    # Supported file formats
    SUPPORTED_FORMATS: List[str] = ['.csv', '.xlsx', '.xls', '.json']
    
    # Natural language processing settings
    NLP_CONFIDENCE_THRESHOLD: float = 0.7
    MAX_COMMAND_LENGTH: int = 500
    
    # Visualization settings
    DEFAULT_PLOT_WIDTH: int = 800
    DEFAULT_PLOT_HEIGHT: int = 600
    MAX_PLOT_POINTS: int = 10000
    
    # Model training settings
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_CV_FOLDS: int = 5
    MAX_TRAINING_TIME_SECONDS: int = 300
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables"""
        return cls(
            SERVER_HOST=os.getenv('GRADIO_HOST', cls.SERVER_HOST),
            SERVER_PORT=int(os.getenv('GRADIO_PORT', cls.SERVER_PORT)),
            DEBUG_MODE=os.getenv('DEBUG', 'true').lower() == 'true',
            MAX_FILE_SIZE_MB=int(os.getenv('MAX_FILE_SIZE_MB', cls.MAX_FILE_SIZE_MB))
        )

# Global configuration instance
config = AppConfig.from_env()