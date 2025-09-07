# Gradio Data Analysis Platform

A comprehensive web-based application for data science workflows using Gradio. This platform allows users to upload datasets, clean data, perform exploratory data analysis, engineer features, and build predictive models through an intuitive multi-tab interface with natural language command support.

## Features

- **Multi-format Data Upload**: Support for CSV, Excel, and JSON files
- **AI-Powered Data Cleaning**: 
  - BeautifulSoup4-based cleaning with AI code generation
  - Direct AI cleaning using Gemini AI
  - Intelligent cleaning suggestions
  - Code validation and safe execution
- **Persistent State Management**: DataFrame state maintained across all tabs
- **Manual Data Cleaning**: Fallback operations for missing values, duplicates, outliers
- **Exploratory Data Analysis**: Interactive visualizations and statistical summaries
- **Feature Engineering**: Scaling, encoding, and feature creation tools
- **Machine Learning**: Model training, evaluation, and comparison
- **Interactive Interface**: Clean, organized multi-tab Gradio interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gradio-data-analysis-platform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AI services (optional but recommended):
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Gemini AI API key
# Get your key from: https://makersuite.google.com/app/apikey
```

4. Run the application:
```bash
python main.py
```

5. Open your browser and navigate to `http://localhost:7860`

## Usage

### Upload Data
- Navigate to the "Upload Data" tab
- Upload CSV, Excel, or JSON files
- View dataset information and preview

### AI-Powered Data Cleaning
- Choose between AI methods:
  - **BeautifulSoup4 Method**: For web-scraped or text-heavy data
  - **Direct Method**: For general data cleaning tasks
- Use natural language requirements like:
  - "Remove missing values and convert dates to proper format"
  - "Clean text columns and remove duplicates"
  - "Handle outliers in numeric columns and standardize formats"
- Get AI suggestions for optimal cleaning strategies
- Review and execute generated Python code
- Fallback to manual operations when needed

### Exploratory Analysis
- Generate insights with commands like:
  - "Show correlation matrix"
  - "Plot histogram of column X"
  - "Create scatter plot of X vs Y"
  - "Display summary statistics"

### Feature Engineering
- Transform data with commands like:
  - "Scale all numeric columns"
  - "One-hot encode categorical columns"
  - "Create polynomial features"
  - "Bin column X into 5 groups"

### Model Building
- Build models with commands like:
  - "Train random forest for classification"
  - "Build linear regression model"
  - "Compare multiple algorithms"
  - "Evaluate model performance"

## Project Structure

```
gradio-data-analysis-platform/
├── main.py                 # Application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── src/                   # Source code
│   ├── state_manager.py   # DataFrame state management
│   ├── command_processor.py # Natural language processing
│   └── controllers/       # Tab controllers
├── tests/                 # Test files
└── README.md             # This file
```

## Configuration

Modify `config.py` or use environment variables to customize:
- Server host and port
- File size limits
- Processing limits
- Visualization settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details