"""
AI Integration Manager for Gradio Data Analysis Platform
Manages Gemini AI integration for intelligent data cleaning and analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import google.generativeai as genai
import ast
import re
import json
from dataclasses import dataclass
import warnings
from bs4 import BeautifulSoup
import requests

@dataclass
class AICodeResult:
    """Result from AI code generation"""
    code: str
    explanation: str
    method_used: str  # 'beautifulsoup' or 'direct'
    confidence: float
    warnings: List[str]

class AIIntegrationManager:
    """
    Manages AI integration for intelligent data processing
    Provides BeautifulSoup4 and direct cleaning code generation via Gemini AI
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AI Integration Manager
        
        Args:
            api_key: Gemini AI API key (if None, will try to get from environment)
        """
        self.api_key = api_key
        self.model = None
        self._initialize_ai()
        
        # Security patterns to block in generated code
        self.blocked_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'import\s+sys',
            r'exec\s*\(',
            r'eval\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
        ]
    
    def _initialize_ai(self) -> bool:
        """Initialize Gemini AI connection"""
        try:
            if self.api_key:
                genai.configure(api_key=self.api_key)
            else:
                # Try to configure from environment
                genai.configure()
            
            # First, let's list available models to see what's actually available
            print("Checking available Gemini models...")
            try:
                available_models = []
                for model in genai.list_models():
                    if 'generateContent' in model.supported_generation_methods:
                        available_models.append(model.name)
                        print(f"Available model: {model.name}")
                
                if not available_models:
                    print("No models with generateContent support found")
                    return False
                    
            except Exception as list_error:
                print(f"Could not list models: {list_error}")
                # Continue with predefined list if listing fails
                available_models = None
            
            # Try different model names in order of preference (prioritizing gemini-2.5-pro)
            model_names = [
                'models/gemini-2.5-pro',        # Primary choice: Most powerful reasoning model
                'models/gemini-2.5-flash',      # Fallback: Faster and more quota-friendly
                'models/gemini-2.5-flash-lite', # Fallback: Most cost effective
                'models/gemini-1.5-flash-latest', # Fallback: 1.5 Flash (latest)
            ]
            
            # If we have available models, prioritize those
            if available_models:
                # Filter our preferred models to only those that are available
                available_preferred = [name for name in model_names if name in available_models]
                if available_preferred:
                    model_names = available_preferred + [m for m in available_models if m not in available_preferred]
                else:
                    model_names = available_models
            
            self.model = None
            self.current_model_name = None
            
            for model_name in model_names:
                try:
                    print(f"Trying to initialize model: {model_name}")
                    test_model = genai.GenerativeModel(model_name)
                    
                    # Test the model with a simple request
                    test_response = test_model.generate_content("Hello")
                    
                    # If we get here, the model works
                    self.model = test_model
                    self.current_model_name = model_name
                    print(f"✅ Successfully initialized Gemini AI with model: {model_name}")
                    return True
                    
                except Exception as model_error:
                    print(f"❌ Failed to initialize {model_name}: {model_error}")
                    continue
            
            print("❌ Failed to initialize any Gemini model")
            return False
            
        except Exception as e:
            print(f"❌ Warning: Could not initialize Gemini AI: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if AI integration is available"""
        return self.model is not None
    
    def get_current_model_name(self) -> str:
        """Get the name of the currently active model"""
        if self.model:
            return getattr(self.model, '_model_name', 'Unknown model')
        return "No model loaded"
    
    def list_available_models(self) -> List[str]:
        """List available Gemini models for debugging"""
        try:
            if self.api_key:
                genai.configure(api_key=self.api_key)
            else:
                genai.configure()
            
            models = genai.list_models()
            model_names = []
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    model_names.append(model.name)
            
            return model_names
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def sample_dataset(self, df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
        """
        Create a representative sample of the dataset for AI processing
        
        Args:
            df: Original DataFrame
            sample_size: Number of rows to sample
            
        Returns:
            Sampled DataFrame
        """
        if len(df) <= sample_size:
            return df.copy()
        
        # Use stratified sampling if possible
        try:
            # Try to maintain distribution of categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                # Sample proportionally from each category in the first categorical column
                col = categorical_cols[0]
                return df.groupby(col, group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max(1, sample_size // df[col].nunique())))
                ).head(sample_size)
        except:
            pass
        
        # Fallback to random sampling
        return df.sample(n=sample_size, random_state=42)
    
    def generate_beautifulsoup_code(self, sample_data: pd.DataFrame, 
                                  user_requirements: str) -> AICodeResult:
        """
        Generate BeautifulSoup4-based cleaning code using AI
        
        Args:
            sample_data: Sample of the dataset
            user_requirements: User's cleaning requirements
            
        Returns:
            AICodeResult with generated code and metadata
        """
        if not self.is_available():
            return self._create_fallback_result("AI not available")
        
        try:
            # Prepare dataset information
            dataset_info = self._prepare_dataset_info(sample_data)
            
            # Create prompt for BeautifulSoup4 cleaning
            prompt = f"""
You are a data cleaning expert. Generate Python code using BeautifulSoup4 and pandas to clean the dataset based on the user requirements.

Dataset Information:
{dataset_info}

User Requirements:
{user_requirements}

Please generate Python code that:
1. Uses BeautifulSoup4 for any text/HTML cleaning if applicable
2. Uses pandas for data manipulation
3. Follows the user requirements
4. Returns a cleaned DataFrame
5. Includes comments explaining each step

Requirements:
- Only use pandas, numpy, BeautifulSoup4, and standard library imports
- Do not use any file I/O operations
- Do not use exec, eval, or similar functions
- The code should work with a DataFrame variable named 'df'
- Return the cleaned DataFrame

Format your response as:
```python
# Your code here
```

Explanation: [Brief explanation of what the code does]
"""
            
            response = self.model.generate_content(prompt)
            code = self._extract_code_from_response(response.text)
            
            # Validate the generated code
            is_valid, warnings_list = self._validate_code(code)
            
            if not is_valid:
                return self._create_fallback_result("Generated code failed validation")
            
            return AICodeResult(
                code=code,
                explanation=self._extract_explanation_from_response(response.text),
                method_used='beautifulsoup',
                confidence=0.8,
                warnings=warnings_list
            )
            
        except Exception as e:
            return self._create_fallback_result(f"Error generating code: {str(e)}")
    
    def generate_direct_cleaning_code(self, dataset: pd.DataFrame, 
                                    user_requirements: str) -> AICodeResult:
        """
        Generate direct pandas cleaning code using AI
        
        Args:
            dataset: Full dataset (will be sampled if too large)
            user_requirements: User's cleaning requirements
            
        Returns:
            AICodeResult with generated code and metadata
        """
        if not self.is_available():
            return self._create_fallback_result("AI not available")
        
        # Try with current model first, then fallback models if rate limited
        fallback_models = [
            'models/gemini-2.5-flash',      # Faster and more quota-friendly
            'models/gemini-2.5-flash-lite', # Most cost effective
            'models/gemini-1.5-flash-latest', # 1.5 Flash (latest)
        ]
        
        try:
            # Sample dataset if too large
            sample_data = self.sample_dataset(dataset, sample_size=50)
            dataset_info = self._prepare_dataset_info(sample_data)
            
            # Create prompt for direct cleaning
            prompt = f"""
You are a data cleaning expert. Generate Python code using pandas to clean the dataset based on the user requirements.

Dataset Information:
{dataset_info}

User Requirements:
{user_requirements}

Please generate Python code that:
1. Uses pandas and numpy for data cleaning
2. Follows the user requirements exactly
3. Handles missing values, duplicates, outliers as needed
4. Returns a cleaned DataFrame
5. Includes comments explaining each step

Requirements:
- Only use pandas, numpy, and standard library imports
- Do not use any file I/O operations
- Do not use exec, eval, or similar functions
- The code should work with a DataFrame variable named 'df'
- Return the cleaned DataFrame as 'cleaned_df'

Format your response as:
```python
# Your code here
```

Explanation: [Brief explanation of what the code does]
"""
            
            # Try current model first
            try:
                response = self.model.generate_content(prompt)
                code = self._extract_code_from_response(response.text)
                
                # Validate the generated code
                is_valid, warnings_list = self._validate_code(code)
                
                if not is_valid:
                    return self._create_fallback_result("Generated code failed validation")
                
                return AICodeResult(
                    code=code,
                    explanation=self._extract_explanation_from_response(response.text),
                    method_used='direct',
                    confidence=0.9,
                    warnings=warnings_list
                )
                
            except Exception as model_error:
                # Check if it's a rate limit error
                if "429" in str(model_error) or "quota" in str(model_error).lower():
                    print(f"Rate limit hit with {self.current_model_name}, trying fallback models...")
                    
                    # Try fallback models
                    for fallback_model in fallback_models:
                        if fallback_model == self.current_model_name:
                            continue  # Skip if it's the same as current model
                            
                        try:
                            print(f"Trying fallback model: {fallback_model}")
                            temp_model = genai.GenerativeModel(fallback_model)
                            response = temp_model.generate_content(prompt)
                            code = self._extract_code_from_response(response.text)
                            
                            # Validate the generated code
                            is_valid, warnings_list = self._validate_code(code)
                            
                            if is_valid:
                                print(f"✅ Successfully generated code with fallback model: {fallback_model}")
                                return AICodeResult(
                                    code=code,
                                    explanation=self._extract_explanation_from_response(response.text),
                                    method_used=f'direct (fallback: {fallback_model})',
                                    confidence=0.8,
                                    warnings=warnings_list + [f"Used fallback model {fallback_model} due to rate limits"]
                                )
                        except Exception as fallback_error:
                            print(f"❌ Fallback model {fallback_model} also failed: {fallback_error}")
                            continue
                    
                    # If all fallback models failed
                    return self._create_fallback_result(f"Rate limit exceeded and all fallback models failed. Original error: {str(model_error)}")
                else:
                    # Re-raise if it's not a rate limit error
                    raise model_error
            
        except Exception as e:
            return self._create_fallback_result(f"Error generating code: {str(e)}")
    
    def _prepare_dataset_info(self, df: pd.DataFrame) -> str:
        """Prepare dataset information for AI prompt"""
        info = []
        info.append(f"Shape: {df.shape}")
        info.append(f"Columns: {list(df.columns)}")
        info.append(f"Data types: {df.dtypes.to_dict()}")
        info.append(f"Missing values: {df.isnull().sum().to_dict()}")
        
        # Add sample data
        info.append("Sample data:")
        info.append(df.head(3).to_string())
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            info.append("Numeric column statistics:")
            info.append(df[numeric_cols].describe().to_string())
        
        return "\n".join(info)
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """Extract Python code from AI response"""
        # Look for code blocks
        code_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_pattern, response_text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Fallback: look for any code-like content
        lines = response_text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if 'import' in line or 'df' in line or line.strip().startswith('#'):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        return '\n'.join(code_lines).strip()
    
    def _extract_explanation_from_response(self, response_text: str) -> str:
        """Extract explanation from AI response"""
        # Look for explanation after code block
        if 'Explanation:' in response_text:
            return response_text.split('Explanation:')[-1].strip()
        
        # Fallback: return first paragraph
        lines = response_text.split('\n')
        explanation_lines = []
        
        for line in lines:
            if not line.strip().startswith('#') and 'import' not in line and 'df' not in line:
                explanation_lines.append(line)
            if len(explanation_lines) > 3:
                break
        
        return ' '.join(explanation_lines).strip()
    
    def _validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate generated code for security and syntax
        
        Returns:
            Tuple of (is_valid, warnings_list)
        """
        warnings_list = []
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                warnings_list.append(f"Blocked pattern detected: {pattern}")
                return False, warnings_list
        
        # Check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            warnings_list.append(f"Syntax error: {str(e)}")
            return False, warnings_list
        
        # Check for required elements
        if 'df' not in code:
            warnings_list.append("Code doesn't reference DataFrame variable 'df'")
        
        return True, warnings_list
    
    def execute_ai_code(self, code: str, dataframe: pd.DataFrame) -> Tuple[bool, pd.DataFrame, str]:
        """
        Safely execute AI-generated code
        
        Args:
            code: Generated Python code
            dataframe: Input DataFrame
            
        Returns:
            Tuple of (success, result_dataframe, error_message)
        """
        try:
            # Validate code first
            is_valid, warnings_list = self._validate_code(code)
            if not is_valid:
                return False, dataframe, f"Code validation failed: {'; '.join(warnings_list)}"
            
            # Create safe execution environment with necessary builtins
            import builtins
            safe_builtins = {
                '__import__': __import__,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'round': round,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'type': type,
                'print': print,  # Allow print for debugging
                're': re,  # Add re module for regex operations
            }
            
            safe_globals = {
                'pd': pd,
                'np': np,
                'BeautifulSoup': BeautifulSoup,
                'df': dataframe.copy(),
                're': re,  # Add re module
                '__builtins__': safe_builtins
            }
            
            # Execute code
            exec(code, safe_globals)
            
            # Get result DataFrame
            if 'cleaned_df' in safe_globals:
                result_df = safe_globals['cleaned_df']
            elif 'df' in safe_globals:
                result_df = safe_globals['df']
            else:
                return False, dataframe, "No result DataFrame found in executed code"
            
            # Validate result
            if not isinstance(result_df, pd.DataFrame):
                return False, dataframe, "Result is not a DataFrame"
            
            if result_df.empty:
                return False, dataframe, "Result DataFrame is empty"
            
            return True, result_df, ""
            
        except Exception as e:
            return False, dataframe, f"Execution error: {str(e)}"
    
    def _create_fallback_result(self, error_message: str) -> AICodeResult:
        """Create a fallback result when AI generation fails"""
        return AICodeResult(
            code="# AI code generation failed, please use manual cleaning",
            explanation=f"Error: {error_message}",
            method_used='fallback',
            confidence=0.0,
            warnings=[error_message]
        )
    
    def get_cleaning_suggestions(self, df: pd.DataFrame) -> List[str]:
        """
        Get AI-powered suggestions for data cleaning
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of cleaning suggestions
        """
        if not self.is_available():
            return ["AI not available for suggestions"]
        
        try:
            dataset_info = self._prepare_dataset_info(df.head(10))
            
            prompt = f"""
Analyze this dataset and provide 3-5 specific data cleaning suggestions:

{dataset_info}

Provide actionable suggestions in this format:
1. [Suggestion 1]
2. [Suggestion 2]
etc.
"""
            
            response = self.model.generate_content(prompt)
            suggestions = []
            
            for line in response.text.split('\n'):
                if re.match(r'^\d+\.', line.strip()):
                    suggestions.append(line.strip())
            
            return suggestions if suggestions else ["No specific suggestions available"]
            
        except Exception as e:
            return [f"Error getting suggestions: {str(e)}"]