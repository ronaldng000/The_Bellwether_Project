"""
Enhanced Cleaning Controller with AI Integration
Provides AI-powered data cleaning using BeautifulSoup4 and direct methods
"""

import gradio as gr
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.state_manager import StateManager
from src.ai_integration_manager import AIIntegrationManager, AICodeResult
from config_ai import ai_config

class CleaningController:
    """
    Enhanced data cleaning controller with AI integration
    Supports BeautifulSoup4 and direct AI-powered cleaning methods
    """
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.ai_manager = AIIntegrationManager(ai_config.gemini_api_key)
        
    def create_interface(self) -> gr.Interface:
        """Create the enhanced cleaning tab interface"""
        
        with gr.Blocks(title="AI-Powered Data Cleaning") as interface:
            gr.Markdown("# 🧹 AI-Powered Data Cleaning")
            
            # Display current dataset info
            with gr.Row():
                dataset_info = gr.Textbox(
                    label="Current Dataset Info",
                    value=self._get_dataset_info(),
                    interactive=False,
                    lines=3
                )
                refresh_btn = gr.Button("🔄 Refresh Info", size="sm")
            
            # AI Configuration Status
            with gr.Row():
                ai_status = gr.Textbox(
                    label="AI Status",
                    value=self._get_ai_status(),
                    interactive=False,
                    lines=2
                )
            
            # Cleaning Method Selection
            with gr.Row():
                cleaning_method = gr.Radio(
                    choices=["AI - BeautifulSoup4 Method", "AI - Direct Method", "Manual Operations"],
                    label="Cleaning Method",
                    value="AI - Direct Method",
                    info="Choose your preferred cleaning approach"
                )
            
            # User Requirements Input
            with gr.Row():
                user_requirements = gr.Textbox(
                    label="Cleaning Requirements",
                    placeholder="Describe what you want to achieve (e.g., 'Remove missing values, convert dates to datetime, remove duplicates')",
                    lines=3
                )
            
            # AI Suggestions
            with gr.Row():
                suggestions_btn = gr.Button("💡 Get AI Suggestions", variant="secondary")
                suggestions_output = gr.Textbox(
                    label="AI Suggestions",
                    lines=4,
                    interactive=False
                )
            
            # Generate and Execute Buttons
            with gr.Row():
                generate_btn = gr.Button("🤖 Generate Cleaning Code", variant="primary")
                execute_btn = gr.Button("▶️ Execute Code", variant="secondary")
            
            # Generated Code Display
            with gr.Row():
                generated_code = gr.Code(
                    label="Generated Cleaning Code",
                    language="python",
                    lines=15
                )
            
            # Code Explanation
            with gr.Row():
                code_explanation = gr.Textbox(
                    label="Code Explanation",
                    lines=3,
                    interactive=False
                )
            
            # Results Display
            with gr.Row():
                results_output = gr.Textbox(
                    label="Execution Results",
                    lines=5,
                    interactive=False
                )
            
            # Manual Operations (fallback)
            with gr.Accordion("Manual Operations", open=False):
                with gr.Row():
                    manual_operation = gr.Dropdown(
                        choices=[
                            "Remove missing values",
                            "Remove duplicates", 
                            "Convert data types",
                            "Remove outliers",
                            "Fill missing values"
                        ],
                        label="Manual Operation"
                    )
                    execute_manual_btn = gr.Button("Execute Manual Operation")
            
            # Event handlers
            refresh_btn.click(
                fn=self._refresh_dataset_info,
                outputs=[dataset_info]
            )
            
            suggestions_btn.click(
                fn=self._get_ai_suggestions,
                outputs=[suggestions_output]
            )
            
            generate_btn.click(
                fn=self._generate_cleaning_code,
                inputs=[cleaning_method, user_requirements],
                outputs=[generated_code, code_explanation, results_output]
            )
            
            execute_btn.click(
                fn=self._execute_cleaning_code,
                inputs=[generated_code],
                outputs=[results_output, dataset_info]
            )
            
            execute_manual_btn.click(
                fn=self._execute_manual_operation,
                inputs=[manual_operation],
                outputs=[results_output, dataset_info]
            )
        
        return interface
    
    def _get_dataset_info(self) -> str:
        """Get current dataset information"""
        if not self.state_manager.has_data():
            return "No dataset loaded. Please upload data first."
        
        return self.state_manager.get_summary_info()
    
    def _refresh_dataset_info(self) -> str:
        """Refresh dataset information display"""
        return self._get_dataset_info()
    
    def _get_ai_status(self) -> str:
        """Get AI integration status"""
        if not ai_config.is_configured():
            return "❌ AI not configured. " + ai_config.get_setup_instructions()
        
        if self.ai_manager.is_available():
            return "✅ AI integration ready (Gemini AI connected)"
        else:
            return "⚠️ AI configured but connection failed"
    
    def _get_ai_suggestions(self) -> str:
        """Get AI-powered cleaning suggestions"""
        if not self.state_manager.has_data():
            return "No dataset available for analysis."
        
        if not self.ai_manager.is_available():
            return "AI not available. Please configure Gemini AI API key."
        
        try:
            df = self.state_manager.get_dataframe()
            suggestions = self.ai_manager.get_cleaning_suggestions(df)
            return "\n".join(suggestions)
        except Exception as e:
            return f"Error getting suggestions: {str(e)}"
    
    def _generate_cleaning_code(self, method: str, requirements: str) -> Tuple[str, str, str]:
        """Generate cleaning code using AI"""
        if not self.state_manager.has_data():
            return "", "", "No dataset available. Please upload data first."
        
        if not requirements.strip():
            return "", "", "Please provide cleaning requirements."
        
        if not self.ai_manager.is_available():
            return "", "", "AI not available. Please configure Gemini AI API key or use manual operations."
        
        try:
            df = self.state_manager.get_dataframe()
            
            # Generate code based on selected method
            if "BeautifulSoup4" in method:
                # Sample data for BeautifulSoup4 method
                sample_df = self.ai_manager.sample_dataset(df, sample_size=100)
                result = self.ai_manager.generate_beautifulsoup_code(sample_df, requirements)
            else:
                # Direct method
                result = self.ai_manager.generate_direct_cleaning_code(df, requirements)
            
            # Prepare results
            warnings_text = ""
            if result.warnings:
                warnings_text = f"\n\nWarnings:\n" + "\n".join(f"⚠️ {w}" for w in result.warnings)
            
            status_text = f"✅ Code generated using {result.method_used} method (confidence: {result.confidence:.1%}){warnings_text}"
            
            return result.code, result.explanation, status_text
            
        except Exception as e:
            return "", "", f"Error generating code: {str(e)}"
    
    def _execute_cleaning_code(self, code: str) -> Tuple[str, str]:
        """Execute the generated cleaning code"""
        if not self.state_manager.has_data():
            return "No dataset available.", self._get_dataset_info()
        
        if not code.strip():
            return "No code to execute.", self._get_dataset_info()
        
        try:
            df = self.state_manager.get_dataframe()
            
            # Execute AI-generated code
            success, result_df, error_msg = self.ai_manager.execute_ai_code(code, df)
            
            if success:
                # Update state with cleaned data
                operation_info = {
                    'tab': 'Data Cleaning',
                    'command': 'AI-Generated Cleaning Code',
                    'operation_type': 'ai_clean'
                }
                
                update_success = self.state_manager.update_dataframe(result_df, operation_info)
                
                if update_success:
                    changes_summary = self._get_changes_summary(df, result_df)
                    return f"✅ Cleaning completed successfully!\n\n{changes_summary}", self._get_dataset_info()
                else:
                    return "❌ Failed to update dataset state.", self._get_dataset_info()
            else:
                return f"❌ Execution failed: {error_msg}", self._get_dataset_info()
                
        except Exception as e:
            return f"❌ Error executing code: {str(e)}", self._get_dataset_info()
    
    def _execute_manual_operation(self, operation: str) -> Tuple[str, str]:
        """Execute manual cleaning operations"""
        if not self.state_manager.has_data():
            return "No dataset available.", self._get_dataset_info()
        
        if not operation:
            return "Please select an operation.", self._get_dataset_info()
        
        try:
            df = self.state_manager.get_dataframe()
            original_shape = df.shape
            
            # Execute manual operation
            if operation == "Remove missing values":
                cleaned_df = df.dropna()
            elif operation == "Remove duplicates":
                cleaned_df = df.drop_duplicates()
            elif operation == "Convert data types":
                cleaned_df = self._auto_convert_types(df)
            elif operation == "Remove outliers":
                cleaned_df = self._remove_outliers(df)
            elif operation == "Fill missing values":
                cleaned_df = df.fillna(method='ffill').fillna(method='bfill')
            else:
                return f"Unknown operation: {operation}", self._get_dataset_info()
            
            # Update state
            operation_info = {
                'tab': 'Data Cleaning',
                'command': f'Manual: {operation}',
                'operation_type': 'manual_clean'
            }
            
            update_success = self.state_manager.update_dataframe(cleaned_df, operation_info)
            
            if update_success:
                new_shape = cleaned_df.shape
                return f"✅ {operation} completed!\nRows: {original_shape[0]} → {new_shape[0]}\nColumns: {original_shape[1]} → {new_shape[1]}", self._get_dataset_info()
            else:
                return "❌ Failed to update dataset state.", self._get_dataset_info()
                
        except Exception as e:
            return f"❌ Error in manual operation: {str(e)}", self._get_dataset_info()
    
    def _get_changes_summary(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> str:
        """Generate summary of changes made during cleaning"""
        summary = []
        
        # Shape changes
        orig_shape = original_df.shape
        new_shape = cleaned_df.shape
        summary.append(f"Shape: {orig_shape} → {new_shape}")
        
        # Missing values changes
        orig_nulls = original_df.isnull().sum().sum()
        new_nulls = cleaned_df.isnull().sum().sum()
        summary.append(f"Missing values: {orig_nulls} → {new_nulls}")
        
        # Memory usage changes
        orig_memory = original_df.memory_usage(deep=True).sum()
        new_memory = cleaned_df.memory_usage(deep=True).sum()
        summary.append(f"Memory usage: {self._format_bytes(orig_memory)} → {self._format_bytes(new_memory)}")
        
        return "Changes Summary:\n" + "\n".join(f"• {s}" for s in summary)
    
    def _format_bytes(self, bytes_size: int) -> str:
        """Format bytes in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"
    
    def _auto_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically convert data types"""
        df_converted = df.copy()
        
        for col in df_converted.columns:
            # Try to convert to numeric
            if df_converted[col].dtype == 'object':
                try:
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='ignore')
                except:
                    pass
                
                # Try to convert to datetime
                try:
                    df_converted[col] = pd.to_datetime(df_converted[col], errors='ignore')
                except:
                    pass
        
        return df_converted
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        return df_clean