"""
Gradio Data Analysis Platform
A comprehensive web-based application for data science workflows with AI integration
"""

import gradio as gr
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Import core components
from src.state_manager import StateManager
from src.ai_integration_manager import AIIntegrationManager
from config_ai import ai_config

# Global state manager instance
global_state_manager = StateManager()
global_ai_manager = AIIntegrationManager(ai_config.gemini_api_key)

def create_app():
    """Create and configure the main Gradio application with shared state"""
    
    print("Initializing Gradio Data Analysis Platform...")
    
    with gr.Blocks(title="🤖 AI-Powered Data Analysis Platform", theme=gr.themes.Soft()) as app:
        
        # Global state components
        state_data = gr.State(value=None)  # Will hold the DataFrame
        
        # Header with dataset info
        with gr.Row():
            gr.Markdown("# 🤖 AI-Powered Data Analysis Platform")
        
        with gr.Row():
            dataset_status = gr.Textbox(
                label="📊 Current Dataset Status",
                value="No dataset loaded. Please upload a file to begin.",
                interactive=False,
                lines=2
            )
        
        # Main tabbed interface
        with gr.Tabs():
            
            # Upload Data Tab
            with gr.TabItem("📁 Upload Data"):
                gr.Markdown("## Upload Dataset")
                gr.Markdown("Upload CSV, Excel, or JSON files to begin your data analysis workflow.")
                
                with gr.Row():
                    file_input = gr.File(
                        label="Choose File",
                        file_types=[".csv", ".xlsx", ".xls", ".json"]
                    )
                    upload_btn = gr.Button("📤 Upload", variant="primary")
                
                upload_output = gr.Textbox(
                    label="Upload Results",
                    lines=8,
                    interactive=False
                )
                
                # Upload event handler
                upload_btn.click(
                    fn=upload_file,
                    inputs=[file_input, state_data],
                    outputs=[upload_output, dataset_status, state_data]
                )
            
            # Data Cleaning Tab
            with gr.TabItem("🧹 Data Cleaning"):
                gr.Markdown("## AI-Powered Data Cleaning")
                
                # AI Status
                with gr.Row():
                    ai_status = gr.Textbox(
                        label="🤖 AI Status",
                        value=get_ai_status(),
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
                
                # User Requirements
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
                
                # Generate and Execute
                with gr.Row():
                    generate_btn = gr.Button("🤖 Generate Code", variant="primary")
                    execute_btn = gr.Button("▶️ Execute Code", variant="secondary")
                
                # Code Display
                generated_code = gr.Code(
                    label="Generated Cleaning Code",
                    language="python",
                    lines=12
                )
                
                code_explanation = gr.Textbox(
                    label="Code Explanation",
                    lines=3,
                    interactive=False
                )
                
                # Results
                cleaning_results = gr.Textbox(
                    label="Execution Results",
                    lines=5,
                    interactive=False
                )
                
                # Data Sampling Section
                with gr.Accordion("📊 Cleaned Data Sample", open=False):
                    with gr.Row():
                        sample_size_slider = gr.Slider(
                            minimum=5,
                            maximum=100,
                            value=10,
                            step=5,
                            label="Sample Size",
                            info="Number of rows to display from cleaned data"
                        )
                        refresh_sample_btn = gr.Button("🔄 Refresh Sample", variant="secondary")
                    
                    cleaned_data_sample = gr.Dataframe(
                        label="Cleaned Data Sample",
                        interactive=False,
                        wrap=True
                    )
                    
                    sample_info = gr.Textbox(
                        label="Sample Information",
                        lines=2,
                        interactive=False,
                        placeholder="Execute cleaning code to see sample information"
                    )
                
                # Manual Operations
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
                        execute_manual_btn = gr.Button("Execute Manual")
                
                # Event handlers for cleaning tab
                suggestions_btn.click(
                    fn=get_ai_suggestions,
                    inputs=[state_data],
                    outputs=[suggestions_output]
                )
                
                generate_btn.click(
                    fn=generate_cleaning_code,
                    inputs=[cleaning_method, user_requirements, state_data],
                    outputs=[generated_code, code_explanation, cleaning_results]
                )
                
                execute_btn.click(
                    fn=execute_cleaning_code,
                    inputs=[generated_code, state_data],
                    outputs=[cleaning_results, dataset_status, state_data, cleaned_data_sample, sample_info]
                )
                
                execute_manual_btn.click(
                    fn=execute_manual_operation,
                    inputs=[manual_operation, state_data],
                    outputs=[cleaning_results, dataset_status, state_data, cleaned_data_sample, sample_info]
                )
                
                refresh_sample_btn.click(
                    fn=refresh_data_sample,
                    inputs=[state_data, sample_size_slider],
                    outputs=[cleaned_data_sample, sample_info]
                )
            
            # Placeholder tabs
            with gr.TabItem("📊 Exploratory Analysis"):
                gr.Markdown("## Exploratory Data Analysis")
                gr.Markdown("This feature will be implemented soon!")
                
                with gr.Row():
                    current_data_info = gr.Textbox(
                        label="Current Dataset Info",
                        value="Upload data to see information here",
                        interactive=False,
                        lines=5
                    )
                
                # Update data info when tab is accessed
                app.load(
                    fn=get_current_dataset_info,
                    inputs=[state_data],
                    outputs=[current_data_info]
                )
            
            with gr.TabItem("⚙️ Feature Engineering"):
                gr.Markdown("## Feature Engineering")
                gr.Markdown("This feature will be implemented soon!")
                
                with gr.Row():
                    current_data_info2 = gr.Textbox(
                        label="Current Dataset Info",
                        value="Upload data to see information here",
                        interactive=False,
                        lines=5
                    )
            
            with gr.TabItem("🤖 Model Building"):
                gr.Markdown("## Machine Learning Models")
                gr.Markdown("This feature will be implemented soon!")
                
                with gr.Row():
                    current_data_info3 = gr.Textbox(
                        label="Current Dataset Info",
                        value="Upload data to see information here",
                        interactive=False,
                        lines=5
                    )
    
    return app

# Helper functions that work with global state
def upload_file(file, current_state):
    """Upload and process file"""
    if file is None:
        return "No file uploaded.", "No dataset loaded.", None
    
    try:
        # Determine file type and load
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.name)
        elif file.name.endswith('.json'):
            df = pd.read_json(file.name)
        else:
            return "Unsupported file format. Please use CSV, Excel, or JSON.", "No dataset loaded.", None
        
        # Validate dataset size
        is_valid, error_msg = global_state_manager.validate_dataframe_size(df)
        if not is_valid:
            return f"Dataset validation failed: {error_msg}", "No dataset loaded.", None
        
        # Update global state
        operation_info = {
            'tab': 'Upload',
            'command': f'Upload {file.name}',
            'operation_type': 'upload'
        }
        
        success = global_state_manager.update_dataframe(df, operation_info)
        
        if success:
            status = global_state_manager.get_summary_info()
            return f"✅ File uploaded successfully!\n\n{status}", status, df
        else:
            return "❌ Failed to load dataset.", "No dataset loaded.", None
            
    except Exception as e:
        return f"❌ Error loading file: {str(e)}", "No dataset loaded.", None

def get_ai_status():
    """Get AI integration status"""
    if not ai_config.is_configured():
        return "❌ AI not configured. Set GEMINI_API_KEY environment variable."
    
    if global_ai_manager.is_available():
        model_name = global_ai_manager.get_current_model_name()
        return f"✅ AI integration ready using: {model_name}"
    else:
        # Try to get more detailed error information
        try:
            available_models = global_ai_manager.list_available_models()
            if available_models:
                models_text = ", ".join(available_models[:3])  # Show first 3 models
                return f"⚠️ AI configured but connection failed. Available models: {models_text}"
            else:
                return "⚠️ AI configured but no models available. Check API key."
        except:
            return "⚠️ AI configured but connection failed. Check API key and internet connection."

def get_ai_suggestions(current_state):
    """Get AI suggestions for data cleaning"""
    if current_state is None or not global_state_manager.has_data():
        return "No dataset available for analysis."
    
    if not global_ai_manager.is_available():
        return "AI not available. Please configure Gemini AI API key."
    
    try:
        df = global_state_manager.get_dataframe()
        suggestions = global_ai_manager.get_cleaning_suggestions(df)
        return "\n".join(suggestions)
    except Exception as e:
        return f"Error getting suggestions: {str(e)}"

def generate_cleaning_code(method, requirements, current_state):
    """Generate AI cleaning code"""
    if current_state is None:
        return "", "", "No dataset available. Please upload data first."
    
    if not requirements.strip():
        return "", "", "Please provide cleaning requirements."
    
    if not global_ai_manager.is_available():
        return "", "", "AI not available. Please configure Gemini AI API key."
    
    try:
        df = current_state  # Use the current state (DataFrame) directly
        
        if "BeautifulSoup4" in method:
            sample_df = global_ai_manager.sample_dataset(df, sample_size=100)
            result = global_ai_manager.generate_beautifulsoup_code(sample_df, requirements)
        else:
            result = global_ai_manager.generate_direct_cleaning_code(df, requirements)
        
        warnings_text = ""
        if result.warnings:
            warnings_text = f"\n\nWarnings:\n" + "\n".join(f"⚠️ {w}" for w in result.warnings)
        
        status_text = f"✅ Code generated using {result.method_used} method (confidence: {result.confidence:.1%}){warnings_text}"
        
        return result.code, result.explanation, status_text
        
    except Exception as e:
        return "", "", f"Error generating code: {str(e)}"

def execute_cleaning_code(code, current_state):
    """Execute AI-generated cleaning code"""
    if current_state is None:
        return "No dataset available.", "No dataset loaded.", None
    
    if not code.strip():
        return "No code to execute.", "No dataset loaded.", current_state
    
    try:
        df = current_state  # Use current state directly
        success, result_df, error_msg = global_ai_manager.execute_ai_code(code, df)
        
        if success:
            operation_info = {
                'tab': 'Data Cleaning',
                'command': 'AI-Generated Cleaning Code',
                'operation_type': 'ai_clean'
            }
            
            update_success = global_state_manager.update_dataframe(result_df, operation_info)
            
            if update_success:
                changes_summary = get_changes_summary(df, result_df)
                status = global_state_manager.get_summary_info()
                
                # Generate sample data
                sample_df, sample_info = get_data_sample(result_df, 10)
                
                return f"✅ Cleaning completed successfully!\n\n{changes_summary}", status, result_df, sample_df, sample_info
            else:
                return "❌ Failed to update dataset state.", global_state_manager.get_summary_info(), current_state, None, "Failed to update dataset"
        else:
            return f"❌ Execution failed: {error_msg}", global_state_manager.get_summary_info(), current_state, None, f"Execution failed: {error_msg}"
            
    except Exception as e:
        return f"❌ Error executing code: {str(e)}", global_state_manager.get_summary_info(), current_state, None, f"Error: {str(e)}"

def execute_manual_operation(operation, current_state):
    """Execute manual cleaning operations"""
    if current_state is None:
        return "No dataset available.", "No dataset loaded.", None, None, "No data available"
    
    if not operation:
        return "Please select an operation.", "No dataset loaded.", current_state, None, "No operation selected"
    
    try:
        df = current_state  # Use current state directly
        original_shape = df.shape
        
        if operation == "Remove missing values":
            cleaned_df = df.dropna()
        elif operation == "Remove duplicates":
            cleaned_df = df.drop_duplicates()
        elif operation == "Convert data types":
            cleaned_df = auto_convert_types(df)
        elif operation == "Remove outliers":
            cleaned_df = remove_outliers(df)
        elif operation == "Fill missing values":
            cleaned_df = df.fillna(method='ffill').fillna(method='bfill')
        else:
            return f"Unknown operation: {operation}", "No dataset loaded.", current_state, None, f"Unknown operation: {operation}"
        
        operation_info = {
            'tab': 'Data Cleaning',
            'command': f'Manual: {operation}',
            'operation_type': 'manual_clean'
        }
        
        update_success = global_state_manager.update_dataframe(cleaned_df, operation_info)
        
        if update_success:
            new_shape = cleaned_df.shape
            status = global_state_manager.get_summary_info()
            
            # Generate sample data
            sample_df, sample_info = get_data_sample(cleaned_df, 10)
            
            return f"✅ {operation} completed!\nRows: {original_shape[0]} → {new_shape[0]}\nColumns: {original_shape[1]} → {new_shape[1]}", status, cleaned_df, sample_df, sample_info
        else:
            return "❌ Failed to update dataset state.", "No dataset loaded.", current_state, None, "Failed to update dataset state"
            
    except Exception as e:
        return f"❌ Error in manual operation: {str(e)}", "No dataset loaded.", current_state, None, f"Error: {str(e)}"

def get_current_dataset_info(current_state):
    """Get current dataset information"""
    if current_state is None or not global_state_manager.has_data():
        return "No dataset loaded. Please upload data first."
    
    return global_state_manager.get_summary_info()

def get_changes_summary(original_df, cleaned_df):
    """Generate summary of changes"""
    summary = []
    orig_shape = original_df.shape
    new_shape = cleaned_df.shape
    summary.append(f"Shape: {orig_shape} → {new_shape}")
    
    orig_nulls = original_df.isnull().sum().sum()
    new_nulls = cleaned_df.isnull().sum().sum()
    summary.append(f"Missing values: {orig_nulls} → {new_nulls}")
    
    return "Changes Summary:\n" + "\n".join(f"• {s}" for s in summary)

def get_data_sample(current_state, sample_size=10):
    """Get a sample of the current dataset"""
    if current_state is None:
        return None, "No dataset available"
    
    try:
        df = current_state
        
        # Get sample
        if len(df) <= sample_size:
            sample_df = df.copy()
            sample_info = f"Showing all {len(df)} rows (dataset smaller than sample size)"
        else:
            sample_df = df.sample(n=sample_size, random_state=42)
            sample_info = f"Showing {sample_size} random rows out of {len(df)} total rows"
        
        # Add basic info
        sample_info += f"\nDataset shape: {df.shape}"
        sample_info += f"\nColumns: {', '.join(df.columns.tolist())}"
        
        return sample_df, sample_info
        
    except Exception as e:
        return None, f"Error generating sample: {str(e)}"

def refresh_data_sample(current_state, sample_size):
    """Refresh the data sample with specified size"""
    return get_data_sample(current_state, sample_size)

def refresh_data_sample(current_state, sample_size):
    """Refresh the data sample display"""
    if current_state is None:
        return None, "No data available. Please upload and clean data first."
    
    try:
        df = current_state
        
        # Get sample
        if len(df) <= sample_size:
            sample_df = df.copy()
            sample_info_text = f"Showing all {len(df)} rows (dataset smaller than sample size)"
        else:
            sample_df = df.sample(n=int(sample_size), random_state=42)
            sample_info_text = f"Showing {int(sample_size)} random rows out of {len(df)} total rows"
        
        # Add basic info
        sample_info_text += f"\nDataset shape: {df.shape} | Columns: {', '.join(df.columns[:5])}"
        if len(df.columns) > 5:
            sample_info_text += f" ... (+{len(df.columns)-5} more)"
        
        return sample_df, sample_info_text
        
    except Exception as e:
        return None, f"Error generating sample: {str(e)}"

def get_data_sample_info(df, sample_size):
    """Generate sample information for cleaned data"""
    if df is None or df.empty:
        return None, "No data available"
    
    try:
        # Get sample
        if len(df) <= sample_size:
            sample_df = df.copy()
            info_text = f"Showing all {len(df)} rows"
        else:
            sample_df = df.head(int(sample_size))  # Use head for consistency after cleaning
            info_text = f"Showing first {int(sample_size)} rows out of {len(df)} total"
        
        # Add summary info
        info_text += f"\nShape: {df.shape} | Missing values: {df.isnull().sum().sum()}"
        info_text += f"\nColumns: {', '.join(df.columns)}"
        
        return sample_df, info_text
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def auto_convert_types(df):
    """Auto convert data types"""
    df_converted = df.copy()
    for col in df_converted.columns:
        if df_converted[col].dtype == 'object':
            try:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='ignore')
            except:
                pass
            try:
                df_converted[col] = pd.to_datetime(df_converted[col], errors='ignore')
            except:
                pass
    return df_converted

def remove_outliers(df):
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

if __name__ == "__main__":
    try:
        print("Creating Gradio Data Analysis Platform...")
        app = create_app()
        print("App created successfully!")
        print("Launching on http://localhost:7860")
        app.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            debug=True,
            show_error=True
        )
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()