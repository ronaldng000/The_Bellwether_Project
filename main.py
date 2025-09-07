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
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Import core components
from src.state_manager import StateManager
from src.ai_integration_manager import AIIntegrationManager
from config_ai import ai_config

# Global state manager instance
global_state_manager = StateManager()
global_ai_manager = AIIntegrationManager(ai_config.gemini_api_key)

# ============================================================================
# EXPLORATORY DATA ANALYSIS FUNCTIONS
# ============================================================================

def generate_statistical_summary(current_state, selected_columns=None):
    """Generate statistical summary of the dataset"""
    if current_state is None:
        return "No dataset available. Please upload and clean data first."
    
    try:
        df = current_state
        
        # Filter columns if specified
        if selected_columns:
            df = df[selected_columns]
        
        summary_parts = []
        
        # Basic info
        summary_parts.append("=== DATASET OVERVIEW ===")
        summary_parts.append(f"Shape: {df.shape}")
        summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
        summary_parts.append("")
        
        # Data types
        summary_parts.append("=== DATA TYPES ===")
        for col, dtype in df.dtypes.items():
            summary_parts.append(f"{col}: {dtype}")
        summary_parts.append("")
        
        # Missing values
        summary_parts.append("=== MISSING VALUES ===")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        for col in df.columns:
            if missing[col] > 0:
                summary_parts.append(f"{col}: {missing[col]} ({missing_pct[col]}%)")
        if missing.sum() == 0:
            summary_parts.append("No missing values found")
        summary_parts.append("")
        
        # Numerical statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary_parts.append("=== NUMERICAL STATISTICS ===")
            desc = df[numeric_cols].describe()
            summary_parts.append(desc.to_string())
            summary_parts.append("")
        
        # Categorical statistics
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            summary_parts.append("=== CATEGORICAL STATISTICS ===")
            for col in cat_cols:
                unique_count = df[col].nunique()
                most_common = df[col].value_counts().head(3)
                summary_parts.append(f"{col}: {unique_count} unique values")
                summary_parts.append(f"  Most common: {most_common.to_dict()}")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
        
    except Exception as e:
        return f"Error generating statistical summary: {str(e)}"

def generate_data_visualization(current_state, viz_type, selected_columns=None):
    """Generate data visualizations"""
    if current_state is None:
        return None, "No dataset available. Please upload and clean data first."
    
    try:
        df = current_state
        
        # Filter columns if specified
        if selected_columns:
            df = df[selected_columns]
        
        plt.style.use('default')
        
        if viz_type == "Histogram":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return None, "No numerical columns found for histogram"
            
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            return fig, f"Generated histograms for {len(numeric_cols)} numerical columns"
        
        elif viz_type == "Correlation Heatmap":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                return None, "Need at least 2 numerical columns for correlation heatmap"
            
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            import seaborn as sns
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, ax=ax)
            ax.set_title('Correlation Heatmap')
            plt.tight_layout()
            return fig, f"Generated correlation heatmap for {len(numeric_cols)} numerical columns"
        
        elif viz_type == "Box Plot":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return None, "No numerical columns found for box plot"
            
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    axes[i].boxplot(df[col].dropna())
                    axes[i].set_title(f'Box Plot of {col}')
                    axes[i].set_ylabel(col)
            
            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            return fig, f"Generated box plots for {len(numeric_cols)} numerical columns"
        
        elif viz_type == "Scatter Plot":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                return None, "Need at least 2 numerical columns for scatter plot"
            
            # Create scatter plots for the first few pairs of columns
            n_pairs = min(6, len(numeric_cols) * (len(numeric_cols) - 1) // 2)
            n_cols = min(3, n_pairs)
            n_rows = (n_pairs + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            plot_idx = 0
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j and plot_idx < len(axes) and plot_idx < n_pairs:
                        # Remove rows with NaN in either column
                        clean_data = df[[col1, col2]].dropna()
                        if len(clean_data) > 0:
                            axes[plot_idx].scatter(clean_data[col1], clean_data[col2], alpha=0.6)
                            axes[plot_idx].set_xlabel(col1)
                            axes[plot_idx].set_ylabel(col2)
                            axes[plot_idx].set_title(f'{col1} vs {col2}')
                        plot_idx += 1
            
            # Hide empty subplots
            for i in range(plot_idx, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            return fig, f"Generated scatter plots for {plot_idx} column pairs"
        
        elif viz_type == "Pair Plot":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                return None, "Need at least 2 numerical columns for pair plot"
            
            # Limit to first 5 columns to avoid overcrowding
            plot_cols = numeric_cols[:5]
            
            try:
                # Create pair plot using seaborn
                fig = plt.figure(figsize=(12, 10))
                
                # Create a subset of data for pair plot
                plot_data = df[plot_cols].dropna()
                
                if len(plot_data) == 0:
                    return None, "No data available after removing missing values"
                
                # Use seaborn pairplot
                import seaborn as sns
                pair_plot = sns.pairplot(plot_data, diag_kind='hist', plot_kws={'alpha': 0.6})
                pair_plot.fig.suptitle('Pair Plot of Numerical Variables', y=1.02)
                
                return pair_plot.fig, f"Generated pair plot for {len(plot_cols)} numerical columns"
                
            except Exception as e:
                return None, f"Error creating pair plot: {str(e)}"
        
        else:
            return None, f"Visualization type '{viz_type}' not yet implemented"
            
    except Exception as e:
        return None, f"Error generating visualization: {str(e)}"

def generate_correlation_analysis(current_state, selected_columns=None):
    """Generate detailed correlation analysis"""
    if current_state is None:
        return None, "No dataset available. Please upload and clean data first."
    
    try:
        df = current_state
        
        # Filter columns if specified
        if selected_columns:
            df = df[selected_columns]
        
        # Get only numeric columns for correlation analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            return None, "Need at least 2 numerical columns for correlation analysis"
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Generate correlation analysis text
        analysis_parts = []
        analysis_parts.append("=== CORRELATION ANALYSIS ===")
        analysis_parts.append(f"Analyzed {len(numeric_cols)} numerical columns")
        analysis_parts.append("")
        
        # Find strong correlations (> 0.7 or < -0.7)
        strong_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append((numeric_cols[i], numeric_cols[j], corr_val))
        
        if strong_correlations:
            analysis_parts.append("=== STRONG CORRELATIONS (|r| > 0.7) ===")
            for col1, col2, corr in strong_correlations:
                direction = "positive" if corr > 0 else "negative"
                analysis_parts.append(f"{col1} ↔ {col2}: {corr:.3f} ({direction})")
        else:
            analysis_parts.append("=== STRONG CORRELATIONS ===")
            analysis_parts.append("No strong correlations (|r| > 0.7) found")
        
        analysis_parts.append("")
        
        # Find moderate correlations (0.3 to 0.7)
        moderate_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if 0.3 <= abs(corr_val) <= 0.7:
                    moderate_correlations.append((numeric_cols[i], numeric_cols[j], corr_val))
        
        if moderate_correlations:
            analysis_parts.append("=== MODERATE CORRELATIONS (0.3 ≤ |r| ≤ 0.7) ===")
            # Show top 10 moderate correlations
            moderate_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            for col1, col2, corr in moderate_correlations[:10]:
                direction = "positive" if corr > 0 else "negative"
                analysis_parts.append(f"{col1} ↔ {col2}: {corr:.3f} ({direction})")
        
        analysis_parts.append("")
        
        # Correlation matrix summary
        analysis_parts.append("=== CORRELATION MATRIX SUMMARY ===")
        analysis_parts.append(corr_matrix.round(3).to_string())
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, ax=ax, fmt='.3f')
        ax.set_title('Correlation Matrix Heatmap')
        plt.tight_layout()
        
        analysis_text = "\n".join(analysis_parts)
        return fig, analysis_text
        
    except Exception as e:
        return None, f"Error generating correlation analysis: {str(e)}"

def generate_distribution_analysis(current_state, selected_columns=None):
    """Generate detailed distribution analysis"""
    if current_state is None:
        return None, "No dataset available. Please upload and clean data first."
    
    try:
        df = current_state
        
        # Filter columns if specified
        if selected_columns:
            df = df[selected_columns]
        
        analysis_parts = []
        analysis_parts.append("=== DISTRIBUTION ANALYSIS ===")
        analysis_parts.append("")
        
        # Analyze numerical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            analysis_parts.append("=== NUMERICAL DISTRIBUTIONS ===")
            
            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                
                # Basic statistics
                mean_val = series.mean()
                median_val = series.median()
                std_val = series.std()
                skewness = series.skew()
                kurtosis = series.kurtosis()
                
                analysis_parts.append(f"\n{col}:")
                analysis_parts.append(f"  Mean: {mean_val:.3f}, Median: {median_val:.3f}")
                analysis_parts.append(f"  Std Dev: {std_val:.3f}")
                analysis_parts.append(f"  Skewness: {skewness:.3f} ({'right-skewed' if skewness > 0.5 else 'left-skewed' if skewness < -0.5 else 'approximately normal'})")
                analysis_parts.append(f"  Kurtosis: {kurtosis:.3f} ({'heavy-tailed' if kurtosis > 0 else 'light-tailed'})")
                
                # Outlier detection using IQR method
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                analysis_parts.append(f"  Outliers (IQR method): {len(outliers)} ({len(outliers)/len(series)*100:.1f}%)")
        
        # Analyze categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            analysis_parts.append("\n=== CATEGORICAL DISTRIBUTIONS ===")
            
            for col in cat_cols:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                
                value_counts = series.value_counts()
                unique_count = len(value_counts)
                
                analysis_parts.append(f"\n{col}:")
                analysis_parts.append(f"  Unique values: {unique_count}")
                analysis_parts.append(f"  Most frequent: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences, {value_counts.iloc[0]/len(series)*100:.1f}%)")
                
                if unique_count <= 10:
                    analysis_parts.append("  Distribution:")
                    for value, count in value_counts.items():
                        percentage = count / len(series) * 100
                        analysis_parts.append(f"    {value}: {count} ({percentage:.1f}%)")
                else:
                    analysis_parts.append(f"  Top 5 values:")
                    for value, count in value_counts.head().items():
                        percentage = count / len(series) * 100
                        analysis_parts.append(f"    {value}: {count} ({percentage:.1f}%)")
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_count = 0
        
        # Plot histograms for numerical columns (up to 4)
        for col in numeric_cols[:4]:
            if plot_count < 4:
                series = df[col].dropna()
                axes[plot_count].hist(series, bins=30, alpha=0.7, edgecolor='black')
                axes[plot_count].axvline(series.mean(), color='red', linestyle='--', label=f'Mean: {series.mean():.2f}')
                axes[plot_count].axvline(series.median(), color='green', linestyle='--', label=f'Median: {series.median():.2f}')
                axes[plot_count].set_title(f'Distribution of {col}')
                axes[plot_count].set_xlabel(col)
                axes[plot_count].set_ylabel('Frequency')
                axes[plot_count].legend()
                plot_count += 1
        
        # Hide unused subplots
        for i in range(plot_count, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        analysis_text = "\n".join(analysis_parts)
        return fig, analysis_text
        
    except Exception as e:
        return None, f"Error generating distribution analysis: {str(e)}"

def get_ai_insights(current_state, selected_columns=None):
    """Get AI-powered insights about the dataset"""
    if current_state is None:
        return "No dataset available. Please upload and clean data first."
    
    if not global_ai_manager.is_available():
        return "AI not available. Please configure Gemini AI API key."
    
    try:
        df = current_state
        
        # Filter columns if specified
        if selected_columns:
            df = df[selected_columns]
        
        # Prepare dataset summary for AI
        dataset_info = global_ai_manager._prepare_dataset_info(df)
        
        prompt = f"""
You are a data analysis expert. Analyze the following dataset and provide insights:

{dataset_info}

Please provide:
1. Key patterns and trends you observe
2. Potential data quality issues
3. Interesting relationships between variables
4. Recommendations for further analysis
5. Suggestions for feature engineering
6. Any anomalies or outliers you notice

Format your response in a clear, structured way with bullet points and sections.
"""
        
        response = global_ai_manager.model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error getting AI insights: {str(e)}"

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def create_new_feature(current_state, source_cols, operation_type, formula, new_name):
    """Create a new feature based on existing columns"""
    if current_state is None:
        # Try to get from global state manager as fallback
        if global_state_manager.has_data():
            current_state = global_state_manager.get_dataframe()
        else:
            return "No dataset available", None, None
    
    if not formula.strip() or not new_name.strip():
        return "Please provide both formula and feature name", None, None
    
    try:
        df = current_state.copy()
        
        # Check if feature name already exists
        if new_name in df.columns:
            return f"❌ Feature '{new_name}' already exists. Please choose a different name.", None, None
        
        # Validate source columns exist
        if source_cols:
            missing_cols = [col for col in source_cols if col not in df.columns]
            if missing_cols:
                return f"❌ Source columns not found: {missing_cols}", None, None
        
        # Create safe execution environment for feature creation
        safe_globals = {
            'pd': pd,
            'np': np,
            'df': df,
            'math': __import__('math'),
            'datetime': __import__('datetime'),
            '__builtins__': {
                'len': len, 'str': str, 'int': int, 'float': float,
                'min': min, 'max': max, 'sum': sum, 'abs': abs,
                'round': round, 'pow': pow, 'sqrt': lambda x: x**0.5
            }
        }
        
        # Pre-built operation templates based on operation type
        if operation_type == "Mathematical Operation" and source_cols and len(source_cols) >= 2:
            # Suggest common mathematical operations
            if "+" in formula or "sum" in formula.lower():
                pass  # User provided custom formula
            elif not any(op in formula for op in ['+', '-', '*', '/', '(', ')']):
                # If no operators, suggest addition
                formula = f"df['{source_cols[0]}'] + df['{source_cols[1]}']"
        
        elif operation_type == "String Operation" and source_cols:
            # Handle string operations
            if not formula.strip():
                # Default string operation - length
                formula = f"df['{source_cols[0]}'].astype(str).str.len()"
            else:
                # Common string operations
                if "length" in formula.lower() or "len" in formula.lower():
                    formula = f"df['{source_cols[0]}'].astype(str).str.len()"
                elif "upper" in formula.lower():
                    formula = f"df['{source_cols[0]}'].astype(str).str.upper()"
                elif "lower" in formula.lower():
                    formula = f"df['{source_cols[0]}'].astype(str).str.lower()"
                elif "contains" in formula.lower():
                    search_term = formula.split("contains")[-1].strip().strip("'\"")
                    formula = f"df['{source_cols[0]}'].astype(str).str.contains('{search_term}', na=False)"
        
        elif operation_type == "Date/Time Operation" and source_cols:
            # Handle date/time operations
            if not formula.strip():
                # Default date operation - extract year
                formula = f"pd.to_datetime(df['{source_cols[0]}'], errors='coerce').dt.year"
            else:
                col = source_cols[0]
                if "year" in formula.lower():
                    formula = f"pd.to_datetime(df['{col}'], errors='coerce').dt.year"
                elif "month" in formula.lower():
                    formula = f"pd.to_datetime(df['{col}'], errors='coerce').dt.month"
                elif "day" in formula.lower():
                    formula = f"pd.to_datetime(df['{col}'], errors='coerce').dt.day"
                elif "weekday" in formula.lower():
                    formula = f"pd.to_datetime(df['{col}'], errors='coerce').dt.weekday"
                elif "quarter" in formula.lower():
                    formula = f"pd.to_datetime(df['{col}'], errors='coerce').dt.quarter"
        
        elif operation_type == "Binning/Categorization" and source_cols:
            # Handle binning operations
            if not formula.strip():
                # Default binning - 5 equal-width bins
                formula = f"pd.cut(df['{source_cols[0]}'], bins=5, labels=['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])"
            else:
                col = source_cols[0]
                if "quantile" in formula.lower():
                    formula = f"pd.qcut(df['{col}'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])"
                elif "bins=" in formula.lower():
                    # User specified number of bins
                    pass  # Keep user formula
                else:
                    # Default equal-width binning
                    formula = f"pd.cut(df['{col}'], bins=5, labels=['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])"
        
        # Execute the formula
        exec(f"df['{new_name}'] = {formula}", safe_globals)
        result_df = safe_globals['df']
        
        # Validate the new feature
        new_feature = result_df[new_name]
        
        # Get statistics about the new feature
        feature_stats = {}
        if pd.api.types.is_numeric_dtype(new_feature):
            feature_stats = {
                'type': 'numeric',
                'mean': new_feature.mean(),
                'std': new_feature.std(),
                'min': new_feature.min(),
                'max': new_feature.max(),
                'null_count': new_feature.isnull().sum()
            }
        else:
            feature_stats = {
                'type': 'categorical',
                'unique_values': new_feature.nunique(),
                'most_common': new_feature.value_counts().head(3).to_dict(),
                'null_count': new_feature.isnull().sum()
            }
        
        # Get preview of the new feature with some context columns
        context_cols = source_cols[:3] if source_cols else df.columns[:3].tolist()
        preview_cols = [col for col in context_cols if col in df.columns] + [new_name]
        preview_df = result_df[preview_cols].head(10)
        
        success_msg = f"✅ Successfully created feature '{new_name}'"
        success_msg += f"\nFormula: {formula}"
        success_msg += f"\nFeature type: {feature_stats['type']}"
        if feature_stats['type'] == 'numeric':
            success_msg += f"\nStatistics: mean={feature_stats['mean']:.3f}, std={feature_stats['std']:.3f}"
        else:
            success_msg += f"\nUnique values: {feature_stats['unique_values']}"
        success_msg += f"\nNull values: {feature_stats['null_count']}"
        success_msg += f"\nNew dataset shape: {result_df.shape}"
        
        return success_msg, preview_df, result_df
        
    except Exception as e:
        return f"❌ Error creating feature: {str(e)}", None, None

def select_features(current_state, method, target_col, num_features):
    """Select important features using various methods"""
    if current_state is None:
        # Try to get from global state manager as fallback
        if global_state_manager.has_data():
            current_state = global_state_manager.get_dataframe()
        else:
            return "No dataset available", None, None
    
    try:
        df = current_state.copy()
        
        if method == "Correlation-based":
            # Select features based on correlation with target
            if not target_col or target_col not in df.columns:
                return "Please select a valid target column", None, None
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if target_col not in numeric_cols:
                return "Target column must be numerical for correlation-based selection", None, None
            
            # Calculate correlations
            correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
            
            # Select top features (excluding target itself)
            selected_features = correlations.drop(target_col).head(num_features).index.tolist()
            
            result_msg = f"✅ Selected {len(selected_features)} features based on correlation with '{target_col}'\n"
            result_msg += "\nSelected features (by correlation):"  
            for i, feat in enumerate(selected_features, 1):
                corr_val = correlations[feat]
                result_msg += f"\n{i}. {feat} (correlation: {corr_val:.3f})"
            
            # Create preview with selected features
            preview_df = df[selected_features + [target_col]].head(10)
            selected_df = df[selected_features + [target_col]]
            
            return result_msg, preview_df, selected_df
        
        elif method == "Variance Threshold":
            # Select features with variance above threshold
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return "No numerical columns found for variance-based selection", None, None
            
            # Calculate variances
            variances = df[numeric_cols].var().sort_values(ascending=False)
            selected_features = variances.head(num_features).index.tolist()
            
            result_msg = f"✅ Selected {len(selected_features)} features with highest variance\n"
            result_msg += "\nSelected features (by variance):"
            for i, feat in enumerate(selected_features, 1):
                var_val = variances[feat]
                result_msg += f"\n{i}. {feat} (variance: {var_val:.3f})"
            
            preview_df = df[selected_features].head(10)
            selected_df = df[selected_features]
            
            return result_msg, preview_df, selected_df
        
        elif method == "Univariate Selection":
            # Select features using univariate statistical tests
            if not target_col or target_col not in df.columns:
                return "Please select a valid target column for univariate selection", None, None
            
            from sklearn.feature_selection import SelectKBest, f_classif, f_regression
            from sklearn.preprocessing import LabelEncoder
            
            # Prepare features (only numeric columns)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            if len(numeric_cols) == 0:
                return "No numerical features found for univariate selection", None, None
            
            X = df[numeric_cols].fillna(df[numeric_cols].mean())
            y = df[target_col].fillna(df[target_col].mean() if pd.api.types.is_numeric_dtype(df[target_col]) else 'missing')
            
            # Choose appropriate test based on target type
            if pd.api.types.is_numeric_dtype(df[target_col]):
                # Regression task
                selector = SelectKBest(score_func=f_regression, k=min(num_features, len(numeric_cols)))
                test_name = "F-regression"
            else:
                # Classification task
                le = LabelEncoder()
                y = le.fit_transform(y.fillna('missing'))
                selector = SelectKBest(score_func=f_classif, k=min(num_features, len(numeric_cols)))
                test_name = "F-classification"
            
            X_selected = selector.fit_transform(X, y)
            selected_features = [numeric_cols[i] for i in selector.get_support(indices=True)]
            scores = selector.scores_[selector.get_support()]
            
            result_msg = f"✅ Selected {len(selected_features)} features using {test_name}\n"
            result_msg += f"\nSelected features (by {test_name} score):"
            for i, (feat, score) in enumerate(zip(selected_features, scores), 1):
                result_msg += f"\n{i}. {feat} (score: {score:.3f})"
            
            preview_df = df[selected_features + [target_col]].head(10)
            selected_df = df[selected_features + [target_col]]
            
            return result_msg, preview_df, selected_df
        
        elif method == "Recursive Feature Elimination":
            # RFE with a simple estimator
            if not target_col or target_col not in df.columns:
                return "Please select a valid target column for RFE", None, None
            
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Prepare features (only numeric columns)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            if len(numeric_cols) == 0:
                return "No numerical features found for RFE", None, None
            
            X = df[numeric_cols].fillna(df[numeric_cols].mean())
            y = df[target_col]
            
            # Choose appropriate estimator based on target type
            if pd.api.types.is_numeric_dtype(y):
                # Handle NaN values in target for regression
                y = y.fillna(y.mean())
                estimator = RandomForestRegressor(n_estimators=10, random_state=42)
                task_type = "regression"
            else:
                le = LabelEncoder()
                y = le.fit_transform(y.fillna('missing'))
                estimator = RandomForestClassifier(n_estimators=10, random_state=42)
                task_type = "classification"
            
            rfe = RFE(estimator=estimator, n_features_to_select=min(num_features, len(numeric_cols)))
            rfe.fit(X, y)
            
            selected_features = [numeric_cols[i] for i in range(len(numeric_cols)) if rfe.support_[i]]
            rankings = [rfe.ranking_[i] for i in range(len(numeric_cols)) if rfe.support_[i]]
            
            result_msg = f"✅ Selected {len(selected_features)} features using RFE ({task_type})\n"
            result_msg += f"\nSelected features (by RFE ranking):"
            for i, (feat, rank) in enumerate(zip(selected_features, rankings), 1):
                result_msg += f"\n{i}. {feat} (rank: {rank})"
            
            preview_df = df[selected_features + [target_col]].head(10)
            selected_df = df[selected_features + [target_col]]
            
            return result_msg, preview_df, selected_df
        
        elif method == "AI-Recommended":
            # AI-powered feature selection
            if not global_ai_manager.is_available():
                return "AI not available. Please configure Gemini AI API key.", None, None
            
            # Get dataset info for AI analysis
            dataset_info = global_ai_manager._prepare_dataset_info(df)
            target_info = f"\nTarget column: {target_col}" if target_col else "\nNo target column specified"
            
            prompt = f"""
You are a feature selection expert. Analyze this dataset and recommend the most important features:

{dataset_info}{target_info}

Based on the data characteristics, please:
1. Identify the {num_features} most important features for analysis
2. Consider correlation, variance, and domain relevance
3. Explain why each feature is important
4. Rank them by importance

Respond with a JSON-like format:
{{
    "selected_features": ["feature1", "feature2", ...],
    "explanations": {{
        "feature1": "explanation for feature1",
        "feature2": "explanation for feature2"
    }}
}}
"""
            
            try:
                response = global_ai_manager.model.generate_content(prompt)
                ai_text = response.text
                
                # Extract feature names from AI response (simple parsing)
                import re
                feature_pattern = r'"([^"]+)"'
                potential_features = re.findall(feature_pattern, ai_text)
                
                # Filter to only existing columns
                selected_features = [f for f in potential_features if f in df.columns][:num_features]
                
                if not selected_features:
                    # Fallback to correlation-based if AI parsing fails
                    if target_col and target_col in df.columns:
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if target_col in numeric_cols:
                            correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
                            selected_features = correlations.drop(target_col).head(num_features).index.tolist()
                
                result_msg = f"✅ AI recommended {len(selected_features)} features\n"
                result_msg += f"\nAI Analysis:\n{ai_text[:500]}..."
                result_msg += f"\n\nSelected features: {', '.join(selected_features)}"
                
                if target_col and target_col in df.columns:
                    preview_df = df[selected_features + [target_col]].head(10)
                    selected_df = df[selected_features + [target_col]]
                else:
                    preview_df = df[selected_features].head(10)
                    selected_df = df[selected_features]
                
                return result_msg, preview_df, selected_df
                
            except Exception as e:
                return f"❌ Error in AI feature selection: {str(e)}", None, None
        
        else:
            return f"Feature selection method '{method}' not yet implemented", None, None
            
    except Exception as e:
        return f"❌ Error in feature selection: {str(e)}", None, None

def transform_features(current_state, transform_type, selected_columns):
    """Apply transformations to selected features"""
    if current_state is None:
        # Try to get from global state manager as fallback
        if global_state_manager.has_data():
            current_state = global_state_manager.get_dataframe()
        else:
            return "No dataset available", None, None
    
    if not selected_columns:
        return "Please select columns to transform", None, None
    
    try:
        df = current_state.copy()
        
        # Validate selected columns exist
        missing_cols = [col for col in selected_columns if col not in df.columns]
        if missing_cols:
            return f"❌ Columns not found: {missing_cols}", None, None
        
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
        import pandas as pd
        
        transformed_cols = []
        
        if transform_type == "Standard Scaling":
            # Apply standard scaling to numeric columns
            numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
            if not numeric_cols:
                return "No numeric columns selected for scaling", None, None
            
            scaler = StandardScaler()
            for col in numeric_cols:
                new_col_name = f"{col}_scaled"
                df[new_col_name] = scaler.fit_transform(df[[col]].fillna(df[col].mean()))
                transformed_cols.append(new_col_name)
            
            result_msg = f"✅ Applied standard scaling to {len(numeric_cols)} columns"
            
        elif transform_type == "Min-Max Scaling":
            # Apply min-max scaling to numeric columns
            numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
            if not numeric_cols:
                return "No numeric columns selected for scaling", None, None
            
            scaler = MinMaxScaler()
            for col in numeric_cols:
                new_col_name = f"{col}_minmax"
                df[new_col_name] = scaler.fit_transform(df[[col]].fillna(df[col].mean()))
                transformed_cols.append(new_col_name)
            
            result_msg = f"✅ Applied min-max scaling to {len(numeric_cols)} columns"
            
        elif transform_type == "Log Transform":
            # Apply log transformation to numeric columns
            numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
            if not numeric_cols:
                return "No numeric columns selected for log transform", None, None
            
            for col in numeric_cols:
                # Handle negative values by adding offset
                min_val = df[col].min()
                offset = abs(min_val) + 1 if min_val <= 0 else 0
                new_col_name = f"{col}_log"
                df[new_col_name] = np.log(df[col].fillna(df[col].mean()) + offset)
                transformed_cols.append(new_col_name)
            
            result_msg = f"✅ Applied log transformation to {len(numeric_cols)} columns"
            
        elif transform_type == "One-Hot Encoding":
            # Apply one-hot encoding to categorical columns
            cat_cols = [col for col in selected_columns if not pd.api.types.is_numeric_dtype(df[col])]
            if not cat_cols:
                return "No categorical columns selected for one-hot encoding", None, None
            
            for col in cat_cols:
                # Create dummy variables
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                df = pd.concat([df, dummies], axis=1)
                transformed_cols.extend(dummies.columns.tolist())
            
            result_msg = f"✅ Applied one-hot encoding to {len(cat_cols)} columns"
            
        elif transform_type == "Binning":
            # Apply binning to numeric columns
            numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]
            if not numeric_cols:
                return "No numeric columns selected for binning", None, None
            
            for col in numeric_cols:
                new_col_name = f"{col}_binned"
                df[new_col_name] = pd.cut(df[col].fillna(df[col].mean()), bins=5, labels=['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])
                transformed_cols.append(new_col_name)
            
            result_msg = f"✅ Applied binning to {len(numeric_cols)} columns"
            
        else:
            return f"Transformation type '{transform_type}' not implemented", None, None
        
        # Create preview with original and transformed columns
        preview_cols = selected_columns + transformed_cols
        preview_df = df[preview_cols].head(10)
        
        result_msg += f"\nCreated {len(transformed_cols)} new features"
        result_msg += f"\nNew dataset shape: {df.shape}"
        result_msg += f"\nTransformed columns: {', '.join(transformed_cols)}"
        
        return result_msg, preview_df, df
        
    except Exception as e:
        return f"❌ Error in feature transformation: {str(e)}", None, None

def encode_categorical_features(current_state, encoding_method, selected_columns):
    """Encode categorical features using various methods"""
    if current_state is None:
        # Try to get from global state manager as fallback
        if global_state_manager.has_data():
            current_state = global_state_manager.get_dataframe()
        else:
            return "No dataset available", None, None
    
    if not selected_columns:
        return "Please select categorical columns to encode", None, None
    
    try:
        df = current_state.copy()
        
        # Filter to only categorical columns
        cat_cols = [col for col in selected_columns if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
        if not cat_cols:
            return "No categorical columns found in selection", None, None
        
        from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
        
        encoded_cols = []
        
        if encoding_method == "Label Encoding":
            for col in cat_cols:
                le = LabelEncoder()
                new_col_name = f"{col}_label_encoded"
                df[new_col_name] = le.fit_transform(df[col].fillna('missing'))
                encoded_cols.append(new_col_name)
            
            result_msg = f"✅ Applied label encoding to {len(cat_cols)} columns"
            
        elif encoding_method == "Target Encoding":
            # Target encoding (requires target column)
            # For now, we'll implement a simple mean encoding
            result_msg = "Target encoding requires implementation with a specific target column."
            
            # Get all numeric columns as potential targets
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                return "Target encoding requires at least one numeric column as target", None, None
            
            # Use the first numeric column as target for demonstration
            target_col = numeric_cols[0]
            encoded_cols = []
            
            for col in cat_cols:
                # Calculate mean of target for each category
                target_means = df.groupby(col)[target_col].mean()
                new_col_name = f"{col}_target_encoded"
                df[new_col_name] = df[col].map(target_means).fillna(df[target_col].mean())
                encoded_cols.append(new_col_name)
            
            result_msg = f"✅ Applied target encoding to {len(cat_cols)} columns using '{target_col}' as target"
            
        else:
            return f"Encoding method '{encoding_method}' not implemented", None, None
        
        # Create preview
        preview_cols = cat_cols + encoded_cols
        preview_df = df[preview_cols].head(10)
        
        result_msg += f"\nCreated {len(encoded_cols)} encoded features"
        result_msg += f"\nNew dataset shape: {df.shape}"
        
        return result_msg, preview_df, df
        
    except Exception as e:
        return f"❌ Error in categorical encoding: {str(e)}", None, None

def get_ai_feature_suggestions(current_state, target_col=None):
    """Get AI-powered feature engineering suggestions"""
    if current_state is None:
        # Try to get from global state manager as fallback
        if global_state_manager.has_data():
            current_state = global_state_manager.get_dataframe()
        else:
            return "No dataset available. Please upload and clean data first."
    
    if not global_ai_manager.is_available():
        return "AI not available. Please configure Gemini AI API key."
    
    try:
        df = current_state
        dataset_info = global_ai_manager._prepare_dataset_info(df)
        
        # Analyze data types and patterns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Check for potential date columns
        potential_date_cols = []
        for col in cat_cols:
            sample_values = df[col].dropna().head(5).astype(str).tolist()
            if any(len(str(val)) > 8 and ('-' in str(val) or '/' in str(val)) for val in sample_values):
                potential_date_cols.append(col)
        
        target_info = f"\nTarget column: {target_col}" if target_col else "\nNo target column specified (unsupervised analysis)"
        
        prompt = f"""
You are a feature engineering expert. Analyze this dataset and suggest specific, actionable feature engineering steps:

{dataset_info}{target_info}

Dataset Analysis:
- Numerical columns: {numeric_cols}
- Categorical columns: {cat_cols}
- Potential date columns: {potential_date_cols}

Please provide SPECIFIC, EXECUTABLE suggestions in the following categories:

1. **MATHEMATICAL FEATURES** (provide exact pandas formulas):
   - Ratios and combinations of numerical features
   - Polynomial features (squares, interactions)
   - Statistical aggregations

2. **TRANSFORMATION FEATURES**:
   - Log, sqrt, or other mathematical transformations
   - Scaling recommendations (StandardScaler, MinMaxScaler)
   - Normalization techniques

3. **CATEGORICAL ENCODING**:
   - One-hot encoding recommendations
   - Label encoding suggestions
   - Target encoding opportunities

4. **DATE/TIME FEATURES** (if applicable):
   - Extract year, month, day, weekday
   - Calculate time differences
   - Create seasonal features

5. **STRING FEATURES** (if applicable):
   - Text length, word count
   - Extract patterns or keywords
   - Create binary flags

6. **BINNING AND DISCRETIZATION**:
   - Convert continuous to categorical
   - Create meaningful ranges
   - Quantile-based binning

For each suggestion, provide:
- Exact feature name
- Complete pandas/numpy formula
- Business rationale
- Expected benefit

Format as numbered, actionable steps with code examples.
"""
        
        response = global_ai_manager.model.generate_content(prompt)
        
        # Add some automatic suggestions based on data analysis
        auto_suggestions = []
        
        # Suggest mathematical combinations for numeric columns
        if len(numeric_cols) >= 2:
            auto_suggestions.append(f"\n=== AUTOMATIC SUGGESTIONS ===")
            auto_suggestions.append(f"Based on your {len(numeric_cols)} numerical columns, consider:")
            for i, col1 in enumerate(numeric_cols[:3]):
                for col2 in numeric_cols[i+1:4]:
                    auto_suggestions.append(f"- Ratio: {col1}_to_{col2}_ratio = df['{col1}'] / (df['{col2}'] + 1)")
                    auto_suggestions.append(f"- Product: {col1}_{col2}_product = df['{col1}'] * df['{col2}']")
        
        # Suggest encoding for categorical columns
        if cat_cols:
            auto_suggestions.append(f"\nFor your {len(cat_cols)} categorical columns:")
            for col in cat_cols[:3]:
                unique_count = df[col].nunique()
                if unique_count <= 10:
                    auto_suggestions.append(f"- One-hot encode '{col}' (only {unique_count} unique values)")
                else:
                    auto_suggestions.append(f"- Label encode '{col}' ({unique_count} unique values)")
        
        combined_response = response.text + "\n" + "\n".join(auto_suggestions)
        return combined_response
        
    except Exception as e:
        return f"Error getting AI feature suggestions: {str(e)}"

def auto_feature_engineering(current_state, target_col=None):
    """Automatically create common feature engineering transformations"""
    if current_state is None:
        # Try to get from global state manager as fallback
        if global_state_manager.has_data():
            current_state = global_state_manager.get_dataframe()
        else:
            return "No dataset available", None, None
    
    try:
        df = current_state.copy()
        created_features = []
        
        # Get column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. Create mathematical combinations for numeric columns
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols[:3]):  # Limit to avoid too many features
                for col2 in numeric_cols[i+1:3]:
                    # Ratio features
                    ratio_name = f"{col1}_to_{col2}_ratio"
                    df[ratio_name] = df[col1] / (df[col2] + 1e-8)  # Add small value to avoid division by zero
                    created_features.append(ratio_name)
                    
                    # Sum features
                    sum_name = f"{col1}_{col2}_sum"
                    df[sum_name] = df[col1] + df[col2]
                    created_features.append(sum_name)
        
        # 2. Create polynomial features (squares) for numeric columns
        for col in numeric_cols[:3]:  # Limit to first 3 columns
            square_name = f"{col}_squared"
            df[square_name] = df[col] ** 2
            created_features.append(square_name)
        
        # 3. Create log transformations for positive numeric columns
        for col in numeric_cols[:3]:
            if df[col].min() > 0:  # Only for positive values
                log_name = f"{col}_log"
                df[log_name] = np.log(df[col] + 1)
                created_features.append(log_name)
        
        # 4. Create binned versions of numeric columns
        for col in numeric_cols[:2]:  # Limit to first 2 columns
            binned_name = f"{col}_binned"
            df[binned_name] = pd.cut(df[col], bins=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
            created_features.append(binned_name)
        
        # 5. Create frequency encoding for categorical columns
        for col in cat_cols[:2]:  # Limit to first 2 columns
            freq_name = f"{col}_frequency"
            freq_map = df[col].value_counts().to_dict()
            df[freq_name] = df[col].map(freq_map)
            created_features.append(freq_name)
        
        # 6. Create target-based statistics if target column is provided
        if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            for col in cat_cols[:2]:
                target_mean_name = f"{col}_target_mean"
                target_means = df.groupby(col)[target_col].mean()
                df[target_mean_name] = df[col].map(target_means).fillna(df[target_col].mean())
                created_features.append(target_mean_name)
        
        result_msg = f"✅ Automatically created {len(created_features)} new features:\n"
        result_msg += "\n".join([f"- {feat}" for feat in created_features])
        result_msg += f"\n\nNew dataset shape: {df.shape}"
        
        # Create preview with some of the new features
        preview_cols = df.columns[-min(5, len(created_features)):].tolist()
        preview_df = df[preview_cols].head(10)
        
        return result_msg, preview_df, df
        
    except Exception as e:
        return f"❌ Error in automatic feature engineering: {str(e)}", None, None

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
            
            # Tab 3: Exploratory Data Analysis
            with gr.TabItem("📊 Exploratory Analysis"):
                gr.Markdown("## Exploratory Data Analysis")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Analysis Options")
                        
                        # Analysis type selection
                        analysis_type = gr.Dropdown(
                            choices=[
                                "Statistical Summary",
                                "Data Visualization", 
                                "Correlation Analysis",
                                "Distribution Analysis",
                                "AI-Powered Insights"
                            ],
                            value="Statistical Summary",
                            label="Analysis Type"
                        )
                        
                        # Column selection for analysis
                        column_selector = gr.Dropdown(
                            choices=[],
                            multiselect=True,
                            label="Select Columns (optional)",
                            info="Leave empty to analyze all columns"
                        )
                        
                        # Visualization type (shown when Data Visualization is selected)
                        viz_type = gr.Dropdown(
                            choices=[
                                "Histogram",
                                "Box Plot", 
                                "Scatter Plot",
                                "Correlation Heatmap",
                                "Pair Plot"
                            ],
                            value="Histogram",
                            label="Visualization Type",
                            visible=False
                        )
                        
                        # Generate analysis button
                        generate_analysis_btn = gr.Button(
                            "🔍 Generate Analysis", 
                            variant="primary"
                        )
                        
                        # AI insights button
                        get_insights_btn = gr.Button(
                            "🤖 Get AI Insights",
                            variant="secondary"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### Analysis Results")
                        
                        # Analysis output
                        analysis_output = gr.Textbox(
                            label="Statistical Analysis",
                            lines=15,
                            max_lines=20,
                            interactive=False
                        )
                        
                        # Visualization output
                        analysis_plot = gr.Plot(
                            label="Visualization",
                            visible=False
                        )
                        
                        # AI insights output
                        ai_insights = gr.Textbox(
                            label="AI-Powered Insights",
                            lines=10,
                            max_lines=15,
                            interactive=False,
                            visible=False
                        )
            
            # Tab 4: Feature Engineering
            with gr.TabItem("⚙️ Feature Engineering"):
                gr.Markdown("## Feature Engineering")
                gr.Markdown("💡 **Tip**: If you see column errors, click 'Refresh Columns' to update the available options.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Feature Engineering Options")
                        
                        # Feature engineering type
                        fe_type = gr.Dropdown(
                            choices=[
                                "Create New Features",
                                "Feature Selection",
                                "Feature Transformation",
                                "Encoding Categorical Variables",
                                "Automatic Feature Engineering",
                                "AI-Powered Feature Engineering"
                            ],
                            value="Create New Features",
                            label="Engineering Type"
                        )
                        
                        # Debug and refresh buttons
                        with gr.Row():
                            debug_state_btn = gr.Button("🔍 Check Dataset Status", variant="secondary", size="sm")
                            refresh_columns_btn = gr.Button("🔄 Refresh Columns", variant="secondary", size="sm")
                        debug_output = gr.Textbox(label="Dataset Status", lines=2, interactive=False)
                        
                        # Feature creation options
                        with gr.Group(visible=True) as feature_creation_group:
                            gr.Markdown("**Feature Creation**")
                            source_columns = gr.Dropdown(
                                choices=[],
                                multiselect=True,
                                label="Source Columns"
                            )
                            
                            operation_type = gr.Dropdown(
                                choices=[
                                    "Mathematical Operation",
                                    "String Operation",
                                    "Date/Time Operation",
                                    "Binning/Categorization",
                                    "Custom Formula"
                                ],
                                value="Mathematical Operation",
                                label="Operation Type"
                            )
                            
                            feature_formula = gr.Textbox(
                                label="Feature Formula",
                                placeholder="e.g., col1 + col2, col1.str.len(), pd.cut(col1, bins=5)",
                                lines=2
                            )
                            
                            new_feature_name = gr.Textbox(
                                label="New Feature Name",
                                placeholder="Enter name for new feature"
                            )
                        
                        # Feature selection options
                        with gr.Group(visible=False) as feature_selection_group:
                            gr.Markdown("**Feature Selection**")
                            selection_method = gr.Dropdown(
                                choices=[
                                    "Correlation-based",
                                    "Variance Threshold",
                                    "Univariate Selection",
                                    "Recursive Feature Elimination",
                                    "AI-Recommended"
                                ],
                                value="Correlation-based",
                                label="Selection Method"
                            )
                            
                            target_column = gr.Dropdown(
                                choices=[],
                                label="Target Column (for supervised selection)"
                            )
                            
                            num_features = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                label="Number of Features to Select"
                            )
                        
                        # Feature transformation options
                        with gr.Group(visible=False) as feature_transformation_group:
                            gr.Markdown("**Feature Transformation**")
                            transform_columns = gr.Dropdown(
                                choices=[],
                                multiselect=True,
                                label="Columns to Transform"
                            )
                            
                            transform_method = gr.Dropdown(
                                choices=[
                                    "Standard Scaling",
                                    "Min-Max Scaling",
                                    "Log Transform",
                                    "One-Hot Encoding",
                                    "Binning"
                                ],
                                value="Standard Scaling",
                                label="Transformation Method"
                            )
                        
                        # Categorical encoding options
                        with gr.Group(visible=False) as categorical_encoding_group:
                            gr.Markdown("**Categorical Encoding**")
                            encoding_columns = gr.Dropdown(
                                choices=[],
                                multiselect=True,
                                label="Categorical Columns to Encode"
                            )
                            
                            encoding_method = gr.Dropdown(
                                choices=[
                                    "Label Encoding",
                                    "One-Hot Encoding",
                                    "Target Encoding"
                                ],
                                value="Label Encoding",
                                label="Encoding Method"
                            )
                        
                        # Generate features button
                        generate_features_btn = gr.Button(
                            "🔧 Generate Features",
                            variant="primary"
                        )
                        
                        # Automatic feature engineering options
                        with gr.Group(visible=False) as automatic_fe_group:
                            gr.Markdown("**Automatic Feature Engineering**")
                            gr.Markdown("Automatically creates common feature transformations:")
                            gr.Markdown("• Mathematical combinations (ratios, sums, products)")
                            gr.Markdown("• Polynomial features (squares)")
                            gr.Markdown("• Log transformations")
                            gr.Markdown("• Binning of numerical features")
                            gr.Markdown("• Frequency encoding of categories")
                            
                            auto_target_col = gr.Dropdown(
                                choices=[],
                                label="Target Column (optional, for target-based features)"
                            )
                        
                        # AI feature suggestions button
                        get_feature_suggestions_btn = gr.Button(
                            "🤖 Get AI Feature Suggestions",
                            variant="secondary"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### Feature Engineering Results")
                        
                        # Feature engineering output
                        fe_output = gr.Textbox(
                            label="Feature Engineering Results",
                            lines=10,
                            max_lines=15,
                            interactive=False
                        )
                        
                        # Feature preview
                        feature_preview = gr.Dataframe(
                            label="Feature Preview",
                            interactive=False
                        )
                        
                        # AI feature suggestions
                        ai_feature_suggestions = gr.Textbox(
                            label="AI Feature Suggestions",
                            lines=8,
                            max_lines=12,
                            interactive=False,
                            visible=False
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
        
        # ============================================================================
        # EXPLORATORY DATA ANALYSIS EVENT HANDLERS
        # ============================================================================
        
        # Update column choices when analysis type changes
        def update_eda_interface(analysis_type, current_state):
            if current_state is None:
                return gr.update(choices=[]), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            df = current_state
            column_choices = df.columns.tolist()
            
            # Show/hide visualization options
            viz_visible = analysis_type == "Data Visualization"
            plot_visible = viz_visible
            insights_visible = analysis_type == "AI-Powered Insights"
            
            return (
                gr.update(choices=column_choices),
                gr.update(visible=viz_visible),
                gr.update(visible=plot_visible),
                gr.update(visible=insights_visible)
            )
        
        analysis_type.change(
            fn=update_eda_interface,
            inputs=[analysis_type, state_data],
            outputs=[column_selector, viz_type, analysis_plot, ai_insights]
        )
        
        # Generate analysis
        def perform_analysis(analysis_type, selected_columns, viz_type, current_state):
            if analysis_type == "Statistical Summary":
                result = generate_statistical_summary(current_state, selected_columns)
                return result, gr.update(visible=False), gr.update(visible=False)
            
            elif analysis_type == "Data Visualization":
                plot, message = generate_data_visualization(current_state, viz_type, selected_columns)
                return message, gr.update(value=plot, visible=True), gr.update(visible=False)
            
            elif analysis_type == "Correlation Analysis":
                plot, analysis_text = generate_correlation_analysis(current_state, selected_columns)
                if plot is not None:
                    return analysis_text, gr.update(value=plot, visible=True), gr.update(visible=False)
                else:
                    return analysis_text, gr.update(visible=False), gr.update(visible=False)
            
            elif analysis_type == "Distribution Analysis":
                plot, analysis_text = generate_distribution_analysis(current_state, selected_columns)
                if plot is not None:
                    return analysis_text, gr.update(value=plot, visible=True), gr.update(visible=False)
                else:
                    return analysis_text, gr.update(visible=False), gr.update(visible=False)
            
            elif analysis_type == "AI-Powered Insights":
                insights = get_ai_insights(current_state, selected_columns)
                return "AI insights generated below:", gr.update(visible=False), gr.update(value=insights, visible=True)
            
            else:
                return f"Analysis type '{analysis_type}' not yet implemented", gr.update(visible=False), gr.update(visible=False)
        
        generate_analysis_btn.click(
            fn=perform_analysis,
            inputs=[analysis_type, column_selector, viz_type, state_data],
            outputs=[analysis_output, analysis_plot, ai_insights]
        )
        
        # Get AI insights
        get_insights_btn.click(
            fn=get_ai_insights,
            inputs=[state_data, column_selector],
            outputs=[ai_insights]
        )
        
        # ============================================================================
        # FEATURE ENGINEERING EVENT HANDLERS
        # ============================================================================
        
        # Update feature engineering interface
        def update_fe_interface(fe_type, current_state):
            # Try to get from global state manager as fallback
            if current_state is None and global_state_manager.has_data():
                current_state = global_state_manager.get_dataframe()
            
            if current_state is None:
                return (
                    gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(choices=[], value=None), 
                    gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), 
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                )
            
            df = current_state
            column_choices = df.columns.tolist()
            
            # Get categorical columns for encoding
            cat_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
            
            # Show/hide different option groups
            creation_visible = fe_type == "Create New Features"
            selection_visible = fe_type == "Feature Selection"
            transformation_visible = fe_type == "Feature Transformation"
            encoding_visible = fe_type == "Encoding Categorical Variables"
            automatic_visible = fe_type == "Automatic Feature Engineering"
            suggestions_visible = fe_type == "AI-Powered Feature Engineering"
            
            return (
                gr.update(choices=column_choices, value=None),
                gr.update(choices=column_choices, value=None),
                gr.update(choices=column_choices, value=None),
                gr.update(choices=cat_columns, value=None),
                gr.update(choices=column_choices, value=None),
                gr.update(visible=creation_visible),
                gr.update(visible=selection_visible),
                gr.update(visible=transformation_visible),
                gr.update(visible=encoding_visible),
                gr.update(visible=automatic_visible),
                gr.update(visible=suggestions_visible)
            )
        
        fe_type.change(
            fn=update_fe_interface,
            inputs=[fe_type, state_data],
            outputs=[source_columns, target_column, transform_columns, encoding_columns, auto_target_col,
                    feature_creation_group, feature_selection_group, feature_transformation_group, 
                    categorical_encoding_group, automatic_fe_group, ai_feature_suggestions]
        )
        
        # Generate features
        def perform_feature_engineering(fe_type, source_cols, operation_type, formula, new_name, 
                                      selection_method, target_col, num_features, 
                                      transform_cols, transform_method, encoding_cols, encoding_method, 
                                      auto_target_col, current_state):
            
            # Get current state with fallback
            if current_state is None and global_state_manager.has_data():
                current_state = global_state_manager.get_dataframe()
            
            if current_state is None:
                return "❌ No dataset available. Please upload data first.", None, current_state
            
            # Validate that selected columns exist in the dataset
            df_columns = current_state.columns.tolist()
            
            if fe_type == "Create New Features":
                # Validate source columns
                if source_cols:
                    invalid_cols = [col for col in source_cols if col not in df_columns]
                    if invalid_cols:
                        return f"❌ Invalid columns selected: {invalid_cols}. Please refresh and select valid columns.", None, current_state
                
                result_msg, preview_df, new_state = create_new_feature(
                    current_state, source_cols, operation_type, formula, new_name
                )
                return result_msg, preview_df, new_state
            
            elif fe_type == "Feature Selection":
                # Validate target column
                if target_col and target_col not in df_columns:
                    return f"❌ Target column '{target_col}' not found. Please select a valid column.", None, current_state
                
                result_msg, preview_df, new_state = select_features(
                    current_state, selection_method, target_col, num_features
                )
                return result_msg, preview_df, new_state
            
            elif fe_type == "Feature Transformation":
                # Validate transform columns
                if transform_cols:
                    invalid_cols = [col for col in transform_cols if col not in df_columns]
                    if invalid_cols:
                        return f"❌ Invalid columns selected: {invalid_cols}. Please refresh and select valid columns.", None, current_state
                
                result_msg, preview_df, new_state = transform_features(
                    current_state, transform_method, transform_cols
                )
                return result_msg, preview_df, new_state
            
            elif fe_type == "Encoding Categorical Variables":
                # Validate encoding columns
                if encoding_cols:
                    invalid_cols = [col for col in encoding_cols if col not in df_columns]
                    if invalid_cols:
                        return f"❌ Invalid columns selected: {invalid_cols}. Please refresh and select valid columns.", None, current_state
                
                result_msg, preview_df, new_state = encode_categorical_features(
                    current_state, encoding_method, encoding_cols, target_col
                )
                return result_msg, preview_df, new_state
            
            elif fe_type == "Automatic Feature Engineering":
                result_msg, preview_df, new_state = auto_feature_engineering(
                    current_state, auto_target_col
                )
                return result_msg, preview_df, new_state
            
            else:
                return f"Feature engineering type '{fe_type}' not yet implemented", None, current_state
        
        generate_features_btn.click(
            fn=perform_feature_engineering,
            inputs=[
                fe_type, source_columns, operation_type, feature_formula, new_feature_name,
                selection_method, target_column, num_features,
                transform_columns, transform_method, encoding_columns, encoding_method,
                auto_target_col, state_data
            ],
            outputs=[fe_output, feature_preview, state_data]
        )
        
        # Debug state check
        def check_fe_state(current_state):
            """Debug function to check feature engineering state"""
            if current_state is None:
                # Try to get from global state manager as fallback
                if global_state_manager.has_data():
                    current_state = global_state_manager.get_dataframe()
                    return f"✅ Dataset found in global state! Shape: {current_state.shape}, Columns: {list(current_state.columns)}"
                else:
                    return "❌ No dataset available. Please upload data in the Upload tab first."
            else:
                return f"✅ Dataset loaded! Shape: {current_state.shape}, Columns: {list(current_state.columns)}"
        
        debug_state_btn.click(
            fn=check_fe_state,
            inputs=[state_data],
            outputs=[debug_output]
        )
        
        # Refresh columns function
        def refresh_fe_columns(current_state):
            """Refresh column choices in feature engineering dropdowns"""
            if current_state is None and global_state_manager.has_data():
                current_state = global_state_manager.get_dataframe()
            
            if current_state is None:
                return (
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    "❌ No dataset available"
                )
            
            df = current_state
            column_choices = df.columns.tolist()
            cat_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
            
            return (
                gr.update(choices=column_choices, value=None),
                gr.update(choices=column_choices, value=None),
                gr.update(choices=column_choices, value=None),
                gr.update(choices=cat_columns, value=None),
                gr.update(choices=column_choices, value=None),
                f"✅ Refreshed! Available columns: {', '.join(column_choices[:5])}{'...' if len(column_choices) > 5 else ''}"
            )
        
        refresh_columns_btn.click(
            fn=refresh_fe_columns,
            inputs=[state_data],
            outputs=[source_columns, target_column, transform_columns, encoding_columns, auto_target_col, debug_output]
        )
        
        # Get AI feature suggestions
        get_feature_suggestions_btn.click(
            fn=get_ai_feature_suggestions,
            inputs=[state_data, target_column],
            outputs=[ai_feature_suggestions]
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