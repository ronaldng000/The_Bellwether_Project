"""
State Manager for Gradio Data Analysis Platform
Manages persistent DataFrame state and metadata across all tabs
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import copy
import sys

@dataclass
class DataFrameMetadata:
    """Metadata information about the current DataFrame"""
    shape: Tuple[int, int]
    columns: List[str]
    dtypes: Dict[str, str]
    memory_usage: str
    null_counts: Dict[str, int]
    last_modified: datetime
    
    def to_summary_string(self) -> str:
        """Convert metadata to a readable summary string"""
        return f"""
Dataset Summary:
- Shape: {self.shape[0]} rows × {self.shape[1]} columns
- Memory Usage: {self.memory_usage}
- Last Modified: {self.last_modified.strftime('%Y-%m-%d %H:%M:%S')}
- Columns: {', '.join(self.columns[:5])}{'...' if len(self.columns) > 5 else ''}
- Missing Values: {sum(self.null_counts.values())} total
"""

@dataclass
class OperationLog:
    """Log entry for operations performed on the DataFrame"""
    timestamp: datetime
    tab: str
    command: str
    operation_type: str
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    
    def to_string(self) -> str:
        """Convert log entry to readable string"""
        status = "✓" if self.success else "✗"
        return f"{status} [{self.timestamp.strftime('%H:%M:%S')}] {self.tab}: {self.command}"

class StateManager:
    """
    Centralized state management for DataFrame and application state
    Ensures data consistency across all tabs and maintains operation history
    """
    
    def __init__(self):
        self._dataframe: Optional[pd.DataFrame] = None
        self._metadata: Optional[DataFrameMetadata] = None
        self._operation_history: List[OperationLog] = []
        self._dataframe_history: List[pd.DataFrame] = []
        self._max_history_size: int = 10
        
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Returns the current DataFrame
        
        Returns:
            Current DataFrame or None if no data loaded
        """
        return self._dataframe.copy() if self._dataframe is not None else None
    
    def update_dataframe(self, df: pd.DataFrame, operation_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Updates the global DataFrame and refreshes metadata
        
        Args:
            df: New DataFrame to store
            operation_info: Optional information about the operation that created this DataFrame
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Validate DataFrame
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            
            if df.empty:
                raise ValueError("DataFrame cannot be empty")
            
            # Store previous DataFrame for history
            if self._dataframe is not None:
                self._add_to_history(self._dataframe.copy())
            
            # Update DataFrame
            self._dataframe = df.copy()
            
            # Update metadata
            self._update_metadata()
            
            # Log the operation if info provided
            if operation_info:
                self.add_operation_log(
                    tab=operation_info.get('tab', 'Unknown'),
                    command=operation_info.get('command', 'DataFrame Update'),
                    operation_type=operation_info.get('operation_type', 'update'),
                    success=True
                )
            
            return True
            
        except Exception as e:
            # Log failed operation
            if operation_info:
                self.add_operation_log(
                    tab=operation_info.get('tab', 'Unknown'),
                    command=operation_info.get('command', 'DataFrame Update'),
                    operation_type=operation_info.get('operation_type', 'update'),
                    success=False,
                    error_message=str(e)
                )
            return False
    
    def get_metadata(self) -> Optional[DataFrameMetadata]:
        """
        Returns current DataFrame metadata
        
        Returns:
            DataFrameMetadata object or None if no data loaded
        """
        return self._metadata
    
    def _update_metadata(self) -> None:
        """Updates metadata based on current DataFrame"""
        if self._dataframe is None:
            self._metadata = None
            return
        
        try:
            # Calculate memory usage
            memory_usage = self._dataframe.memory_usage(deep=True).sum()
            memory_str = self._format_memory_usage(memory_usage)
            
            # Get null counts
            null_counts = self._dataframe.isnull().sum().to_dict()
            
            # Create metadata object
            self._metadata = DataFrameMetadata(
                shape=self._dataframe.shape,
                columns=list(self._dataframe.columns),
                dtypes={col: str(dtype) for col, dtype in self._dataframe.dtypes.items()},
                memory_usage=memory_str,
                null_counts=null_counts,
                last_modified=datetime.now()
            )
            
        except Exception as e:
            print(f"Error updating metadata: {e}")
            self._metadata = None
    
    def _format_memory_usage(self, bytes_size: int) -> str:
        """Format memory usage in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"
    
    def add_operation_log(self, tab: str, command: str, operation_type: str, 
                         success: bool, error_message: Optional[str] = None,
                         execution_time: float = 0.0) -> None:
        """
        Adds an operation to the history log
        
        Args:
            tab: Tab where operation was performed
            command: Command that was executed
            operation_type: Type of operation (clean, analyze, engineer, model)
            success: Whether operation was successful
            error_message: Error message if operation failed
            execution_time: Time taken to execute operation
        """
        log_entry = OperationLog(
            timestamp=datetime.now(),
            tab=tab,
            command=command,
            operation_type=operation_type,
            success=success,
            error_message=error_message,
            execution_time=execution_time
        )
        
        self._operation_history.append(log_entry)
        
        # Keep only recent history to prevent memory issues
        if len(self._operation_history) > 100:
            self._operation_history = self._operation_history[-50:]
    
    def get_operation_history(self, limit: Optional[int] = None) -> List[OperationLog]:
        """
        Returns operation history
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of OperationLog entries
        """
        history = self._operation_history.copy()
        if limit:
            history = history[-limit:]
        return history
    
    def _add_to_history(self, df: pd.DataFrame) -> None:
        """Add DataFrame to history for undo functionality"""
        self._dataframe_history.append(df)
        
        # Keep only recent history
        if len(self._dataframe_history) > self._max_history_size:
            self._dataframe_history = self._dataframe_history[-self._max_history_size:]
    
    def undo_last_operation(self) -> bool:
        """
        Undo the last operation by restoring previous DataFrame
        
        Returns:
            True if undo was successful, False otherwise
        """
        if not self._dataframe_history:
            return False
        
        try:
            # Restore previous DataFrame
            previous_df = self._dataframe_history.pop()
            self._dataframe = previous_df.copy()
            self._update_metadata()
            
            # Log the undo operation
            self.add_operation_log(
                tab="System",
                command="Undo Last Operation",
                operation_type="undo",
                success=True
            )
            
            return True
            
        except Exception as e:
            self.add_operation_log(
                tab="System",
                command="Undo Last Operation",
                operation_type="undo",
                success=False,
                error_message=str(e)
            )
            return False
    
    def has_data(self) -> bool:
        """Check if DataFrame is loaded"""
        return self._dataframe is not None and not self._dataframe.empty
    
    def get_summary_info(self) -> str:
        """Get a summary of current state"""
        if not self.has_data():
            return "No data loaded. Please upload a dataset to begin."
        
        metadata = self.get_metadata()
        if metadata:
            return metadata.to_summary_string()
        else:
            return "Data loaded but metadata unavailable."
    
    def get_recent_operations(self, count: int = 5) -> str:
        """Get recent operations as formatted string"""
        recent_ops = self.get_operation_history(limit=count)
        if not recent_ops:
            return "No operations performed yet."
        
        return "Recent Operations:\n" + "\n".join([op.to_string() for op in recent_ops])
    
    def clear_data(self) -> None:
        """Clear all data and reset state"""
        self._dataframe = None
        self._metadata = None
        self._dataframe_history.clear()
        
        self.add_operation_log(
            tab="System",
            command="Clear All Data",
            operation_type="clear",
            success=True
        )
    
    def validate_dataframe_size(self, df: pd.DataFrame, max_rows: int = 100000, 
                               max_cols: int = 1000) -> Tuple[bool, str]:
        """
        Validate DataFrame size constraints
        
        Args:
            df: DataFrame to validate
            max_rows: Maximum allowed rows
            max_cols: Maximum allowed columns
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if df.shape[0] > max_rows:
            return False, f"Dataset too large: {df.shape[0]} rows (max: {max_rows})"
        
        if df.shape[1] > max_cols:
            return False, f"Too many columns: {df.shape[1]} (max: {max_cols})"
        
        # Check memory usage
        try:
            memory_usage = df.memory_usage(deep=True).sum()
            max_memory = 500 * 1024 * 1024  # 500MB limit
            
            if memory_usage > max_memory:
                return False, f"Dataset too large: {self._format_memory_usage(memory_usage)} (max: 500MB)"
        
        except Exception:
            pass  # Skip memory check if it fails
        
        return True, ""