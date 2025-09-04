"""
Unit tests for StateManager class
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from state_manager import StateManager, DataFrameMetadata, OperationLog

class TestStateManager:
    """Test cases for StateManager class"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.state_manager = StateManager()
        self.sample_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5],
            'D': [True, False, True, False, True]
        })
    
    def test_initial_state(self):
        """Test initial state of StateManager"""
        assert self.state_manager.get_dataframe() is None
        assert self.state_manager.get_metadata() is None
        assert not self.state_manager.has_data()
        assert len(self.state_manager.get_operation_history()) == 0
    
    def test_update_dataframe_success(self):
        """Test successful DataFrame update"""
        operation_info = {
            'tab': 'Upload',
            'command': 'Load CSV',
            'operation_type': 'upload'
        }
        
        result = self.state_manager.update_dataframe(self.sample_df, operation_info)
        
        assert result is True
        assert self.state_manager.has_data()
        
        # Check DataFrame is stored correctly
        stored_df = self.state_manager.get_dataframe()
        pd.testing.assert_frame_equal(stored_df, self.sample_df)
        
        # Check metadata is created
        metadata = self.state_manager.get_metadata()
        assert metadata is not None
        assert metadata.shape == (5, 4)
        assert len(metadata.columns) == 4
        assert 'A' in metadata.columns
    
    def test_update_dataframe_invalid_input(self):
        """Test DataFrame update with invalid input"""
        # Test with non-DataFrame input
        result = self.state_manager.update_dataframe("not a dataframe")
        assert result is False
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = self.state_manager.update_dataframe(empty_df)
        assert result is False
    
    def test_metadata_generation(self):
        """Test metadata generation"""
        # Add some missing values
        df_with_nulls = self.sample_df.copy()
        df_with_nulls.loc[0, 'A'] = np.nan
        df_with_nulls.loc[1, 'B'] = None
        
        self.state_manager.update_dataframe(df_with_nulls)
        metadata = self.state_manager.get_metadata()
        
        assert metadata is not None
        assert metadata.shape == (5, 4)
        assert metadata.null_counts['A'] == 1
        assert metadata.null_counts['B'] == 1
        assert metadata.null_counts['C'] == 0
        assert 'MB' in metadata.memory_usage or 'KB' in metadata.memory_usage or 'B' in metadata.memory_usage
    
    def test_operation_logging(self):
        """Test operation logging functionality"""
        # Add some operations
        self.state_manager.add_operation_log(
            tab="Upload",
            command="Load data",
            operation_type="upload",
            success=True,
            execution_time=1.5
        )
        
        self.state_manager.add_operation_log(
            tab="Cleaning",
            command="Remove nulls",
            operation_type="clean",
            success=False,
            error_message="Column not found"
        )
        
        history = self.state_manager.get_operation_history()
        assert len(history) == 2
        
        # Check first operation
        assert history[0].tab == "Upload"
        assert history[0].success is True
        assert history[0].execution_time == 1.5
        
        # Check second operation
        assert history[1].tab == "Cleaning"
        assert history[1].success is False
        assert history[1].error_message == "Column not found"
    
    def test_undo_functionality(self):
        """Test undo functionality"""
        # Load initial data
        self.state_manager.update_dataframe(self.sample_df)
        
        # Modify data
        modified_df = self.sample_df.copy()
        modified_df['E'] = [6, 7, 8, 9, 10]
        self.state_manager.update_dataframe(modified_df)
        
        # Check modified data is current
        current_df = self.state_manager.get_dataframe()
        assert 'E' in current_df.columns
        
        # Undo operation
        undo_result = self.state_manager.undo_last_operation()
        assert undo_result is True
        
        # Check original data is restored
        restored_df = self.state_manager.get_dataframe()
        assert 'E' not in restored_df.columns
        pd.testing.assert_frame_equal(restored_df, self.sample_df)
    
    def test_undo_without_history(self):
        """Test undo when no history exists"""
        result = self.state_manager.undo_last_operation()
        assert result is False
    
    def test_dataframe_size_validation(self):
        """Test DataFrame size validation"""
        # Test valid DataFrame
        is_valid, message = self.state_manager.validate_dataframe_size(self.sample_df)
        assert is_valid is True
        assert message == ""
        
        # Test DataFrame with too many rows
        large_df = pd.DataFrame({'A': range(200000)})  # Exceeds default max_rows
        is_valid, message = self.state_manager.validate_dataframe_size(large_df, max_rows=100000)
        assert is_valid is False
        assert "too large" in message.lower()
        
        # Test DataFrame with too many columns
        wide_df = pd.DataFrame({f'col_{i}': [1, 2, 3] for i in range(2000)})  # Exceeds default max_cols
        is_valid, message = self.state_manager.validate_dataframe_size(wide_df, max_cols=1000)
        assert is_valid is False
        assert "too many columns" in message.lower()
    
    def test_summary_info(self):
        """Test summary information generation"""
        # Test with no data
        summary = self.state_manager.get_summary_info()
        assert "No data loaded" in summary
        
        # Test with data
        self.state_manager.update_dataframe(self.sample_df)
        summary = self.state_manager.get_summary_info()
        assert "5 rows" in summary
        assert "4 columns" in summary
    
    def test_recent_operations_display(self):
        """Test recent operations display"""
        # Test with no operations
        recent = self.state_manager.get_recent_operations()
        assert "No operations performed" in recent
        
        # Add some operations
        for i in range(3):
            self.state_manager.add_operation_log(
                tab=f"Tab{i}",
                command=f"Command{i}",
                operation_type="test",
                success=True
            )
        
        recent = self.state_manager.get_recent_operations(count=2)
        assert "Recent Operations:" in recent
        assert "Tab1" in recent
        assert "Tab2" in recent
        assert "Tab0" not in recent  # Should only show last 2
    
    def test_clear_data(self):
        """Test data clearing functionality"""
        # Load data
        self.state_manager.update_dataframe(self.sample_df)
        assert self.state_manager.has_data()
        
        # Clear data
        self.state_manager.clear_data()
        assert not self.state_manager.has_data()
        assert self.state_manager.get_dataframe() is None
        assert self.state_manager.get_metadata() is None
    
    def test_dataframe_copy_isolation(self):
        """Test that returned DataFrames are copies and don't affect internal state"""
        self.state_manager.update_dataframe(self.sample_df)
        
        # Get DataFrame and modify it
        df_copy = self.state_manager.get_dataframe()
        df_copy.loc[0, 'A'] = 999
        
        # Check internal DataFrame is unchanged
        internal_df = self.state_manager.get_dataframe()
        assert internal_df.loc[0, 'A'] != 999
        assert internal_df.loc[0, 'A'] == 1  # Original value

class TestDataFrameMetadata:
    """Test cases for DataFrameMetadata class"""
    
    def test_metadata_creation(self):
        """Test DataFrameMetadata creation and methods"""
        metadata = DataFrameMetadata(
            shape=(100, 5),
            columns=['A', 'B', 'C', 'D', 'E'],
            dtypes={'A': 'int64', 'B': 'object'},
            memory_usage="1.2 KB",
            null_counts={'A': 0, 'B': 5},
            last_modified=datetime.now()
        )
        
        summary = metadata.to_summary_string()
        assert "100 rows" in summary
        assert "5 columns" in summary
        assert "1.2 KB" in summary

class TestOperationLog:
    """Test cases for OperationLog class"""
    
    def test_operation_log_creation(self):
        """Test OperationLog creation and string representation"""
        log = OperationLog(
            timestamp=datetime.now(),
            tab="Upload",
            command="Load CSV file",
            operation_type="upload",
            success=True,
            execution_time=2.5
        )
        
        log_string = log.to_string()
        assert "✓" in log_string  # Success indicator
        assert "Upload" in log_string
        assert "Load CSV file" in log_string
        
        # Test failed operation
        failed_log = OperationLog(
            timestamp=datetime.now(),
            tab="Cleaning",
            command="Remove nulls",
            operation_type="clean",
            success=False,
            error_message="Error occurred"
        )
        
        failed_string = failed_log.to_string()
        assert "✗" in failed_string  # Failure indicator

if __name__ == "__main__":
    pytest.main([__file__])