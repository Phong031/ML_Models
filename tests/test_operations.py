# tests/test_operations.py
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.operations import FeatureOperations, OperationRegistry

class TestFeatureOperations:
    """Test suite for FeatureOperations class"""
    
    def test_subtract_operation(self):
        df = pd.DataFrame({'A': [10, 20, 30], 'B': [3, 5, 7]})
        result = FeatureOperations.subtract(df, ['A', 'B'], 'C')
        assert 'C' in result.columns
        assert result['C'].tolist() == [7, 15, 23]
    
    def test_add_operation(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = FeatureOperations.add(df, ['A', 'B'], 'C')
        assert result['C'].tolist() == [5, 7, 9]
    
    def test_multiply_operation(self):
        df = pd.DataFrame({'A': [2, 3, 4], 'B': [5, 6, 7]})
        result = FeatureOperations.multiply(df, ['A', 'B'], 'C')
        assert result['C'].tolist() == [10, 18, 28]
    
    def test_divide_operation(self):
        df = pd.DataFrame({'A': [10, 20, 30], 'B': [2, 4, 6]})
        result = FeatureOperations.divide(df, ['A', 'B'], 'C')
        assert result['C'].tolist() == [5.0, 5.0, 5.0]
    
    def test_divide_by_zero(self):
        df = pd.DataFrame({'A': [10, 20], 'B': [0, 5]})
        result = FeatureOperations.divide(df, ['A', 'B'], 'C')
        assert pd.isna(result['C'].iloc[0])
        assert result['C'].iloc[1] == 4.0
    
    def test_divide_percent(self):
        df = pd.DataFrame({'A': [10, 20], 'B': [100, 200]})
        result = FeatureOperations.divide_percent(df, ['A', 'B'], 'C')
        assert result['C'].tolist() == [10.0, 10.0]
    
class TestOperationRegistry:
    """Test suite for OperationRegistry class"""
    
    def test_get_valid_operation(self):
        subtract_func = OperationRegistry.get('subtract')
        assert subtract_func == FeatureOperations.subtract
    
    def test_get_invalid_operation(self):
        with pytest.raises(ValueError, match="Operation 'invalid' not found"):
            OperationRegistry.get('invalid')
    
    def test_all_operations_available(self):
        operations = ['subtract', 'add', 'multiply', 'divide', 
                     'divide_percent']
        for op in operations:
            func = OperationRegistry.get(op)
            assert callable(func)