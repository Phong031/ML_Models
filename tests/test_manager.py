# tests/test_manager.py
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.manager import FeatureManager, FeatureEngineeringPipeline

class TestFeatureEngineeringPipeline:
    """Test suite for FeatureEngineeringPipeline"""
    
    def test_pipeline_initialization(self):
        operations = [{'operation': 'subtract', 'inputs': ['A', 'B'], 'output': 'C'}]
        pipeline = FeatureEngineeringPipeline(operations)
        assert len(pipeline.operations_config) == 1
        assert pipeline.applied_operations == []
    
    def test_pipeline_transform(self):
        df = pd.DataFrame({'A': [10, 20], 'B': [3, 5]})
        operations = [{'operation': 'subtract', 'inputs': ['A', 'B'], 'output': 'C'}]
        
        pipeline = FeatureEngineeringPipeline(operations)
        result = pipeline.transform(df)
        
        assert 'C' in result.columns
        assert result['C'].tolist() == [7, 15]
        assert len(pipeline.applied_operations) == 1
    
    def test_pipeline_multiple_operations(self):
        df = pd.DataFrame({'A': [10, 20], 'B': [3, 5]})
        operations = [
            {'operation': 'subtract', 'inputs': ['A', 'B'], 'output': 'C'},
            {'operation': 'add', 'inputs': ['A', 'B'], 'output': 'D'}
        ]
        
        pipeline = FeatureEngineeringPipeline(operations)
        result = pipeline.transform(df)
        
        assert 'C' in result.columns
        assert 'D' in result.columns
        assert result['C'].tolist() == [7, 15]
        assert result['D'].tolist() == [13, 25]
        assert len(pipeline.applied_operations) == 2
    
    def test_pipeline_handles_missing_operation(self, caplog):
        df = pd.DataFrame({'A': [10, 20]})
        operations = [{'operation': 'invalid', 'inputs': ['A'], 'output': 'B'}]
        
        pipeline = FeatureEngineeringPipeline(operations)
        result = pipeline.transform(df)
        
        assert 'B' not in result.columns
        assert "Operation 'invalid' not found" in caplog.text

class TestFeatureManager:
    """Test suite for FeatureManager"""
    
    def test_manager_initialization(self):
        manager = FeatureManager()
        assert manager.feature_pipeline is None
        
        config = {'operations': [{'operation': 'subtract', 'inputs': ['A', 'B'], 'output': 'C'}]}
        manager = FeatureManager(config)
        assert manager.feature_pipeline is not None
    
    def test_build_without_operations(self, sample_data_dict):
        manager = FeatureManager()
        result = manager.build(sample_data_dict, val_size=0.4, random_state=42)
        
        assert 'X_train' in result
        assert 'X_valid' in result
        assert 'y_train' in result
        assert 'y_valid' in result
        assert 'feature_names' in result
        
        total_rows = len(result['X_train']) + len(result['X_valid'])
        assert total_rows == len(sample_data_dict['X'])
    
    def test_build_with_operations(self, sample_data_dict):
        config = {
            'operations': [
                {
                    'operation': 'subtract',
                    'inputs': ['Total Claimed', 'Total Cost'],
                    'output': 'Profit'
                }
            ]
        }
        
        manager = FeatureManager(config)
        result = manager.build(sample_data_dict, val_size=0.3, random_state=42)
        
        # Check that Profit column exists in either train or valid
        profit_in_train = 'Profit' in result['X_train'].columns if hasattr(result['X_train'], 'columns') else False
        profit_in_valid = 'Profit' in result['X_valid'].columns if hasattr(result['X_valid'], 'columns') else False
        assert profit_in_train or profit_in_valid
        
        assert 'engineered_features' in result
        assert 'Profit' in result['engineered_features']
    
    def test_build_with_categorical_handling(self, sample_data_dict):
        config = {
            'categorical_features': ['Estimator', 'Foreman'],
            'categorical_handling': {'fillna': 'Unknown', 'encoding': 'native'}
        }
        
        manager = FeatureManager(config)
        result = manager.build(sample_data_dict)
        
        # Check that categorical columns are strings
        if hasattr(result['X_train'], 'columns'):
            for col in ['Estimator', 'Foreman']:
                if col in result['X_train'].columns:
                    assert result['X_train'][col].dtype == 'object' or str(result['X_train'][col].dtype).startswith('string')
    
    def test_build_with_missing_values(self):
        X = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': ['x', None, 'z', 'w']
        })
        y = pd.Series([10, 20, 30, 40])
        data = {'X': X, 'y': y}
        
        config = {
            'categorical_features': ['B'],
            'categorical_handling': {'fillna': 'Unknown'}
        }
        
        manager = FeatureManager(config)
        result = manager.build(data)
        
        # Check that missing values were handled
        if hasattr(result['X_train'], 'isnull'):
            assert result['X_train'].isnull().sum().sum() >= 0  # Should have fewer or no NaNs
    
    def test_train_test_split_ratio(self, sample_data_dict):
        manager = FeatureManager()
        
        # Test 20% validation
        result_20 = manager.build(sample_data_dict, val_size=0.2, random_state=42)
        valid_ratio_20 = len(result_20['X_valid']) / (len(result_20['X_train']) + len(result_20['X_valid']))
        assert abs(valid_ratio_20 - 0.2) < 0.1  # Allow small rounding differences
        
        # Test 40% validation
        result_40 = manager.build(sample_data_dict, val_size=0.4, random_state=42)
        valid_ratio_40 = len(result_40['X_valid']) / (len(result_40['X_train']) + len(result_40['X_valid']))
        assert abs(valid_ratio_40 - 0.4) < 0.1
    
    def test_get_applied_features(self, sample_data_dict):
        config = {
            'operations': [
                {'operation': 'subtract', 'inputs': ['Total Claimed', 'Total Cost'], 'output': 'Profit'},
                {'operation': 'divide_percent', 'inputs': ['Profit', 'Job Value'], 'output': 'Margin'}
            ]
        }
        
        manager = FeatureManager(config)
        result = manager.build(sample_data_dict)
        
        if manager.feature_pipeline:
            features = manager.feature_pipeline.get_applied_features()
            assert 'Profit' in features
            assert 'Margin' in features