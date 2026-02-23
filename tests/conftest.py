# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing"""
    return pd.DataFrame({
        'Job Number': [1, 2, 3, 4, 5],
        'Job Value': [5000, 8000, 10000, 12000, 15000],
        'Total Claimed': [1000, 2000, 3000, 4000, 5000],
        'Total Cost': [800, 1500, 2500, 3200, 4000],
        'Estimator': ['John', 'Jane', 'John', 'Bob', 'Jane'],
        'Foreman': ['Mike', 'Mike', 'Steve', 'Steve', 'Mike'],
        'Job Area': ['North', 'South', 'East', 'West', 'North'],
        'Main Contractor': ['A', 'B', 'A', 'C', 'B'],
        'Suburb': ['X', 'Y', 'Z', 'X', 'Y'],
        'Supervisor': ['Tom', 'Dick', 'Harry', 'Tom', 'Dick'],
        'Job Description': ['Desc1', 'Desc2', 'Desc3', 'Desc4', 'Desc5'],
        'Saving or Loss': [200, 500, 500, 800, 1000]
    })

@pytest.fixture
def sample_features(sample_dataframe):
    """Create sample features for training tests"""
    X = sample_dataframe.drop('Saving or Loss', axis=1)
    y = sample_dataframe['Saving or Loss']
    return X, y

@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary"""
    return {
        'experiment_name': 'Test_Experiment',
        'random_state': 42,
        'val_size': 0.3,
        'cv_folds': 2,
        'track_path': 'test_log.json',
        'save_path': 'test_model.pkl',
        'data': {
            'categorical_features': ['Estimator', 'Foreman', 'Job Area'],
            'target_column': 'Saving or Loss'
        },
        'feature_engineering': {
            'operations': [
                {
                    'name': 'test_profit',
                    'operation': 'subtract',
                    'inputs': ['Total Claimed', 'Total Cost'],
                    'output': 'Test Profit'
                }
            ]
        },
        'model': {
            'type': 'randomforest',
            'params': {
                'n_estimators': 5,
                'max_depth': 3
            }
        }
    }

@pytest.fixture
def temp_config_file(tmp_path, sample_config_dict):
    """Create a temporary config file"""
    import yaml
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config_dict, f)
    return str(config_file)

@pytest.fixture
def sample_data_dict(sample_features):
    """Create data dictionary for FeatureManager"""
    X, y = sample_features
    return {'X': X, 'y': y}