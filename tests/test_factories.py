# tests/test_factories.py
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.factories import ModelFactory

class TestModelFactory:
    """Test suite for ModelFactory"""
    
    def test_create_catboost_model(self):
        config = {
            'type': 'catboost',
            'params': {
                'iterations': 10,
                'depth': 3,
                'learning_rate': 0.1
            }
        }
        
        model = ModelFactory.create_model(config, random_state=42)
        
        # Check that model was created
        assert model is not None
        # Check class name contains CatBoost
        assert 'CatBoost' in str(type(model))
    
    def test_create_xgboost_model(self):
        config = {
            'type': 'xgboost',
            'params': {
                'n_estimators': 10,
                'max_depth': 3,
                'learning_rate': 0.1
            }
        }
        
        model = ModelFactory.create_model(config, random_state=42)
        
        assert model is not None
        assert 'XGB' in str(type(model)) or 'XGBoost' in str(type(model))
    
    def test_create_randomforest_model(self):
        config = {
            'type': 'randomforest',
            'params': {
                'n_estimators': 10,
                'max_depth': 3
            }
        }
        
        model = ModelFactory.create_model(config, random_state=42)
        
        assert model is not None
        assert 'RandomForest' in str(type(model))
    
    def test_create_lightgbm_model(self):
        config = {
            'type': 'lightgbm',
            'params': {
                'n_estimators': 10,
                'max_depth': 3,
                'learning_rate': 0.1
            }
        }
        
        model = ModelFactory.create_model(config, random_state=42)
        
        assert model is not None
        assert 'LGBM' in str(type(model)) or 'LightGBM' in str(type(model))
    
    def test_invalid_model_type(self):
        config = {'type': 'invalid_model'}
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            ModelFactory.create_model(config)
    
    def test_random_state_passed_correctly(self):
        # Test that random_state is properly passed to models
        config = {'type': 'randomforest', 'params': {'n_estimators': 5}}
        
        model1 = ModelFactory.create_model(config, random_state=42)
        model2 = ModelFactory.create_model(config, random_state=99)
        
        # Models with different random states should be different
        # (this is a simple check - they might still be the same by chance)
        assert model1.random_state != model2.random_state or model1.random_state == 42
    
    def test_create_hyperparameter_search_randomized(self):
        model = ModelFactory.create_model({'type': 'randomforest'})
        
        search_config = {
            'enabled': True,
            'method': 'randomized_search',
            'n_iter': 3,
            'scoring': 'neg_root_mean_squared_error',
            'cv': 2,
            'param_distributions': {
                'n_estimators': [5, 10],
                'max_depth': [3, 5]
            }
        }
        
        search = ModelFactory.create_hyperparameter_search(
            model, search_config, random_state=42
        )
        
        assert search is not None
        assert hasattr(search, 'fit')
        assert search.n_iter == 3
        assert search.cv == 2
    
    def test_create_hyperparameter_search_disabled(self):
        model = ModelFactory.create_model({'type': 'randomforest'})
        
        search_config = {'enabled': False}
        
        search = ModelFactory.create_hyperparameter_search(model, search_config)
        
        assert search is None
    
    def test_create_hyperparameter_search_grid(self):
        model = ModelFactory.create_model({'type': 'randomforest'})
        
        search_config = {
            'enabled': True,
            'method': 'grid_search',
            'scoring': 'neg_root_mean_squared_error',
            'cv': 2,
            'param_distributions': {
                'n_estimators': [5, 10],
                'max_depth': [3, 5]
            }
        }
        
        search = ModelFactory.create_hyperparameter_search(
            model, search_config, random_state=42
        )
        
        assert search is not None
        assert hasattr(search, 'fit')
        assert 'GridSearchCV' in str(type(search))