# tests/test_pipeline.py
import pytest
import pandas as pd
import numpy as np
import sys
import os
import json
import tempfile
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import (
    ConfigLoader, DataManager, Trainer, Evaluator, 
    Tracker, Registry, TrainingOrchestrator
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

class TestConfigLoader:
    """Test suite for ConfigLoader"""
    
    def test_init_with_dict(self, sample_config_dict):
        loader = ConfigLoader(sample_config_dict)
        assert loader.config_dict == sample_config_dict
    
    def test_load(self, sample_config_dict):
        loader = ConfigLoader(sample_config_dict)
        result = loader.load()
        assert result == sample_config_dict
        assert result['experiment_name'] == 'Test_Experiment'
    
    def test_load_empty(self):
        loader = ConfigLoader()
        result = loader.load()
        assert result == {}

class TestDataManager:
    """Test suite for DataManager"""
    
    def test_load(self):
        manager = DataManager()
        X = pd.DataFrame({'a': [1, 2]})
        y = pd.Series([3, 4])
        
        result = manager.load(X, y)
        
        assert 'X' in result
        assert 'y' in result
        assert result['X'].equals(X)
        assert result['y'].equals(y)

class TestTrainer:
    """Test suite for Trainer"""
    
    def test_init(self):
        model = RandomForestRegressor()
        trainer = Trainer(model=model, cv=3)
        
        assert trainer.model == model
        assert trainer.cv == 3
        assert trainer.param_search is None
    
    def test_cross_validate(self, sample_features):
        X, y = sample_features
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        trainer = Trainer(model=model, cv=2)
        
        # Convert to numpy for sklearn compatibility
        X_np = X.select_dtypes(include=[np.number]).values
        y_np = y.values
        
        result = trainer.cross_validate(X_np, y_np)
        
        assert 'CV_RMSE_mean' in result
        assert 'CV_RMSE_std' in result
        assert result['CV_RMSE_mean'] > 0  # RMSE should be positive
    
    def test_train_without_search(self, sample_features):
        X, y = sample_features
        X_np = X.select_dtypes(include=[np.number]).values
        y_np = y.values
        
        features = {
            'X_train': X_np,
            'y_train': y_np,
            'X_valid': X_np,
            'y_valid': y_np,
            'feature_names': ['f1', 'f2']
        }
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        trainer = Trainer(model=model, cv=2)
        
        trained_model = trainer.train(features)
        
        assert trained_model is not None
        assert trainer.cv_results is not None
    
    def test_train_with_search(self, sample_features):
        X, y = sample_features
        X_np = X.select_dtypes(include=[np.number]).values
        y_np = y.values
        
        features = {
            'X_train': X_np,
            'y_train': y_np,
            'X_valid': X_np,
            'y_valid': y_np,
            'feature_names': ['f1', 'f2']
        }
        
        base_model = RandomForestRegressor(random_state=42)
        param_dist = {
            'n_estimators': [3, 5],
            'max_depth': [2, 3]
        }
        
        search = RandomizedSearchCV(
            base_model, param_dist, n_iter=2, cv=2, random_state=42
        )
        
        trainer = Trainer(model=base_model, param_search=search)
        trained_model = trainer.train(features)
        
        assert trained_model is not None
        assert trainer.cv_results is not None
        if 'best_params' in trainer.cv_results:
            assert 'n_estimators' in trainer.cv_results['best_params']

class TestEvaluator:
    """Test suite for Evaluator"""
    
    def test_evaluate(self, sample_features):
        X, y = sample_features
        X_np = X.select_dtypes(include=[np.number]).values
        y_np = y.values
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X_np, y_np)
        
        features = {
            'X_train': X_np,
            'y_train': y_np,
            'X_valid': X_np,
            'y_valid': y_np,
            'feature_names': list(X.select_dtypes(include=[np.number]).columns)
        }
        
        evaluator = Evaluator()
        metrics = evaluator.evaluate(model, features)
        
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'R2' in metrics
        assert metrics['MAE'] > 0
        assert metrics['RMSE'] > 0
    
    def test_evaluate_with_cv_results(self, sample_features):
        X, y = sample_features
        X_np = X.select_dtypes(include=[np.number]).values
        y_np = y.values
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X_np, y_np)
        
        features = {
            'X_train': X_np,
            'y_train': y_np,
            'X_valid': X_np,
            'y_valid': y_np,
            'feature_names': ['f1', 'f2']
        }
        
        cv_results = {'CV_RMSE_mean': 100.5, 'CV_RMSE_std': 10.2}
        
        evaluator = Evaluator()
        metrics = evaluator.evaluate(model, features, cv_results=cv_results)
        
        assert metrics['CV_RMSE_mean'] == 100.5
        assert metrics['CV_RMSE_std'] == 10.2
    
    def test_feature_importance(self, sample_features):
        X, y = sample_features
        X_np = X.select_dtypes(include=[np.number]).values
        y_np = y.values
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X_np, y_np)
        
        feature_names = list(X.select_dtypes(include=[np.number]).columns)
        
        evaluator = Evaluator()
        importance = evaluator.extract_feature_importance(model, feature_names)
        
        assert importance is not None
        assert 'Feature' in importance.columns
        assert 'Importance' in importance.columns
        assert len(importance) == len(feature_names)

class TestTracker:
    """Test suite for Tracker"""
    
    def test_log(self, tmp_path):
        tracker = Tracker()
        model = RandomForestRegressor(n_estimators=5)
        metrics = {'MAE': 10.5, 'RMSE': 15.2}
        
        log_path = tmp_path / "test_log.json"
        
        tracker.log('test_experiment', model, metrics, path=str(log_path))
        
        assert log_path.exists()
        
        with open(log_path, 'r') as f:
            logs = json.load(f)
        
        assert len(logs) == 1
        assert logs[0]['experiment_name'] == 'test_experiment'
        assert logs[0]['metrics']['MAE'] == 10.5
    
    def test_log_multiple_runs(self, tmp_path):
        tracker = Tracker()
        model = RandomForestRegressor(n_estimators=5)
        
        log_path = tmp_path / "test_log.json"
        
        # First log
        tracker.log('exp1', model, {'MAE': 10}, path=str(log_path))
        # Second log
        tracker.log('exp2', model, {'MAE': 20}, path=str(log_path))
        
        with open(log_path, 'r') as f:
            logs = json.load(f)
        
        assert len(logs) == 2
        assert logs[0]['experiment_name'] == 'exp1'
        assert logs[1]['experiment_name'] == 'exp2'

class TestRegistry:
    """Test suite for Registry"""
    
    def test_save_and_load(self, tmp_path):
        registry = Registry()
        model = RandomForestRegressor(n_estimators=5)
        
        model_path = tmp_path / "test_model.pkl"
        
        # Save model
        registry.save(model, path=str(model_path))
        assert model_path.exists()
        
        # Load model (using joblib)
        import joblib
        loaded_model = joblib.load(str(model_path))
        assert loaded_model.n_estimators == 5

class TestTrainingOrchestrator:
    """Test suite for TrainingOrchestrator"""
    
    def test_orchestrator_init(self, sample_config_dict, sample_features):
        X, y = sample_features
        
        # Create mock components
        config_loader = ConfigLoader(sample_config_dict)
        data_manager = DataManager()
        
        from feature_engineering.manager import FeatureManager
        feature_manager = FeatureManager(sample_config_dict.get('feature_engineering', {}))
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        trainer = Trainer(model=model, cv=2)
        evaluator = Evaluator()
        tracker = Tracker()
        registry = Registry()
        
        orchestrator = TrainingOrchestrator(
            config_loader, data_manager, feature_manager,
            trainer, evaluator, tracker, registry
        )
        
        assert orchestrator.config_loader == config_loader
        assert orchestrator.trainer == trainer
    
    def test_orchestrator_run(self, sample_config_dict, sample_features, tmp_path):
        X, y = sample_features
        
        config_loader = ConfigLoader(sample_config_dict)
        data_manager = DataManager()
        
        from feature_engineering.manager import FeatureManager
        feature_manager = FeatureManager(sample_config_dict.get('feature_engineering', {}))
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        trainer = Trainer(model=model, cv=2)
        evaluator = Evaluator()
        tracker = Tracker()
        registry = Registry()
        
        orchestrator = TrainingOrchestrator(
            config_loader, data_manager, feature_manager,
            trainer, evaluator, tracker, registry
        )
        
        # Run with temporary paths
        track_path = tmp_path / "test_log.json"
        save_path = tmp_path / "test_model.pkl"
        
        model, metrics, feature_importance = orchestrator.run(
            X, y, 
            val_size=0.3,
            track_path=str(track_path),
            save_path=str(save_path)
        )
        
        assert model is not None
        assert metrics is not None
        assert 'MAE' in metrics
        assert track_path.exists()
        assert save_path.exists()