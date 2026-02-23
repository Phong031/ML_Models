# ML_Models/core/pipeline.py
import numpy as np
import pandas as pd
import logging
import uuid
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ICBMLPipeline")

# --------------------------------------------------------------------------------------------------
# Config Loader
# --------------------------------------------------------------------------------------------------
class ConfigLoader:
    def __init__(self, config_dict=None):
        self.config_dict = config_dict or {}

    def load(self):
        logger.info("✅ Config loaded")
        return self.config_dict

# --------------------------------------------------------------------------------------------------
# Data Manager
# --------------------------------------------------------------------------------------------------
class DataManager:
    def load(self, X, y):
        logger.info("✅ Data loaded")
        return {"X": X, "y": y}

# --------------------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, model, param_search=None, fit_params=None, cv=5, scoring="neg_root_mean_squared_error"):
        self.model = model
        self.param_search = param_search
        self.fit_params = fit_params or {}
        self.cv = cv
        self.scoring = scoring
        self.best_model = None
        self.cv_results = None

    def cross_validate(self, X, y):
        logger.info("🔁 Running cross-validation")
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, scoring=self.scoring, cv=kf)
        rmse_scores = -scores
        self.cv_results = {
            "CV_RMSE_mean": rmse_scores.mean(),
            "CV_RMSE_std": rmse_scores.std()
        }
        logger.info(f"CV RMSE Mean: {rmse_scores.mean():.4f}")
        logger.info(f"CV RMSE Std: {rmse_scores.std():.4f}")
        return self.cv_results

    def train(self, features):
        X_train, y_train = features["X_train"], features["y_train"]
        
        if self.param_search:
            logger.info("🔎 Running hyperparameter tuning")
            self.param_search.fit(X_train, y_train, **self.fit_params)
            self.best_model = self.param_search.best_estimator_
            if hasattr(self.param_search, 'best_score_'):
                self.cv_results = {
                    "CV_RMSE_mean": -self.param_search.best_score_,
                    "best_params": self.param_search.best_params_
                }
            logger.info(f"Best params: {self.param_search.best_params_}")
        # skip the cross_validate if hyperparameter tuning activate
        else:
            logger.info("🚀 Training base model")
            self.cross_validate(X_train, y_train)
            self.model.fit(X_train, y_train, **self.fit_params)
            self.best_model = self.model
            
        return self.best_model

# --------------------------------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------------------------------
class Evaluator:
    def evaluate(self, model, features, cv_results=None):
        X_valid, y_valid = features["X_valid"], features["y_valid"]
        y_pred = model.predict(X_valid)

        mae = mean_absolute_error(y_valid, y_pred)
        mse = mean_squared_error(y_valid, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_valid, y_pred)
        baseline_pred = np.full_like(y_valid, features["y_train"].mean())
        baseline_mae = mean_absolute_error(y_valid, baseline_pred)
        improvement = (baseline_mae - mae) / baseline_mae * 100
        residuals = y_valid - y_pred

        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Baseline_MAE": baseline_mae,
            "Improvement_percent": improvement,
            "Residual_mean": residuals.mean(),
            "Residual_std": residuals.std()
        }

        if cv_results:
            metrics.update(cv_results)
        logger.info("✅ Evaluation completed")
        return metrics

    def extract_feature_importance(self, model, feature_names):
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)
            logger.info("📊 Feature importance extracted")
            return fi
        logger.info("⚠ Model does not support feature importance")
        return None

# --------------------------------------------------------------------------------------------------
# Tracker
# --------------------------------------------------------------------------------------------------
class Tracker:
    def log(self, experiment_name, model, metrics, path="experiment_log.json"):
        new_entry = {
            "run_id": str(uuid.uuid4()),
            "experiment_name": experiment_name,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "model_params": model.get_params()
        }
        
        existing_logs = []
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    existing_logs = json.load(f)
                    if isinstance(existing_logs, dict):
                        existing_logs = [existing_logs]
            except:
                existing_logs = []
        
        if isinstance(existing_logs, list):
            existing_logs.append(new_entry)
        else:
            existing_logs = [new_entry]
        
        with open(path, "w") as f:
            json.dump(existing_logs, f, indent=4)
        
        logger.info(f"📝 Experiment logged at {path} (run #{len(existing_logs)})")

# --------------------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------------------
class Registry:
    def save(self, model, path="model.pkl"):
        joblib.dump(model, path)
        logger.info(f"💾 Model saved at {path}")

# --------------------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------------------
class TrainingOrchestrator:
    def __init__(
        self, 
        config_loader, 
        data_manager, 
        feature_manager, 
        trainer, 
        evaluator, 
        tracker, 
        registry
    ):
        self.config_loader = config_loader
        self.data_manager = data_manager
        self.feature_manager = feature_manager
        self.trainer = trainer
        self.evaluator = evaluator
        self.tracker = tracker
        self.registry = registry

    def run(self, X, y, val_size=0.2, track_path="experiment_log.json", save_path="model.pkl"):
        config = self.config_loader.load()
        data = self.data_manager.load(X, y)
        features = self.feature_manager.build(
            data,
            val_size,
            config.get("random_state", 42)
        )
        model = self.trainer.train(features)
        metrics = self.evaluator.evaluate(
            model,
            features,
            cv_results=self.trainer.cv_results
        )
        feature_importance = self.evaluator.extract_feature_importance(
            model,
            features["feature_names"]
        )
        self.tracker.log(
            config.get("experiment_name", "default"),
            model,
            metrics,
            path=track_path
        )
        self.registry.save(model, path=save_path)
        return model, metrics, feature_importance