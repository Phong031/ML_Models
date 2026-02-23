# ML_Models/models/factories.py
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating models based on configuration"""
    # With @staticmethod - just use class directly
    # Without @staticmethod - need instance
    @staticmethod
    def create_model(model_config: Dict, random_state: int = 42):
        """Create a model instance based on config"""
        model_type = model_config.get("type", "").lower()
        
        if model_type == "catboost":
            from catboost import CatBoostRegressor, CatBoostClassifier
            
            params = model_config.get("params", {}).copy()
            params["random_seed"] = random_state
            
            if "cat_features" in model_config:
                params["cat_features"] = model_config["cat_features"]
            
            task = model_config.get("task", "regression")
            if task == "regression":
                logger.info("📊 Creating CatBoost Regressor")
                return CatBoostRegressor(**params)
            else:
                logger.info("📊 Creating CatBoost Classifier")
                return CatBoostClassifier(**params)
        
        elif model_type == "xgboost":
            import xgboost as xgb
            
            params = model_config.get("params", {}).copy()
            params["random_state"] = random_state
            
            task = model_config.get("task", "regression")
            if task == "regression":
                logger.info("📊 Creating XGBoost Regressor")
                return xgb.XGBRegressor(**params)
            else:
                logger.info("📊 Creating XGBoost Classifier")
                return xgb.XGBClassifier(**params)
        
        elif model_type == "lightgbm":
            import lightgbm as lgb
            
            params = model_config.get("params", {}).copy()
            params["random_state"] = random_state
            
            task = model_config.get("task", "regression")
            if task == "regression":
                logger.info("📊 Creating LightGBM Regressor")
                return lgb.LGBMRegressor(**params)
            else:
                logger.info("📊 Creating LightGBM Classifier")
                return lgb.LGBMClassifier(**params)
        
        elif model_type == "neuralnetwork" or model_type == "nn" or model_type == "mlp":
            from sklearn.neural_network import MLPRegressor, MLPClassifier
            
            params = model_config.get("params", {}).copy()
            
            # MLP uses random_state, not random_seed
            params["random_state"] = random_state
            
            task = model_config.get("task", "regression")
            if task == "regression":
                logger.info("🧠 Creating Neural Network Regressor (MLP)")
                return MLPRegressor(**params)
            else:
                logger.info("🧠 Creating Neural Network Classifier (MLP)")
                return MLPClassifier(**params)
        
        elif model_type == "randomforest":
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            params = model_config.get("params", {}).copy()
            params["random_state"] = random_state
            
            task = model_config.get("task", "regression")
            if task == "regression":
                logger.info("🌲 Creating Random Forest Regressor")
                return RandomForestRegressor(**params)
            else:
                logger.info("🌲 Creating Random Forest Classifier")
                return RandomForestClassifier(**params)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def create_hyperparameter_search(model, search_config: Dict, random_state: int = 42):
        """Create hyperparameter search object"""
        if not search_config.get("enabled", False):
            logger.info("⏭️ Hyperparameter search disabled")
            return None
        
        method = search_config.get("method", "randomized_search").lower()
        
        if method == "randomized_search":
            from sklearn.model_selection import RandomizedSearchCV
            
            logger.info(f"🔍 Creating RandomizedSearchCV with {search_config.get('n_iter', 10)} iterations")
            
            return RandomizedSearchCV(
                estimator=model,
                param_distributions=search_config.get("param_distributions", {}),
                n_iter=search_config.get("n_iter", 10),
                scoring=search_config.get("scoring", "neg_root_mean_squared_error"),
                cv=search_config.get("cv", 5),
                random_state=random_state,
                n_jobs=search_config.get("n_jobs", -1),
                refit=True,
                verbose=search_config.get("verbose", 0)
            )
        
        elif method == "grid_search":
            from sklearn.model_selection import GridSearchCV
            
            logger.info(f"🔍 Creating GridSearchCV")
            
            return GridSearchCV(
                estimator=model,
                param_grid=search_config.get("param_distributions", {}),
                scoring=search_config.get("scoring", "neg_root_mean_squared_error"),
                cv=search_config.get("cv", 5),
                n_jobs=search_config.get("n_jobs", -1),
                refit=True,
                verbose=search_config.get("verbose", 0)
            )
        
        else:
            raise ValueError(f"Unsupported search method: {method}")