# ML_Models/feature_engineering/manager.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from .operations import OperationRegistry

logger = logging.getLogger(__name__)

class FeatureEngineeringPipeline:
    """Pipeline that applies configured feature engineering operations"""
    
    def __init__(self, operations_config: List[Dict]):
        """
        Args:
            operations_config: List of operation configs from config file
        """
        self.operations_config = operations_config
        self.applied_operations = []
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured operations"""
        X = X.copy()
        
        for op_config in self.operations_config:
            op_name = op_config.get("operation") or op_config.get("type")
            if not op_name:
                logger.warning(f"⚠ Operation missing 'operation' field: {op_config}")
                continue
            
            try:
                operation_func = OperationRegistry.get(op_name)
                
                X = operation_func(
                    X, 
                    input_cols=op_config["inputs"],
                    output_col=op_config["output"]
                )
                
                self.applied_operations.append({
                    "name": op_config.get("name", op_name),
                    "output": op_config["output"],
                    "inputs": op_config["inputs"]
                })
                
                logger.info(f"✅ Applied feature: {op_config.get('name', op_name)} -> {op_config['output']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to apply operation {op_name}: {e}")
        
        return X
    
    def get_applied_features(self) -> List[str]:
        """Get list of features created by this pipeline"""
        return [op["output"] for op in self.applied_operations]


class PreprocessingPipeline:
    """Handles scaling and encoding for different model types"""
    
    def __init__(self, preprocessing_config: Dict, model_type: str, categorical_features: List[str] = None):
        """
        Args:
            preprocessing_config: Configuration for preprocessing
            model_type: Type of model (catboost, lightgbm, neuralnetwork, etc.)
            categorical_features: List of categorical column names
        """
        self.config = preprocessing_config
        self.model_type = model_type.lower() if model_type else ""
        self.categorical_features = categorical_features or []
        self.scaler = None
        self.encoders = {}
        self.fitted = False
        self.numeric_columns = None
        self.log_transformations = []
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit preprocessing and transform data"""
        logger.info("🔄 Fitting preprocessing pipeline")
        X = X.copy()
        
        # Store numeric columns for later use
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Step 1: Handle missing values
        X = self._handle_missing(X)
        
        # Step 2: Handle categorical features based on model type
        X = self._encode_categorical(X)
        
        # Step 3: Scale numerical features if needed
        X = self._scale_features(X)
        
        self.fitted = True
        logger.info(f"✅ Preprocessing fitted. Transformations: {self.log_transformations}")
        return X
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessing"""
        if not self.fitted:
            raise ValueError("❌ Must call fit_transform before transform")
        
        logger.info("🔄 Applying fitted preprocessing")
        X = X.copy()
        
        # Apply same transformations as fit
        X = self._handle_missing(X, fitted=True)
        X = self._encode_categorical(X, fitted=True)
        X = self._scale_features(X, fitted=True)
        
        return X
    
    def _handle_missing(self, X: pd.DataFrame, fitted: bool = False) -> pd.DataFrame:
        """Handle missing values in both numeric and categorical columns"""
        strategy = self.config.get("missing_strategy", "mean")
        
        for col in X.columns:
            if X[col].isnull().any():
                missing_count = X[col].isnull().sum()
                
                # Case 1: Categorical column
                if col in self.categorical_features:
                    # Always fill categorical with 'Unknown'
                    X[col] = X[col].fillna('Unknown')
                    if not fitted:
                        self.log_transformations.append(f"Filled {missing_count} missing in categorical '{col}' with 'Unknown'")
                    logger.debug(f"   Filled {missing_count} missing in categorical '{col}' with 'Unknown'")
                
                # Case 2: Numeric column
                elif pd.api.types.is_numeric_dtype(X[col]):
                    if strategy == "mean" and not fitted:
                        fill_value = X[col].mean()
                        setattr(self, f"{col}_fill", fill_value)
                        X[col] = X[col].fillna(fill_value)
                        self.log_transformations.append(f"Filled {missing_count} missing in numeric '{col}' with mean ({fill_value:.2f})")
                    
                    elif strategy == "mean" and fitted:
                        fill_value = getattr(self, f"{col}_fill", 0)
                        X[col] = X[col].fillna(fill_value)
                    
                    elif strategy == "median" and not fitted:
                        fill_value = X[col].median()
                        setattr(self, f"{col}_fill", fill_value)
                        X[col] = X[col].fillna(fill_value)
                        self.log_transformations.append(f"Filled {missing_count} missing in numeric '{col}' with median ({fill_value:.2f})")
                    
                    elif strategy == "median" and fitted:
                        fill_value = getattr(self, f"{col}_fill", 0)
                        X[col] = X[col].fillna(fill_value)
                    
                    else:  # constant
                        X[col] = X[col].fillna(0)
                        if not fitted:
                            self.log_transformations.append(f"Filled {missing_count} missing in numeric '{col}' with 0")
                
                # Case 3: Other types (object, etc.)
                else:
                    X[col] = X[col].fillna('Unknown')
                    if not fitted:
                        self.log_transformations.append(f"Filled {missing_count} missing in column '{col}' with 'Unknown'")
        
        return X
    
    def _get_encoding_method(self) -> str:
        """Determine encoding method based on model type and config"""
        # Neural networks need one-hot encoding
        if self.model_type in ["neuralnetwork", "nn", "mlp"]:
            logger.info("🧠 Neural network detected - using one-hot encoding")
            return "onehot"
        
        # CatBoost can handle native categoricals
        if self.model_type == "catboost":
            logger.info("🐱 CatBoost detected - using native categorical handling")
            return "native"
        
        # Tree-based models can use label encoding
        if self.model_type in ["lightgbm", "xgboost", "randomforest"]:
            logger.info(f"🌲 {self.model_type} detected - using label encoding")
            return "label"
        
        # Default from config
        return self.config.get("categorical_encoding", "label")
    
    def _encode_categorical(self, X: pd.DataFrame, fitted: bool = False) -> pd.DataFrame:
        """Encode categorical features based on model type"""
        if not self.categorical_features:
            return X
        
        encoding_method = self._get_encoding_method()
        
        # Native encoding - just ensure strings, no NaN
        if encoding_method == "native":
            for col in self.categorical_features:
                if col in X.columns:
                    # Convert to string and handle NaN
                    X[col] = X[col].astype(str).replace('nan', 'Unknown')
                    if not fitted:
                        self.log_transformations.append(f"Native encoding for '{col}'")
        
        # One-hot encoding
        elif encoding_method == "onehot":
            X = self._onehot_encode(X, fitted)
        
        # Label encoding
        elif encoding_method == "label":
            X = self._label_encode(X, fitted)
        
        return X
    
    def _onehot_encode(self, X: pd.DataFrame, fitted: bool = False) -> pd.DataFrame:
        """Apply one-hot encoding to categorical features"""
        for col in self.categorical_features:
            if col not in X.columns:
                continue
            
            # Ensure no NaN and convert to string
            X[col] = X[col].fillna('Unknown').astype(str)
            
            if not fitted:
                # Create encoder
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(X[[col]])
                
                # Create feature names
                feature_names = [f"{col}_{val}" for val in encoder.categories_[0]]
                
                # Add encoded columns
                for i, name in enumerate(feature_names):
                    X[name] = encoded[:, i]
                
                # Store encoder
                self.encoders[col] = encoder
                self.log_transformations.append(f"One-hot encoded '{col}' -> {len(feature_names)} features")
                
            else:
                # Use fitted encoder
                encoder = self.encoders[col]
                encoded = encoder.transform(X[[col]])
                feature_names = [f"{col}_{val}" for val in encoder.categories_[0]]
                
                for i, name in enumerate(feature_names):
                    X[name] = encoded[:, i]
            
            # Drop original column
            X = X.drop(columns=[col])
        
        return X
    
    def _label_encode(self, X: pd.DataFrame, fitted: bool = False) -> pd.DataFrame:
        """Apply label encoding to categorical features"""
        for col in self.categorical_features:
            if col not in X.columns:
                continue
            
            # Ensure no NaN and convert to string
            X[col] = X[col].fillna('Unknown').astype(str)
            
            if not fitted:
                # Create encoder
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col])
                self.encoders[col] = encoder
                self.log_transformations.append(f"Label encoded '{col}'")
                
            else:
                # Use fitted encoder
                encoder = self.encoders[col]
                
                # Handle unseen labels
                try:
                    X[col] = encoder.transform(X[col])
                except ValueError:
                    # For unseen labels, set to -1
                    def safe_transform(x):
                        if x in encoder.classes_:
                            return encoder.transform([x])[0]
                        else:
                            return -1
                    
                    X[col] = X[col].apply(safe_transform)
                    logger.debug(f"   Handled unseen labels in '{col}'")
        
        return X
    
    def _scale_features(self, X: pd.DataFrame, fitted: bool = False) -> pd.DataFrame:
        """Scale numerical features"""
        if not self.config.get("scale_features", False):
            return X
        
        # For neural networks, force scaling
        if self.model_type in ["neuralnetwork", "nn", "mlp"] and not self.config.get("scale_features", False):
            logger.info("🧠 Neural network detected - forcing feature scaling")
            self.config["scale_features"] = True
        
        method = self.config.get("scaling_method", "standard")
        
        # Get numeric columns (excluding those that might have been one-hot encoded)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return X
        
        if not fitted:
            # Create scaler
            if method == "standard":
                self.scaler = StandardScaler()
                self.log_transformations.append(f"Applied StandardScaler to {len(numeric_cols)} numeric features")
            elif method == "minmax":
                self.scaler = MinMaxScaler()
                self.log_transformations.append(f"Applied MinMaxScaler to {len(numeric_cols)} numeric features")
            elif method == "robust":
                self.scaler = RobustScaler()
                self.log_transformations.append(f"Applied RobustScaler to {len(numeric_cols)} numeric features")
            else:
                logger.warning(f"⚠ Unknown scaling method '{method}', using StandardScaler")
                self.scaler = StandardScaler()
            
            # Fit and transform
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            
        else:
            # Transform using fitted scaler
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        return X
    
    def get_preprocessing_summary(self) -> Dict:
        """Get summary of preprocessing applied"""
        return {
            "model_type": self.model_type,
            "encoding_method": self._get_encoding_method(),
            "scaling": self.config.get("scale_features", False),
            "scaling_method": self.config.get("scaling_method", "standard"),
            "missing_strategy": self.config.get("missing_strategy", "mean"),
            "transformations": self.log_transformations
        }


class FeatureManager:
    """Enhanced FeatureManager with configurable feature engineering and preprocessing"""
    
    def __init__(self, feature_config: Optional[Dict] = None, model_type: str = None):
        """
        Args:
            feature_config: Configuration for feature engineering and preprocessing
            model_type: Type of model (for intelligent preprocessing decisions)
        """
        self.feature_config = feature_config or {}
        self.model_type = model_type.lower() if model_type else ""
        self.feature_pipeline = None
        self.preprocessing_pipeline = None
        self.transformations_log = []
        self.categorical_features = self.feature_config.get("categorical_features", [])
        
        # Initialize feature engineering pipeline
        if "operations" in self.feature_config and self.feature_config["operations"]:
            self.feature_pipeline = FeatureEngineeringPipeline(
                self.feature_config["operations"]
            )
            logger.info(f"🔧 Feature engineering pipeline initialized with {len(self.feature_config['operations'])} operations")
        
        # Initialize preprocessing pipeline if configured
        if "preprocessing" in self.feature_config:
            logger.info(f"📊 Preprocessing pipeline initialized for model type: {self.model_type}")
            self.preprocessing_pipeline = PreprocessingPipeline(
                self.feature_config["preprocessing"],
                self.model_type,
                self.categorical_features
            )
        else:
            logger.info(f"⏭️ No preprocessing configured - will handle categorical features directly")
    
    def build(self, data: Dict, val_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Build features with optional engineering, preprocessing, and split
        
        Args:
            data: Dictionary with 'X' and 'y' keys
            val_size: Validation set size (default 0.2)
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with train/validation splits and metadata
        """
        X = data["X"].copy()
        y = data["y"]
        
        logger.info(f"📊 Initial data: {X.shape[0]} rows, {X.shape[1]} columns")
        
        # ------------------------------------------------------------------
        # STEP 1: Handle categorical features for all models (baseline)
        # ------------------------------------------------------------------
        # This ensures NO NaN values reach any model
        X = self._handle_categorical_baseline(X)
        
        # ------------------------------------------------------------------
        # STEP 2: Feature engineering (creating new features)
        # ------------------------------------------------------------------
        if self.feature_pipeline:
            logger.info("🔨 Applying feature engineering operations...")
            X = self.feature_pipeline.transform(X)
            engineered = self.feature_pipeline.get_applied_features()
            self.transformations_log.append({
                "step": "engineering",
                "features_created": engineered
            })
            logger.info(f"✅ Created {len(engineered)} new features: {', '.join(engineered) if engineered else 'none'}")
        
        # ------------------------------------------------------------------
        # STEP 3: Advanced preprocessing (scaling, encoding) if configured
        # ------------------------------------------------------------------
        if self.preprocessing_pipeline:
            logger.info("🔄 Applying advanced preprocessing...")
            X = self.preprocessing_pipeline.fit_transform(X, y)
            prep_summary = self.preprocessing_pipeline.get_preprocessing_summary()
            self.transformations_log.append({
                "step": "preprocessing",
                "summary": prep_summary
            })
            
            # Log preprocessing details
            if prep_summary["scaling"]:
                logger.info(f"📏 Features scaled using {prep_summary['scaling_method']}")
            logger.info(f"🏷️ Categorical encoding: {prep_summary['encoding_method']}")
        
        else:
            # If no preprocessing pipeline, just ensure categoricals are strings
            logger.info("🔄 Applying basic categorical handling...")
            for col in self.categorical_features:
                if col in X.columns:
                    # Final safety check - ensure no NaN
                    X[col] = X[col].fillna('Unknown').astype(str)
                    logger.debug(f"   Ensured '{col}' is string type with no NaN")
        
        # ------------------------------------------------------------------
        # STEP 4: Final check for any remaining NaN values
        # ------------------------------------------------------------------
        nan_columns = X.columns[X.isna().any()].tolist()
        if nan_columns:
            logger.warning(f"⚠ Warning: Found NaN values in columns: {nan_columns}")
            # Fill any remaining NaN with appropriate values
            for col in nan_columns:
                if col in self.categorical_features or X[col].dtype == 'object':
                    X[col] = X[col].fillna('Unknown')
                else:
                    X[col] = X[col].fillna(0)
            logger.info("✅ Filled all remaining NaN values")
        
        # ------------------------------------------------------------------
        # STEP 5: Split data
        # ------------------------------------------------------------------
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=val_size, random_state=random_state
        )
        
        # Get feature names
        feature_names = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
        
        logger.info(f"✅ Final features: {len(feature_names)} features")
        logger.info(f"   Train set: {len(X_train)} rows, Validation set: {len(X_valid)} rows")
        
        # Count feature types
        numeric_count = len(X.select_dtypes(include=[np.number]).columns)
        categorical_count = len(X.select_dtypes(include=['object', 'category']).columns)
        logger.info(f"   Feature types: {numeric_count} numeric, {categorical_count} categorical")
        
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "feature_names": feature_names,
            "engineered_features": self.feature_pipeline.get_applied_features() if self.feature_pipeline else [],
            "transformations": self.transformations_log,
            "preprocessing_summary": self.preprocessing_pipeline.get_preprocessing_summary() if self.preprocessing_pipeline else None
        }
    
    def _handle_categorical_baseline(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Basic handling of categorical features to ensure NO NaN values.
        This is a safety net that runs for ALL models.
        """
        X = X.copy()
        
        # Get all categorical columns (from config or auto-detect)
        cat_cols = self.categorical_features.copy()
        
        # Also auto-detect object/string columns not in categorical_features
        auto_cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in auto_cat_cols:
            if col not in cat_cols:
                cat_cols.append(col)
        
        if not cat_cols:
            return X
        
        # Process each categorical column
        for col in cat_cols:
            if col not in X.columns:
                continue
            
            # Check for NaN values
            nan_count = X[col].isna().sum()
            if nan_count > 0:
                logger.debug(f"   Found {nan_count} NaN values in categorical column '{col}'")
                
                # Fill NaN with 'Unknown'
                X[col] = X[col].fillna('Unknown')
                
                # Log warning if significant missing data
                missing_pct = (nan_count / len(X)) * 100
                if missing_pct > 5:  # More than 5% missing
                    logger.warning(f"⚠ Column '{col}' has {missing_pct:.1f}% missing values (filled with 'Unknown')")
            
            # Convert to string to ensure CatBoost compatibility
            X[col] = X[col].astype(str)
            
            # Replace string 'nan' if any (from previous conversions)
            X[col] = X[col].replace('nan', 'Unknown')
        
        return X
    
    def get_feature_summary(self) -> Dict:
        """Get summary of all feature engineering and preprocessing"""
        summary = {
            "model_type": self.model_type,
            "categorical_features": self.categorical_features,
            "transformations": self.transformations_log
        }
        
        if self.feature_pipeline:
            summary["engineered_features"] = self.feature_pipeline.get_applied_features()
        
        if self.preprocessing_pipeline:
            summary["preprocessing"] = self.preprocessing_pipeline.get_preprocessing_summary()
        
        return summary


















# ======================================================================================
# ======================================================================================
# ML_Models/feature_engineering/manager.py
# Manager without Preprocessing to process cat encode and scaling
'''
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from sklearn.model_selection import train_test_split

# Use relative import
from .operations import OperationRegistry

logger = logging.getLogger(__name__)

class FeatureEngineeringPipeline:
    """Pipeline that applies configured feature engineering operations"""
    
    def __init__(self, operations_config: List[Dict]):
        self.operations_config = operations_config
        self.applied_operations = []
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured operations"""
        X = X.copy()
        
        for op_config in self.operations_config:
            op_name = op_config.get("operation") or op_config.get("type")
            if not op_name:
                continue
            
            try:
                operation_func = OperationRegistry.get(op_name)
                
                X = operation_func(
                    X, 
                    input_cols=op_config["inputs"],
                    output_col=op_config["output"]
                )
                
                self.applied_operations.append({
                    "name": op_config.get("name", op_name),
                    "output": op_config["output"]
                })
                
                logger.info(f"✅ Applied feature: {op_config.get('name', op_name)}")
                
            except Exception as e:
                logger.error(f"Failed to apply operation {op_name}: {e}")
        
        return X
    
    def get_applied_features(self) -> List[str]:
        """Get list of features created by this pipeline"""
        return [op["output"] for op in self.applied_operations]

class FeatureManager:
    """Enhanced FeatureManager with configurable feature engineering"""
    
    def __init__(self, feature_config: Optional[Dict] = None):
        self.feature_config = feature_config or {}
        self.feature_pipeline = None
        self.transformations_log = []
        
        if "operations" in self.feature_config:
            self.feature_pipeline = FeatureEngineeringPipeline(
                self.feature_config["operations"]
            )
    
    def build(self, data: Dict, val_size: float = 0.2, random_state: int = 42) -> Dict:
        """Build features with optional engineering and split"""
        X = data["X"].copy()
        y = data["y"]
        
        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)
            self.transformations_log.append({
                "features_created": self.feature_pipeline.get_applied_features()
            })
        
        if "categorical_handling" in self.feature_config:
            X = self._handle_categorical(X)
        
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=val_size, random_state=random_state
        )
        
        feature_names = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
        
        logger.info(f"✅ Built {len(feature_names)} features")
        
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "feature_names": feature_names,
            "engineered_features": self.feature_pipeline.get_applied_features() if self.feature_pipeline else []
        }
    
    def _handle_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle categorical features"""
        X = X.copy()
        cat_features = self.feature_config.get("categorical_features", [])
        
        for col in cat_features:
            if col in X.columns:
                X[col] = X[col].fillna("Unknown").astype(str)
        
        return X
'''