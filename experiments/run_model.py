# ML_Models/experiments/run_model.py
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
import yaml
from datetime import datetime

# Now import using simple names (no ML_Models prefix)
from core.pipeline import (
    ConfigLoader, DataManager, Trainer, Evaluator, 
    Tracker, Registry, TrainingOrchestrator
)
from feature_engineering.manager import FeatureManager as EnhancedFeatureManager
from models.factories import ModelFactory

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("MLExperiment")

def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_experiment(config_path: str):
    """Run ML experiment with configuration file"""
    
    print("\n" + "="*60)
    print("🚀 ML PIPELINE EXPERIMENT")
    print("="*60)
    print(f"📋 Config: {config_path}")
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ------------------------------------------------------------------
    # 1. LOAD CONFIGURATION
    # ------------------------------------------------------------------
    print("\n📋 Step 1: Loading configuration")
    config_dict = load_yaml_config(config_path)
    experiment_name = config_dict.get('experiment_name', 'Unnamed Experiment')
    model_type = config_dict.get('model', {}).get('type', 'unknown')
    
    print(f"   Experiment: {experiment_name}")
    print(f"   Model type: {model_type}")
    print(f"   Random state: {config_dict.get('random_state', 42)}")
    
    # ------------------------------------------------------------------
    # 2. LOAD DATA
    # ------------------------------------------------------------------
    print("\n📂 Step 2: Loading data")
    data_config = config_dict.get("data", {})
    
    file_path = data_config.get("file_path")
    if not os.path.exists(file_path):
        # Try relative path from project root: # parent_dir = "C:/Projects/ML_Models"
                                               # file_path = "data/Jobs Data.xlsx"  (from config)
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), file_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"   File: {file_path}")
    
    # Load based on file type
    if data_config.get("file_type") == "excel":
        df = pd.read_excel(
            file_path,
            usecols=data_config.get("columns_needed")
        )
    elif data_config.get("file_type") == "csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {data_config.get('file_type')}")
    
    target_column = data_config["target_column"]
    
    # Remove rows with missing target
    initial_rows = len(df)
    df_clean = df.dropna(subset=[target_column]).copy()
    dropped_rows = initial_rows - len(df_clean)
    
    if dropped_rows > 0:
        print(f"   ⚠ Dropped {dropped_rows} rows with missing target")
    
    # Split features and target
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    print(f"   Loaded {len(df_clean)} rows, {len(X.columns)} features")
    print(f"   Target: {target_column}")
    
    # Show data types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"   Numeric features: {len(numeric_cols)}")
    print(f"   Categorical features: {len(categorical_cols)}")
    
    # ------------------------------------------------------------------
    # 3. CREATE MODEL USING FACTORY
    # ------------------------------------------------------------------
    print("\n🤖 Step 3: Creating model")
    model_config = config_dict.get("model", {})
    
    # Add categorical features to model config if present
    if "categorical_features" in data_config:
        model_config["cat_features"] = data_config["categorical_features"]
        print(f"   Categorical features for model: {len(data_config['categorical_features'])}")
    
    # Create base model
    model = ModelFactory.create_model(
        model_config,
        random_state=config_dict.get("random_state", 42)
    )
    print(f"   Created: {model.__class__.__name__}")
    
    # Create hyperparameter search if enabled
    search_config = model_config.get("hyperparameter_search", {})
    param_search = ModelFactory.create_hyperparameter_search(
        model,
        search_config,
        random_state=config_dict.get("random_state", 42)
    )
    
    if param_search:
        print(f"   Hyperparameter search: {search_config.get('method', 'randomized')}")
        print(f"   Search iterations: {search_config.get('n_iter', 10)}")
    else:
        print("   Hyperparameter search: Disabled")
    
    # ------------------------------------------------------------------
    # 4. INITIALIZE FEATURE MANAGER WITH PREPROCESSING
    # ------------------------------------------------------------------
    print("\n🔧 Step 4: Initializing feature engineering")
    
    # Check if preprocessing is configured
    feature_engineering_config = config_dict.get("feature_engineering", {})
    has_preprocessing = "preprocessing" in feature_engineering_config
    
    if has_preprocessing:
        print(f"   📊 Preprocessing enabled for {model_type}")
        preprocess_config = feature_engineering_config.get("preprocessing", {})
        if preprocess_config.get("scale_features", False):
            print(f"      - Scaling: {preprocess_config.get('scaling_method', 'standard')}")
        if preprocess_config.get("categorical_encoding") == "onehot":
            print(f"      - Encoding: One-hot")
    else:
        print(f"   ⏭️ No preprocessing - using raw features for {model_type}")
    
    # Create feature manager with model type awareness
    feature_manager = EnhancedFeatureManager(
        feature_config={
            "operations": feature_engineering_config.get("operations", []),
            "preprocessing": feature_engineering_config.get("preprocessing", {}),
            "categorical_features": data_config.get("categorical_features", []),
            "categorical_handling": feature_engineering_config.get("categorical_handling", {})
        },
        model_type=model_type  # Pass model type for intelligent preprocessing
    )
    
    # Log feature engineering operations
    if feature_engineering_config.get("operations"):
        print(f"   Feature engineering operations: {len(feature_engineering_config['operations'])}")
        for op in feature_engineering_config['operations']:
            print(f"      - {op.get('name')}: {op.get('operation')} → {op.get('output')}")
    
    # ------------------------------------------------------------------
    # 5. INITIALIZE PIPELINE COMPONENTS
    # ------------------------------------------------------------------
    print("\n⚙️ Step 5: Initializing pipeline components")
    
    config_loader = ConfigLoader(config_dict=config_dict)
    data_manager = DataManager()
    
    trainer = Trainer(
        model=model,
        param_search=param_search,
        cv=config_dict.get("cv_folds", 5),
        scoring=search_config.get("scoring", "neg_root_mean_squared_error")
    )
    
    evaluator = Evaluator()
    tracker = Tracker()
    registry = Registry()
    
    print(f"   CV Folds: {config_dict.get('cv_folds', 5)}")
    print(f"   Validation size: {config_dict.get('val_size', 0.2)}")
    
    # ------------------------------------------------------------------
    # 6. CREATE AND RUN ORCHESTRATOR
    # ------------------------------------------------------------------
    print("\n🚀 Step 6: Running pipeline")
    
    orchestrator = TrainingOrchestrator(
        config_loader=config_loader,
        data_manager=data_manager,
        feature_manager=feature_manager,
        trainer=trainer,
        evaluator=evaluator,
        tracker=tracker,
        registry=registry
    )
    
    # Prepare data for orchestrator
    data = {"X": X, "y": y}
    
    # Run pipeline
    try:
        model, metrics, feature_importance = orchestrator.run(
            X, y,  # Pass X and y directly
            val_size=config_dict.get("val_size", 0.2),
            track_path=config_dict.get("track_path", "experiment_log.json"),
            save_path=config_dict.get("save_path", "model.pkl")
        )
        
        # ------------------------------------------------------------------
        # 7. DISPLAY RESULTS
        # ------------------------------------------------------------------
        print("\n" + "="*60)
        print("✅ EXPERIMENT COMPLETE")
        print("="*60)
        
        print("\n📊 Model Performance:")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                if isinstance(v, float):
                    print(f"   {k:25s}: {v:.4f}")
                else:
                    print(f"   {k:25s}: {v}")
        
        if feature_importance is not None:
            print("\n📈 Top 10 Feature Importances:")
            print(feature_importance.head(10).to_string(index=False))
        
        # Show engineered features
        if hasattr(feature_manager, 'feature_pipeline') and feature_manager.feature_pipeline:
            engineered = feature_manager.feature_pipeline.get_applied_features()
            if engineered:
                print(f"\n🔧 Engineered features: {', '.join(engineered)}")
        
        # Show preprocessing info
        if hasattr(feature_manager, 'preprocessing_pipeline') and feature_manager.preprocessing_pipeline:
            print(f"\n📊 Preprocessing applied:")
            if feature_manager.preprocessing_pipeline.config.get("scale_features"):
                print(f"   - Feature scaling: {feature_manager.preprocessing_pipeline.config.get('scaling_method', 'standard')}")
            encoding = feature_manager.preprocessing_pipeline._get_encoding_method()
            print(f"   - Categorical encoding: {encoding}")
        
        print(f"\n💾 Model saved to: {config_dict.get('save_path', 'model.pkl')}")
        print(f"📝 Log saved to: {config_dict.get('track_path', 'experiment_log.json')}")
        
        return model, metrics, feature_importance
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def print_config_summary(config_dict):
    """Print a summary of the configuration"""
    print("\n📋 Configuration Summary:")
    print(f"   Model: {config_dict.get('model', {}).get('type', 'unknown')}")
    
    data_config = config_dict.get('data', {})
    print(f"   Data file: {os.path.basename(data_config.get('file_path', 'unknown'))}")
    
    feature_config = config_dict.get('feature_engineering', {})
    if feature_config.get('operations'):
        print(f"   Feature engineering: {len(feature_config['operations'])} operations")
    if feature_config.get('preprocessing'):
        prep = feature_config['preprocessing']
        print(f"   Preprocessing: Scale={prep.get('scale_features', False)}, "
              f"Encoding={prep.get('categorical_encoding', 'none')}")

# Set how Command line run
if __name__ == "__main__":  # This line checks if the script is being run directly vs being imported:
    # Get config path from command line or use default, Command: python run_model.py --> defaul catboost
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Default to catboost config
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "configs", 
            "catboost_config.yaml"
        )
    
    # Check if file exists
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("\nAvailable configs:")
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
        if os.path.exists(config_dir):
            for file in os.listdir(config_dir):
                if file.endswith('.yaml') or file.endswith('.yml'):
                    print(f"   - {file}")
        sys.exit(1)
    
    # Run experiment
    run_experiment(config_path)





# ======================================================================================
# ======================================================================================
# ML_Models/experiments/run_model.py
# run_model without Preprocessing to process cat encode and scaling
'''
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
import yaml
from datetime import datetime

# Now import using simple names (no ML_Models prefix)
from core.pipeline import (
    ConfigLoader, DataManager, Trainer, Evaluator, 
    Tracker, Registry, TrainingOrchestrator
)
from feature_engineering.manager import FeatureManager as EnhancedFeatureManager
from models.factories import ModelFactory

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("MLExperiment")

def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_experiment(config_path: str):
    """Run ML experiment with configuration file"""
    
    print("\n" + "="*60)
    print("🚀 ML PIPELINE EXPERIMENT")
    print("="*60)
    print(f"📋 Config: {config_path}")
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load configuration
    print("\n📋 Step 1: Loading configuration")
    config_dict = load_yaml_config(config_path)
    print(f"   Experiment: {config_dict.get('experiment_name')}")
    print(f"   Model type: {config_dict.get('model', {}).get('type', 'unknown')}")
    
    # 2. Load data
    print("\n📂 Step 2: Loading data")
    data_config = config_dict.get("data", {})
    
    file_path = data_config.get("file_path")
    if not os.path.exists(file_path):
        # Try relative path from project root
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), file_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    if data_config.get("file_type") == "excel":
        df = pd.read_excel(
            file_path,
            usecols=data_config.get("columns_needed")
        )
    elif data_config.get("file_type") == "csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {data_config.get('file_type')}")
    
    target_column = data_config["target_column"]
    df_clean = df.dropna(subset=[target_column]).copy()
    
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    print(f"   Loaded {len(df_clean)} rows, {len(X.columns)} features")
    print(f"   Target: {target_column}")
    
    # 3. Create model
    print("\n🤖 Step 3: Creating model")
    model_config = config_dict.get("model", {})
    
    if "categorical_features" in data_config:
        model_config["cat_features"] = data_config["categorical_features"]
    
    model = ModelFactory.create_model(
        model_config,
        random_state=config_dict.get("random_state", 42)
    )
    
    search_config = model_config.get("hyperparameter_search", {})
    param_search = ModelFactory.create_hyperparameter_search(
        model,
        search_config,
        random_state=config_dict.get("random_state", 42)
    )
    
    print(f"   Model: {model_config.get('type')}")
    if param_search:
        print(f"   Hyperparameter search: {search_config.get('method')}")
    
    # 4. Initialize pipeline components
    print("\n🔧 Step 4: Initializing pipeline")
    
    config_loader = ConfigLoader(config_dict=config_dict)
    data_manager = DataManager()
    
    feature_manager = EnhancedFeatureManager(
        feature_config={
            "operations": config_dict.get("feature_engineering", {}).get("operations", []),
            "categorical_features": data_config.get("categorical_features", []),
            "categorical_handling": config_dict.get("feature_engineering", {}).get("categorical_handling", {})
        }
    )
    
    trainer = Trainer(
        model=model,
        param_search=param_search,
        cv=config_dict.get("cv_folds", 5),
        scoring=search_config.get("scoring", "neg_root_mean_squared_error")
    )
    
    evaluator = Evaluator()
    tracker = Tracker()
    registry = Registry()
    
    # 5. Create and run orchestrator
    print("\n🚀 Step 5: Running pipeline")
    
    orchestrator = TrainingOrchestrator(
        config_loader=config_loader,
        data_manager=data_manager,
        feature_manager=feature_manager,
        trainer=trainer,
        evaluator=evaluator,
        tracker=tracker,
        registry=registry
    )
    
    # Run pipeline
    model, metrics, feature_importance = orchestrator.run(
        X, y,
        val_size=config_dict.get("val_size", 0.2),
        track_path=config_dict.get("track_path", "experiment_log.json"),
        save_path=config_dict.get("save_path", "model.pkl")
    )
    
    # 6. Display results
    print("\n" + "="*60)
    print("✅ EXPERIMENT COMPLETE")
    print("="*60)
    
    print("\n📊 Model Performance:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"   {k:25s}: {v:.4f}" if isinstance(v, float) else f"   {k:25s}: {v}")
    
    if feature_importance is not None:
        print("\n📈 Top 5 Feature Importances:")
        print(feature_importance.head(5).to_string(index=False))
    
    if hasattr(feature_manager, 'feature_pipeline') and feature_manager.feature_pipeline:
        engineered = feature_manager.feature_pipeline.get_applied_features()
        if engineered:
            print(f"\n🔧 Engineered features: {', '.join(engineered)}")
    
    print(f"\n💾 Model saved to: {config_dict.get('save_path')}")
    print(f"📝 Log saved to: {config_dict.get('track_path')}")
    
    return model, metrics, feature_importance

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "catboost_config.yaml")
    
    run_experiment(config_path)
'''