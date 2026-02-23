# ML_Models/run.py
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run_model import run_experiment

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔬 ML MODELS PIPELINE")
    print("="*60)
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1] # Takes whatever you type
    else:
        # Only used if you type nothing
        config_path = os.path.join(os.path.dirname(__file__), "configs", "catboost_config.yaml")
        print(f"\n📋 Using default config: {config_path}")
    '''catboost_config.yaml just the default model - run any model with command line argument below'''

    if not os.path.exists(config_path):
        print(f"\n❌ Config file not found: {config_path}")
        sys.exit(1)
    
    run_experiment(config_path)


'''
# From ML_Models folder
python run.py configs/neuralnetwork_config.yaml
python run.py configs/catboost_config.yaml  
python run.py configs/lightgbm_config.yaml
python run.py configs/xgboost_config.yaml


# Test Block
python -m pytest tests/test_operations.py -v
python -m pytest tests/test_manager.py -v
python -m pytest tests/test_pipeline.py -v

# Quick comparison
python run.py configs/catboost_config.yaml && python run.py configs/neuralnetwork_config.yaml
'''