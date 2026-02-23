# ML_Models/setup.py           --> Like shopping list for tools needed in your workshop
from setuptools import setup, find_packages

setup(
    name="ml_models",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "catboost>=1.0.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "pyyaml>=5.4.0",
        "joblib>=1.1.0",
        "openpyxl>=3.0.0",  # For Excel files
    ],
    author="Phong",
    description="ML Pipeline for training various models",
    python_requires=">=3.8",
)



# ============ HOW PROJECT RUN =========================
'''
# When you run: python run.py

# Step 1: run.py executes: python run.py configs/catboost_config.yaml 
from experiments.run_model import run_experiment
run_experiment("configs/catboost_config.yaml")

# Step 2: run_model.py reads config
config = yaml.load("configs/catboost_config.yaml")
# Gets: {"model": {"type": "catboost"}, ...}

# Step 3: Creates model based on config
from models.factories import ModelFactory
model = ModelFactory.create_model(config["model"])
# Creates a CatBoost model automatically!

# Step 4: Uses feature engineering
from feature_engineering.manager import FeatureManager
fm = FeatureManager(config["feature_engineering"])
# Creates Temp Profit feature because config said so

# Step 5: Trains using core pipeline
from core.pipeline import Trainer
trainer = Trainer(model)
trainer.train(features)  # All connected!


DATA FLOW:
1. YOU RUN: python run.py
              ↓
2. run.py → calls experiments/run_model.py
              ↓
3. run_model.py → reads configs/catboost_config.yaml
              ↓
4. Gets settings: "use CatBoost, split 80/20, create Temp Profit feature"
              ↓
5. run_model.py → imports from core/pipeline.py
              ↓
6. core/pipeline.py → uses feature_engineering/manager.py
              ↓
7. feature_engineering/manager.py → uses operations.py
              ↓
8. All tools work together → trains model → saves results
'''

'''
=========== init =============================
# Without __init__.py - Python is BLIND
ML_Models/
├── core/ 
│   └── pipeline.py  ← Python can't find this!

# With __init__.py - Python can SEE everything
ML_Models/
├── core/
│   ├── __init__.py  ← "Open for business!"
│   └── pipeline.py  ← Python can find this now

========= STRUCTURE: =========================
configs/        ← Recipe books (what to cook)
core/           ← Oven, stove (main appliances)
feature_engineering/ ← Knives, peelers (specialized tools)
models/         ← Cake pans, muffin trays (specific to what you're making)
experiments/    ← Test kitchen (try new recipes)
run.py          ← Head chef (coordinates everything)
'''