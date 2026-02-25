# 🏗️ ML_Models

A production-ready machine learning pipeline for construction project cost prediction. Built with CatBoost, featuring comprehensive logging, cross-validation, and hyperparameter tuning.

## 📖 Overview

This pipeline predicts **Saving or Loss** for construction projects using historical data. It handles categorical features natively, performs feature engineering, and includes extensive testing.

### ✨ Key Features

- **Multiple Model Support**: CatBoost (with XGBoost/LightGBM ready to add)
- **Feature Engineering**: Configurable operations (subtract, divide, etc.)
- **Hyperparameter Tuning**: RandomizedSearchCV with YAML configs
- **Cross-Validation**: Robust performance estimation
- **Comprehensive Logging**: Console + file logging with timestamps
- **Experiment Tracking**: JSON-based experiment logging
- **Unit Testing**: 90%+ test coverage with pytest
- **Modular Design**: Easy to extend and maintain

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

### ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

Step 1: Clone the repository

```bash
git clone https://github.com/Phong031/ML_Models.git
cd ML_Models
```

```bash
Step 2: Create virtual environment (recommended)
# Windows
python -m venv venv
venv\Scripts\activate
# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

Step 3: Install dependencies

```bash
# For Users
pip install -e .

# For Developer - They get your package + dev tools:
pip install -r requirements.txt
```

Step 4: Install in development mode

```bash
pip install -e .
```

### 🚀 Quick Start

1. Prepare your data
   Place your Excel file in the data/ folder:

text
data/Jobs Data.xlsx

2. Run the pipeline
   bash
   python run.py

### ⚙️ Configuration

All settings are in YAML files.

### 📊 Usage Examples

Basic Usage
python
from experiments.run_model import run_experiment

#### Run with default config

run_experiment("configs/catboost_config.yaml")
Using in Jupyter Notebook
python

#### In your notebook

import sys
sys.path.append('path/to/ML_Models')

from core.pipeline import Trainer
from feature_engineering.manager import FeatureManager

#### Custom training

trainer = Trainer(model=my_model)
results = trainer.train(features)

### 📁 Project Structure

text

```bash
ML_Models/
├── 📂 core/
│ ├── pipeline.py # Main pipeline classes
│ └── **init**.py
├── 📂 feature_engineering/
│ ├── operations.py # Feature operations
│ └── manager.py # Feature management
├── 📂 models/
│ └── factories.py # Model factory
├── 📂 experiments/
│ └── run_model.py # Experiment runner
├── 📂 configs/
│ └── catboost_config.yaml # Configuration
├── 📂 tests/
│ ├── test_operations.py # Unit tests
│ ├── test_pipeline.py
│ └── conftest.py # Test fixtures
├── 📂 logs/ # Created at runtime
├── .gitignore
├── requirements.txt
├── setup.py
└── run.py # Main entry point
```

###🧪 Testing

```bash
# Run all tests
pytest tests/ -v
# Run with coverage
pytest --cov=core --cov=feature_engineering tests/
# Run specific test file
pytest tests/test_operations.py -v
# Generate HTML coverage report
pytest --cov-report=html tests/
```

### 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
📞 Contact & Support
Author: Phong
GitHub: @Phong031
Issues: Report a bug
