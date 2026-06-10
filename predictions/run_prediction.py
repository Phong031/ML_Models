# ML_Models/run_prediction.py
#!/usr/bin/env python
"""
Universal prediction script for trained models
Handles CatBoost, XGBoost, LightGBM, and Neural Networks
"""

import argparse
import sys
import os
import yaml
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import feature operations
from feature_engineering.operations import OperationRegistry


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def clean_column_names(df):
    """Clean column names: replace spaces with underscores"""
    df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
    return df


def apply_feature_operations(df, operations_config):
    """Apply feature engineering operations from config"""
    print("   🔧 Applying feature engineering from config...")
    
    for op_config in operations_config:
        op_name = op_config.get("operation") or op_config.get("type")
        if not op_name:
            continue
        
        # Clean column names
        inputs = [col.replace(' ', '_') for col in op_config["inputs"]]
        output = op_config["output"].replace(' ', '_')
        
        try:
            operation_func = OperationRegistry.get(op_name)
            df = operation_func(df, input_cols=inputs, output_col=output)
            print(f"      ✅ Created '{output}'")
        except Exception as e:
            print(f"      ❌ Failed: {e}")
            raise
    
    return df


def encode_categorical_features(df, categorical_features, encoders=None):
    """
    Encode categorical features using saved encoders or apply label encoding
    """
    print("   🏷️ Encoding categorical features...")
    
    if encoders:
        # Use saved encoders from training
        print("      Using saved encoders from training")
        for col in categorical_features:
            col_clean = col.replace(' ', '_')
            if col_clean in df.columns and col_clean in encoders:
                encoder = encoders[col_clean]
                # Handle unseen labels
                df[col_clean] = df[col_clean].fillna('Unknown').astype(str)
                try:
                    df[col_clean] = encoder.transform(df[col_clean])
                except ValueError:
                    # For unseen labels, map to -1 or most frequent
                    df[col_clean] = df[col_clean].apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                    )
                print(f"      ✅ Encoded '{col_clean}'")
    else:
        # No encoders - apply label encoding on the fly
        print("      No saved encoders - applying label encoding")
        for col in categorical_features:
            col_clean = col.replace(' ', '_')
            if col_clean in df.columns:
                df[col_clean] = df[col_clean].fillna('Unknown').astype(str)
                df[col_clean] = df[col_clean].astype('category').cat.codes
                print(f"      ✅ Label encoded '{col_clean}'")
    
    return df


def handle_categorical_features(df, categorical_features, model_type, encoders=None):
    """
    Handle categorical features based on model type
    """
    if not categorical_features:
        return df
    
    # Clean column names in categorical_features list
    categorical_features_clean = [col.replace(' ', '_') for col in categorical_features]
    
    if model_type == 'catboost':
        # CatBoost: keep as strings
        print("   🏷️ CatBoost: keeping categorical features as strings")
        for col in categorical_features_clean:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)
                print(f"      ✅ Processed '{col}'")
    
    elif model_type in ['xgboost', 'lightgbm', 'randomforest']:
        # XGBoost/LightGBM: need numeric encoding
        df = encode_categorical_features(df, categorical_features, encoders)
    
    elif model_type in ['neuralnetwork', 'nn', 'mlp']:
        # Neural networks: should already be one-hot encoded
        print("   🧠 Neural network: ensuring categorical features are encoded")
        df = encode_categorical_features(df, categorical_features, encoders)
    
    return df


def load_encoders(model_path):
    """Try to load saved encoders"""
    encoder_path = model_path.replace('.pkl', '_encoders.pkl')
    if os.path.exists(encoder_path):
        return joblib.load(encoder_path)
    return None


def get_model_type(model):
    """Determine model type from loaded model"""
    model_class = model.__class__.__name__.lower()
    
    if 'catboost' in model_class:
        return 'catboost'
    elif 'xgb' in model_class:
        return 'xgboost'
    elif 'lgbm' in model_class or 'lightgbm' in model_class:
        return 'lightgbm'
    elif 'randomforest' in model_class:
        return 'randomforest'
    elif 'mlp' in model_class or 'neural' in model_class:
        return 'neuralnetwork'
    else:
        return 'unknown'


def predict(model_path, data_path, output_path, config_path):
    """Main prediction function"""
    print("\n" + "="*60)
    print("🔮 PREDICTION PIPELINE")
    print("="*60)
    
    # ------------------------------------------------------------------
    # Step 1: Load configuration
    # ------------------------------------------------------------------
    print(f"\n📋 Step 1: Loading configuration")
    config = load_config(config_path)
    data_config = config.get('data', {})
    feature_config = config.get('feature_engineering', {})
    
    # Get feature columns
    feature_columns = data_config.get('feature_columns')
    columns_needed = data_config.get('columns_needed', [])
    target_column = data_config.get('target_column')
    
    if feature_columns:
        required_columns = feature_columns
    elif columns_needed and target_column:
        required_columns = [col for col in columns_needed if col != target_column]
    else:
        raise ValueError("Config must have 'feature_columns' or 'columns_needed'")
    
    categorical_features = data_config.get('categorical_features', [])
    operations_config = feature_config.get('operations', [])
    
    print(f"   Experiment: {config.get('experiment_name', 'Unknown')}")
    print(f"   Feature columns: {len(required_columns)}")
    print(f"   Categorical features: {len(categorical_features)}")
    print(f"   Feature operations: {len(operations_config)}")
    
    # ------------------------------------------------------------------
    # Step 2: Load model
    # ------------------------------------------------------------------
    print(f"\n📂 Step 2: Loading model from {model_path}")
    model = joblib.load(model_path)
    model_type = get_model_type(model)
    print(f"   ✅ Model loaded: {model.__class__.__name__} (Type: {model_type})")
    
    # Try to load saved encoders
    encoders = load_encoders(model_path)
    if encoders:
        print(f"   ✅ Loaded {len(encoders)} saved encoders")
    
    # ------------------------------------------------------------------
    # Step 3: Load data
    # ------------------------------------------------------------------
    print(f"\n📂 Step 3: Loading data from {data_path}")
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)
    
    df = clean_column_names(df)
    print(f"   ✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # ------------------------------------------------------------------
    # Step 4: Select required columns
    # ------------------------------------------------------------------
    print(f"\n📋 Step 4: Selecting required columns")
    required_columns_clean = [col.replace(' ', '_') for col in required_columns]
    
    missing_cols = [col for col in required_columns_clean if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    df = df[required_columns_clean].copy()
    print(f"   ✅ Selected {len(df.columns)} feature columns")
    
    # ------------------------------------------------------------------
    # Step 5: Apply feature engineering
    # ------------------------------------------------------------------
    print(f"\n🔧 Step 5: Applying feature engineering")
    if operations_config:
        df = apply_feature_operations(df, operations_config)
    else:
        print("   ⚠ No feature engineering operations")
    
    # ------------------------------------------------------------------
    # Step 6: Handle categorical features
    # ------------------------------------------------------------------
    df = handle_categorical_features(df, categorical_features, model_type, encoders)
    
    # ------------------------------------------------------------------
    # Step 7: Validate model features
    # ------------------------------------------------------------------
    print(f"\n🔍 Step 6: Validating model features")
    
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        print(f"   Model expects {len(model_features)} features")
        
        missing = [f for f in model_features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        df = df[model_features]
        print(f"   ✅ Features match model expectations")
    
    # Check dtypes for XGBoost/LightGBM
    if model_type in ['xgboost', 'lightgbm']:
        print(f"\n🔍 Checking data types for {model_type.upper()}...")
        for col in df.columns:
            if df[col].dtype == 'object':
                raise ValueError(f"Column '{col}' is still object type. Encoding failed.")
        
        # Ensure numeric types
        df = df.astype(float)
        print(f"   ✅ All features are numeric")
    
    # ------------------------------------------------------------------
    # Step 8: Make predictions
    # ------------------------------------------------------------------
    print(f"\n🤖 Step 7: Making predictions")
    predictions = model.predict(df)
    print(f"   ✅ Generated {len(predictions)} predictions")
    
    print(f"\n📊 Prediction Summary:")
    print(f"   Min: {predictions.min():.2f}")
    print(f"   Max: {predictions.max():.2f}")
    print(f"   Mean: {predictions.mean():.2f}")
    print(f"   Std: {predictions.std():.2f}")
    
    # ------------------------------------------------------------------
    # Step 9: Save results
    # ------------------------------------------------------------------
    print(f"\n💾 Step 8: Saving results")
    
    # Reload original data for output
    if data_path.endswith('.csv'):
        output_df = pd.read_csv(data_path)
    else:
        output_df = pd.read_excel(data_path)
    
    prediction_col = f"Predicted_{target_column.replace(' ', '_') if target_column else 'Target'}"
    output_df[prediction_col] = predictions
    output_df["Prediction_Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_df["Model_Type"] = model_type
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if output_path.endswith('.csv'):
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_excel(output_path, index=False)
    
    print(f"   ✅ Saved to {output_path}")
    print("\n" + "="*60)
    print("✅ PREDICTION COMPLETE")
    print("="*60)
    
    return output_df


def main():
    parser = argparse.ArgumentParser(description='Universal prediction with trained model')
    parser.add_argument('--model', '-m', required=True, help='Path to trained model (.pkl file)')
    parser.add_argument('--config', '-c', required=True, help='Path to training config YAML')
    parser.add_argument('--data', '-d', required=True, help='Path to new data file')
    parser.add_argument('--output', '-o', required=True, help='Path to save predictions')
    
    args = parser.parse_args()
    
    # Check files
    for f in [args.model, args.config, args.data]:
        if not os.path.exists(f):
            print(f"\n❌ Error: File not found: {f}")
            sys.exit(1)
    
    # Run prediction
    try:
        predict(args.model, args.data, args.output, args.config)
    except Exception as e:
        print(f"\n❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
