# ML_Models/run_prediction.py
#!/usr/bin/env python
"""
Simple prediction script for trained models
python run_prediction.py --model catboost_model.pkl --data data.xlsx --output results.xlsx
"""

import argparse
import sys
import os
import pandas as pd
import joblib
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def predict(model_path, data_path, output_path):
    """
    Load model, apply feature engineering, and predict on new data
    """
    print("\n" + "="*60)
    print("🔮 PREDICTION PIPELINE")
    print("="*60)
    
    # Step 1: Load model
    print(f"\n📂 Step 1: Loading model from {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"   ✅ Model loaded: {model.__class__.__name__}")
    
    # Step 2: Load new data
    print(f"\n📂 Step 2: Loading data from {data_path}")
    columns_needed = [
        "Job Number", "Job Value", "Total Claimed", "Total Cost",
        "Estimator", "Foreman", "Job Area", "Main Contractor",
        "Suburb", "Supervisor", "Job Description"
    ]
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path, usecols=columns_needed)
    else:
        df = pd.read_excel(data_path, usecols=columns_needed)
    
    print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Step 3: Apply feature engineering
    print("\n🔧 Step 3: Applying feature engineering")
    df["Temp Profit"] = df["Total Claimed"] - df["Total Cost"]
    print(f"   ✅ Created 'Temp Profit' feature")
    
    # Step 4: Handle categorical features
    print("\n🏷️ Step 4: Handling categorical features")
    cat_features = [
        'Estimator', 'Foreman', 'Job Area', 'Main Contractor',
        'Suburb', 'Supervisor', 'Job Description'
    ]
    
    for col in cat_features:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"   ⚠ Column '{col}' had {nan_count} missing values")
            df[col] = df[col].fillna('Unknown').astype(str)
    
    print(f"   ✅ Categorical features processed")
    
    # Step 5: Make predictions
    print("\n🤖 Step 5: Making predictions")
    predictions = model.predict(df)
    print(f"   ✅ Generated {len(predictions)} predictions")
    
    # Summary statistics
    print(f"\n📊 Prediction Summary:")
    print(f"   Min: {predictions.min():.2f}")
    print(f"   Max: {predictions.max():.2f}")
    print(f"   Mean: {predictions.mean():.2f}")
    print(f"   Std: {predictions.std():.2f}")
    
    # Step 6: Prepare results
    print("\n💾 Step 6: Preparing results")
    df["Predicted Saving or Loss"] = predictions
    df["Prediction_Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Step 7: Save results
    print(f"\n💾 Step 7: Saving results to {output_path}")
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    else:
        df.to_excel(output_path, index=False)
    
    print(f"   ✅ Saved {len(df)} rows with predictions")
    print("\n" + "="*60)
    print("✅ PREDICTION COMPLETE")
    print("="*60)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model', '-m', required=True, help='Path to trained model (.pkl file)')
    parser.add_argument('--data', '-d', required=True, help='Path to new data file (Excel or CSV)')
    parser.add_argument('--output', '-o', required=True, help='Path to save predictions')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.model):
        print(f"\n❌ Error: Model not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"\n❌ Error: Data not found: {args.data}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n📁 Created output directory: {output_dir}")
    
    # Run prediction
    try:
        predict(args.model, args.data, args.output)
    except Exception as e:
        print(f"\n❌ Error during prediction: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Run Prediction with data path
'''
python run_prediction.py 
--model catboost_model.pkl 
--data "C:/Users/Phong/OneDrive - ICB Construction/Phong/data/Python_ETL/DS/Database/Jobs to be Predicted.xlsx" 
--output "C:/Users/Phong/OneDrive - ICB Construction/Phong/data/Python_ETL/DS/Database/CatBoost Result Project.xlsx"
'''