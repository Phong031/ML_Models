
import joblib, pandas as pd
model = joblib.load('decision_tree_pipeline.joblib')
raw = pd.read_excel('C:\Users\Phong\OneDrive - ICB Construction\Phong\data\Python_ETL\DS\ML_Models\data\Quotation Data.xlsx', sheet_name='Data').sample(3, random_state=7)
print('LOADED OK in a fresh process:', type(model).__name__)
print('Predictions:', list(model.predict(raw)))
print('Win probabilities:', list(model.predict_proba(raw)[:, 1].round(3)))
