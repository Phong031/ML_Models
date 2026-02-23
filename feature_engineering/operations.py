# ML_Models/feature_engineering/operations.py
import pandas as pd
import numpy as np
from typing import List, Dict, Callable
import logging

logger = logging.getLogger(__name__)

class FeatureOperations:
    """Library of generic feature engineering operations"""
    
    @staticmethod
    def subtract(df: pd.DataFrame, input_cols: List[str], output_col: str) -> pd.DataFrame:
        """Subtract second column from first"""
        if all(col in df.columns for col in input_cols):
            df[output_col] = df[input_cols[0]] - df[input_cols[1]]
        return df
    
    @staticmethod
    def add(df: pd.DataFrame, input_cols: List[str], output_col: str) -> pd.DataFrame:
        """Add columns"""
        if all(col in df.columns for col in input_cols):
            df[output_col] = sum(df[col] for col in input_cols)
        return df
    
    @staticmethod
    def multiply(df: pd.DataFrame, input_cols: List[str], output_col: str) -> pd.DataFrame:
        """Multiply columns"""
        if all(col in df.columns for col in input_cols):
            result = 1
            for col in input_cols:
                result *= df[col]
            df[output_col] = result
        return df
    
    @staticmethod
    def divide(df: pd.DataFrame, input_cols: List[str], output_col: str) -> pd.DataFrame:
        """Divide first column by second"""
        if all(col in df.columns for col in input_cols):
            df[output_col] = df[input_cols[0]] / df[input_cols[1]].replace(0, np.nan)
        return df
    
    @staticmethod
    def divide_percent(df: pd.DataFrame, input_cols: List[str], output_col: str) -> pd.DataFrame:
        """Divide first by second and multiply by 100"""
        if all(col in df.columns for col in input_cols):
            df[output_col] = (df[input_cols[0]] / df[input_cols[1]].replace(0, np.nan)) * 100
        return df
    
    @staticmethod
    def log_transform(df: pd.DataFrame, input_cols: List[str], output_col: str) -> pd.DataFrame:
        """Apply log1p transformation"""
        if input_cols[0] in df.columns:
            df[output_col] = np.log1p(df[input_cols[0]])
        return df

class OperationRegistry:
    """Registry of available operations"""
    
    _operations = {
        "subtract": FeatureOperations.subtract,
        "add": FeatureOperations.add,
        "multiply": FeatureOperations.multiply,
        "divide": FeatureOperations.divide,
        "divide_percent": FeatureOperations.divide_percent,
        "log": FeatureOperations.log_transform,
    }
    
    @classmethod
    def get(cls, name: str) -> Callable:
        """Get operation by name"""
        if name not in cls._operations:
            raise ValueError(f"Operation '{name}' not found. Available: {list(cls._operations.keys())}")
        return cls._operations[name]