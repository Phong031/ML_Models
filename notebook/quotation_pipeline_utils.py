"""
Shared cleaning/feature-engineering step for the Quotation Win/Loss project.

Import QuotationCleaner (and the two helper functions) from this file rather than
redefining them in each notebook. This is required, not just tidier: a class defined
inside a notebook lives in that notebook's throwaway namespace, so joblib/pickle can save
a *reference* to it but a fresh Python process has nowhere to resolve that reference from
when loading the model back -- it needs to import the real class from a real module.
Keep this file in the same folder as the notebooks and any prediction script.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

WALL_TYPE_COLS = ['Timber_RW','RC_Pile','Steel_Beam','Sheetpile','Anchor','Block','Shotcrete',
                  'Capping_Beam','Earthwork','Concrete_Slab','Precast','Culvert','Slip_Repair',
                  'Soil_Nail','Rock_RW','Bridge','Concrete','Design_and_Build','Budget','Drill_Only',
                  'Labour_Only','Driven_Pile','Palisade','Boardwalk','Soldier','Insitu','Barrier',
                  'Noise_RW','Base','Casing','Crib','DayWork','Flood_Repair','Micro_Pile','Reno',
                  'Temp_RW','Other']
NUMERIC_FEATURES = ['Value_Log', 'No_Wall_Types', 'Quote_Year', 'Month_Sin', 'Month_Cos']
CATEGORICAL_FEATURES = ['Client_Clean', 'Suburb', 'Priced_By']
DROP_COLS = ['Successful', 'number_of_successful', 'JOB NO', 'CLIENT', 'ADDRESS',
             'CONTACTS', 'Contact_Clean', 'Contact_Number_Clean', 'DESCRIPTION',
             'Due_Date', 'Date_Sent']
SENTINELS = {'missing', 'n/a', 'na', 'unknown', 'none', 'null', '-', 'tbc', 'tbd', '?', ''}


class QuotationCleaner(BaseEstimator, TransformerMixin):
    """fit() learns which wall-type flags are 'rare' (< rare_walltype_pct% prevalence) from
    training data only, and freezes that decision. transform() applies all cleaning/feature
    engineering -- safe to call on training data, the test set, or a brand new raw quotation,
    always consistently.

    Deliberately does NOT drop rows (e.g. for an invalid Date): a transformer that changes
    row count breaks the implicit X/y alignment sklearn pipelines assume, and at prediction
    time every input row should get a prediction, not a silent drop. See filter_valid_rows()
    below for the training-time-only row filter.
    """
    def __init__(self, rare_walltype_pct=1.0):
        self.rare_walltype_pct = rare_walltype_pct

    def _clean_sentinels(self, df):
        df = df.copy()
        text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        for c in text_cols:
            s = df[c].astype('string').str.strip()
            mask = s.str.lower().isin(SENTINELS)
            df.loc[mask, c] = pd.NA
        return df

    def fit(self, X, y=None):
        df = self._clean_sentinels(X)
        wall_prevalence = df[WALL_TYPE_COLS].mean() * 100
        self.common_walls_ = wall_prevalence[wall_prevalence >= self.rare_walltype_pct].index.tolist()
        self.rare_walls_ = wall_prevalence[wall_prevalence < self.rare_walltype_pct].index.tolist()
        self.binary_features_ = self.common_walls_ + ['Rare_WorkType', 'Multi_Estimator']
        self.feature_cols_ = NUMERIC_FEATURES + self.binary_features_ + CATEGORICAL_FEATURES
        return self

    def transform(self, X):
        df = self._clean_sentinels(X)
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Quote_Year'] = df['Date'].dt.year
        df['Month_Sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
        df['Value_Log'] = np.log1p(df['Value'].clip(lower=0))

        df['Rare_WorkType'] = (df[self.rare_walls_].sum(axis=1) > 0).astype(int)

        n_est = df['Priced_By'].fillna('').astype(str).str.split('/').apply(
            lambda parts: len([p for p in parts if p.strip() != ''])
        )
        df['Multi_Estimator'] = (n_est > 1).astype(int)

        df['Client_Clean'] = df['Client_Clean'].fillna('Unknown_Client')
        df['Suburb'] = df['Suburb'].fillna('Unknown_Suburb')
        df['Priced_By'] = df['Priced_By'].fillna('Unknown_Estimator').astype('string').str.strip()

        return df[self.feature_cols_]


def filter_valid_rows(df_raw, date_col='Date'):
    """Training-time-only data quality gate -- NOT part of the reusable prediction pipeline.
    Drops rows with no parseable quote Date. At prediction time a row with a bad Date instead
    flows through QuotationCleaner and gets its date features median-imputed downstream by
    the model's own imputer, rather than dropped -- a reasonable fallback, but worth flagging
    such rows for manual review rather than trusting that prediction blindly."""
    d = pd.to_datetime(df_raw[date_col], errors='coerce')
    before = len(df_raw)
    out = df_raw[d.notna()].copy()
    print(f'Dropped {before - len(out)} rows with missing/invalid quote Date ({(before-len(out))/before*100:.1f}%)')
    return out


def select_binary_features(X):
    """Column selector for ColumnTransformer: everything QuotationCleaner produced that
    isn't numeric or categorical. A plain module-level function rather than a lambda --
    lambdas can't be pickled by joblib, which would silently break model saving/loading."""
    return [c for c in X.columns if c not in NUMERIC_FEATURES and c not in CATEGORICAL_FEATURES]
