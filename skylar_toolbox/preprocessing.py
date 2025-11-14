# =============================================================================
# Load libraries
# =============================================================================

import pandas as pd
from sklearn import base as snbe

# =============================================================================
# Replacer 
# =============================================================================

class Replacer(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self, to_replace, value): self.to_replace, self.value = to_replace, value
    def fit(self, X, y=None):
        self.values_ss = X.isin(values=self.to_replace).sum().sort_values(ascending=False)
        return self
    def transform(self, X): return X.replace(to_replace=self.to_replace, value=self.value)
    def get_feature_names_out(): pass

# =============================================================================
# TypeCaster
# =============================================================================

class TypeCaster(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self, dtype, features_lt=None): self.dtype, self.features_lt = dtype, features_lt
    def fit(self, X, y=None):
        self.old_dtypes_dt = X.dtypes.to_dict()
        if self.features_lt: self.new_dtypes_dt = {
            feature_sr: self.dtype if feature_sr in self.features_lt else dtype
            for feature_sr, dtype in self.old_dtypes_dt.items()}
        else: self.new_dtypes_dt = {feature_sr: self.dtype for feature_sr in X.columns}
        return self
    def transform(self, X): return X.astype(dtype=self.new_dtypes_dt)
    def get_feature_names_out(): pass

# =============================================================================
# TypeConverter
# =============================================================================

class TypeConverter(snbe.BaseEstimator, snbe.TransformerMixin):
    def fit(self, X, y=None): 
        self.dtypes_ss = X.convert_dtypes().dtypes
        return self
    def transform(self, X): return X.astype(dtype=self.dtypes_ss)
    def get_feature_names_out(): pass

# =============================================================================
# TypeDowncaster
# =============================================================================

class TypeDowncaster(snbe.BaseEstimator, snbe.TransformerMixin):
    def fit(self, X, y=None): 
        self.dtypes_ss = X.pipe(func=downcast_types).dtypes
        return self
    def transform(self, X): return X.astype(dtype=self.dtypes_ss)
    def get_feature_names_out(): pass

# =============================================================================
# downcast_types
# =============================================================================

def downcast_types(df): return df.pipe(func=lambda x: pd.concat(objs=[
    x.select_dtypes(exclude='number').apply(func=pd.to_datetime, errors='ignore'),
    x.select_dtypes(include=int).apply(func=pd.to_numeric, errors='ignore', downcast='integer'),
    x.select_dtypes(include=float).apply(func=pd.to_numeric, errors='ignore', downcast='float')], axis=1))
