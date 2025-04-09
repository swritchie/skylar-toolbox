# =============================================================================
# Load libraries
# =============================================================================

import itertools
import pandas as pd
import tqdm
from sklearn import base as snbe

# =============================================================================
# ConstantDropper
# =============================================================================

class ConstantDropper(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self): pass
    def fit(self, X, y=None):
        self.nunique_ss = X.nunique().sort_values()
        self.constant_ix = self.nunique_ss.pipe(func=lambda x: x[x.lt(other=2)]).index.sort_values()
        return self
    def transform(self, X): return X.drop(columns=self.constant_ix)
    def get_feature_names_out(): pass

# =============================================================================
# DuplicatedDropper
# =============================================================================

class DuplicatedDropper(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self): pass
    def fit(self, X, y=None):
        duplicated_lt = []
        for column_sr, column_sr2 in tqdm.tqdm(iterable=sorted(itertools.combinations(iterable=X.columns.sort_values(), r=2))):
            if X[column_sr].equals(other=X[column_sr2]): duplicated_lt.append(column_sr2)
        self.duplicated_ix = pd.Index(data=duplicated_lt).sort_values()
        return self
    def transform(self, X): return X.drop(columns=self.duplicated_ix)
    def get_feature_names_out(): pass
