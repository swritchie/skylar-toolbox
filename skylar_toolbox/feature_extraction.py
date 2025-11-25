# =============================================================================
# Load libraries
# =============================================================================

import pandas as pd
from sklearn import base as snbe
from sklearn.feature_extraction import text as snfett

# =============================================================================
# TfidfVectorizer
# =============================================================================

class TfidfVectorizer(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self, **kwargs): self.vectorizer = snfett.TfidfVectorizer(**kwargs)
    def fit(self, X, y=None):
        self.vectorizer.fit(raw_documents=X)
        return self
    def transform(self, X): return pd.DataFrame(
        data=self.vectorizer.transform(raw_documents=X).todense(), 
        columns=self.vectorizer.get_feature_names_out(),
        index=X.index)
    def get_feature_names_out(): pass
