# =============================================================================
# Load libraries
# =============================================================================

import pandas as pd
from sklearn.feature_extraction import text as snfett

# =============================================================================
# TfidfVectorizer
# =============================================================================

class TfidfVectorizer(snfett.TfidfVectorizer):
    def fit(self, X, y=None):
        super().fit(raw_documents=X)
        return self
    def transform(self, X): return pd.DataFrame(
        data=super().transform(raw_documents=X).toarray(),
        columns=self.get_feature_names_out(),
        index=X.index)
    def fit_transform(self, X, y=None):
        self.fit(X=X, y=y)
        return self.transform(X=X)
    def set_output(self, *, transform=None): pass
