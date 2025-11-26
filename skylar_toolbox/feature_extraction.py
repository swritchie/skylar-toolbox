# =============================================================================
# Load libraries
# =============================================================================

import pandas as pd
import re
import string
from sklearn.feature_extraction import text as snfett

# =============================================================================
# CountVectorizer
# =============================================================================

class CountVectorizer(snfett.CountVectorizer):
    def fit(self, X, y=None):
        super().fit(raw_documents=X)
        return self
    def transform(self, X): return pd.DataFrame(
        data=super().transform(raw_documents=X).toarray(),
        columns=self.get_feature_names_out(),
        index=X.index)
    def fit_transform(self, X, y=None): return pd.DataFrame(
        data=super().fit_transform(raw_documents=X).toarray(),
        columns=self.get_feature_names_out(),
        index=X.index)
    def set_output(self, *, transform=None): pass

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

# =============================================================================
# remove_digits
# =============================================================================

def remove_digits(x, **kwargs): return re.sub(pattern=r'\d', repl='', string=x, **kwargs)

# =============================================================================
# remove_multiple_spaces
# =============================================================================

def remove_multiple_spaces(x, **kwargs): return re.sub(pattern=r'\s{2,}', repl=' ', string=x, **kwargs)

# =============================================================================
# remove_punctuation
# =============================================================================

def remove_punctuation(x, punctuation_sr=string.punctuation, **kwargs):
    return re.sub(pattern=fr'[{string.punctuation}]', repl='', string=x, **kwargs)

# =============================================================================
# remove_stop_words
# =============================================================================

def remove_stop_words(x, stop_words_st=snfett.ENGLISH_STOP_WORDS): 
    return ' '.join(y for y in x.split() if y not in stop_words_st)
