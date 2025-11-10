# =============================================================================
# Load libraries
# =============================================================================

import pandas as pd
from sklearn import metrics as snmes

# =============================================================================
# get_cumulative_predictions
# =============================================================================

def get_cumulative_predictions(predictions_df, classification_bl):
    inner_fn = lambda x: x.mode().iloc[0] if classification_bl else x.mean()
    outer_fn = lambda x: predictions_df.loc[:, :x].apply(func=inner_fn, axis=1)
    return predictions_df.pipe(func=lambda x: pd.concat(objs=map(outer_fn, range(1, x.shape[1] + 1)), axis=1, keys=x.columns))

# =============================================================================
# get_cumulative_scores
# =============================================================================

def get_cumulative_scores(cumulative_predictions_df, y, classification_bl):
    fn = lambda x: snmes.balanced_accuracy_score(y_true=y, y_pred=x) if classification_bl else snmes.r2_score(y_true=y, y_pred=x)
    return (cumulative_predictions_df
        .apply(func=fn)
        .to_frame(name='scores')
        .assign(**{'cummax': lambda x: x['scores'].cummax()}))

# =============================================================================
# get_predictions
# =============================================================================

def get_predictions(random_forest, X, classification_bl):
    fn = lambda x: pd.Series(data=x.predict(X=X), index=X.index)
    return (pd.concat(objs=map(fn, random_forest.estimators_), axis=1)
        .rename(columns=lambda x: x + 1)
        .pipe(func=lambda x: x.astype(dtype=int) if classification_bl else x))
