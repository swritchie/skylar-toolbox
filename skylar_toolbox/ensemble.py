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

# =============================================================================
# plot_cumulative_predictions 
# =============================================================================

def plot_cumulative_scores(cumulative_scores_df):
    fewest_best_dt = cumulative_scores_df['cummax'].pipe(func=lambda x: x[x.eq(other=x.max())]).iloc[[0]].to_dict()
    title_sr = tz.pipe(fewest_best_dt.items(), tz.curried.map(lambda x: 'Estimators: %d\nScores: %.3f' % x), '\n'.join)
    ax = cumulative_scores_df.plot(drawstyle='steps-mid', title=title_sr)
    ax.axvline(x=next(iter(fewest_best_dt)), c='k', ls=':')
    return ax
