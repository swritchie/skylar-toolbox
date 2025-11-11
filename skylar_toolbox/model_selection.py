# =============================================================================
# Load libraries
# =============================================================================

import pandas as pd
import toolz as tz
from matplotlib import pyplot as plt
from sklearn import model_selection as snmos 

# =============================================================================
# plot_warm_start_validation_curve 
# =============================================================================

def plot_warm_start_validation_curve(scores_df):
    # Get features/scores associated with best number of features (and number within 1 SEM)
    best_dt = scores_df.query(expr='best')['mean'].to_dict()
    wi_1_sem_dt = scores_df.query(expr='wi_1_sem')['mean'].iloc[[0]].to_dict()
    # Get title
    best_sr = tz.pipe(best_dt.items(), tz.curried.map(lambda x: 'Best: %d / %.3f score' % x), next)
    wi_1_sem_sr = tz.pipe(wi_1_sem_dt.items(), tz.curried.map(lambda x: 'W/i 1 SEM: %d / %.3f score' % x), next)
    title_sr = '\n'.join([best_sr, wi_1_sem_sr])
    # Plot
    fig, ax = plt.subplots()
    list(map(lambda x: ax.axvline(x=next(iter(x)), c='k', ls=':'), [best_dt, wi_1_sem_dt]))
    ax.axhline(y=scores_df['best_lower'].iloc[0], c='k', ls=':')
    scores_df.plot(y='mean', yerr='sem', marker='.', legend=False, ylabel='Mean +- SEM', title=title_sr, ax=ax)
    return ax

# =============================================================================
# warm_start_validation_curve 
# =============================================================================

def warm_start_validation_curve(estimator, X, y, param_name_sr, param_range, cross_val_score_dt=dict()):
    scores_dt = {}
    for i_it, param_it in enumerate(iterable=param_range):
        estimator.set_params(**{param_name_sr: param_it})
        estimator.fit(X=X, y=y)
        scores_ay = snmos.cross_val_score(estimator=estimator, X=X, y=y, **cross_val_score_dt)
        scores_dt[param_it] = pd.Series(data=scores_ay).agg(func=['mean', 'sem']).to_dict()
    return (pd.DataFrame(data=scores_dt)
        .T
        .rename_axis(index=param_name_sr)
        .assign(**{
            'best': lambda x: x['mean'].pipe(func=lambda x: x.eq(other=x.max())),
            'best_mean': lambda x: x['mean'].where(cond=x['best']).bfill().ffill(),
            'best_sem': lambda x: x['sem'].where(cond=x['best']).bfill().ffill(),
            'best_lower': lambda x: x['best_mean'].sub(other=x['best_sem']),
            'wi_1_sem': lambda x: x['mean'].gt(other=x['best_lower'])}))
