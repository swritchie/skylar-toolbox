# =============================================================================
# Load libraries
# =============================================================================

import pandas as pd
import toolz as tz
from matplotlib import pyplot as plt

# =============================================================================
# aggregate_scores
# =============================================================================

def aggregate_scores(scores_df, test_bl): return (
    scores_df
    .agg(func=['mean', 'sem'], axis=1)
    .pipe(func=lambda x: x if not test_bl else x.assign(**{
        'best': lambda x: x['mean'].pipe(func=lambda x: x.eq(other=x.max())),
        'best_mean': lambda x: x['mean'].where(cond=x['best']).bfill().ffill(),
        'best_sem': lambda x: x['sem'].where(cond=x['best']).bfill().ffill(),
        'best_lower': lambda x: x['best_mean'].sub(other=x['best_sem']),
        'wi_1_sem': lambda x: x['mean'].gt(other=x['best_lower'])})))

# =============================================================================
# get_learning_curve_scores
# =============================================================================

def get_learning_curve_scores(lcd): return (
    pd.DataFrame(data=lcd.train_scores, index=lcd.train_sizes).pipe(func=aggregate_scores, test_bl=False),
    pd.DataFrame(data=lcd.test_scores, index=lcd.train_sizes).pipe(func=aggregate_scores, test_bl=True))

# =============================================================================
# get_validation_curve_scores
# =============================================================================

def get_validation_curve_scores(vcd): return (
    pd.DataFrame(data=vcd.train_scores, index=vcd.param_range).pipe(func=aggregate_scores, test_bl=False),
    pd.DataFrame(data=vcd.test_scores, index=vcd.param_range).pipe(func=aggregate_scores, test_bl=True))

# =============================================================================
# plot_test_scores
# =============================================================================

def plot_test_scores(test_scores_df):
    # Get best and simplest
    best_dt = test_scores_df.query(expr='best')['mean'].to_dict()
    wi_1_sem_dt = test_scores_df.query(expr='wi_1_sem')['mean'].iloc[[0]].to_dict()
    # Get title
    best_sr = tz.pipe(best_dt.items(), tz.curried.map(lambda x: 'Best: %d / %.3f score' % x), next)
    wi_1_sem_sr = tz.pipe(wi_1_sem_dt.items(), tz.curried.map(lambda x: 'W/i 1 SEM: %d / %.3f score' % x), next)
    title_sr = '\n'.join([best_sr, wi_1_sem_sr])
    # Plot
    fig, ax = plt.subplots()
    list(map(lambda x: ax.axvline(x=next(iter(x)), c='k', ls=':'), [best_dt, wi_1_sem_dt]))
    ax.axhline(y=test_scores_df['best_lower'].iloc[0], c='k', ls=':')
    test_scores_df.plot(y='mean', yerr='sem', marker='.', legend=False, ylabel='Mean +- SEM', title=title_sr, ax=ax)
    return ax
