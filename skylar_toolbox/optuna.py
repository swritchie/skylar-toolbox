# =============================================================================
# Load libraries
# =============================================================================

from matplotlib import pyplot as plt
from optuna.visualization import matplotlib as oavnmb

# =============================================================================
# TrialsSinceBestCallback
# =============================================================================

class TrialsSinceBestCallback:
    def __init__(self, burn_in_period_it, wait_period_it, weights_lt=None): 
        self.burn_in_period_it, self.wait_period_it, self.weights_lt = \
            burn_in_period_it, wait_period_it, weights_lt
    def __call__(self, study, trial):
        # Check whether burn-in has occurred
        current_trial_number_it = trial.number
        burned_in_bl = current_trial_number_it >= self.burn_in_period_it
        # Check whether wait has expired
        trials_df, best_trials_df = get_best_trial(study=study, weights_lt=self.weights_lt)
        best_trial_number_it = best_trials_df.query(expr='best')['number'].squeeze()
        waited_bl = current_trial_number_it - best_trial_number_it >= self.wait_period_it
        if burned_in_bl and waited_bl: study.stop()

# =============================================================================
# get_best_trial
# =============================================================================

def get_best_trial(study, weights_lt=None):
    # Get complete trials
    trials_df = study.trials_dataframe().query(expr='state.eq(other="COMPLETE")')
    # Get best trial numbers
    best_trial_numbers_lt = list(map(lambda x: x.number, study.best_trials))
    # Sort, identify best trial, and filter
    n_objectives_it = len(study.directions)
    lower_is_better_dt = dict(map(
        lambda x: [f'values_{x[0]}' if n_objectives_it > 1 else 'value', x[1].name == 'MINIMIZE'],
        enumerate(iterable=study.directions)))
    sort_sr = list(lower_is_better_dt.keys())[0]
    ranks_dt = dict(map(
        lambda x: [f'{x[0]}_rank', lambda y: y[x[0]].rank(ascending=x[1], pct=True)],
        lower_is_better_dt.items()))
    weights_lt = weights_lt if weights_lt else [1] * n_objectives_it 
    ranks_dt.update(
        weighted_rank=lambda x: x.filter(like='_rank').dot(other=weights_lt), 
        best=lambda x: x['weighted_rank'].pipe(func=lambda y: y.eq(other=y.min())))
    best_trials_df = (trials_df
        .sort_values(by=sort_sr)
        .assign(**ranks_dt)
        .query(expr=f'number.isin(values={best_trial_numbers_lt})'))
    return trials_df, best_trials_df

# =============================================================================
# plot_study
# =============================================================================

def plot_study(study, outputs_directory_ph, notebook_bl):
    for function_sr in filter(lambda x: x.startswith('plot'), dir(oavnmb)):
        try:
            getattr(oavnmb, function_sr)(study=study)
            plt.savefig(fname=outputs_directory_ph/function_sr[5:], bbox_inches='tight')
        except Exception as en:
            print(en.__class__, en)
            try:
                for i_it in range(len(study.directions)):
                    getattr(oavnmb, function_sr)(study=study, target=lambda x: x.values[i_it])
                    plt.savefig(fname=outputs_directory_ph/f'{function_sr[5:]}_target_{i_it}', bbox_inches='tight')
            except Exception as en: print(en.__class__, en)
        if notebook_bl: plt.show()
        plt.close()
