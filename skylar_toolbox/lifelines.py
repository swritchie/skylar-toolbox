# =============================================================================
# Load libraries
# =============================================================================

import lifelines
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import base as snbe

# =============================================================================
# DurationCalculator
# =============================================================================

class DurationCalculator(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self, start_sr, end_sr, datetimes_to_durations_dt=dict()):
        self.start_sr, self.end_sr, self.datetimes_to_durations_dt = start_sr, end_sr, datetimes_to_durations_dt
    def fit(self, X, y=None):
        self.duration_sr = self._get_duration(start_sr=self.start_sr, end_sr=self.end_sr)
        self.event_flag_sr = self._get_event_flag(end_sr=self.end_sr)
        return self
    def transform(self, X):
        durations_te = lifelines.utils.datetimes_to_durations(
            start_times=X[self.start_sr], end_times=X[self.end_sr], **self.datetimes_to_durations_dt)
        durations_df = pd.DataFrame(data=dict(zip([self.duration_sr, self.event_flag_sr], durations_te)), index=X.index)
        return X.join(other=durations_df, how='left', validate='one_to_one')
    def get_feature_names_out(): pass
    @staticmethod
    def _get_duration(start_sr, end_sr): return f'time_from_{start_sr}_to_{end_sr}'
    @staticmethod
    def _get_event_flag(end_sr): return f'{end_sr}_flag'

# =============================================================================
# plot_feature
# =============================================================================

def plot_feature(
    feature_ss, duration_ss, event_flag_ss,
    ax=None, Fitter=lifelines.KaplanMeierFitter, init_dt={}, fit_dt={}, plot_dt={}, add_at_risk_counts_dt={}):
    fitters_lt = []
    ax = ax if ax else plt.subplot(1, 1, 1)
    for value in feature_ss.drop_duplicates().sort_values():
        # Filter
        filtered_feature_ss = feature_ss.pipe(func=lambda x: x[x.eq(other=value)])
        fn = lambda x: x.loc[filtered_feature_ss.index]
        filtered_duration_ss, filtered_event_flag_ss = map(fn, [duration_ss, event_flag_ss])
        # Initialize
        fitter = Fitter(**init_dt)
        # Fit
        fitter.fit(durations=filtered_duration_ss, event_observed=filtered_event_flag_ss, label=value, **fit_dt)
        fitters_lt.append(fitter)
        # Plot
        fitter.plot(ax=ax, **plot_dt)
    lifelines.plotting.add_at_risk_counts(*fitters_lt, ax=ax, **add_at_risk_counts_dt)
    return fitters_lt, ax
