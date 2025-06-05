# =============================================================================
# Load libraries
# =============================================================================

import lifelines
import pandas as pd
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
