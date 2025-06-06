# =============================================================================
# Load libraries
# =============================================================================

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sksurv import metrics as ssmes
from skylar_toolbox import exploratory_data_analysis as steda

# =============================================================================
# SurvivalModelEvaluator
# =============================================================================

class SurvivalModelEvaluator:
    def __init__(self, estimator, times_ay=None): self.estimator, self.times_ay = estimator, times_ay
    def fit(self, X_train, y_train, X_test, y_test):
        # Save targets
        self.y_train, self.y_test = targets_te = y_train, y_test
        # Get event flag and duration
        self.event_flag_sr, self.duration_sr = y_test.dtype.names
        # Get concordance (C) index
        features_te = X_train, X_test
        c_index_train_ft, c_index_test_ft = map(lambda x, y: self._get_score(X=x, y=y), features_te, targets_te)
        # Get predictions
        self.y_train_pred, self.y_test_pred = map(lambda x: self._get_predictions(X=x), features_te)
        # Get times at which to evaluate model
        self.train_range_ss, self.test_range_ss = map(lambda x: self._get_range(y=x), targets_te)
        if self.times_ay is None: self.times_ay = np.linspace(*self.test_range_ss)
        # Predict cumulative hazard
        self.y_train_pred_ch, self.y_test_pred_ch = map(lambda x: self._get_cumulative_hazard_predictions(X=x), features_te)
        # Get time-dependent ROC AUC
        shared_dt = dict(survival_train=y_train, survival_test=y_test, times=self.times_ay)
        roc_auc_ay, roc_auc_ft = ssmes.cumulative_dynamic_auc(estimate=self.y_test_pred_ch.pipe(func=np.vstack), **shared_dt)
        self.roc_auc_ss = pd.Series(data=roc_auc_ay, index=self.times_ay, name='roc_auc')
        # Predict survival
        self.y_train_pred_s, self.y_test_pred_s = map(lambda x: self._get_survival_predictions(X=x), features_te)
        # Get time-dependent Brier score
        brier_score_ft = ssmes.integrated_brier_score(estimate=self.y_test_pred_s.pipe(func=np.vstack), **shared_dt)
        # Get and compare eval metrics
        eval_metrics_dt = {
            'train': {'c_index': c_index_train_ft},
            'test': {'c_index': c_index_test_ft, 'roc_auc': roc_auc_ft, 'brier_score': brier_score_ft}}
        self.eval_metrics_df = steda.get_differences(df=pd.DataFrame(data=eval_metrics_dt), columns_lt=['train', 'test'])
        return self
    def plot_predictions(self):
        fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(1.5e1, 1e1))
        for split_sr, ax in zip(['train', 'test'], axes.ravel()):
            y_pred = self.y_train_pred if split_sr == 'train' else self.y_test_pred
            data_ay = (self.y_train if split_sr == 'train' else self.y_test)[self.duration_sr]
            y_true = pd.Series(data=data_ay, index=y_pred.index, name=split_sr)
            data_df = pd.concat(objs=[y_true, y_pred], axis=1).describe().round(decimals=3)
            plot_dt = dict(kind='kde')
            y_true.plot(ax=ax, title=split_sr, **plot_dt)
            y_pred.plot(ax=ax, title=split_sr, **plot_dt)
            ax.legend()
            pd.plotting.table(ax=ax, data=data_df, bbox=[1.25, 0, 5e-1, 1])
        fig.tight_layout()
        return fig
    def plot_eval_metrics(self):
        columns_lt = ['train', 'test', 'pct_diff']
        plot_df = self.eval_metrics_df[columns_lt[:-1]]
        data_df = self.eval_metrics_df[columns_lt].round(decimals=3)
        ax = plot_df.plot(kind='bar')
        pd.plotting.table(ax=ax, data=data_df, bbox=[1.5, 0, 5e-1, 5e-1])
        return ax.figure
    def plot_roc_auc(self):
        ax = self.roc_auc_ss.plot(marker='.', xlabel='time', ylabel='roc_auc', title='Time-dependent ROC AUC')
        ax.axhline(y=5e-1, c='k', ls=':')
        pd.plotting.table(ax=ax, data=self.roc_auc_ss.describe().round(decimals=3), bbox=[1.25, 0, 2.5e-1, 1])
        return ax.figure
    def delete_predictions_and_targets(self):
        attributes_lt = [
            'y_train', 'y_train_pred', 'y_train_pred_ch', 'y_train_pred_s',
            'y_test', 'y_test_pred', 'y_test_pred_ch', 'y_test_pred_s']
        for attribute_sr in attributes_lt: self.__delattr__(attribute_sr)
        return self
    def _get_score(self, X, y): return self.estimator.score(X=X, y=y)
    def _get_predictions(self, X): return pd.Series(data=self.estimator.predict(X=X), index=X.index, name='predictions')
    def _get_range(self, y): return pd.Series(data=y[self.duration_sr]).agg(func=['min', 'max'])
    def _get_cumulative_hazard_predictions(self, X): return (
        pd.Series(data=self.estimator.predict_cumulative_hazard_function(X=X), index=X.index, name='ch_predictions')
        .apply(func=lambda x: x(self.times_ay)))
    def _get_survival_predictions(self, X): return (
        pd.Series(data=self.estimator.predict_survival_function(X=X), index=X.index, name='s_predictions')
        .apply(func=lambda x: x(self.times_ay)))
