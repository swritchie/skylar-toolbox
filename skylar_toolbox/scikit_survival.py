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
    def __init__(self, estimator, times_ay=None): 
        self.estimator, self.times_ay = estimator, times_ay
        self.split_sr, self.splits_lt = 'split', ['train', 'test'] # For consistent usage later
        self.roc_auc_sr, self.predictions_sr, self.time_sr = 'roc_auc', 'predictions', 'time'
        self.ch_predictions_sr, self.s_predictions_sr = map(lambda x: f'{x}_{self.predictions_sr}', ['ch', 's'])
    def fit(self, X_train, y_train, X_test, y_test):
        # Get event flag and duration
        self.event_flag_sr, self.duration_sr = y_test.dtype.names
        # Get concordance (C) index
        features_te = X_train, X_test
        targets_te = y_train, y_test
        c_index_train_ft, c_index_test_ft = map(lambda x, y: self._get_score(X=x, y=y), features_te, targets_te)
        # Get predictions
        y_train_pred, y_test_pred = preds_lt = list(map(lambda x: self._get_predictions(X=x), features_te))
        # Get times at which to evaluate model
        self.train_range_ss, self.test_range_ss = map(lambda x: self._get_range(y=x), targets_te)
        if self.times_ay is None: self.times_ay = np.linspace(*self.test_range_ss)
        # Predict cumulative hazard
        y_train_pred_ch, y_test_pred_ch = ch_preds_lt = list(map(lambda x: self._get_cumulative_hazard_predictions(X=x), features_te))
        # Get time-dependent ROC AUC
        shared_dt = dict(survival_train=y_train, survival_test=y_test, times=self.times_ay)
        roc_auc_ay, roc_auc_ft = ssmes.cumulative_dynamic_auc(estimate=y_test_pred_ch.pipe(func=np.vstack), **shared_dt)
        self.roc_auc_ss = pd.Series(data=roc_auc_ay, index=self.times_ay, name=self.roc_auc_sr)
        # Predict survival
        y_train_pred_s, y_test_pred_s = s_preds_lt = list(map(lambda x: self._get_survival_predictions(X=x), features_te))
        # Combine
        fn = lambda x, y, z: pd.DataFrame(data=y, index=x.index).assign(**{self.split_sr: z})
        self.y_combined = (pd.concat(objs=map(fn, features_te, targets_te, self.splits_lt))
            .join(other=pd.concat(objs=preds_lt))
            .sort_values(by=self.predictions_sr)
            .join(other=pd.concat(objs=ch_preds_lt))
            .join(other=pd.concat(objs=s_preds_lt)))
        # Get time-dependent Brier score
        brier_score_ft = ssmes.integrated_brier_score(estimate=y_test_pred_s.pipe(func=np.vstack), **shared_dt)
        # Get and compare eval metrics
        eval_metrics_dt = {
            'train': {'c_index': c_index_train_ft},
            'test': {'c_index': c_index_test_ft, self.roc_auc_sr: roc_auc_ft, 'brier_score': brier_score_ft}}
        self.eval_metrics_df = steda.get_differences(df=pd.DataFrame(data=eval_metrics_dt), columns_lt=self.splits_lt)
        return self
    def plot_actuals(self, displot_dt=dict(kind='kde', common_norm=False, clip=(0, None), aspect=4, height=2)):
        sns.displot(data=self.y_combined, x=self.duration_sr, hue=self.event_flag_sr, row=self.split_sr, **displot_dt)
        return plt.gcf()
    def plot_predictions(self, displot_dt=dict(kind='kde', common_norm=False, clip=(0, None), aspect=4, height=2)):
        sns.displot(data=self.y_combined, x=self.predictions_sr, hue=self.event_flag_sr, row=self.split_sr, **displot_dt)
        return plt.gcf()
    def plot_sample_cumulative_hazard_predictions(self, sample_dt=dict(frac=1e-2), plot_dt=dict(alpha=1e-2)):
        ax = plt.subplot(1, 1, 1)
        self.y_combined[self.ch_predictions_sr].sample(**sample_dt).apply(func=lambda x: ax.plot(self.times_ay, x, **plot_dt))
        ax.set(xlabel=self.time_sr, ylabel=self.ch_predictions_sr)
        return plt.gcf()
    def plot_sample_survival_predictions(self, sample_dt=dict(frac=1e-2), plot_dt=dict(alpha=1e-2)):
        ax = plt.subplot(1, 1, 1)
        self.y_combined[self.s_predictions_sr].sample(**sample_dt).apply(func=lambda x: ax.plot(self.times_ay, x, **plot_dt))
        ax.set(xlabel=self.time_sr, ylabel=self.s_predictions_sr)
        return plt.gcf()
    def plot_sample_prediction(self, index):
        # Get sample data
        values_ss = self.y_combined.loc[index, :]
        # Create figure and grid spec
        fig = plt.figure(figsize=plt.figaspect(arg=1/2))
        gs = plt.GridSpec(nrows=2, ncols=2)
        # Plot distribution of time-independent risk scores (and this one)
        ax = fig.add_subplot(gs[:, 0])
        self.y_combined[self.predictions_sr].plot(kind='hist', label='all', ax=ax)
        ax.axvline(x=values_ss[self.predictions_sr], label='this')
        ax.legend()
        ax.set(xlabel=self.predictions_sr, title='time-independent risk scores')
        # Plot cumulative hazard predictions alongside actuals
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.times_ay, values_ss[self.ch_predictions_sr], label='predicted')
        c_sr = 'r' if values_ss[self.event_flag_sr] else 'g'
        ax2.axvline(x=values_ss[self.duration_sr], c=c_sr, label='actual')
        ax2.legend()
        ax2.set(ylabel=self.ch_predictions_sr, title='time-dependent risk scores')
        # Plot survival predictions alongside actuals
        ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
        ax3.plot(self.times_ay, values_ss[self.s_predictions_sr], label='predicted')
        ax3.axvline(x=values_ss[self.duration_sr], c=c_sr, label='actual')
        ax3.legend()
        ax3.set(xlabel=self.time_sr, ylabel=self.s_predictions_sr)
        fig.tight_layout()
        return fig
    def plot_eval_metrics(self):
        columns_lt = self.splits_lt + ['pct_diff']
        plot_df = self.eval_metrics_df[columns_lt[:-1]]
        data_df = self.eval_metrics_df[columns_lt].round(decimals=3)
        ax = plot_df.plot(kind='bar')
        pd.plotting.table(ax=ax, data=data_df, bbox=[1.25, 0, 5e-1, 5e-1])
        return ax.figure
    def plot_roc_auc(self):
        ax = self.roc_auc_ss.plot(marker='.', xlabel=self.time_sr, ylabel=self.roc_auc_sr)
        ax.axhline(y=5e-1, c='k', ls=':')
        pd.plotting.table(ax=ax, data=self.roc_auc_ss.describe().round(decimals=3), bbox=[1.25, 0, 2.5e-1, 1])
        return ax.figure
    def delete_predictions_and_targets(self):
        attributes_lt = ['y_combined']
        for attribute_sr in attributes_lt: self.__delattr__(attribute_sr)
        return self
    def _get_score(self, X, y): return self.estimator.score(X=X, y=y)
    def _get_predictions(self, X): return pd.Series(data=self.estimator.predict(X=X), index=X.index, name=self.predictions_sr)
    def _get_range(self, y): return pd.Series(data=y[self.duration_sr]).agg(func=['min', 'max'])
    def _get_cumulative_hazard_predictions(self, X): return (
        pd.Series(data=self.estimator.predict_cumulative_hazard_function(X=X), index=X.index, name=self.ch_predictions_sr)
        .apply(func=lambda x: x(self.times_ay)))
    def _get_survival_predictions(self, X): return (
        pd.Series(data=self.estimator.predict_survival_function(X=X), index=X.index, name=self.s_predictions_sr)
        .apply(func=lambda x: x(self.times_ay)))
