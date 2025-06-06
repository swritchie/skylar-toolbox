# =============================================================================
# Load libraries
# =============================================================================

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as syss
from sklearn import calibration as sncn
from sklearn import metrics as snmes
from skylar_toolbox import exploratory_data_analysis as steda

# =============================================================================
# ClassificationEvaluator
# =============================================================================

class ClassificationEvaluator:
    def __init__(
        self, estimator, metrics_lt=['accuracy','average_precision','balanced_accuracy','f1','neg_log_loss','precision','recall','roc_auc']):
        self.estimator, self.metrics_lt = estimator, metrics_lt
    def fit(self, X_train, y_train, X_test, y_test):
        # Get predictions
        self.y_train_pred, self.y_train_pred_probas, self.y_train_pred_proba = self._get_predictions(X=X_train)
        self.y_test_pred, self.y_test_pred_probas, self.y_test_pred_proba = self._get_predictions(X=X_test)
        # Save targets
        self.y_train, self.y_test = y_train, y_test
        # Get and compare eval metrics
        self.eval_metrics_df = self._compare_eval_metrics(X_train=X_train, X_test=X_test)
        # Get classification report
        self._get_classification_report()
        return self
    def plot_predictions(self):
        fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(1.5e1, 1e1))
        for split_sr, ax in zip(['train', 'test'], axes.ravel()):
            y_true = self.y_train if split_sr == 'train' else self.y_test
            y_pred = self.y_train_pred_proba if split_sr == 'train' else self.y_test_pred_proba
            data_df = pd.concat(objs=[y_true, y_pred], axis=1).describe().round(decimals=3)
            plot_dt = dict(kind='hist', bins=int(3e1))
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
        pd.plotting.table(ax=ax, data=data_df, bbox=[1.5, 0, 5e-1, 1])
        return ax.figure
    def plot_confusion_matrix(self, from_predictions_dt=dict(normalize='all')):
        fig, axes = plt.subplots(ncols=2, figsize=(1e1, 5))
        from_predictions_dt.update(y_true=self.y_train, y_pred=self.y_train_pred, ax=axes[0])
        snmes.ConfusionMatrixDisplay.from_predictions(**from_predictions_dt)
        from_predictions_dt.update(y_true=self.y_test, y_pred=self.y_test_pred, ax=axes[1])
        snmes.ConfusionMatrixDisplay.from_predictions(**from_predictions_dt)
        set_titles(axes=axes)
        for ax in axes.ravel(): ax.grid(None)
        fig.tight_layout()
        return fig
    def plot_roc_curve(self, from_predictions_dt=dict(plot_chance_level=True)):
        fig, ax = plt.subplots(figsize=(5, 5))
        from_predictions_dt.update(y_true=self.y_train, y_pred=self.y_train_pred_proba, ax=ax)
        snmes.RocCurveDisplay.from_predictions(**from_predictions_dt)
        from_predictions_dt.update(y_true=self.y_test, y_pred=self.y_test_pred_proba, ax=ax)
        snmes.RocCurveDisplay.from_predictions(**from_predictions_dt)
        return fig
    def plot_pr_curve(self, from_predictions_dt=dict(plot_chance_level=True)):
        fig, ax = plt.subplots(figsize=(5, 5))
        from_predictions_dt.update(y_true=self.y_train, y_pred=self.y_train_pred_proba, ax=ax)
        snmes.PrecisionRecallDisplay.from_predictions(**from_predictions_dt)
        from_predictions_dt.update(y_true=self.y_test, y_pred=self.y_test_pred_proba, ax=ax)
        snmes.PrecisionRecallDisplay.from_predictions(**from_predictions_dt)
        return fig
    def plot_det_curve(self, from_predictions_dt=dict()):
        fig, ax = plt.subplots(figsize=(5, 5))
        from_predictions_dt.update(y_true=self.y_train, y_pred=self.y_train_pred_proba, ax=ax)
        snmes.DetCurveDisplay.from_predictions(**from_predictions_dt)
        from_predictions_dt.update(y_true=self.y_test, y_pred=self.y_test_pred_proba, ax=ax)
        snmes.DetCurveDisplay.from_predictions(**from_predictions_dt)
        return fig
    def plot_calibration(self, from_predictions_dt=dict(n_bins=int(1e1))):
        fig, ax = plt.subplots(figsize=(5, 5))
        from_predictions_dt.update(y_true=self.y_train, y_prob=self.y_train_pred_proba, ax=ax)
        sncn.CalibrationDisplay.from_predictions(**from_predictions_dt)
        from_predictions_dt.update(y_true=self.y_test, y_prob=self.y_test_pred_proba, ax=ax)
        sncn.CalibrationDisplay.from_predictions(**from_predictions_dt)
        return fig
    def delete_predictions_and_targets(self):
        attributes_lt = [
            'y_train_pred', 'y_train_pred_probas', 'y_train_pred_proba', 'y_train',
            'y_test_pred', 'y_test_pred_probas', 'y_test_pred_proba', 'y_test']
        for attribute_sr in attributes_lt: self.__delattr__(attribute_sr)
        return self
    def _get_predictions(self, X):
        y_pred = pd.Series(data=self.estimator.predict(X), index=X.index, name='predictions')
        y_pred_probas = pd.DataFrame(data=self.estimator.predict_proba(X), index=X.index)
        y_pred_proba = y_pred_probas[1].rename(index='predictions')
        return y_pred, y_pred_probas, y_pred_proba
    def _get_eval_metrics(self, X, y): return {
        scorer_sr: snmes.get_scorer(scoring=scorer_sr).__call__(estimator=self.estimator, X=X, y_true=y)
        for scorer_sr in self.metrics_lt}
    def _compare_eval_metrics(self, X_train, X_test):
        eval_metrics_dt = {
            split_sr: self._get_eval_metrics(X=X, y=y) 
            for split_sr, X, y in [('train', X_train, self.y_train), ('test', X_test, self.y_test)]}
        return steda.get_differences(df=pd.DataFrame(data=eval_metrics_dt), columns_lt=['train', 'test'])
    def _get_classification_report(self):
        args_lt = [
            ('train', self.y_train, self.y_train_pred), 
            ('test', self.y_test, self.y_test_pred)]
        self.classification_report_dt = {
            split_sr: snmes.classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
            for split_sr, y_true, y_pred in args_lt}
        for split_sr, y_true, y_pred in args_lt: print(f'{split_sr}:', snmes.classification_report(y_true=y_true, y_pred=y_pred), sep='\n')
        return self

# =============================================================================
# RegressionEvaluator
# =============================================================================
    
class RegressionEvaluator:
    def __init__(
        self, estimator, metrics_lt=['max_error','neg_mean_absolute_error','neg_mean_squared_error','neg_median_absolute_error','r2']):
        self.estimator, self.metrics_lt = estimator, metrics_lt
    def fit(self, X_train, y_train, X_test, y_test):
        # Get predictions
        self.y_train_pred, self.y_test_pred = map(lambda x: self._get_predictions(X=x), [X_train, X_test])
        # Save targets
        self.y_train, self.y_test = y_train, y_test
        # Get and compare eval metrics
        self.eval_metrics_df = self._compare_eval_metrics(X_train=X_train, X_test=X_test)
        return self
    def plot_predictions(self):
        fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(1.5e1, 1e1))
        for split_sr, ax in zip(['train', 'test'], axes.ravel()):
            y_true = self.y_train if split_sr == 'train' else self.y_test
            y_pred = self.y_train_pred if split_sr == 'train' else self.y_test_pred
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
        pd.plotting.table(ax=ax, data=data_df, bbox=[1.5, 0, 5e-1, 1])
        return ax.figure
    def plot_prediction_error(self, from_predictions_dt=dict()):
        fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(1e1, 5))
        from_predictions_dt.update(y_true=self.y_train, y_pred=self.y_train_pred, ax=axes[0])
        snmes.PredictionErrorDisplay.from_predictions(**from_predictions_dt)
        from_predictions_dt.update(y_true=self.y_test, y_pred=self.y_test_pred, ax=axes[1])
        snmes.PredictionErrorDisplay.from_predictions(**from_predictions_dt)
        set_titles(axes=axes)
        fig.tight_layout()
        return fig
    def delete_predictions_and_targets(self):
        attributes_lt = ['y_train_pred', 'y_train', 'y_test_pred', 'y_test']
        for attribute_sr in attributes_lt: self.__delattr__(attribute_sr)
        return self
    def _get_predictions(self, X): return pd.Series(data=self.estimator.predict(X), index=X.index, name='predictions')
    def _get_eval_metrics(self, X, y): return {
            scorer_sr: snmes.get_scorer(scoring=scorer_sr).__call__(estimator=self.estimator, X=X, y_true=y)
            for scorer_sr in self.metrics_lt}
    def _compare_eval_metrics(self, X_train, X_test):
        eval_metrics_dt = {
            split_sr: self._get_eval_metrics(X=X, y=y) 
            for split_sr, X, y in [('train', X_train, self.y_train), ('test', X_test, self.y_test)]}
        return steda.get_differences(df=pd.DataFrame(data=eval_metrics_dt), columns_lt=['train', 'test'])
 
# =============================================================================
# ThresholdEvaluator
# =============================================================================

class ThresholdEvaluator:
    def __init__(self): pass
    def fit(self, y_true, y_score, target_names_lt=[0, 1], **kgargs):
        fpr_ay, tpr_ay, thresh_ay = snmes.roc_curve(y_true=y_true, y_score=y_score)
        n_it, p_it = map(lambda x: y_true.value_counts().loc[x], target_names_lt)
        self.thresholded_metrics_df = (
            pd.DataFrame(data={
                'fpr': fpr_ay, # Type I error
                'tpr': tpr_ay  # Recall, sensitivity, power, hit rate
            }, index=thresh_ay)
            .assign(**{
                'tnr': lambda x: 1 - x['fpr'],                         # Specificity, selectivity
                'fnr': lambda x: 1 - x['tpr'],                         # Type II error, miss rate
                'fp': lambda x: (x['fpr'] * n_it).astype(dtype=int),   # False alarm
                'tp': lambda x: (x['tpr'] * p_it).astype(dtype=int),   # Hit
                'tn': lambda x: (x['tnr'] * n_it).astype(dtype=int),
                'fn': lambda x: (x['fnr'] * p_it).astype(dtype=int),   # Miss
                'ppv': lambda x: x['tp'] / (x['tp'] + x['fp']),        # 'Positive predictive value', precision
                'fdr': lambda x: 1 - x['ppv'],                         # 'False discovery rate'
                'npv': lambda x: x['tn'] / (x['tn'] + x['fn']),        # 'Negative predictive value'
                'for': lambda x: 1 - x['npv'],                         # 'False omission rate'
                'f1': lambda x: x[['tpr', 'ppv']].apply(func=syss.hmean, axis=1),
                'acc': lambda x: x[['tp', 'tn']].sum(axis=1) / x.select_dtypes(include=int).sum(axis=1),
                'bacc': lambda x: x[['tpr', 'tnr']].mean(axis=1)})
            .mask(cond=lambda x: x == float('inf')))
        return self
    def plot_rates(self): return self.thresholded_metrics_df[['tpr', 'tnr', 'fnr', 'fpr']].plot()
    def plot_values(self): return self.thresholded_metrics_df[['ppv', 'npv', 'fdr', 'for']].plot()
    def plot_bacc_f1(self):
        metrics_lt = ['bacc', 'f1']
        ax = self.thresholded_metrics_df[metrics_lt].plot()
        for index_it, metric_sr in enumerate(iterable=metrics_lt):
            ss = self.thresholded_metrics_df[metric_sr]
            ss.pipe(func=lambda x: ax.scatter(x=x.idxmax(), y=x.max(), c=f'C{index_it}'))
            ss.pipe(func=lambda x: ax.axvline(x=x.idxmax(), c=f'C{index_it}', ls='--'))
        ax.axhline(c='k', ls=':')
        return ax
    def plot_counts(self): return (self.thresholded_metrics_df
        .select_dtypes(include=int)
        .sort_index()
        .rename(index=lambda x: round(number=x, ndigits=3))
        .pipe(func=lambda x: x.iloc[::x.shape[0] // int(2e1), :])
        .plot(kind='bar', stacked=True))
    
# =============================================================================
# get_cross_entropy
# =============================================================================
    
def get_cross_entropy(y_trues, y_pred_probas):
    eps_ft = np.finfo(dtype=y_pred_probas.iloc[:, 0].dtype).eps
    return -(y_trues * y_pred_probas.clip(lower=eps_ft, upper=1 - eps_ft).apply(func=np.log)).sum(axis=1)

# =============================================================================
# set_titles
# =============================================================================

def set_titles(axes):
    for index_it, title_sr in enumerate(iterable=['train', 'test']): axes[index_it].set(title=title_sr)
    return axes
