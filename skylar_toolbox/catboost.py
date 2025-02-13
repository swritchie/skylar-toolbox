# =============================================================================
# Load libraries
# =============================================================================

import catboost as cb
import numpy as np
import pandas as pd
from catboost import monoforest as cbmf
from catboost import utils as cbus
from matplotlib import pyplot as plt
from sklearn import base as snbe

# =============================================================================
# CatBoostClassifier
# =============================================================================

class CatBoostClassifier(cb.CatBoostClassifier):
    def fit(self, X, y, **kwargs):
        params_dt = self.get_params()
        params_dt = update_params(params_dt=params_dt, X=X)
        self.set_params(**params_dt)
        return super().fit(X=X, y=y, **kwargs)

# =============================================================================
# CatBoostInspector
# =============================================================================

class CatBoostInspector:
    def __init__(self, cbm, metrics_lt): self.cbm, self.metrics_lt = cbm, metrics_lt
    def fit(self, X, y, pool_dt):
        # Get pool
        pool_dt['cat_features'] = self.cbm.get_params().get('cat_features', [])
        pl = cb.Pool(data=X, label=y, **pool_dt)
        # Get eval metrics
        self.eval_metrics_df = pd.DataFrame(data=self.cbm.eval_metrics(data=pl, metrics=self.metrics_lt))
        # Get thresholded metrics
        self.thresholded_metrics_df = pd.concat(objs=[
            pd.DataFrame(
                data=np.array(object=cbus.get_roc_curve(model=self.cbm, data=pl)), 
                index=['fpr', 'tpr', 'thresholds']).T.set_index(keys='thresholds'),
            pd.DataFrame(
                data=np.array(object=cbus.get_fnr_curve(model=self.cbm, data=pl)), 
                index=['thresholds', 'fnr']).T.set_index(keys='thresholds')], axis=1)
        # Get feature importances
        features_lt = self.cbm.feature_names_
        feature_importances_df = pd.concat(objs=[pd.Series(
            data=self.cbm.get_feature_importance(data=pl, type=type_sr), 
            index=features_lt, 
            name=type_sr) for type_sr in ['PredictionValuesChange', 'LossFunctionChange']], axis=1)
        feature_importances_df.sort_values(by='LossFunctionChange', ascending=False, inplace=True)
        self.feature_importances_df = feature_importances_df
        # Get interactions
        features_dt = {index_it: feature_sr for index_it, feature_sr in enumerate(iterable=features_lt)}
        interactions_df = pd.DataFrame(
            data=self.cbm.get_feature_importance(data=pl, type='Interaction'),
            columns=['first_feature', 'second_feature', 'interactions'])
        self.interactions_df = interactions_df.apply(func=lambda x: x.map(arg=features_dt) if x.name != 'interactions' else x)
        return self
    def plot_eval_metrics(self, last_bl):
        if last_bl:
            eval_metrics_ss = self.eval_metrics_df.iloc[-1, :].sort_values().rename(index='metrics')
            ax = eval_metrics_ss.plot(kind='barh')
            ax.axvline(c='k', ls=':')
            pd.plotting.table(ax=ax, data=eval_metrics_ss.round(decimals=3), bbox=[1.5, 0, 2.5e-1, 1e-1 * eval_metrics_ss.shape[0]])
        else: self.eval_metrics_df.plot(subplots=True, layout=(-1, 1), figsize=(1e1, 3 * self.eval_metrics_df.shape[1]));
        return plt.gcf()
    def plot_roc_curve(self):
        ax = self.thresholded_metrics_df.plot(x='fpr', y='tpr', ylabel='tpr', label='ROC curve', figsize=(5, 5))
        ax.plot([0, 1], [0, 1], 'k:', label='Chance')
        ax.legend()
        return ax.figure
    def plot_fpr_fnr(self): return self.thresholded_metrics_df.plot(y=['fpr', 'fnr']).get_figure()
    def plot_feature_importances(self):
        axes = self.feature_importances_df.plot(subplots=True, layout=(-1, 1), figsize=(1e1, 1e1))
        for ax in axes.ravel():
            ax.axhline(c='k', ls=':')
            ax.set(xticks=[])
            column_sr = ax.get_legend().get_texts()[0]._text
            data_ss = self.feature_importances_df[column_sr].describe().round(decimals=3)
            pd.plotting.table(ax=ax, data=data_ss, bbox=[1.25, 0, 2.5e-1, 1])
        return plt.gcf()
    def plot_top_feature_importances(self, nlargest_n_it=int(1e1), nlargest_columns_lt=['LossFunctionChange']): return (
        self.feature_importances_df
        .nlargest(n=nlargest_n_it, columns=nlargest_columns_lt)
        [::-1]
        .plot(kind='barh', subplots=True, layout=(1, -1), sharex=False, sharey=True, legend=False)
        .get_figure())
    def plot_interactions(self):
        interactions_ss = self.interactions_df['interactions']
        ax = interactions_ss.plot()
        ax.axhline(c='k', ls=':')
        data_ss = interactions_ss.describe().round(decimals=3)
        pd.plotting.table(ax=ax, data=data_ss, bbox=[1.25, 0, 2.5e-1, 1])
        return ax.figure
    def plot_top_interactions(self, nlargest_n_it=int(1e1)): return (
        self.interactions_df
        .set_index(keys=['first_feature', 'second_feature'])
        .squeeze()
        .nlargest(n=nlargest_n_it)
        [::-1]
        .plot(kind='barh')
        .get_figure())
    
# =============================================================================
# CatBoostRegressor
# =============================================================================

class CatBoostRegressor(cb.CatBoostRegressor):
    def fit(self, X, y, **kwargs):
        params_dt = self.get_params()
        params_dt = update_params(params_dt=params_dt, X=X)
        self.set_params(**params_dt)
        return super().fit(X=X, y=y, **kwargs)
    
# =============================================================================
# CatBoostSelector
# =============================================================================
   
class CatBoostSelector(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(
        self, 
        model_type_sr, exact_elimination_bl=False,
        depth_it=None, eval_fraction_ft=None, learning_rate_ft=None, init_params_dt=dict(),
        algorithm_sr=None, num_features_to_select_it=None, steps_it=None, fit_params_dt=dict(
            algorithm='RecursiveByLossFunctionChange', num_features_to_select=1, steps=1)):
        assert model_type_sr in ['classification', 'regression']
        assert exact_elimination_bl in [True, False]
        self.model_type_sr, self.exact_elimination_bl = model_type_sr, exact_elimination_bl
        self.depth_it, self.eval_fraction_ft, self.learning_rate_ft, self.init_params_dt = \
            depth_it, eval_fraction_ft, learning_rate_ft, init_params_dt
        self.algorithm_sr, self.num_features_to_select_it, self.steps_it, self.fit_params_dt = \
            algorithm_sr, num_features_to_select_it, steps_it, fit_params_dt
    def fit(self, X, y):
        # Get init params
        init_params_dt = self.init_params_dt
        # Update them
        init_params_dt = update_params(params_dt=init_params_dt, X=X)
        # Set them
        for param_sr in ['depth_it', 'eval_fraction_ft', 'learning_rate_ft']:
            param = getattr(self, param_sr)
            if param: init_params_dt.update({param_sr[:-3]: param})
        # Initialize estimator
        self.cbm = cb.CatBoostClassifier(**init_params_dt) if self.model_type_sr == 'classification' \
            else cb.CatBoostRegressor(**init_params_dt)
        # Get fit params
        fit_params_dt = self.fit_params_dt
        # Set them
        for param_sr in ['algorithm_sr', 'num_features_to_select_it', 'steps_it']:
            param = getattr(self, param_sr)
            if param: fit_params_dt.update({param_sr[:-3]: param})
        # Select
        select_features_dt = self.cbm.select_features(
            X=X, y=y, 
            features_for_select=range(X.shape[1]), train_final_model=False, **fit_params_dt)
        # Get eliminated features
        self.eliminated_features_ss = pd.Series(
            data=select_features_dt['loss_graph']['loss_values'][1:], 
            index=select_features_dt['eliminated_features_names'], name='losses')
        self.eliminated_features_ix = self.eliminated_features_ss.index if self.exact_elimination_bl \
            else self.eliminated_features_ss.pipe(func=lambda x: x[:x[x == x.min()].index[-1]].index)
        return self
    def transform(self, X): return X.drop(columns=self.eliminated_features_ix)
    def get_feature_names_out(self): pass
    def plot(self):
        ax = self.eliminated_features_ss.reset_index(drop=True).plot(c='lightgrey')
        self.eliminated_features_ss[self.eliminated_features_ix].reset_index(drop=True).plot(c='tab:blue', lw=2, ax=ax)
        return ax
    
# =============================================================================
# get_default_params
# =============================================================================
   
def get_default_params(): return dict(
    early_stopping_rounds=int(1e1),
    eval_fraction=2e-1,
    learning_rate=3e-2,
    use_best_model=True,
    verbose=int(1e2))

# =============================================================================
# get_evals_result
# =============================================================================

def get_evals_result(cbm): return (
    pd.DataFrame(data=cbm.evals_result_)
    .stack()
    .apply(func=pd.Series)
    .T)

# =============================================================================
# MonoForestInspector
# =============================================================================

class MonoForestInspector:
    def __init__(self, cbm): self.cbm = cbm
    def fit(self):
        # Get polynomials list
        polynom_lt = cbmf.to_polynom(model=self.cbm)
        # Get it as data frame
        self.polynom_df = (
            pd.DataFrame(data=list(map(lambda x: x.__dict__, polynom_lt)))
            .assign(value = lambda x: x['value'].apply(func=lambda x: x[0])))
        # Get splits
        splits_dt = (
            self.polynom_df['splits']
            .apply(func=pd.Series)
            .stack()
            .apply(func=lambda x: x.__dict__)
            .to_dict())
        # Get them as data frame
        self.features_dt = {
            index_it: feature_sr 
            for index_it, feature_sr in enumerate(iterable=self.cbm.feature_names_)}
        self.splits_df = (
            pd.DataFrame(data=splits_dt)
            .T
            .assign(feature_idx = lambda x: x['feature_idx'].map(arg=self.features_dt)))
        return self
    def plot_weight(self):
        weight_ss = self.polynom_df['weight']
        ax = weight_ss.sort_values(ascending=False).reset_index(drop=True).plot()
        ax.axhline(c='k', ls=':')
        data_ss = weight_ss.describe().round(decimals=3)
        pd.plotting.table(ax=ax, data=data_ss, bbox=[1.25, 0, 2.5e-1, 1])
        return ax.figure
    
# =============================================================================
# plot_evals_result
# =============================================================================

def plot_evals_result(evals_result_df):
    metrics_ix = (
        evals_result_df.columns
        .get_level_values(level=0)
        .drop_duplicates())
    fig, axes = plt.subplots(
        nrows=metrics_ix.shape[0], 
        sharex=True, 
        squeeze=False, 
        figsize=(1e1, 3 * metrics_ix.shape[0]))
    for index_it, metric_sr in enumerate(iterable=metrics_ix):
        evals_result_df[metric_sr].plot(title=metric_sr, ax=axes.ravel()[index_it])
    fig.tight_layout()
    return fig

# =============================================================================
# update_params
# =============================================================================

def update_params(params_dt, X):
    # Get cat features
    old_cat_features_lt = params_dt.get('cat_features', [])
    new_cat_features_lt = (
        X.columns
        .intersection(other=old_cat_features_lt)              # May have been dropped
        .union(other=X.select_dtypes(include=object).columns) # May have been added
        .tolist())
    # Get monotone contraints
    old_monotone_constraints_dt = params_dt.get('monotone_constraints', dict())
    if old_monotone_constraints_dt: new_monotone_constraints_dt = {
        key_sr: value_it for key_sr, value_it in old_monotone_constraints_dt.items() if key_sr in X.columns}
    else: new_monotone_constraints_dt = old_monotone_constraints_dt
    # Update
    params_dt.update(cat_features=new_cat_features_lt, monotone_constraints=new_monotone_constraints_dt)
    return params_dt

# =============================================================================
# ValidationChangeCallback
# =============================================================================

class ValidationChangeCallback:
    def __init__(self, metric_sr, threshold_ft=1e-4): self.metric_sr, self.threshold_ft = metric_sr, threshold_ft
    def after_iteration(self, info):
        try:
            first_ft, second_ft = map(lambda x: info.metrics['validation'][self.metric_sr][x], [-2, -1])
            change_ft = abs(second_ft - first_ft)
            continue_bl = change_ft > self.threshold_ft
        except Exception as en: continue_bl = True
        return continue_bl

# =============================================================================
# ValidationDifferenceCallback
# =============================================================================

class ValidationDifferenceCallback:
    def __init__(self, metric_sr, threshold_ft=1e-2): self.metric_sr, self.threshold_ft = metric_sr, threshold_ft
    def after_iteration(self, info):
        learn_metric_ft = info.metrics['learn'][self.metric_sr][-1]
        validation_metric_ft = info.metrics['validation'][self.metric_sr][-1]
        difference_ft = abs(validation_metric_ft - learn_metric_ft)
        continue_bl = difference_ft < self.threshold_ft
        return continue_bl
