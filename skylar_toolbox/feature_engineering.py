# =============================================================================
# Load libraries
# =============================================================================

import pandas as pd
from sklearn import base as snbe
from sklearn import feature_selection as snfs

# =============================================================================
# AggregationEngineer
# =============================================================================

class AggregationEngineer(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(
        self, model_type_sr, features_ix, fit_dt=dict(random_state=0), aggregations_lt=['min','median','mean','max','std','sum'], name_sr=None):
        assert model_type_sr in ['classification', 'regression']
        self.model_type_sr, self.features_ix, self.fit_dt, self.aggregations_lt, self.name_sr = \
            model_type_sr, features_ix, fit_dt, aggregations_lt, name_sr
    def fit(self, X, y):
        # Get function for model type
        mutual_info_fn = snfs.mutual_info_classif if self.model_type_sr == 'classification' else snfs.mutual_info_regression
        # Subset feature matrix
        X_subset = X.loc[:, self.features_ix]
        # Get mutual info
        subset_mutual_info_ss = pd.Series(data=mutual_info_fn(X=X_subset, y=y, **self.fit_dt), index=X_subset.columns, name='mutual_info')
        # Engineer aggregations
        X_agg = X_subset.agg(func=self.aggregations_lt, axis=1)
        # Get mutual info
        agg_mutual_info_ss = pd.Series(data=mutual_info_fn(X=X_agg, y=y, **self.fit_dt), index=X_agg.columns, name='mutual_info')
        # Select best
        self.best_ix = agg_mutual_info_ss.index[agg_mutual_info_ss > subset_mutual_info_ss.max()]
        self.mutual_info_ss = pd.concat(objs=[subset_mutual_info_ss, agg_mutual_info_ss]).sort_values()
        return self
    def transform(self, X):
        # If any aggregated features are better than originals...
        if not self.best_ix.empty:
            # Subset feature matrix
            X_subset = X.loc[:, self.features_ix]
            # Engineer aggregations
            labels_ix = self.best_ix + '-' + self.name_sr if self.name_sr else self.best_ix + '-' + '-'.join(self.features_ix)
            X_agg = X_subset.agg(func=self.best_ix.tolist(), axis=1).set_axis(labels=labels_ix, axis=1)
            # Join
            return X.join(other=X_agg)
        return X
    def get_feature_names_out(): pass
    def plot(self): return self.mutual_info_ss.plot(kind='barh')

# =============================================================================
# InteractionEngineer
# =============================================================================

class InteractionEngineer(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(self, model_type_sr, feature_sr, feature_sr2, fit_dt=dict(random_state=0)):
        assert model_type_sr in ['classification', 'regression']
        self.model_type_sr, self.feature_sr, self.feature_sr2, self.fit_dt = model_type_sr, feature_sr, feature_sr2, fit_dt
    def fit(self, X, y):
        # Get function for model type
        mutual_info_fn = snfs.mutual_info_classif if self.model_type_sr == 'classification' else snfs.mutual_info_regression
        # Subset feature matrix and engineer all interactions
        self.assign_dt = {
            f'{self.feature_sr}-add-{self.feature_sr2}': self._add,
            f'{self.feature_sr}-sub-{self.feature_sr2}': self._sub,
            f'{self.feature_sr2}-sub-{self.feature_sr}': self._sub2,
            f'{self.feature_sr}-mul-{self.feature_sr2}': self._mul,
            f'{self.feature_sr}-div-{self.feature_sr2}': self._div,
            f'{self.feature_sr2}-div-{self.feature_sr}': self._div2}
        X_subset = X.loc[:, [self.feature_sr, self.feature_sr2]].assign(**self.assign_dt)
        # Select best
        self.mutual_info_ss = (
            pd.Series(data=mutual_info_fn(X=X_subset, y=y, **self.fit_dt), index=X_subset.columns, name='mutual_info')
            .sort_values())
        self.best_sr = (self.mutual_info_ss
            .drop(labels=[self.feature_sr, self.feature_sr2])
            .nlargest(n=1)
            .index[0])
        return self
    def transform(self, X): return X.assign(**{self.best_sr: self.assign_dt[self.best_sr]})
    def get_feature_names_out(): pass
    def plot(self): return self.mutual_info_ss.plot(kind='barh')
    def _add(self, x): return x[self.feature_sr] + x[self.feature_sr2]
    def _sub(self, x): return x[self.feature_sr] - x[self.feature_sr2]
    def _sub2(self, x): return x[self.feature_sr2] - x[self.feature_sr]
    def _mul(self, x): return x[self.feature_sr] * x[self.feature_sr2]
    def _div(self, x): return x[self.feature_sr] / (x[self.feature_sr2] + 1e-10)
    def _div2(self, x): return x[self.feature_sr2] / (x[self.feature_sr] + 1e-10)
