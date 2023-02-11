# =============================================================================
# Load libraries
# =============================================================================

import catboost as cb
import os
import pandas as pd
from catboost import utils as cbus
from matplotlib import pyplot as plt
from scipy import special as sysl
from skylar_toolbox import exploratory_data_analysis as steda

# =============================================================================
# CustomCatBoost
# =============================================================================

class CustomCatBoost:
    def __init__(
            self, 
            cat_boost_dt: dict, 
            binary_bl: bool):
        '''
        Wraps CatBoost model to provide additional tools for model inspection

        Parameters
        ----------
        cat_boost_dt : dict
            Parameters passed to cb.CatBoost.
        binary_bl : bool
            Flag for binary classification.

        Returns
        -------
        None.

        '''
        required_keys_lt = [
            'loss_function', 'eval_metric', 'custom_metric', 'random_seed', 'iterations', 'verbose',
            'early_stopping_rounds', 'use_best_model', 'task_type', 'cat_features']
        for key_sr in required_keys_lt:
            assert key_sr in cat_boost_dt.keys(), f'{key_sr} not in cat_boost_dt'
        self.cat_boost_dt = cat_boost_dt
        self.cbm = cb.CatBoost(params=cat_boost_dt)
        self.binary_bl = binary_bl
    
    def fit(
            self, 
            X_train: pd.DataFrame, 
            y_train: pd.Series, 
            X_valid: pd.DataFrame, 
            y_valid: pd.Series):
        '''
        Fits model and collects metadata

        Parameters
        ----------
        X_train : pd.DataFrame
            Train feature matrix.
        y_train : pd.Series
            Train target vector.
        X_valid : pd.DataFrame
            Validation feature matrix.
        y_valid : pd.Series
            Validation target vector.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        # Fit model
        self.cbm.fit(X=X_train, y=y_train, eval_set=(X_valid, y_valid))
        
        # Get evals result
        self.evals_result_df = self._get_evals_result()
        
        # Get predictions
        self.y_train_pred = self._get_predictions(X=X_train)
        self.y_valid_pred = self._get_predictions(X=X_valid)
        
        # Get and compare eval metrics
        self.eval_metrics_df = self._compare_eval_metrics(y_train=y_train, y_valid=y_valid)
        
        # Get and compare LFC feature importances
        self.lfc_feature_importances_df = self._compare_feature_importances(
            X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, type_sr='LossFunctionChange')
        
        # Get and compare PVC feature importances
        self.pvc_feature_importances_df = self._compare_feature_importances(
            X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, type_sr='PredictionValuesChange')
        
        # Get interaction strengths
        if X_train.shape[1] > 1:
            self.interaction_strengths_df = self._get_interaction_strengths()
        return self
    
    def plot_evals_result(self):
        '''
        Plots learning curve of eval metrics v. iterations

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        fig, axes = plt.subplots(nrows=len(self.metrics_lt), sharex=True, figsize=(6, 3 * len(self.metrics_lt)))
        for metric_sr, ax in zip(self.metrics_lt, axes.ravel()):
            self.evals_result_df.filter(like=f'_{metric_sr}').plot(ax=ax)
        fig.tight_layout()
        return fig
    
    def plot_eval_metrics(self):
        '''
        Plots train and validation eval metrics as bars along with table

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        ax = self.eval_metrics_df.iloc[:, :2].plot(kind='bar')
        pd.plotting.table(ax=ax, data=self.eval_metrics_df.iloc[:, [0, 1, -1]].round(decimals=3), bbox=[1.25, 0, 0.5, 1])
        fig = ax.figure
        return fig
    
    def plot_predictions(
            self, 
            y_train: pd.Series, 
            y_valid: pd.Series):
        '''
        Plots histograms of train and validation targets and predictions with table

        Parameters
        ----------
        y_train : pd.Series
            Train target vector.
        y_valid : pd.Series
            Validation target vector.

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(10, 5))
        for split_sr, ax in zip(['learn', 'validation'], axes.ravel()):
            y_true = y_train if split_sr == 'learn' else y_valid
            y_pred = self.y_train_pred if split_sr == 'learn' else self.y_valid_pred
            data_df = pd.concat(objs=[y_true, y_pred], axis=1).describe().round(decimals=3)
            plot_dt = dict(kind='hist' if self.binary_bl else 'kde')
            if self.binary_bl: 
                plot_dt['bins'] = 30
            y_true.plot(ax=ax, **plot_dt)
            y_pred.plot(ax=ax, **plot_dt)
            ax.legend()
            ax.set(title=split_sr)
            pd.plotting.table(ax=ax, data=data_df, bbox=[1.25, 0, 0.5, 1])
        fig.tight_layout()
        return fig
    
    def plot_feature_importances(
            self, 
            importance_type_sr: str, 
            plot_type_sr: str):
        '''
        Plots train and validation feature importances (either 'LossFunctionChange' or 'PredictionValuesChange'):
            - 'all': line with all importances and table with description
            - 'top_bottom': horizontal bar with top and bottom features
            - 'abs_diff': horizontal bar of features with largest difference in train and validation importances
            - 'pct_diff': horizontal bar of features with largest percent difference in train and validation importances

        Parameters
        ----------
        importance_type_sr : str
            Type of feature importance. Must be one of ['LossFunctionChange', 'PredictionValuesChange']
        plot_type_sr : str
            Type of plot. Must be one of ['all', 'top_bottom', 'abs_diff', 'pct_diff'].

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        if importance_type_sr == 'LossFunctionChange':
            feature_importances_df = self.lfc_feature_importances_df
        elif importance_type_sr == 'PredictionValuesChange':
            feature_importances_df = self.pvc_feature_importances_df
        else:
            assert False, f'{importance_type_sr} not in ["LossFunctionChange", "PredictionValuesChange"]'
        if plot_type_sr == 'all':
            ax = feature_importances_df.iloc[:, :2].plot()
            ax.set(xticks=[])
            ax.axhline(y=0, c='k', ls=':')
            pd.plotting.table(ax=ax, data=feature_importances_df.iloc[:, :2].describe().round(decimals=3), bbox=[1.25, 0, 0.5, 1])
            fig = ax.figure
            return fig
        elif plot_type_sr == 'top_bottom':
            fig, axes = plt.subplots(nrows=2, sharex=True)
            for index_it, split_sr in enumerate(iterable=['learn', 'validation']):
                pd.concat(objs=[
                    feature_importances_df.iloc[:, index_it].nsmallest(),
                    feature_importances_df.iloc[:, index_it].nlargest()[::-1]]).plot(kind='barh', ax=axes[index_it])
                axes[index_it].axvline(x=0, c='k', ls=':')
                axes[index_it].set(title=split_sr)
            fig.tight_layout()
            return fig
        elif plot_type_sr in ['abs_diff', 'pct_diff']:
            top_bottom_df = feature_importances_df.nlargest(n=10, columns=plot_type_sr) if plot_type_sr == 'abs_diff' \
                else feature_importances_df.nsmallest(n=10, columns=plot_type_sr)
            ax = (
                top_bottom_df
                .sort_values(by=plot_type_sr, ascending=True if plot_type_sr == 'abs_diff' else False)
                .iloc[:, :2]
                .plot(kind='barh'))
            ax.axvline(x=0, c='k', ls=':')
            fig = ax.figure
            return fig
        else:
            assert False, f'{plot_type_sr} not in ["all", "top_bottom", "abs_diff", "pct_diff"]'
            
    def plot_interaction_strengths(self):
        '''
        Plots interaction strengths

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        interaction_strengths_ss = self.interaction_strengths_df.iloc[:, -1]
        ax = interaction_strengths_ss.plot()
        ax.set(xticks=[])
        ax.axhline(y=0, c='k', ls=':')
        pd.plotting.table(ax=ax, data=interaction_strengths_ss.describe().round(decimals=3), bbox=[1.25, 0, 0.25, 1])
        fig = ax.figure
        return fig
    
    def delete_predictions(self):
        '''
        Deletes predictions from instance

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        for attribute_sr in ['y_train_pred', 'y_valid_pred']:
            self.__delattr__(attribute_sr)
        return self
    
    def _get_evals_result(self):
        '''
        Gets evals result data frame

        Returns
        -------
        evals_result_df : pd.DataFrame
            DESCRIPTION.

        '''
        evals_result_df = pd.DataFrame(data={
            f'{key_sr}_{key_sr2}': value_lt
            for key_sr, value_dt in self.cbm.evals_result_.items()
            for key_sr2, value_lt in value_dt.items()})
        return evals_result_df
    
    def _get_predictions(
            self, 
            X: pd.DataFrame):
        '''
        Gets predictions as series

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        predictions_ss : pd.Series
            Predictions.

        '''
        if self.binary_bl:
            predictions_ay = self.cbm.predict(data=X, prediction_type='Probability')[:, 1]
        else:
            predictions_ay = self.cbm.predict(data=X)
        predictions_ss = pd.Series(data=predictions_ay, index=X.index, name='predictions')
        return predictions_ss
    
    def _get_eval_metrics(
            self, 
            y_true: pd.Series, 
            y_pred: pd.Series):
        '''
        Gets eval metrics

        Parameters
        ----------
        y_true : pd.Series
            Target vector.
        y_pred : pd.Series
            Predictions.

        Returns
        -------
        eval_metrics_dt : dict
            Eval metrics.

        '''
        label_ss = y_true
        approx_ss = y_pred.apply(func=sysl.logit) if self.binary_bl else y_pred
        self.metrics_lt = self.cat_boost_dt['custom_metric']
        eval_metrics_dt = {
            metric_sr: cbus.eval_metric(label=label_ss.values, approx=approx_ss.values, metric=metric_sr)[0] 
            for metric_sr in self.metrics_lt}
        return eval_metrics_dt
    
    def _compare_eval_metrics(
            self, 
            y_train: pd.Series, 
            y_valid: pd.Series):
        '''
        Gets difference in eval metrics for train and validation

        Parameters
        ----------
        y_train : pd.Series
            Train target vector.
        y_valid : pd.Series
            Validation target vector.

        Returns
        -------
        eval_metrics_df : pd.DataFrame
            Eval metrics.

        '''
        eval_metrics_dt = {
            'learn': self._get_eval_metrics(y_true=y_train, y_pred=self.y_train_pred),
            'validation': self._get_eval_metrics(y_true=y_valid, y_pred=self.y_valid_pred)}
        eval_metrics_df = steda.get_differences(df=pd.DataFrame(data=eval_metrics_dt), columns_lt=list(eval_metrics_dt.keys()))
        return eval_metrics_df
    
    def _get_feature_importances(
            self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            type_sr: str, 
            name_sr: str):
        '''
        Gets feature importances

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        type_sr : str
            Feature importance type.
        name_sr : str
            Split name.

        Returns
        -------
        feature_importances_ss : pd.Series
            Feature importances.

        '''
        feature_importances_ay = self.cbm.get_feature_importance(
            data=cb.Pool(data=X, label=y, cat_features=self.cat_boost_dt['cat_features']), 
            type=type_sr)
        feature_importances_ss = pd.Series(
            data=feature_importances_ay,
            index=X.columns,
            name=name_sr)
        return feature_importances_ss

    def _compare_feature_importances(
            self, 
            X_train: pd.DataFrame, 
            y_train: pd.Series, 
            X_valid: pd.DataFrame, 
            y_valid: pd.Series, 
            type_sr: str):
        '''
        Gets difference in feature importances for train and validation

        Parameters
        ----------
        X_train : pd.DataFrame
            Train feature matrix.
        y_train : pd.Series
            Train target vector.
        X_valid : pd.DataFrame
            Validation feature matrix.
        y_valid : pd.Series
            Validation target vector.
        type_sr : str
            Feature importance type.

        Returns
        -------
        feature_importances_df : pd.DataFrame
            Feature importances.

        '''
        objs_lt = [
            self._get_feature_importances(X=X, y=y, type_sr=type_sr, name_sr=name_sr)
            for X, y, name_sr in [(X_train, y_train, 'learn'), (X_valid, y_valid, 'validation')]]
        feature_importances_df = steda.get_differences(
            df=pd.concat(objs=objs_lt, axis=1).sort_values(by='validation', ascending=False),
            columns_lt=['learn', 'validation'])
        return feature_importances_df
                     
    def _get_interaction_strengths(self):
        '''
        Gets interaction strengths

        Returns
        -------
        interaction_strengths_df : pd.DataFrame
            Interaction strengths.

        '''
        map_dt = {index_it: feature_sr for index_it, feature_sr in enumerate(iterable=self.cbm.feature_names_)}
        interaction_strengths_df = (
            pd.DataFrame(data=self.cbm.get_feature_importance(type='Interaction'))
            .set_axis(labels=['first_features', 'second_features', 'strengths'], axis=1)
            .assign(
                first_features = lambda x: x['first_features'].map(arg=map_dt),
                second_features = lambda x: x['second_features'].map(arg=map_dt)))
        return interaction_strengths_df

# =============================================================================
# CustomCatBoostCV
# =============================================================================

class CustomCatBoostCV:
    def __init__(
            self, 
            cat_boost_dt: dict, 
            binary_bl: bool, 
            sklearn_splitter):
        '''
        Cross-validates wrapped model and provides additional model inspection tools

        Parameters
        ----------
        cat_boost_dt : dict
            Parameters passed to cb.CatBoost.
        binary_bl : bool
            Flag for binary classification.
        sklearn_splitter : TYPE
            Instance of scikit-learn splitter class.

        Returns
        -------
        None.

        '''
        required_keys_lt = [
            'loss_function', 'eval_metric', 'custom_metric', 'random_seed', 'iterations', 'verbose',
            'early_stopping_rounds', 'use_best_model', 'task_type', 'cat_features', 'train_dir']
        for key_sr in required_keys_lt:
            assert key_sr in cat_boost_dt.keys(), f'{key_sr} not in cat_boost_dt'
        self.cat_boost_dt = cat_boost_dt
        assert 'n_splits' in sklearn_splitter.__dict__.keys()
        self.models_lt = [
            CustomCatBoost(cat_boost_dt=cat_boost_dt, binary_bl=binary_bl) 
            for _ in range(sklearn_splitter.__getattribute__('n_splits'))]
        self.binary_bl = binary_bl
        self.sklearn_splitter = sklearn_splitter
        
    def fit(
            self, 
            X: pd.DataFrame, 
            y: pd.Series):
        '''
        Fits model and collects metadata

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        for index_it, (train_ay, test_ay) in enumerate(iterable=self.sklearn_splitter.split(X=X, y=y)):
            # Log
            print('-' * 80)
            print(f'Split: {index_it}')
            print('Shapes:\n- {}'.format('\n- '.join(str(tmp.shape) for tmp in [train_ay, test_ay])))

            # Update CatBoost params
            train_dir_sr = os.path.join(self.cat_boost_dt['train_dir'], str(index_it))
            self.cat_boost_dt.update(train_dir=train_dir_sr)

            # Fit model
            self.models_lt[index_it].fit(
                X_train=X.iloc[train_ay, :],
                y_train=y.iloc[train_ay], 
                X_valid=X.iloc[test_ay, :], 
                y_valid=y.iloc[test_ay])
        
        # Compare eval metrics
        self.eval_metrics_df = self._compare_eval_metrics()
        
        # Compare LFC feature importances
        self.lfc_feature_importances_df = self._compare_feature_importances(type_sr='LossFunctionChange')
        
        # Compare PVC feature importances
        self.pvc_feature_importances_df = self._compare_feature_importances(type_sr='PredictionValuesChange')
        return self
    
    def plot_eval_metrics(self):
        '''
        Plots eval metrics with standard errors

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        # Get data frames
        ys_df = self.eval_metrics_df.filter(like='mean').rename(columns=lambda x: x.split('_')[0])
        yerrs_df = self.eval_metrics_df.filter(like='se2').rename(columns=lambda x: x.split('_')[0])
        data_df = self.eval_metrics_df.filter(regex='mean|se2').round(decimals=3)
        # Plot
        ax = ys_df.plot(kind='bar', yerr=yerrs_df)
        pd.plotting.table(ax=ax, data=data_df, bbox=[1.25, 0, 1, 1])
        fig = ax.figure
        return fig
    
    def plot_feature_importances(
            self, 
            importance_type_sr: str, 
            plot_type_sr: str):
        '''
        Plots feature importances with standard errors

        Parameters
        ----------
        importance_type_sr : str
            Type of feature importance. Must be one of ['LossFunctionChange', 'PredictionValuesChange']
        plot_type_sr : str
            Type of plot. Must be one of ['all', 'top_bottom'].

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        if importance_type_sr == 'LossFunctionChange':
            feature_importances_df = self.lfc_feature_importances_df
        elif importance_type_sr == 'PredictionValuesChange':
            feature_importances_df = self.pvc_feature_importances_df
        else:
            assert False, f'{importance_type_sr} not in ["LossFunctionChange", "PredictionValuesChange"]'
        xs_df = ys_df = feature_importances_df.filter(like='mean').rename(columns=lambda x: x.split('_')[0])
        xerrs_df = yerrs_df = feature_importances_df.filter(like='se2').rename(columns=lambda x: x.split('_')[0])
        data_df = ys_df.describe().round(decimals=3)
        if plot_type_sr == 'all':
            ax = ys_df.plot(yerr=yerrs_df)
            ax.set(xticks=[])
            pd.plotting.table(ax=ax, data=data_df, bbox=[1.25, 0, 0.5, 1])
            fig = ax.figure
            return fig
        elif plot_type_sr == 'top_bottom':
            fig, axes = plt.subplots(nrows=2, sharex=True)
            for index_it, split_sr in enumerate(iterable=['learn', 'validation']):
                concat_df = pd.concat(objs=[
                    xs_df.iloc[:, index_it].nsmallest(),
                    xs_df.iloc[:, index_it].nlargest()[::-1]])
                concat_df.plot(kind='barh', xerr=xerrs_df.loc[concat_df.index, :], ax=axes[index_it])
                axes[index_it].axvline(x=0, c='k', ls=':')
                axes[index_it].set(title=split_sr)
            fig.tight_layout()
            return fig
        else:
            assert False, f'{plot_type_sr} not in ["all", "top_bottom"]'
            
    def delete_predictions(self):
        '''
        Deletes predictions from all model instances

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        for ccb in self.models_lt:
            for attribute_sr in ['y_train_pred', 'y_valid_pred']:
                ccb.__delattr__(attribute_sr)
        return self
    
    def _compare_eval_metrics(self):
        '''
        Gets mean and standard error for eval metrics

        Returns
        -------
        eval_metrics_df : pd.DataFrame
            Eval metrics.

        '''
        # Concatenate them
        eval_metrics_df = pd.concat(objs=[
            ccb.eval_metrics_df.iloc[:, :2].rename(columns=lambda x: f'{x}_{index_it}')
            for index_it, ccb in enumerate(iterable=self.models_lt)
        ], axis=1)
        # Get means
        for split_sr in ['learn', 'validation']:
            eval_metrics_df = (
                steda.get_means(df=eval_metrics_df, columns_lt=eval_metrics_df.filter(like=split_sr).columns.tolist())
                .rename(columns=lambda x: f'{split_sr}_{x}' if '_' not in x else x))
        return eval_metrics_df
    
    def _compare_feature_importances(
            self, 
            type_sr: str):
        '''
        Gets mean and standard error for feature importances

        Parameters
        ----------
        type_sr : str
            Type of feature importance. Must be one of ['LossFunctionChange', 'PredictionValuesChange']

        Returns
        -------
        feature_importances_df : pd.DataFrame
            Feature importances.

        '''
        # Concatenate them
        if type_sr == 'LossFunctionChange':
            feature_importances_df = pd.concat(objs=[
                ccb.lfc_feature_importances_df.iloc[:, :2].rename(columns=lambda x: f'{x}_{index_it}')
                for index_it, ccb in enumerate(iterable=self.models_lt)
            ], axis=1)
        elif type_sr == 'PredictionValuesChange':
            feature_importances_df = pd.concat(objs=[
                ccb.pvc_feature_importances_df.iloc[:, :2].rename(columns=lambda x: f'{x}_{index_it}')
                for index_it, ccb in enumerate(iterable=self.models_lt)
            ], axis=1)
        # Get means
        for split_sr in ['learn', 'validation']:
            feature_importances_df = (
                steda.get_means(df=feature_importances_df, columns_lt=feature_importances_df.filter(like=split_sr).columns.tolist())
                .rename(columns=lambda x: f'{split_sr}_{x}' if '_' not in x else x))
        # Sort
        feature_importances_df.sort_values(by='validation_mean', ascending=False, inplace=True)
        return feature_importances_df