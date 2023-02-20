# =============================================================================
# Load libraries
# =============================================================================

import catboost as cb
import numpy as np
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
            model_type_sr: str,
            cat_boost_dt: dict):
        '''
        Wraps CatBoost model...
        - Storing metadata
        - Providing methods for plotting, etc.

        Parameters
        ----------
        model_type_sr : str
            Model type.
        cat_boost_dt : dict
            Parameters passed to CatBoost.
            
        Raises
        ------
        NotImplementedError
            Implemented values of model_type_sr are ['classification', 'regression'].
        KeyError
            Required key "{required_key_sr}" is not in cat_boost_dt (e.g., train_dir).

        Returns
        -------
        None.

        '''
        # model_type_sr
        implemented_model_types_lt = ['classification', 'regression']
        if model_type_sr not in implemented_model_types_lt:
            raise NotImplementedError(f'Implemented values of model_type_sr are {implemented_model_types_lt}')
        self.model_type_sr = model_type_sr

        # cat_boost_dt
        required_keys_lt = ['train_dir']
        for required_key_sr in required_keys_lt:
             if required_key_sr not in cat_boost_dt.keys():
                 raise KeyError(f'Required key "{required_key_sr}" is not in cat_boost_dt')
        general_defaults_dt = {
            'cat_features': [],
            'early_stopping_rounds': 100,
            'iterations': 1_000,
            'monotone_constraints': {},
            'random_seed': 0,
            'task_type': 'CPU',
            'use_best_model': True}
        if model_type_sr == 'classification':
            model_defaults_dt = {
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'custom_metric': ['Logloss', 'AUC', 'PRAUC', 'F1', 'Precision', 'Recall']}
            general_defaults_dt.update(model_defaults_dt)
        elif model_type_sr == 'regression':
            model_defaults_dt = {
                'loss_function': 'RMSE',
                'eval_metric': 'R2',
                'custom_metric': ['RMSE', 'R2', 'MSLE', 'MAE', 'MAPE', 'MedianAbsoluteError']}
            general_defaults_dt.update(model_defaults_dt)
        general_defaults_dt.update(cat_boost_dt)
        general_defaults_dt['verbose'] = general_defaults_dt['iterations'] // 10
        self.cat_boost_dt = general_defaults_dt
    
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
        self.cbm = cb.CatBoost(params=self.cat_boost_dt).fit(X=X_train, y=y_train, eval_set=(X_valid, y_valid))
        
        # Get evals result
        self.evals_result_df = self._get_evals_result()
        
        # Get predictions
        self.y_train_pred = self._get_predictions(X=X_train)
        self.y_valid_pred = self._get_predictions(X=X_valid)
        
        # Save targets
        self.y_train = y_train
        self.y_valid = y_valid
        
        # Get and compare eval metrics
        self.eval_metrics_df = self._compare_eval_metrics()
        
        # Get and compare LFC feature importances
        self.lfc_feature_importances_df = self._compare_feature_importances(
            X_train=X_train, X_valid=X_valid, type_sr='LossFunctionChange')
        
        # Get and compare PVC feature importances
        self.pvc_feature_importances_df = self._compare_feature_importances(
            X_train=X_train, X_valid=X_valid, type_sr='PredictionValuesChange')
        
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
        permitted_importance_types_lt = ['LossFunctionChange', 'PredictionValuesChange']
        if importance_type_sr == 'LossFunctionChange':
            feature_importances_df = self.lfc_feature_importances_df
        elif importance_type_sr == 'PredictionValuesChange':
            feature_importances_df = self.pvc_feature_importances_df
        else:
            raise ValueError(f'importance_type_sr must be one of {permitted_importance_types_lt}')
        implemented_plot_types_lt = ['all', 'top_bottom', 'abs_diff', 'pct_diff']
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
            raise NotImplementedError(f'plot_type_sr must be one of {implemented_plot_types_lt}')
            
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
    
    def plot_predictions(self):
        '''
        Plots histograms or KDEs of train and validation targets and predictions with table

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(10, 5))
        for split_sr, ax in zip(['learn', 'validation'], axes.ravel()):
            y_true = self.y_train if split_sr == 'learn' else self.y_valid
            y_pred = self.y_train_pred if split_sr == 'learn' else self.y_valid_pred
            data_df = pd.concat(objs=[y_true, y_pred], axis=1).describe().round(decimals=3)
            plot_dt = dict(kind='hist' if self.model_type_sr == 'classification' else 'kde')
            if self.model_type_sr == 'classification': 
                plot_dt['bins'] = 30
            y_true.plot(ax=ax, **plot_dt)
            y_pred.plot(ax=ax, **plot_dt)
            ax.legend()
            ax.set(title=split_sr)
            pd.plotting.table(ax=ax, data=data_df, bbox=[1.25, 0, 0.5, 1])
        fig.tight_layout()
        return fig
    
    def delete_predictions_and_targets(self):
        '''
        Deletes predictions and targets from instance

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        for attribute_sr in ['y_train_pred', 'y_valid_pred', 'y_train', 'y_valid']:
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
        if self.model_type_sr == 'classification':
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
        approx_ss = y_pred.apply(func=sysl.logit) \
            if self.model_type_sr == 'classification' else y_pred
        self.metrics_lt = self.cat_boost_dt['custom_metric']
        eval_metrics_dt = {
            metric_sr: cbus.eval_metric(label=label_ss.values, approx=approx_ss.values, metric=metric_sr)[0] 
            for metric_sr in self.metrics_lt}
        return eval_metrics_dt
    
    def _compare_eval_metrics(self):
        '''
        Gets difference in eval metrics for train and validation

        Returns
        -------
        eval_metrics_df : pd.DataFrame
            Eval metrics.

        '''
        eval_metrics_dt = {
            'learn': self._get_eval_metrics(y_true=self.y_train, y_pred=self.y_train_pred),
            'validation': self._get_eval_metrics(y_true=self.y_valid, y_pred=self.y_valid_pred)}
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
            X_valid: pd.DataFrame, 
            type_sr: str):
        '''
        Gets difference in feature importances for train and validation

        Parameters
        ----------
        X_train : pd.DataFrame
            Train feature matrix.
        X_valid : pd.DataFrame
            Validation feature matrix.
        type_sr : str
            Feature importance type.

        Returns
        -------
        feature_importances_df : pd.DataFrame
            Feature importances.

        '''
        objs_lt = [
            self._get_feature_importances(X=X, y=y, type_sr=type_sr, name_sr=name_sr)
            for X, y, name_sr in [(X_train, self.y_train, 'learn'), (X_valid, self.y_valid, 'validation')]]
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
            model_type_sr: str,
            cat_boost_dt: dict, 
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
        self.model_type_sr = model_type_sr
        required_keys_lt = ['train_dir']
        for required_key_sr in required_keys_lt:
            if required_key_sr not in cat_boost_dt.keys():
                raise KeyError(f'Required key "{required_key_sr}" is not in cat_boost_dt')
        self.cat_boost_dt = cat_boost_dt
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
        # Make directory
        output_directory_sr = self.cat_boost_dt['train_dir']
        os.mkdir(path=output_directory_sr)
        
        # Initialize
        self.models_lt = []
        
        for index_it, (train_ay, test_ay) in enumerate(iterable=self.sklearn_splitter.split(X=X, y=y)):
            # Log
            print('-' * 80)
            print(f'Split: {index_it}')
            print('Shapes:\n- {}'.format('\n- '.join(str(tmp.shape) for tmp in [train_ay, test_ay])))
            
            # Make subdirectory
            output_subdirectory_sr = '{}/{:03d}'.format(output_directory_sr, index_it)
            os.mkdir(path=output_subdirectory_sr)
            
            # Update params
            self.cat_boost_dt['train_dir'] = output_subdirectory_sr

            # Fit model
            ccb = CustomCatBoost(model_type_sr=self.model_type_sr, cat_boost_dt=self.cat_boost_dt)
            ccb.fit(
                X_train=X.iloc[train_ay, :],
                y_train=y.iloc[train_ay], 
                X_valid=X.iloc[test_ay, :], 
                y_valid=y.iloc[test_ay])
            self.models_lt.append(ccb)
        
        # Compare eval metrics
        self.eval_metrics_df = self._compare_eval_metrics()
        
        # Compare LFC feature importances
        self.lfc_feature_importances_df = self._compare_feature_importances(type_sr='LossFunctionChange')
        
        # Compare PVC feature importances
        self.pvc_feature_importances_df = self._compare_feature_importances(type_sr='PredictionValuesChange')
        return self
    
    def sum_models(
            self, 
            strategy_sr: str):
        '''
        Ensembles models trained on different subsets

        Parameters
        ----------
        strategy_sr : str
            How models should be weighted. Must be one of ['weight_by_score', 'weight_equally']

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        models_lt = [ccb.cbm for ccb in self.models_lt]
        implemented_strategies_lt = ['weight_by_score', 'weight_equally']
        if strategy_sr == 'weight_by_score':
            weights_lt = self.eval_metrics_df.filter(regex='validation_\d').loc[self.cat_boost_dt['eval_metric'], :].tolist()
            self.cbm = cb.sum_models(models=models_lt, weights=weights_lt)
        elif strategy_sr == 'weight_equally':
            self.cbm = cb.sum_models(models=models_lt)
        else:
            raise NotImplementedError(f'strategy_sr must be one of {implemented_strategies_lt}')
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
        permitted_importance_types_lt = ['LossFunctionChange', 'PredictionValuesChange']
        if importance_type_sr == 'LossFunctionChange':
            feature_importances_df = self.lfc_feature_importances_df
        elif importance_type_sr == 'PredictionValuesChange':
            feature_importances_df = self.pvc_feature_importances_df
        else:
            raise ValueError(f'importance_type_sr must be one of {permitted_importance_types_lt}')
        xs_df = ys_df = feature_importances_df.filter(like='mean').rename(columns=lambda x: x.split('_')[0])
        xerrs_df = yerrs_df = feature_importances_df.filter(like='se2').rename(columns=lambda x: x.split('_')[0])
        data_df = ys_df.describe().round(decimals=3)
        implemented_plot_types_lt = ['all', 'top_bottom']
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
            raise NotImplementedError(f'plot_type_sr must be one of {implemented_plot_types_lt}')
            
    def delete_predictions_and_targets(self):
        '''
        Deletes predictions and targets from all model instances

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        for ccb in self.models_lt:
            ccb.delete_predictions_and_targets()
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
    
# =============================================================================
# ExampleInspector
# =============================================================================
    
class ExampleInspector:
    def __init__(
            self, 
            ccbcv: CustomCatBoostCV, 
            losses_nlargest_n_it: int):
        '''
        Gets losses and importances (of train on valid) per example

        Parameters
        ----------
        ccbcv : CustomCatBoostCV
            CV model.
        losses_nlargest_n_it : int
            Number of validation examples used to get train example importances. 

        Returns
        -------
        None.

        '''
        self.ccbcv = ccbcv
        self.losses_nlargest_n_it = losses_nlargest_n_it
    
    def fit(
            self, 
            X: pd.DataFrame, 
            y: pd.Series):
        '''
        Gets losses and example importances

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
        losses_lt, example_importances_lt = [], []
        
        for index_it, ccb in enumerate(iterable=self.ccbcv.models_lt):
            # Log
            print('-' * 80)
            print(f'Split: {index_it}')
            
            # Get losses
            losses_ss = self._get_losses(ccb=ccb)
            losses_lt.append(losses_ss)
            
            # Get largest
            largest_losses_ss = losses_ss.nlargest(n=self.losses_nlargest_n_it)
            
            # Get example importances
            example_importances_ss = self._get_example_importances(largest_losses_ss=largest_losses_ss, ccb=ccb, X=X, y=y)
            example_importances_lt.append(example_importances_ss)
            
        # Combine losses
        self.losses_ss = pd.concat(objs=losses_lt).sort_values(ascending=False)
        
        # Compare example importances
        self.example_importances_df = self._compare_example_importances(example_importances_lt=example_importances_lt)
        return self
    
    def plot_losses(self):
        '''
        Plots histogram of losses (with table)

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        ax = steda.plot_histogram(ss=self.losses_ss)
        fig = ax.figure
        return fig
    
    def plot_example_importances(self):
        '''
        Plots histogram of example importances (with table)

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        ax = steda.plot_histogram(ss=self.example_importances_df['mean'])
        fig = ax.figure
        return fig
    
    def delete_predictions_and_targets(self):
        '''
        Deletes predictions and targets from all model instances

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        for ccb in self.ccbcv.models_lt:
            ccb.delete_predictions_and_targets()
        return self
    
    def _get_losses(
            self, 
            ccb: CustomCatBoost):
        '''
        Gets validation losses for models with loss functions of...
        - Logloss
        - RMSE
        - MAE

        Parameters
        ----------
        ccb : CustomCatBoost
            Model.

        Returns
        -------
        losses_ss : pd.Series
            Validation losses.

        '''
        loss_function_sr = ccb.cat_boost_dt['loss_function']
        implemented_loss_functions_lt = ['Logloss', 'RMSE', 'MAE']
        if loss_function_sr == 'Logloss':
            # Get targets
            y_valid = pd.get_dummies(data=ccb.y_valid)
            # Get predictions
            y_valid_pred = (
                ccb.y_valid_pred
                .to_frame(name='_1')
                .assign(_0 = lambda x: 1 - x['_1'])
                .rename(columns=lambda x: int(x[-1]))
                .sort_index(axis=1))
            # Get losses
            losses_ss = -(y_valid * np.log(y_valid_pred)).sum(axis=1).rename(index='losses')
            return losses_ss
        elif loss_function_sr == 'RMSE':
            losses_ss = ((ccb.y_train - ccb.y_train_pred)**2).rename(index='losses')
            return losses_ss
        elif loss_function_sr == 'MAE':
            losses_ss = np.abs(ccb.y_train - ccb.y_train_pred).rename(index='losses')
            return losses_ss
        else:
            raise NotImplementedError(f'loss_function_sr must be one of {implemented_loss_functions_lt}')
    
    def _get_example_importances(
            self, 
            largest_losses_ss: pd.Series, 
            ccb: CustomCatBoost, 
            X: pd.DataFrame, 
            y: pd.Series):
        '''
        Gets train example importances for largest validation losses

        Parameters
        ----------
        largest_losses_ss : pd.Series
            Largest validation losses (where "largest" is defined at init).
        ccb : CustomCatBoost
            Model.
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.

        Returns
        -------
        example_importances_ss : pd.Series
            Train example importances.

        '''
        largest_losses_ix = largest_losses_ss.index
        train_ix = ccb.y_train.index
        pool_dt = dict(cat_features=ccb.cat_boost_dt['cat_features'])
        indices_lt, scores_lt = ccb.cbm.get_object_importance(
            pool=cb.Pool(data=X.loc[largest_losses_ix, :], label=y.loc[largest_losses_ix], **pool_dt),
            train_pool=cb.Pool(data=X.loc[train_ix, :], label=y.loc[train_ix], **pool_dt), 
            verbose=train_ix.shape[0] // 10)
        example_importances_ss = pd.Series(data=scores_lt, index=train_ix[indices_lt], name='importances')
        return example_importances_ss
    
    def _compare_example_importances(
            self, 
            example_importances_lt: list):
        '''
        Compare example importances across splits

        Parameters
        ----------
        example_importances_lt : list
            List of example importances.

        Returns
        -------
        example_importances_df : pd.DataFrame
            Data frame of example importances with means, etc.

        '''
        example_importances_df = pd.concat(objs=[
            example_importances_ss.rename(index=index_it)
            for index_it, example_importances_ss in enumerate(iterable=example_importances_lt)
        ], axis=1)
        example_importances_df = (
            steda.get_means(df=example_importances_df, columns_lt=example_importances_df.columns.tolist())
            .sort_values(by='mean', ascending=False))
        return example_importances_df
    
# =============================================================================
# ExampleSelector
# =============================================================================

class ExampleSelector:
    def __init__(
            self,
            model_type_sr: str,
            cat_boost_dt: dict,
            sklearn_splitter,
            objective_sr: str,
            losses_nlargest_n_it: int,
            example_importances_nlargest_n_it: int,
            wait_it: int):
        '''
        Iteratively removes examples

        Parameters
        ----------
        cat_boost_dt : dict
            Parameters passed to CatBoost.
        binary_bl : bool
            Flag for binary classification.
        sklearn_splitter : TYPE
            Instance of scikit-learn splitter class.
        objective_sr : str
            One of ['minimize', 'maximize'].
        losses_nlargest_n_it : int
            Number of validation examples with largest losses (on which to calculate importances).
        example_importances_nlargest_n_it : int
            Number of examples to drop per iteration.
        wait_it : int
            Iterations to wait before stopping procedure.

        Raises
        ------
        ValueError
            objective_sr must be one of ['minimize', 'maximize'].

        Returns
        -------
        None.

        '''
        self.model_type_sr = model_type_sr
        required_keys_lt = ['train_dir']
        for required_key_sr in required_keys_lt:
            if required_key_sr not in cat_boost_dt.keys():
                raise KeyError(f'Required key "{required_key_sr}" is not in cat_boost_dt')
        self.cat_boost_dt = cat_boost_dt
        self.sklearn_splitter = sklearn_splitter
        permitted_objectives_lt = ['minimize', 'maximize']
        if objective_sr not in permitted_objectives_lt:
            raise ValueError(f'Permitted values of objective_sr are {permitted_objectives_lt}')
        self.objective_sr = objective_sr
        self.best_score_ft = np.inf if objective_sr == 'minimize' else -np.inf
        self.best_iteration_it = 0
        self.losses_nlargest_n_it = losses_nlargest_n_it
        self.example_importances_nlargest_n_it = example_importances_nlargest_n_it
        self.wait_it = wait_it

    def fit(
                self,
            X: pd.DataFrame,
            y: pd.Series):
        '''
        Iteratively fits models

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
        # Make directory
        output_directory_sr = self.cat_boost_dt['train_dir']
        os.mkdir(path=output_directory_sr)

        # Initialize
        iteration_it = 0
        self.models_lt = []
        self.inspectors_lt = []
        results_lt = []

        # Loop
        while True:
            # Log
            print('=' * 80)
            print(f'Iteration: {iteration_it}')

            # Update params
            output_subdirectory_sr = '{}/{:03d}'.format(output_directory_sr, iteration_it)
            self._update_params(train_dir_sr=output_subdirectory_sr)

            # Fit model
            ccbcv = CustomCatBoostCV(model_type_sr=self.model_type_sr, cat_boost_dt=self.cat_boost_dt, sklearn_splitter=self.sklearn_splitter)
            ccbcv.fit(X=X, y=y)
            self.models_lt.append(ccbcv)

            # Get score and update bests
            score_ft = ccbcv.eval_metrics_df.loc[self.cat_boost_dt['eval_metric'], 'validation_mean']
            self._update_best_score_and_iteration(score_ft=score_ft, iteration_it=iteration_it)

            # Fit inspector
            ei = ExampleInspector(ccbcv=ccbcv, losses_nlargest_n_it=self.losses_nlargest_n_it)
            ei.fit(X=X, y=y)
            self.inspectors_lt.append(ei)

            # Get features to drop and keep
            examples_ix, drop_ix, keep_ix = self._get_examples(X=X, ei=ei)

            # Get and print result
            result_dt = self._get_result(iteration_it=iteration_it, score_ft=score_ft, examples_ix=examples_ix, drop_ix=drop_ix, keep_ix=keep_ix)
            results_lt.append(result_dt)
            self._print_result(result_dt=result_dt)

            # Evaluate whether to continue
            if ((iteration_it - self.best_iteration_it == self.wait_it) or
            (examples_ix.shape[0] == 1) or
            (drop_ix.empty)):
                break
            else:
                X.drop(index=drop_ix, inplace=True)
                y.drop(index=drop_ix, inplace=True)
                iteration_it += 1

        # Get results
        self.results_df = self._get_results(results_lt=results_lt)

        # Get ranks
        self.ranks_df = self._get_ranks()
        return self

    def weight_ranks(
            self,
            weights_dt: dict):
        '''
        Weights score and example count ranks to arrive at new combined rank

        Parameters
        ----------
        weights_dt : dict
            Dict with...
            - Keys of 'scores' and 'cnt_examples'
            - Values corresponding to importances (higher = more important).

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        required_keys_lt = ['scores', 'pct_diffs', 'cnt_examples']
        for required_key_sr in required_keys_lt:
            if required_key_sr not in weights_dt.keys():
                raise KeyError(f'Required key "{required_key_sr}" is not in weights_dt')
        self.ranks_df['weighted_rank'] = (
            self.ranks_df['scores_rank'] * weights_dt['scores'] +
            self.ranks_df['pct_diffs_rank'] * weights_dt['pct_diffs'] +
            self.ranks_df['cnt_examples_rank'] * weights_dt['cnt_examples'])
        return self

    def get_best_examples(self):
        '''
        Gets examples from best model according to rank (including weighted if it exists)

        Returns
        -------
        best_examples_ix : pd.Index
            Examples from best model.

        '''
        best_iteration_it = self._get_best_iteration()
        best_examples_ix = self.results_df.loc[best_iteration_it, 'examples']
        return best_examples_ix

    def get_best_model(self):
        '''
        Gets best model according to rank (including weighted if it exists)

        Returns
        -------
        best_ccbcv : CustomCatBoostCV
            Best model.

        '''
        best_iteration_it = self._get_best_iteration()
        best_ccbcv = self.models_lt[best_iteration_it]
        return best_ccbcv

    def get_best_inspector(self):
        '''
        Gets best inspector according to rank (including weighted if it exists)

        Returns
        -------
        best_ei : ExampleInspector
            Best inspector.

        '''
        best_iteration_it = self._get_best_iteration()
        best_ei = self.inspectors_lt[best_iteration_it]
        return best_ei

    def plot_results(self):
        '''
        Plots results (original scale)

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        fig, axes = plt.subplots(nrows=2, sharex=True)
        self.results_df.iloc[:, :2].plot(marker='.', subplots=True, ax=axes)
        for index_it, ax in enumerate(iterable=axes.ravel()):
            data_ss = self.results_df.iloc[:, index_it].describe().round(decimals=3)
            pd.plotting.table(ax=ax, data=data_ss, bbox=[1.25, 0, 0.25, 1])
        fig.tight_layout()
        return fig

    def plot_ranks(self):
        '''
        Plots ranks (common scale)

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        ax = self.ranks_df.plot(marker='.')
        for _, column_ss in self.ranks_df.items():
            ax.scatter(x=column_ss.idxmin(), y=column_ss.min())
        fig = ax.figure
        return fig

    def delete_predictions_and_targets(self):
        '''
        Deletes predictions and targets from all model and inspector instances

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        for ccbcv in self.models_lt:
            ccbcv.delete_predictions_and_targets()
        return self

    def _update_params(
            self,
            train_dir_sr: str):
        '''
        Updates parameters passed to CustomCatBoostCV

        Parameters
        ----------
        train_dir_sr : str
            Output directory.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        self.cat_boost_dt['train_dir'] = train_dir_sr
        return self

    def _update_best_score_and_iteration(
            self,
            score_ft: float,
            iteration_it: int):
        '''
        Updates best score and iteration after fitting model

        Parameters
        ----------
        score_ft : float
            Current score on eval metric.
        iteration_it : int
            Current iteration.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if (self.objective_sr == 'minimize') and (score_ft < self.best_score_ft):
            self.best_score_ft = score_ft
            self.best_iteration_it = iteration_it
        elif (self.objective_sr == 'maximize') and (score_ft > self.best_score_ft):
            self.best_score_ft = score_ft
            self.best_iteration_it = iteration_it
        return self

    def _get_examples(
            self,
            X: pd.DataFrame,
            ei: ExampleInspector):
        '''
        Gets examples used in model and those to be dropped and kept based on it

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        ei : ExampleInspector
            Example inspector object.

        Returns
        -------
        examples_ix : pd.Index
            Examples used.
        drop_ix : pd.Index
            Examples to drop.
        keep_ix : pd.Index
            Examples to keep.

        '''
        examples_ix = X.index
        drop_ix = (
            ei.example_importances_df
            .query(expr='lci > 0')['mean']
            .nlargest(n=self.example_importances_nlargest_n_it)
            .index)
        keep_ix = examples_ix.difference(other=drop_ix)
        return examples_ix, drop_ix, keep_ix

    def _get_result(
            self,
            iteration_it: int,
            score_ft: float,
            examples_ix: pd.Index,
            drop_ix: pd.Index,
            keep_ix: pd.Index):
        '''
        Gets current result as dict

        Parameters
        ----------
        iteration_it : int
            Current iteration.
        score_ft : float
            Current score on eval metric.
        examples_ix : pd.Index
            Examples used.
        drop_ix : pd.Index
            Examples to drop.
        keep_ix : pd.Index
            Examples to keep.

        Returns
        -------
        result_dt : dict
            Current result.

        '''
        result_dt = {
            'iterations': iteration_it,
            'scores': score_ft,
            'cnt_examples': examples_ix.shape[0],
            'cnt_drop': drop_ix.shape[0],
            'cnt_keep': keep_ix.shape[0],
            'best_iterations': self.best_iteration_it,
            'best_scores': self.best_score_ft,
            'examples': examples_ix}
        return result_dt

    def _print_result(
            self,
            result_dt: dict):
        '''
        Prints current result

        Parameters
        ----------
        result_dt : dict
            Current result.

        Returns
        -------
        None.

        '''
        print('Result:')
        for key_sr in result_dt.keys():
            if key_sr != 'examples':
                print('- {}: {}'.format(key_sr, result_dt[key_sr]))

    def _get_results(
            self,
            results_lt: list):
        '''
        Gets all results as data frame

        Parameters
        ----------
        results_lt : list
            Results from all iterations.

        Returns
        -------
        results_df : pd.DataFrame
            Results from all iterations.

        '''
        results_df = pd.DataFrame(data=results_lt).set_index(keys='iterations')
        return results_df

    def _get_ranks(self):
        '''
        Ranks results based on scores and example counts

        Returns
        -------
        ranks_df : pd.DataFrame
            Ranked results from all iterations.

        '''
        ranks_df = (
            self.results_df
            .iloc[:, :2]
            .assign(
                scores = lambda x: x['scores'].rank(pct=True, ascending=False if self.objective_sr == 'maximize' else True),
                cnt_examples = lambda x: x['cnt_examples'].rank(pct=True),
                combined = lambda x: x.sum(axis=1))
            .rename(columns=lambda x: f'{x}_rank'))
        return ranks_df

    def _get_best_iteration(self):
        '''
        Gets best iteration according to rank (including weighted if it exists)

        Returns
        -------
        best_iteration_it : int
            Best iteration.

        '''
        best_iteration_it = self.ranks_df.iloc[:, -1].idxmin()
        return best_iteration_it
    
# =============================================================================
# FeatureSelector
# =============================================================================

class FeatureSelector:
    def __init__(
            self,
            model_type_sr: str,
            cat_boost_dt: dict, 
            sklearn_splitter, 
            objective_sr: str, 
            strategy_sr: str, 
            wait_it: int):
        '''
        Iteratively removes features

        Parameters
        ----------
        cat_boost_dt : dict
            Parameters passed to cb.CatBoost.
        binary_bl : bool
            Flag for binary classification.
        sklearn_splitter : TYPE
            Instance of scikit-learn splitter class.
        objective_sr : str
            One of ['minimize', 'maximize'].
        strategy_sr : str
            How features should be dropped. Must be one of ['drop_mean_at_or_below_zero', 'drop_uci_below_zero', 'drop_lowest_mean'] for removing...
            - Either all features with mean LFC <= 0
            - Or all features with UCI < 0
            - Or lowest feature by mean LFC.
        wait_it : int
            Iterations to wait before stopping procedure.

        Returns
        -------
        None.

        '''
        self.model_type_sr = model_type_sr
        required_keys_lt = ['train_dir']
        for required_key_sr in required_keys_lt:
            if required_key_sr not in cat_boost_dt.keys():
                raise KeyError(f'Required key "{required_key_sr}" is not in cat_boost_dt')
        self.cat_boost_dt = cat_boost_dt
        self.sklearn_splitter = sklearn_splitter
        permitted_objectives_lt = ['minimize', 'maximize']
        if objective_sr not in permitted_objectives_lt:
            raise ValueError(f'Permitted values of objective_sr are {permitted_objectives_lt}')
        self.objective_sr = objective_sr
        self.best_score_ft = np.inf if objective_sr == 'minimize' else -np.inf
        self.best_iteration_it = 0
        implemented_strategies_lt = ['drop_mean_at_or_below_zero', 'drop_uci_below_zero', 'drop_lowest_mean']
        if strategy_sr not in implemented_strategies_lt:
            raise NotImplementedError(f'Implemented values of strategy_sr are {implemented_strategies_lt}')
        self.strategy_sr = strategy_sr
        self.wait_it = wait_it
        
    def fit(
            self, 
            X: pd.DataFrame, 
            y: pd.Series):
        '''
        Iteratively fits models

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
        # Make directory
        output_directory_sr = self.cat_boost_dt['train_dir']
        os.mkdir(path=output_directory_sr)
        
        # Initialize
        iteration_it = 0
        self.models_lt = []
        results_lt = []
        
        # Loop
        while True:
            # Log
            print('=' * 80)
            print(f'Iteration: {iteration_it}')
            
            # Update params
            output_subdirectory_sr = '{}/{:03d}'.format(output_directory_sr, iteration_it)
            self._update_params(X=X, train_dir_sr=output_subdirectory_sr)
            
            # Fit model
            ccbcv = CustomCatBoostCV(model_type_sr=self.model_type_sr, cat_boost_dt=self.cat_boost_dt, sklearn_splitter=self.sklearn_splitter)
            ccbcv.fit(X=X, y=y)
            self.models_lt.append(ccbcv)
            
            # Get score and update bests
            score_ft = ccbcv.eval_metrics_df.loc[self.cat_boost_dt['eval_metric'], 'validation_mean']
            self._update_best_score_and_iteration(score_ft=score_ft, iteration_it=iteration_it)
            
            # Get features to drop and keep
            features_ix, drop_ix, keep_ix = self._get_features(X=X, ccbcv=ccbcv)
            
            # Get and print result
            result_dt = self._get_result(iteration_it=iteration_it, score_ft=score_ft, features_ix=features_ix, drop_ix=drop_ix, keep_ix=keep_ix)
            results_lt.append(result_dt)
            self._print_result(result_dt=result_dt)
            
            # Evaluate whether to continue
            if ((iteration_it - self.best_iteration_it == self.wait_it) or 
            (features_ix.shape[0] == 1) or 
            (drop_ix.empty)):
                break
            else:
                X.drop(columns=drop_ix, inplace=True)
                iteration_it += 1
        
        # Get results
        self.results_df = self._get_results(results_lt=results_lt)
        
        # Get ranks
        self.ranks_df = self._get_ranks()
        return self

    def weight_ranks(
            self, 
            weights_dt: dict):
        '''
        Weights score and feature count ranks to arrive at new combined rank

        Parameters
        ----------
        weights_dt : dict
            Dict with...
            - Keys of 'scores' and 'cnt_features'
            - Values corresponding to importances (higher = more important).

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        required_keys_lt = ['scores', 'pct_diffs', 'cnt_features']
        for required_key_sr in required_keys_lt:
            if required_key_sr not in weights_dt.keys():
                raise KeyError(f'Required key "{required_key_sr}" is not in weights_dt')
        self.ranks_df['weighted_rank'] = (
            self.ranks_df['scores_rank'] * weights_dt['scores'] +
            self.ranks_df['pct_diffs_rank'] * weights_dt['pct_diffs'] +
            self.ranks_df['cnt_features_rank'] * weights_dt['cnt_features'])
        return self
    
    def get_best_features(self):
        '''
        Gets features from best model according to rank (including weighted if it exists)

        Returns
        -------
        best_features_ix : pd.Index
            Features from best model.

        '''
        best_iteration_it = self._get_best_iteration()
        best_features_ix = self.results_df.loc[best_iteration_it, 'features']
        return best_features_ix
    
    def get_best_model(self):
        '''
        Gets best model according to rank (including weighted if it exists)

        Returns
        -------
        best_ccbcv : CustomCatBoostCV
            Best model.

        '''
        best_iteration_it = self._get_best_iteration()
        best_ccbcv = self.models_lt[best_iteration_it]
        return best_ccbcv
    
    def plot_results(self):
        '''
        Plots results (original scale)

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        fig, axes = plt.subplots(nrows=2, sharex=True)
        self.results_df.iloc[:, :2].plot(marker='.', subplots=True, ax=axes)
        for index_it, ax in enumerate(iterable=axes.ravel()):
            data_ss = self.results_df.iloc[:, index_it].describe().round(decimals=3)
            pd.plotting.table(ax=ax, data=data_ss, bbox=[1.25, 0, 0.25, 1])
        fig.tight_layout()
        return fig
    
    def plot_ranks(self):
        '''
        Plots ranks (common scale)

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        ax = self.ranks_df.plot(marker='.')
        for _, column_ss in self.ranks_df.items():
            ax.scatter(x=column_ss.idxmin(), y=column_ss.min())
        fig = ax.figure
        return fig
    
    def delete_predictions_and_targets(self):
        '''
        Deletes predictions and targets from all model instances

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        for ccbcv in self.models_lt:
            ccbcv.delete_predictions_and_targets()
        return self
    
    def _update_params(
            self, 
            X: pd.DataFrame, 
            train_dir_sr: str):
        '''
        Updates parameters passed to CustomCatBoostCV

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        train_dir_sr : str
            Output directory.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        self.cat_boost_dt['cat_features'] = [
            feature_sr for feature_sr in self.cat_boost_dt['cat_features'] if feature_sr in X.columns]
        self.cat_boost_dt['monotone_constraints'] = {
            feature_sr: direction_it for feature_sr, direction_it in self.cat_boost_dt['monotone_constraints'].items()
            if feature_sr in X.columns}
        self.cat_boost_dt['train_dir'] = train_dir_sr
        return self
            
    def _update_best_score_and_iteration(
            self, 
            score_ft: float, 
            iteration_it: int):
        '''
        Updates best score and iteration after fitting model

        Parameters
        ----------
        score_ft : float
            Current score on eval metric.
        iteration_it : int
            Current iteration.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if (self.objective_sr == 'minimize') and (score_ft < self.best_score_ft):
            self.best_score_ft = score_ft
            self.best_iteration_it = iteration_it
        elif (self.objective_sr == 'maximize') and (score_ft > self.best_score_ft):
            self.best_score_ft = score_ft
            self.best_iteration_it = iteration_it
        return self
    
    def _get_features(
            self, 
            X: pd.DataFrame, 
            ccbcv: CustomCatBoostCV):
        '''
        Gets features used in model and those to be dropped and kept based on it

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        ccbcv : CustomCatBoostCV
            CV model object.

        Returns
        -------
        features_ix : pd.Index
            Features used.
        drop_ix : pd.Index
            Features to drop.
        keep_ix : pd.Index
            Features to keep.

        '''
        features_ix = X.columns
        if self.strategy_sr == 'drop_mean_at_or_below_zero':
            drop_ix = ccbcv.lfc_feature_importances_df.query(expr='validation_mean <= 0').index
        elif self.strategy_sr == 'drop_uci_below_zero':
            drop_ix = ccbcv.lfc_feature_importances_df.query(expr='validation_uci < 0').index
        elif self.strategy_sr == 'drop_lowest_mean':
            drop_ix = pd.Index(data=[ccbcv.lfc_feature_importances_df['validation_mean'].idxmin()])
        keep_ix = features_ix.difference(other=drop_ix)
        return features_ix, drop_ix, keep_ix
    
    def _get_result(
            self, 
            iteration_it: int, 
            score_ft: float, 
            features_ix: pd.Index, 
            drop_ix: pd.Index, 
            keep_ix: pd.Index):
        '''
        Gets current result as dict

        Parameters
        ----------
        iteration_it : int
            Current iteration.
        score_ft : float
            Current score on eval metric.
        features_ix : pd.Index
            Features used.
        drop_ix : pd.Index
            Features to drop.
        keep_ix : pd.Index
            Features to keep.

        Returns
        -------
        result_dt : dict
            Current result.

        '''
        result_dt = {
            'iterations': iteration_it,
            'scores': score_ft,
            'cnt_features': features_ix.shape[0],
            'cnt_drop': drop_ix.shape[0],
            'cnt_keep': keep_ix.shape[0],
            'best_iterations': self.best_iteration_it,
            'best_scores': self.best_score_ft,
            'features': features_ix}
        return result_dt
    
    def _print_result(
            self, 
            result_dt: dict):
        '''
        Prints current result

        Parameters
        ----------
        result_dt : dict
            Current result.

        Returns
        -------
        None.

        '''
        print('Result:')
        for key_sr in result_dt.keys():
            if key_sr != 'features':
                print('- {}: {}'.format(key_sr, result_dt[key_sr]))
    
    def _get_results(
            self, 
            results_lt: list):
        '''
        Gets all results as data frame

        Parameters
        ----------
        results_lt : list
            Results from all iterations.

        Returns
        -------
        results_df : pd.DataFrame
            Results from all iterations.

        '''
        results_df = pd.DataFrame(data=results_lt).set_index(keys='iterations')
        return results_df
    
    def _get_ranks(self):
        '''
        Ranks results based on scores and feature counts

        Returns
        -------
        ranks_df : pd.DataFrame
            Ranked results from all iterations.

        '''
        ranks_df = (
            self.results_df
            .iloc[:, :2]
            .assign(
                scores = lambda x: x['scores'].rank(pct=True, ascending=False if self.objective_sr == 'maximize' else True),
                cnt_features = lambda x: x['cnt_features'].rank(pct=True),
                combined = lambda x: x.sum(axis=1))
            .rename(columns=lambda x: f'{x}_rank'))
        return ranks_df
    
    def _get_best_iteration(self):
        '''
        Gets best iteration according to rank (including weighted if it exists)

        Returns
        -------
        best_iteration_it : int
            Best iteration.

        '''
        best_iteration_it = self.ranks_df.iloc[:, -1].idxmin()
        return best_iteration_it