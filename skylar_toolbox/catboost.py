# =============================================================================
# Load libraries
# =============================================================================

import catboost as cb
import json
import numpy as np
import os
import pandas as pd
import seaborn as sns; sns.set()
import tqdm
from catboost import utils as cbus
from matplotlib import pyplot as plt
from scipy import special as sysl
from sklearn import inspection as snin
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
        Wraps cb.CatBoost, storing metadata and providing methods for plotting

        Parameters
        ----------
        model_type_sr : str
            Model type.
        cat_boost_dt : dict
            Parameters passed to CatBoost.

        Raises
        ------
        NotImplementedError
            Implemented values of model_type_sr are  ['classification', 'regression']
        KeyError
            Required key "['train_dir']" is not in cat_boost_dt.

        Returns
        -------
        None.

        '''
        implemented_model_types_lt = ['classification', 'regression']
        if model_type_sr not in implemented_model_types_lt:
            raise NotImplementedError(f'Implemented values of model_type_sr are {implemented_model_types_lt}')
        self.model_type_sr = model_type_sr
        required_keys_lt = ['train_dir']
        for required_key_sr in required_keys_lt:
             if required_key_sr not in cat_boost_dt.keys():
                 raise KeyError(f'Required key "{required_key_sr}" is not in cat_boost_dt')
        self.cat_boost_dt = get_parameters(model_type_sr=model_type_sr, cat_boost_dt=cat_boost_dt)
    
    def fit(
            self, 
            X_train: pd.DataFrame, 
            y_train: pd.Series, 
            X_valid: pd.DataFrame, 
            y_valid: pd.Series, 
            fit_dt: dict = dict(),
            sample_dt: dict = None):
        '''
        Fits model and stores metadata

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
        fit_dt : dict, optional
            Fit params. The default is dict().
        sample_dt : dict, optional
            Sample params for getting example importances. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        # Fit model
        self.cbm = cb.CatBoost(params=self.cat_boost_dt)
        self.cbm.fit(X=X_train, y=y_train, eval_set=(X_valid, y_valid), **fit_dt)

        # Save best iteration
        self.best_iteration_it = self.cbm.best_iteration_

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
        
        # Get and compare feature importances
        self.feature_importances_df = self._compare_feature_importances(
            X_train=X_train, X_valid=X_valid)
        
        # Get interaction strengths
        if X_train.shape[1] > 1:
            self.interaction_strengths_df = self._get_interaction_strengths()
            
        # Get example importances
        if sample_dt:
            valid_ix = y_valid.sample(**sample_dt).index
            self.example_importances_ss = self._get_example_importances(
                X_train=X_train, X_valid=X_valid, valid_ix=valid_ix)
        return self
    
    def plot_evals_result(self):
        '''
        Plots evals result per iteration

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
        Plots bars of eval metrics on train and validation with table

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        columns_lt = ['learn', 'validation', 'pct_diff']
        plot_df = self.eval_metrics_df[columns_lt[:-1]]
        data_df = self.eval_metrics_df[columns_lt].round(decimals=3)
        ax = plot_df.plot(kind='bar')
        pd.plotting.table(ax=ax, data=data_df, bbox=[1.25, 0, 0.5, 1])
        fig = ax.figure
        return fig
    
    def plot_feature_importances(
            self, 
            plot_type_sr: str):
        '''
        Plots feature importances

        Parameters
        ----------
        plot_type_sr : str
            Plot type.

        Raises
        ------
        NotImplementedError
            Implemented values of plot_type_sr are ['all', 'top_bottom', 'abs_diff', 'pct_diff']

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        implemented_plot_types_lt = ['all', 'top_bottom', 'abs_diff', 'pct_diff']
        if plot_type_sr == 'all':
            columns_lt = ['learn', 'validation', 'pct_diff']
            plot_df = self.feature_importances_df[columns_lt[:-1]]
            data_df = self.feature_importances_df[columns_lt].describe().round(decimals=3)
            ax = plot_df.plot()
            ax.set(xticks=[])
            ax.axhline(y=0, c='k', ls=':')
            pd.plotting.table(ax=ax, data=data_df, bbox=[1.25, 0, 0.5, 1])
            fig = ax.figure
            return fig
        elif plot_type_sr == 'top_bottom':
            fig, axes = plt.subplots(nrows=2, sharex=True)
            for index_it, split_sr in enumerate(iterable=['learn', 'validation']):
                pd.concat(objs=[
                    self.feature_importances_df[split_sr].nsmallest(),
                    self.feature_importances_df[split_sr].nlargest()[::-1]]).plot(kind='barh', ax=axes[index_it])
                axes[index_it].axvline(x=0, c='k', ls=':')
                axes[index_it].set(title=split_sr)
            fig.tight_layout()
            return fig
        elif plot_type_sr in ['abs_diff', 'pct_diff']:
            top_bottom_df = self.feature_importances_df.nlargest(n=10, columns=plot_type_sr) if plot_type_sr == 'abs_diff' \
                else self.feature_importances_df.nsmallest(n=10, columns=plot_type_sr)
            ax = (
                top_bottom_df
                .sort_values(by=plot_type_sr, ascending=True if plot_type_sr == 'abs_diff' else False)
                .loc[:, ['learn', 'validation']]
                .plot(kind='barh'))
            ax.axvline(x=0, c='k', ls=':')
            fig = ax.figure
            return fig
        else:
            raise NotImplementedError(f'Implemented values of plot_type_sr are {implemented_plot_types_lt}')
            
    def plot_interaction_strengths(self):
        '''
        Plots interaction strengths

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        ax = steda.plot_line(ss=self.interaction_strengths_df['strengths'])
        fig = ax.figure
        return fig
    
    def plot_example_importances(self):
        '''
        Plots example importances

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        ax = steda.plot_line(ss=self.example_importances_ss)
        fig = ax.figure
        return fig
    
    def plot_predictions(self):
        '''
        Plots predictions as histogram (classification) or KDE (regression)

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(15, 10))
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
        Deletes predictions and targets

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
        Gets evals result as data frame

        Returns
        -------
        evals_result_df : pd.DataFrame
            Evals result.

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
        Gets predictions

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
        Compares train and validation eval metrics

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
            name_sr: str):
        '''
        Gets feature importances

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        name_sr : str
            Split name.

        Returns
        -------
        feature_importances_ss : pd.Series
            Feature importances.

        '''
        feature_importances_ay = self.cbm.get_feature_importance(
            data=cb.Pool(data=X, label=y, cat_features=self.cat_boost_dt['cat_features']), 
            type='LossFunctionChange')
        feature_importances_ss = pd.Series(
            data=feature_importances_ay,
            index=X.columns,
            name=name_sr)
        return feature_importances_ss

    def _compare_feature_importances(
            self, 
            X_train: pd.DataFrame, 
            X_valid: pd.DataFrame):
        '''
        Compares train and validation feature importances

        Parameters
        ----------
        X_train : pd.DataFrame
            Train feature matrix.
        X_valid : pd.DataFrame
            Validation feature matrix.

        Returns
        -------
        feature_importances_df : pd.DataFrame
            Feature importances.

        '''
        objs_lt = [
            self._get_feature_importances(X=X, y=y, name_sr=name_sr)
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
        interaction_strengths_df = pd.DataFrame(data=self.cbm.get_feature_importance(type='Interaction'))
        if interaction_strengths_df.shape[0] > 0:
            map_dt = {index_it: feature_sr for index_it, feature_sr in enumerate(iterable=self.cbm.feature_names_)}
            interaction_strengths_df = (
                interaction_strengths_df 
                .set_axis(labels=['first_features', 'second_features', 'strengths'], axis=1)
                .assign(
                    first_features = lambda x: x['first_features'].map(arg=map_dt),
                    second_features = lambda x: x['second_features'].map(arg=map_dt)))
        return interaction_strengths_df
    
    def _get_example_importances(
        self, 
        X_train: pd.DataFrame, 
        X_valid: pd.Series, 
        valid_ix: pd.Index):
        '''
        Gets example importances

        Parameters
        ----------
        X_train : pd.DataFrame
            Train feature matrix.
        X_valid : pd.DataFrame
            Validation feature matrix.
        valid_ix : pd.Index
            Sampled validation index.

        Returns
        -------
        example_importances_ss : pd.Series
            Example importances.

        '''
        train_ix = self.y_train.index
        pool_dt = dict(cat_features=self.cat_boost_dt['cat_features'])
        indices_lt, scores_lt = self.cbm.get_object_importance(
            pool=cb.Pool(data=X_valid.loc[valid_ix, :], label=self.y_valid.loc[valid_ix], **pool_dt),
            train_pool=cb.Pool(data=X_train, label=self.y_train, **pool_dt), 
            thread_count=1,
            verbose=train_ix.shape[0] // 10)
        example_importances_ss = pd.Series(data=scores_lt, index=train_ix[indices_lt], name='importances')
        return example_importances_ss

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
        Cross-validates wrapped model

        Parameters
        ----------
        model_type_sr : str
            Model type.
        cat_boost_dt : dict
            Parameters passed to CatBoost.
        sklearn_splitter : TYPE
            Splitter from scikit-learn.

        Raises
        ------
        NotImplementedError
            Implemented values of model_type_sr are ['classification', 'regression']
        KeyError
            Required key "['train_dir']" is not in cat_boost_dt.

        Returns
        -------
        None.

        '''        
        implemented_model_types_lt = ['classification', 'regression']
        if model_type_sr not in implemented_model_types_lt:
            raise NotImplementedError(f'Implemented values of model_type_sr are {implemented_model_types_lt}')
        self.model_type_sr = model_type_sr
        required_keys_lt = ['train_dir']
        for required_key_sr in required_keys_lt:
             if required_key_sr not in cat_boost_dt.keys():
                 raise KeyError(f'Required key "{required_key_sr}" is not in cat_boost_dt')
        self.cat_boost_dt = get_parameters(model_type_sr=model_type_sr, cat_boost_dt=cat_boost_dt)
        self.sklearn_splitter = sklearn_splitter
        
    def fit(
            self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            split_dt: dict = dict(),
            fit_dt: dict = dict(), 
            sample_dt: dict = None):
        '''
        Fits model and stores metadata

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        split_dt : dict, optional
            Split params (e.g., groups). The default is dict().
        fit_dt : dict, optional
            Fit params. The default is dict().
        sample_dt : dict, optional
            Sample params for getting example importances. The default is None.

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
        
        for index_it, (train_ay, test_ay) in enumerate(iterable=self.sklearn_splitter.split(X=X, y=y, **split_dt)):
            # Log
            print('-' * 80)
            print(f'Split: {index_it}')
            print('Shapes:\n- {}'.format('\n- '.join(str(tmp.shape) for tmp in [train_ay, test_ay])))
            
            # Make subdirectory
            output_subdirectory_sr = '{}/{:03d}'.format(output_directory_sr, index_it)
            os.mkdir(path=output_subdirectory_sr)
            
            # Update parameters
            self.cat_boost_dt['train_dir'] = output_subdirectory_sr

            # Fit model
            ccb = CustomCatBoost(model_type_sr=self.model_type_sr, cat_boost_dt=self.cat_boost_dt)
            ccb.fit(
                X_train=X.iloc[train_ay, :], y_train=y.iloc[train_ay], 
                X_valid=X.iloc[test_ay, :], y_valid=y.iloc[test_ay], 
                fit_dt=fit_dt, 
                sample_dt=sample_dt)
            self.models_lt.append(ccb)
        
        # Compare eval metrics
        self.eval_metrics_df = self._compare_eval_metrics()
        
        # Compare feature importances
        self.feature_importances_df = self._compare_feature_importances()
        
        # Compare example importances
        if sample_dt:
            self.example_importances_df = self._compare_example_importances()
        return self
    
    def sum_models(
            self, 
            strategy_sr: str = 'weight_by_score'):
        '''
        Ensembles models trained on different subsets of data

        Parameters
        ----------
        strategy_sr : str, optional
            Strategy for weighting models in ensemble. The default is 'weight_by_score'.

        Raises
        ------
        NotImplementedError
            Implemented values of strategy_sr are ['weight_by_score', 'weight_equally']

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        models_lt = [ccb.cbm for ccb in self.models_lt]
        implemented_strategies_lt = ['weight_by_score', 'weight_equally']
        if strategy_sr == 'weight_by_score':
            weights_lt = self.eval_metrics_df.filter(regex='validation_\d').loc[self.cat_boost_dt['eval_metric'], :].tolist()
            return cb.sum_models(models=models_lt, weights=weights_lt)
        elif strategy_sr == 'weight_equally':
            return cb.sum_models(models=models_lt)
        else:
            raise NotImplementedError(f'Implemented values of strategy_sr are {implemented_strategies_lt}')
    
    def plot_eval_metrics(self):
        '''
        Plots eval metrics with error bars

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        # Get data frames
        ys_df = self.eval_metrics_df.filter(like='_mean').rename(columns=lambda x: x.split('_')[0])
        yerrs_df = self.eval_metrics_df.filter(like='se2').rename(columns=lambda x: x.split('_')[0])
        data_df = self.eval_metrics_df[['learn_mean', 'validation_mean', 'pct_diff']].round(decimals=3)
        # Plot
        ax = ys_df.plot(kind='bar', yerr=yerrs_df)
        pd.plotting.table(ax=ax, data=data_df, bbox=[1.25, 0, 1, 1])
        fig = ax.figure
        return fig
    
    def plot_feature_importances(
            self, 
            plot_type_sr: str):
        '''
        Plots feature importances with error bars

        Parameters
        ----------
        plot_type_sr : str
            Plot type.

        Raises
        ------
        NotImplementedError
            Implemented values of plot_type_sr are ['all', 'top_bottom']

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        xs_df = ys_df = self.feature_importances_df.filter(like='_mean').rename(columns=lambda x: x.split('_')[0])
        xerrs_df = yerrs_df = self.feature_importances_df.filter(like='se2').rename(columns=lambda x: x.split('_')[0])
        implemented_plot_types_lt = ['all', 'top_bottom']
        if plot_type_sr == 'all':
            data_df = ys_df.describe().round(decimals=3)
            ax = ys_df.plot(yerr=yerrs_df)
            ax.set(xticks=[])
            ax.axhline(y=0, c='k', ls=':')
            pd.plotting.table(ax=ax, data=data_df, bbox=[1.25, 0, 0.5, 1])
            fig = ax.figure
            return fig
        elif plot_type_sr == 'top_bottom':
            fig, axes = plt.subplots(nrows=2, sharex=True)
            for index_it, split_sr in enumerate(iterable=['learn', 'validation']):
                concat_df = pd.concat(objs=[
                    xs_df[split_sr].nsmallest(),
                    xs_df[split_sr].nlargest()[::-1]])
                concat_df.plot(kind='barh', xerr=xerrs_df.loc[concat_df.index, :], ax=axes[index_it])
                axes[index_it].axvline(x=0, c='k', ls=':')
                axes[index_it].set(title=split_sr)
            fig.tight_layout()
            return fig
        else:
            raise NotImplementedError(f'Implemented values of plot_type_sr are {implemented_plot_types_lt}')
            
    def plot_example_importances(self):
        '''
        Plots example importances with error bars

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        data_ss = self.example_importances_df['mean'].describe().round(decimals=3)
        ax = (
            self.example_importances_df
            .sort_values(by='mean', ascending=False)
            .plot(y='mean', yerr='se2'))
        ax.set(xticks=[])
        ax.axhline(y=0, c='k', ls=':')
        pd.plotting.table(ax=ax, data=data_ss, bbox=[1.25, 0, 0.25, 1])
        fig = ax.figure
        return fig
            
    def delete_predictions_and_targets(self):
        '''
        Deletes predictions and targets

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
        Compares train and validation eval metrics

        Returns
        -------
        eval_metrics_df : pd.DataFrame
            Eval metrics.

        '''
        # Concatenate them
        eval_metrics_df = pd.concat(objs=[
            ccb.eval_metrics_df[['learn', 'validation']].rename(columns=lambda x: f'{x}_{index_it}')
            for index_it, ccb in enumerate(iterable=self.models_lt)
        ], axis=1)
        # Get means
        for split_sr in ['learn', 'validation']:
            eval_metrics_df = (
                steda.get_means(df=eval_metrics_df, columns_lt=eval_metrics_df.filter(like=split_sr).columns.tolist())
                .rename(columns=lambda x: f'{split_sr}_{x}' if '_' not in x else x))
        # Get differences
        eval_metrics_df = steda.get_differences(df=eval_metrics_df, columns_lt=['learn_mean', 'validation_mean'])
        return eval_metrics_df
    
    def _compare_feature_importances(self):
        '''
        Compares train and validation feature importances

        Returns
        -------
        feature_importances_df : pd.DataFrame
            Feature importances.

        '''
        # Concatenate them
        feature_importances_df = pd.concat(objs=[
            ccb.feature_importances_df[['learn', 'validation']].rename(columns=lambda x: f'{x}_{index_it}')
            for index_it, ccb in enumerate(iterable=self.models_lt)
        ], axis=1)
        # Get means
        for split_sr in ['learn', 'validation']:
            feature_importances_df = (
                steda.get_means(df=feature_importances_df, columns_lt=feature_importances_df.filter(like=split_sr).columns.tolist())
                .rename(columns=lambda x: f'{split_sr}_{x}' if '_' not in x else x))
        # Sort
        feature_importances_df.sort_values(by='validation_mean', ascending=False, inplace=True)
        # Get differences
        feature_importances_df = steda.get_differences(df=feature_importances_df, columns_lt=['learn_mean', 'validation_mean'])
        return feature_importances_df
    
    def _compare_example_importances(self):
        '''
        Compares example importances

        Returns
        -------
        example_importances_df : pd.DataFrame
            Example importances.

        '''
        # Concatenate them
        example_importances_df = pd.concat(objs=[
            ccb.example_importances_ss.rename(index=index_it)
            for index_it, ccb in enumerate(iterable=self.models_lt)
        ], axis=1)
        # Get means
        example_importances_df = steda.get_means(
            df=example_importances_df, 
            columns_lt=example_importances_df.columns.tolist())
        # Sort
        example_importances_df.sort_values(by='mean', ascending=False, inplace=True)
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
            objective_sr: str = 'minimize', 
            strategy_sr: str = 'drop_positive_means', 
            wait_it: int = 10, 
            store_models_bl: bool = False, 
            fit_only_wait_bl: bool = False):
        '''
        Selects examples by iteratively removing those contributing most to validation losses

        Parameters
        ----------
        model_type_sr : str
            Model type.
        cat_boost_dt : dict
            Parameters passed to CatBoost.
        sklearn_splitter : TYPE
            Splitter from scikit-learn.
        objective_sr : str, optional
            Objective for eval metric. The default is 'minimize'.
        strategy_sr : str, optional
            Strategy for dropping examples. The default is 'drop_positive_means'.
        wait_it : int, optional
            Number of iterations to wait before terminating procedure. The default is 10.
        store_models_bl : bool, optional
            Flag for whether to store during procedure to save memory. The default is False.
        fit_only_wait_bl : bool, optional
            Flag for whether to fit only as many iterations as wait. The default is False.

        Raises
        ------
        KeyError
            Required key "['train_dir']" is not in cat_boost_dt.
        ValueError
            Permitted values of objective_sr are ['minimize', 'maximize']
        NotImplementedError
            Implemented values of strategy_sr are ['drop_positive_means', 'drop_positive_lcis']

        Returns
        -------
        None.

        '''
        self.model_type_sr = model_type_sr
        required_keys_lt = ['train_dir']
        for required_key_sr in required_keys_lt:
            if required_key_sr not in cat_boost_dt.keys():
                raise KeyError(f'Required key "{required_key_sr}" is not in cat_boost_dt')
        self.cat_boost_dt = get_parameters(model_type_sr=model_type_sr, cat_boost_dt=cat_boost_dt)
        self.sklearn_splitter = sklearn_splitter
        permitted_objectives_lt = ['minimize', 'maximize']
        if objective_sr not in permitted_objectives_lt:
            raise ValueError(f'Permitted values of objective_sr are {permitted_objectives_lt}')
        self.objective_sr = objective_sr
        self.best_score_ft = np.inf if objective_sr == 'minimize' else -np.inf
        self.best_iteration_it = 0
        implemented_strategies_lt = ['drop_positive_means', 'drop_positive_lcis']
        if strategy_sr not in implemented_strategies_lt:
            raise NotImplementedError(f'Implemented values of strategy_sr are {implemented_strategies_lt}')
        self.strategy_sr = strategy_sr
        self.wait_it = wait_it
        self.store_models_bl = store_models_bl
        self.fit_only_wait_bl = fit_only_wait_bl
        
    def fit(
            self, 
            X: pd.DataFrame, 
            y: pd.Series,
            split_dt: dict = dict(),
            fit_dt: dict = dict(), 
            sample_dt: dict = dict(n=1_000)):
        '''
        Fits models, stores metadata, and drops examples

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        split_dt : dict, optional
            Split params (e.g., groups). The default is dict().
        fit_dt : dict, optional
            Fit params. The default is dict().
        sample_dt : dict, optional
            Sample params. The default is dict(n=1_000).

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
        self.results_lt = []
        
        # Loop
        while True:
            # Log
            print('=' * 80)
            print(f'Iteration: {iteration_it}')
            
            # Update parameters
            output_subdirectory_sr = '{}/{:03d}'.format(output_directory_sr, iteration_it)
            self._update_params(train_dir_sr=output_subdirectory_sr)
            
            # Fit model
            ccbcv = CustomCatBoostCV(model_type_sr=self.model_type_sr, cat_boost_dt=self.cat_boost_dt, sklearn_splitter=self.sklearn_splitter)
            try:
                ccbcv.fit(X=X, y=y, split_dt=split_dt, fit_dt=fit_dt, sample_dt=sample_dt)
            except:
                break
            if self.store_models_bl:
                self.models_lt.append(ccbcv)
            
            # Get score and update bests
            score_ft = ccbcv.eval_metrics_df.loc[self.cat_boost_dt['eval_metric'], 'validation_mean']
            self._update_best_score_and_iteration(score_ft=score_ft, iteration_it=iteration_it)
            
            # Get examples to drop and keep
            examples_ix, drop_ix, keep_ix = self._get_examples(X=X, ccbcv=ccbcv)
            
            # Get and print result
            pct_diff_ft = ccbcv.eval_metrics_df.loc[self.cat_boost_dt['eval_metric'], 'pct_diff']
            result_dt = self._get_result(iteration_it=iteration_it, score_ft=score_ft, pct_diff_ft=pct_diff_ft, examples_ix=examples_ix, drop_ix=drop_ix, keep_ix=keep_ix)
            self.results_lt.append(result_dt)
            self._print_result(result_dt=result_dt)
            
            # Get results
            self.results_df = self._get_results()
            
            # Get ranks
            self.ranks_df = self._get_ranks()
            
            # Checkpoint
            pd.to_pickle(obj=self, filepath_or_buffer=os.path.join(output_directory_sr, 'es.pkl'), protocol=4)
            
            # Evaluate whether to continue
            if ((iteration_it - self.best_iteration_it == self.wait_it) or 
            (drop_ix.empty) or
            (keep_ix.empty) or 
            ((self.fit_only_wait_bl) and (iteration_it == self.wait_it - 1))):
                break
            else:
                X.drop(index=drop_ix, inplace=True)
                y.drop(index=drop_ix, inplace=True)
                iteration_it += 1
        return self

    def weight_ranks(
            self,
            weights_dt: dict = {'scores': 1, 'pct_diffs': 0, 'cnt_examples': 0}):
        '''
        Creates new combined rank by weighting components

        Parameters
        ----------
        weights_dt : dict, optional
            Weights. The default is {'scores': 1, 'pct_diffs': 0, 'cnt_examples': 0}.

        Raises
        ------
        KeyError
            Required key "['scores', 'pct_diffs', 'cnt_examples']" is not in weights_dt.

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
        Gets examples from best iteration

        Returns
        -------
        best_examples_ix : pd.Index
            Best examples.

        '''
        best_iteration_it = self._get_best_iteration()
        best_examples_ix = self.results_df.loc[best_iteration_it, 'examples']
        return best_examples_ix

    def get_best_model(self):
        '''
        Gets model from best iteration

        Returns
        -------
        best_ccbcv : CatBoostCV
            Best cross-validated model.

        '''
        best_iteration_it = self._get_best_iteration()
        best_ccbcv = self.models_lt[best_iteration_it]
        return best_ccbcv
    
    def plot_results(self):
        '''
        Plots results

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        columns_lt = ['scores', 'pct_diffs', 'cnt_examples']
        fig, axes = plt.subplots(nrows=len(columns_lt), sharex=True, figsize=(10, 10))
        self.results_df[columns_lt].plot(marker='.', subplots=True, ax=axes)
        for index_it, column_sr in enumerate(iterable=columns_lt):
            data_ss = self.results_df[column_sr].describe().round(decimals=3)
            pd.plotting.table(ax=axes[index_it], data=data_ss, bbox=[1.25, 0, 0.25, 1])
        fig.tight_layout()
        return fig
    
    def plot_ranks(self):
        '''
        Plots ranks

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
        Deletes predictions and targets

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
        Updates parameters

        Parameters
        ----------
        train_dir_sr : str
            Train directory.

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
        Updates best score and iteration

        Parameters
        ----------
        score_ft : float
            Current score.
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
        ccbcv: CustomCatBoostCV):
        '''
        Gets examples

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        ccbcv : CustomCatBoostCV
            Cross-validated model.

        Returns
        -------
        examples_ix : pd.Index
            Examples.
        drop_ix : pd.Index
            Examples to drop.
        keep_ix : pd.Index
            Examples to keep.

        '''
        examples_ix = X.index
        if self.strategy_sr == 'drop_positive_means':
            drop_ix = ccbcv.example_importances_df.query(expr='mean > 0').index
        elif self.strategy_sr == 'drop_positive_lcis':
            drop_ix = ccbcv.example_importances_df.query(expr='lci > 0').index
        keep_ix = examples_ix.difference(other=drop_ix)
        return examples_ix, drop_ix, keep_ix
    
    def _get_result(
            self, 
            iteration_it: int, 
            score_ft: float, 
            pct_diff_ft: float,
            examples_ix: pd.Index, 
            drop_ix: pd.Index, 
            keep_ix: pd.Index):
        '''
        Gets result

        Parameters
        ----------
        iteration_it : int
            Current iteration.
        score_ft : float
            Current score.
        pct_diff_ft : float
            Current percent difference.
        examples_ix : pd.Index
            Examples.
        drop_ix : pd.Index
            Examples to drop.
        keep_ix : pd.Index
            Examples to keep.

        Returns
        -------
        result_dt : dict
            Result.

        '''
        result_dt = {
            'iterations': iteration_it,
            'scores': score_ft,
            'pct_diffs': pct_diff_ft,
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
        Prints result

        Parameters
        ----------
        result_dt : dict
            Result.

        Returns
        -------
        None.

        '''
        print('Result:')
        for key_sr in result_dt.keys():
            if key_sr != 'examples':
                print('- {}: {}'.format(key_sr, result_dt[key_sr]))
    
    def _get_results(self):
        '''
        Gets results

        Returns
        -------
        results_df : pd.DataFrame
            Results.

        '''
        results_df = pd.DataFrame(data=self.results_lt).set_index(keys='iterations')
        return results_df
    
    def _get_ranks(self):
        '''
        Gets ranks

        Returns
        -------
        ranks_df : pd.DataFrame
            Ranks.

        '''
        ranks_df = (
            self.results_df[['scores', 'pct_diffs', 'cnt_examples']]
            .assign(
                scores = lambda x: x['scores'].rank(pct=True, ascending=False if self.objective_sr == 'maximize' else True),
                pct_diffs = lambda x: x['pct_diffs'].rank(pct=True),
                cnt_examples = lambda x: x['cnt_examples'].rank(pct=True),
                combined = lambda x: x.sum(axis=1))
            .rename(columns=lambda x: f'{x}_rank'))
        return ranks_df
    
    def _get_best_iteration(self):
        '''
        Gets best iteration

        Returns
        -------
        best_iteration_it : int
            Best iteration.

        '''
        best_iteration_it = self.ranks_df.iloc[:, -1].idxmin()
        return best_iteration_it

# =============================================================================
# FeatureInspector
# =============================================================================

class FeatureInspector:
    def __init__(
            self, 
            ccb: CustomCatBoost, 
            strengths_nlargest_n_it: int = 10):
        '''
        Stores metadata and provides plotting methods

        Parameters
        ----------
        ccb : CustomCatBoost
            Model.
        strengths_nlargest_n_it : int, optional
            Interactions to plot. The default is 10

        Returns
        -------
        None.

        '''
        self.ccb = ccb
        self.strengths_nlargest_n_it = strengths_nlargest_n_it
    
    def fit(
            self, 
            X: pd.DataFrame, 
            y: pd.Series):
        '''
        Stores metadata

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
        # Save model
        fname_sr = 'cbm.json'
        self.ccb.cbm.save_model(fname=fname_sr, format='json')
        
        # Read in model
        with open(file=fname_sr) as fe:
            self.model_dt = json.load(fp=fe)
            
        # Remove model
        os.remove(path=fname_sr)
        
        # Get float features
        self.float_features_df = (
            pd.DataFrame(data=self.model_dt['features_info']['float_features'])
            .astype(dtype={'feature_index': pd.Int32Dtype()}))
        
        # Get splits
        self.splits_df = self._get_splits()
        
        # Get features
        self.features_lt = self.ccb.cbm.feature_names_
        self.cat_features_lt = self.ccb.cat_boost_dt['cat_features']
        self.other_features_lt = list(set(self.features_lt).difference(set(self.cat_features_lt)))
        
        # Sort columns
        X = X[self.features_lt]
        
        # Get feature importances by tree
        self.feature_importances_by_tree_df = self._get_feature_importances_by_tree(X=X, y=y)
        
        # Join importances to splits
        self._join_importances_to_splits()
        
        # Get feature importances by feature, tree, and border
        self.feature_importances_by_feature_tree_and_border_df = \
            self._get_feature_importances_by_feature_tree_and_border()
        
        # Get feature importances by feature and border
        self.feature_importances_by_feature_and_border_df = self._get_feature_importances_by_feature_and_border()
        
        # Get SHAPs
        shaps_df = self._get_shaps(X=X, y=y)
        
        # Get SHAPs by feature and border
        self.oneway_shaps_dt = self._get_shaps_by_feature_and_border(X=X, shaps_df=shaps_df)
        
        # Flag categorical in interactions
        self._flag_categorical_interactions()
        
        # Get SHAPs by features
        self.twoway_shaps_dt = self._get_shaps_by_features(X=X, shaps_df=shaps_df)
        return self
    
    def plot_feature_importances_by_tree(
            self, 
            feature_sr: str):
        '''
        Plots feature importance by tree for given feature

        Parameters
        ----------
        feature_sr : str
            Feature.

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        plot_ss = self.feature_importances_by_tree_df[feature_sr]
        data_ss = plot_ss.describe().round(decimals=3)
        ax = plot_ss.plot(marker='.', drawstyle='steps-mid')
        pd.plotting.table(ax=ax, data=data_ss, bbox=[1.25, 0, 0.25, 1])
        fig = ax.figure
        return fig
    
    def plot_feature_importances_by_border(
            self, 
            feature_sr: str):
        '''
        Plots feature importance by border for given float feature

        Parameters
        ----------
        feature_sr : str
            Feature.

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        grouped_df = self.feature_importances_by_feature_and_border_df.loc[feature_sr, :]
        ax = self._plot_group_means(grouped_df=grouped_df, sort_bl=False)
        ax.set(title=feature_sr)
        fig = ax.figure
        return fig
    
    def plot_oneway_shaps(
            self, 
            feature_sr: str, 
            sort_bl: bool):
        '''
        Plots mean SHAP values by border for given feature

        Parameters
        ----------
        feature_sr : str
            Feature.
        sort_bl : bool
            Flag for whether to sort borders (e.g., for categorical features).

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        grouped_df = self.oneway_shaps_dt.get(feature_sr)
        ax = self._plot_group_means(grouped_df=grouped_df, sort_bl=sort_bl)
        ax.set(title=feature_sr)
        fig = ax.figure
        return fig
    
    def plot_twoway_shaps(
            self, 
            features_te: tuple, 
            heatmap_dt: dict = dict()):
        '''
        Plots mean SHAP values for given pair of features with high interaction strengths

        Parameters
        ----------
        features_te : tuple
            Pair of features.
        heatmap_dt : dict, optional
            Optional arguments passed to sns.heatmap(). The default is dict().

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        crosstab_df = self.twoway_shaps_dt.get(features_te)
        fig, ax = plt.subplots()
        sns.heatmap(data=crosstab_df, cmap=plt.cm.coolwarm, ax=ax, **heatmap_dt)
        ax.grid(visible=False)
        return fig
    
    def plot_partial_dependence(
            self, 
            X: pd.DataFrame, 
            features_lt: list, 
            from_estimator_dt: dict = dict()):
        '''
        Plots partial dependence for one or pair of features

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        features_lt : list
            List of feature(s) to plot.
        from_estimator_dt : dict, optional
            Optional arguments passed to snin.PartialDependenceDisplay.from_estimator(). The default is dict().

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        pdd = snin.PartialDependenceDisplay.from_estimator(
            estimator=cb.to_classifier(model=self.ccb.cbm) if self.ccb.model_type_sr == 'classification' \
                else cb.to_regressor(model=self.ccb.cbm), 
            X=X[self.features_lt], 
            features=features_lt,
            contour_kw=dict(cmap=plt.cm.coolwarm),
            **from_estimator_dt)
        fig = pdd.figure_
        return fig
    
    def _get_splits(self):
        '''
        Gets splits for each tree from model dict

        Returns
        -------
        splits_df : pd.DataFrame
            Splits.

        '''
        depth_it = self.ccb.cbm.get_all_params().get('depth')
        splits_df = pd.concat(objs=[
            pd.DataFrame(data=self.model_dt['oblivious_trees'][tree_index_it]['splits'])
            .assign(
                tree_index = tree_index_it,
                depth_index = lambda x: depth_it - x.index.to_numpy() - 1)
            .sort_values(by='depth_index')
            for tree_index_it in range(len(self.model_dt['oblivious_trees']))
        ], ignore_index=True)
        if 'float_feature_index' in splits_df.columns:
            splits_df = (
                splits_df
                .astype(dtype={'float_feature_index': pd.Int32Dtype()})
                .merge(
                    right=self.float_features_df[['feature_id', 'feature_index']], 
                    how='left', 
                    left_on='float_feature_index', 
                    right_on='feature_index', 
                    validate='many_to_one')
                .drop(columns='feature_index'))
        return splits_df
    
    def _get_feature_importances_by_tree(
            self, 
            X: pd.DataFrame, 
            y: pd.Series):
        '''
        Gets feature importances by tree

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
        def _get_feature_importances(
                cbm: cb.CatBoost, 
                X: pd.DataFrame, 
                y: pd.Series, 
                cat_features_lt: list):
            '''
            Gets feature importances

            Parameters
            ----------
            cbm : cb.CatBoost
                Model.
            X : pd.DataFrame
                Feature matrix.
            y : pd.Series
                Target vector.
            cat_features_lt : list
                Categorical features.

            Returns
            -------
            feature_importances_ss : pd.Series
                feature importances.

            '''
            feature_importances_ay = cbm.get_feature_importance(
                data=cb.Pool(data=X, label=y, cat_features=cat_features_lt), 
                type='LossFunctionChange')
            feature_importances_ss = pd.Series(data=feature_importances_ay, index=X.columns, name='importances')
            return feature_importances_ss
        feature_importances_lt = []
        desc_sr = 'Get feature importances by tree'
        for tree_index_it in tqdm.tqdm(iterable=range(self.ccb.cbm.tree_count_), desc=desc_sr):
            cbm2 = self.ccb.cbm.copy()
            cbm2.shrink(ntree_start=tree_index_it, ntree_end=tree_index_it + 1)
            feature_importances_ss = _get_feature_importances(cbm=cbm2, X=X, y=y, cat_features_lt=self.cat_features_lt)
            feature_importances_lt.append(feature_importances_ss)
        feature_importances_by_tree_df = pd.concat(objs=feature_importances_lt, axis=1, ignore_index=True).T
        return feature_importances_by_tree_df
    
    def _join_importances_to_splits(self):
        '''
        Joins importances to splits

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        desc_sr = 'Join importances to splits'
        for float_feature_sr in tqdm.tqdm(iterable=self.float_features_df['feature_id'], desc=desc_sr):
            float_feature_importances_ss = self.feature_importances_by_tree_df[float_feature_sr]
            float_feature_importances_df = (
                float_feature_importances_ss
                .reset_index()
                .set_axis(labels=['tree_index', 'importances'], axis=1)
                .assign(feature_id = float_feature_sr))
            self.splits_df = self.splits_df.merge(
                right=float_feature_importances_df, 
                how='left', 
                on=['tree_index', 'feature_id'], 
                validate='many_to_one')
            if 'importances_x' in self.splits_df.columns:
                self.splits_df = (
                    self.splits_df
                    .assign(importances = lambda x: x.filter(like='importances').sum(axis=1))
                    .drop(columns=['importances_x', 'importances_y']))
        return self
    
    def _get_feature_importances_by_feature_tree_and_border(self):
        '''
        Gets feature importances by feature, tree, and border

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        def _apply_func(in_df: pd.DataFrame):
            '''
            Prepares data frame for each group

            Parameters
            ----------
            in_df : pd.DataFrame
                Input data frame.

            Returns
            -------
            out_df : pd.DataFrame
                Output data frame.

            '''
            out_df = (
                in_df[['tree_index', 'border', 'importances']]
                .assign(bad_flag = lambda x: x['importances'] < 0))
            return out_df
        feature_importances_by_feature_tree_and_border_df = (
            self.splits_df
            .groupby(by='feature_id', group_keys=True)
            .apply(func=_apply_func))
        return feature_importances_by_feature_tree_and_border_df
    
    def _get_feature_importances_by_feature_and_border(self):
        '''
        Gets feature importances by feature and border

        Returns
        -------
        feature_importances_by_feature_and_border_df : pd.DataFrame
            Feature importances.

        '''
        feature_importances_by_feature_and_border_df = (
            steda.get_group_means(
                df=self.feature_importances_by_feature_tree_and_border_df.reset_index(), 
                groupby_by_lt=['feature_id', 'border'], 
                column_sr='importances')
            .assign(bad_mean_flag = lambda x: x['mean'] < 0))
        return feature_importances_by_feature_and_border_df
    
    def _get_shaps(
            self, 
            X: pd.DataFrame, 
            y: pd.Series):
        '''
        Gets SHAP values

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.

        Returns
        -------
        shaps_df : pd.DataFrame
            SHAP values.

        '''
        shaps_df = pd.DataFrame(
            data=self.ccb.cbm.get_feature_importance(
                data=cb.Pool(data=X, label=y, cat_features=self.cat_features_lt), 
                type='ShapValues', 
                shap_calc_type='Exact'), 
            index=X.index, 
            columns=X.columns.tolist() + ['bias'])
        return shaps_df
    
    def _cut_feature(
            self, 
            ss: pd.Series):
        '''
        Cuts feature (by borders if float)

        Parameters
        ----------
        ss : pd.Series
            Feature.

        Returns
        -------
        cut_ss : pd.Series
            Cut feature.

        '''
        feature_sr = ss.name
        if feature_sr in self.cat_features_lt:
            cut_ss = ss
        else:
            bins_lt = (
                [-np.inf] + 
                self.float_features_df.query(expr=f'feature_id == "{feature_sr}"')['borders'].squeeze() + 
                [np.inf])
            cut_ss = pd.cut(x=ss, bins=bins_lt)
        return cut_ss
    
    def _get_shaps_by_feature_and_border(
            self, 
            X: pd.DataFrame, 
            shaps_df: pd.DataFrame):
        '''
        Gets SHAP values by feature and border

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        shaps_df : pd.DataFrame
            SHAP values.

        Returns
        -------
        oneway_shaps_dt : dict
            SHAP values by borders.

        '''
        oneway_shaps_dt = {}
        desc_sr = 'Get SHAPs by feature and border'
        for feature_sr in tqdm.tqdm(iterable=X.columns, desc=desc_sr):
            df = pd.concat(objs=[
                self._cut_feature(ss=X[feature_sr]), 
                shaps_df[feature_sr].rename(index='shaps')
            ], axis=1)
            oneway_shaps_dt[feature_sr] = steda.get_group_means(
                df=df, 
                groupby_by_lt=[feature_sr], 
                column_sr='shaps')
        return oneway_shaps_dt
    
    def _flag_categorical_interactions(self):
        '''
        Flags categorical features in interactions

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        self.ccb.interaction_strengths_df = self.ccb.interaction_strengths_df.assign(
            first_categorical_flag = lambda x: x['first_features'].isin(values=self.cat_features_lt),
            second_categorical_flag = lambda x: x['second_features'].isin(values=self.cat_features_lt),
            neither_categorical_flag = lambda x: ~x.filter(like='flag').any(axis=1))
        return self
    
    def _get_shaps_by_features(
            self, 
            X: pd.DataFrame, 
            shaps_df: pd.DataFrame):
        '''
        Gets SHAP values by pair of features

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        shaps_df : pd.DataFrame
            SHAP values.

        Returns
        -------
        twoway_shaps_dt : TYPE
            SHAP values by pair of borders.

        '''
        twoway_shaps_dt = {}
        interactions_ra = (
            self.ccb.interaction_strengths_df
            .nlargest(n=self.strengths_nlargest_n_it, columns='strengths')
            .loc[:, ['first_features', 'second_features']]
            .to_records(index=False))
        desc_sr = 'Get SHAPs by features'
        for features_rd in tqdm.tqdm(iterable=interactions_ra, desc=desc_sr):
            twoway_shaps_dt[tuple(features_rd)] = pd.crosstab(
                index=self._cut_feature(ss=X[features_rd[0]]), 
                columns=self._cut_feature(ss=X[features_rd[1]]), 
                values=shaps_df[list(features_rd)].sum(axis=1), 
                aggfunc='mean')
        return twoway_shaps_dt
    
    def _plot_group_means(
            self, 
            grouped_df: pd.DataFrame, 
            sort_bl: bool):
        '''
        Plots group means

        Parameters
        ----------
        grouped_df : pd.DataFrame
            Grouped data frame.
        sort_bl : bool
            Flag for whether to sort.

        Returns
        -------
        ax : plt.Axes
            Axis.

        '''
        if sort_bl:
            grouped_df.sort_values(by='mean', inplace=True)
        ax = grouped_df.plot(y='mean', yerr='se2')
        ax.axhline(y=0, c='k', ls=':')
        ax.tick_params(axis='x', labelrotation=90)
        return ax

# =============================================================================
# FeatureSelector
# =============================================================================

class FeatureSelector:
    def __init__(
            self,
            model_type_sr: str,
            cat_boost_dt: dict, 
            sklearn_splitter, 
            objective_sr: str = 'minimize', 
            strategy_sr: str = 'drop_negative_means', 
            wait_it: int = 10, 
            store_models_bl: bool = False,
            fit_only_wait_bl: bool = False,
            losses_nsmallest_n_it: int = 1):
        '''
        Selects features by iteratively removing those with highest validation losses

        Parameters
        ----------
        model_type_sr : str
            Model type.
        cat_boost_dt : dict
            Parameters passed to CatBoost.
        sklearn_splitter : TYPE
            Splitter from scikit-learn.
        objective_sr : str, optional
            Objective for eval metric. The default is 'minimize'.
        strategy_sr : str, optional
            Strategy for dropping features. The default is 'drop_negative_means'.
        wait_it : int, optional
            Number of iterations to wait before terminating procedure. The default is 10.
        store_models_bl : bool, optional
            Flag for whether to store during procedure to save memory. The default is False.
        fit_only_wait_bl : bool, optional
            Flag for whether to fit only as many iterations as wait. The default is False.
        losses_nsmallest_n_it : int, optional
            Number of features to drop. The default is 1.

        Raises
        ------
        KeyError
            Required key "['train_dir']" is not in cat_boost_dt.
        ValueError
            Permitted values of objective_sr are ['minimize', 'maximize']
        NotImplementedError
            Implemented values of strategy_sr are ['drop_nonpositive_means', 'drop_negative_means', 'drop_negative_ucis', 'drop_nsmallest_means']

        Returns
        -------
        None.

        '''
        self.model_type_sr = model_type_sr
        required_keys_lt = ['train_dir']
        for required_key_sr in required_keys_lt:
            if required_key_sr not in cat_boost_dt.keys():
                raise KeyError(f'Required key "{required_key_sr}" is not in cat_boost_dt')
        self.cat_boost_dt = get_parameters(model_type_sr=model_type_sr, cat_boost_dt=cat_boost_dt)
        self.sklearn_splitter = sklearn_splitter
        permitted_objectives_lt = ['minimize', 'maximize']
        if objective_sr not in permitted_objectives_lt:
            raise ValueError(f'Permitted values of objective_sr are {permitted_objectives_lt}')
        self.objective_sr = objective_sr
        self.best_score_ft = np.inf if objective_sr == 'minimize' else -np.inf
        self.best_iteration_it = 0
        implemented_strategies_lt = ['drop_nonpositive_means', 'drop_negative_means', 'drop_negative_ucis', 'drop_nsmallest_means']
        if strategy_sr not in implemented_strategies_lt:
            raise NotImplementedError(f'Implemented values of strategy_sr are {implemented_strategies_lt}')
        self.strategy_sr = strategy_sr
        self.wait_it = wait_it
        self.store_models_bl = store_models_bl
        self.fit_only_wait_bl = fit_only_wait_bl
        self.losses_nsmallest_n_it = losses_nsmallest_n_it
        
    def fit(
            self, 
            X: pd.DataFrame, 
            y: pd.Series,
            split_dt: dict = dict(),
            fit_dt: dict = dict(), 
            sample_dt: dict = None):
        '''
        Fits models, stores metadata, and drops features

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        split_dt : dict, optional
            Split params (e.g., groups). The default is dict().
        fit_dt : dict, optional
            Fit params. The default is dict().
        sample_dt : dict, optional
            Sample params. The default is None.

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
        self.results_lt = []
        
        # Loop
        while True:
            # Log
            print('=' * 80)
            print(f'Iteration: {iteration_it}')
            
            # Update parameters
            output_subdirectory_sr = '{}/{:03d}'.format(output_directory_sr, iteration_it)
            self._update_params(X=X, train_dir_sr=output_subdirectory_sr)
            
            # Fit model
            ccbcv = CustomCatBoostCV(model_type_sr=self.model_type_sr, cat_boost_dt=self.cat_boost_dt, sklearn_splitter=self.sklearn_splitter)
            ccbcv.fit(X=X, y=y, split_dt=split_dt, fit_dt=fit_dt, sample_dt=sample_dt)
            if self.store_models_bl:
                self.models_lt.append(ccbcv)
            
            # Get score and update bests
            score_ft = ccbcv.eval_metrics_df.loc[self.cat_boost_dt['eval_metric'], 'validation_mean']
            self._update_best_score_and_iteration(score_ft=score_ft, iteration_it=iteration_it)
            
            # Get features to drop and keep
            features_ix, drop_ix, keep_ix = self._get_features(X=X, ccbcv=ccbcv)
            
            # Get and print result
            pct_diff_ft = ccbcv.eval_metrics_df.loc[self.cat_boost_dt['eval_metric'], 'pct_diff']
            result_dt = self._get_result(iteration_it=iteration_it, score_ft=score_ft, pct_diff_ft=pct_diff_ft, features_ix=features_ix, drop_ix=drop_ix, keep_ix=keep_ix)
            self.results_lt.append(result_dt)
            self._print_result(result_dt=result_dt)
            
            # Get results
            self.results_df = self._get_results()
            
            # Get ranks
            self.ranks_df = self._get_ranks()
            
            # Checkpoint
            pd.to_pickle(obj=self, filepath_or_buffer=os.path.join(output_directory_sr, 'fs.pkl'), protocol=4)
            
            # Evaluate whether to continue
            if ((iteration_it - self.best_iteration_it == self.wait_it) or 
            (drop_ix.empty) or
            (keep_ix.difference(other=ccbcv.cat_boost_dt['ignored_features']).empty) or 
            ((self.fit_only_wait_bl) and (iteration_it == self.wait_it - 1))):
                break
            else:
                X.drop(columns=drop_ix, inplace=True)
                iteration_it += 1
        return self

    def weight_ranks(
            self, 
            weights_dt: dict = {'scores': 1, 'pct_diffs': 0, 'cnt_features': 0}):
        '''
        Creates new combined rank by weighting components

        Parameters
        ----------
        weights_dt : dict, optional
            Weights. The default is {'scores': 1, 'pct_diffs': 0, 'cnt_features': 0}.

        Raises
        ------
        KeyError
            Required key "['scores', 'pct_diffs', 'cnt_features']" is not in weights_dt.

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
        Gets features from best iteration

        Returns
        -------
        best_features_ix : pd.Index
            Best features.

        '''
        best_iteration_it = self._get_best_iteration()
        best_features_ix = self.results_df.loc[best_iteration_it, 'features']
        return best_features_ix
    
    def get_best_model(self):
        '''
        Gets model from best iteration

        Returns
        -------
        best_ccbcv : CustomCatBoostCV
            Best cross-validated model.

        '''
        best_iteration_it = self._get_best_iteration()
        best_ccbcv = self.models_lt[best_iteration_it]
        return best_ccbcv
    
    def plot_results(self):
        '''
        Plots results

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        columns_lt = ['scores', 'pct_diffs', 'cnt_features']
        fig, axes = plt.subplots(nrows=len(columns_lt), sharex=True, figsize=(10, 10))
        self.results_df[columns_lt].plot(marker='.', subplots=True, ax=axes)
        for index_it, column_sr in enumerate(iterable=columns_lt):
            data_ss = self.results_df[column_sr].describe().round(decimals=3)
            pd.plotting.table(ax=axes[index_it], data=data_ss, bbox=[1.25, 0, 0.25, 1])
        fig.tight_layout()
        return fig
    
    def plot_ranks(self):
        '''
        Plots ranks

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
        Deletes predictions and targets

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
        Updates parameters

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        train_dir_sr : str
            Train directory.

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
        Updates best score and iteration

        Parameters
        ----------
        score_ft : float
            Current score.
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
        Gets features

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        ccbcv : CustomCatBoostCV
            Cross-validated model.

        Returns
        -------
        features_ix : pd.Index
            Features.
        drop_ix : pd.Index
            Features to drop.
        keep_ix : pd.Index
            Feature to keep.

        '''
        features_ix = X.columns
        if self.strategy_sr == 'drop_nonpositive_means':
            drop_ix = (
                ccbcv.feature_importances_df
                .drop(index=ccbcv.cat_boost_dt['ignored_features'])
                .query(expr='validation_mean <= 0').index)
        elif self.strategy_sr == 'drop_negative_means':
            drop_ix = ccbcv.feature_importances_df.query(expr='validation_mean < 0').index
        elif self.strategy_sr == 'drop_negative_ucis':
            drop_ix = ccbcv.feature_importances_df.query(expr='validation_uci < 0').index
        elif self.strategy_sr == 'drop_nsmallest_means':
            drop_ix = (
                ccbcv.feature_importances_df
                .drop(index=ccbcv.cat_boost_dt['ignored_features'])
                .loc[:, 'validation_mean']
                .nsmallest(n=self.losses_nsmallest_n_it).index)
        keep_ix = features_ix.difference(other=drop_ix)
        return features_ix, drop_ix, keep_ix
    
    def _get_result(
            self, 
            iteration_it: int, 
            score_ft: float, 
            pct_diff_ft: float,
            features_ix: pd.Index, 
            drop_ix: pd.Index, 
            keep_ix: pd.Index):
        '''
        Gets result

        Parameters
        ----------
        iteration_it : int
            Current iteration.
        score_ft : float
            Current score.
        pct_diff_ft : float
            Current percent difference.
        features_ix : pd.Index
            Features.
        drop_ix : pd.Index
            Features to drop.
        keep_ix : pd.Index
            Features to keep.

        Returns
        -------
        result_dt : dict
            Result.

        '''
        result_dt = {
            'iterations': iteration_it,
            'scores': score_ft,
            'pct_diffs': pct_diff_ft,
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
        Prints result

        Parameters
        ----------
        result_dt : dict
            Result.

        Returns
        -------
        None.

        '''
        print('Result:')
        for key_sr in result_dt.keys():
            if key_sr != 'features':
                print('- {}: {}'.format(key_sr, result_dt[key_sr]))
    
    def _get_results(self):
        '''
        Gets results

        Returns
        -------
        results_df : pd.DataFrame
            Results.

        '''
        results_df = pd.DataFrame(data=self.results_lt).set_index(keys='iterations')
        return results_df
    
    def _get_ranks(self):
        '''
        Gets ranks

        Returns
        -------
        ranks_df : pd.DataFrame
            Ranks.

        '''
        ranks_df = (
            self.results_df[['scores', 'pct_diffs', 'cnt_features']]
            .assign(
                scores = lambda x: x['scores'].rank(pct=True, ascending=False if self.objective_sr == 'maximize' else True),
                pct_diffs = lambda x: x['pct_diffs'].rank(pct=True),
                cnt_features = lambda x: x['cnt_features'].rank(pct=True),
                combined = lambda x: x.sum(axis=1))
            .rename(columns=lambda x: f'{x}_rank'))
        return ranks_df
    
    def _get_best_iteration(self):
        '''
        Gets best iteration

        Returns
        -------
        best_iteration_it : int
            Best iteration.

        '''
        best_iteration_it = self.ranks_df.iloc[:, -1].idxmin()
        return best_iteration_it
    
# =============================================================================
# get_parameters
# =============================================================================

def get_parameters(
        model_type_sr: str,
        cat_boost_dt: dict):
    '''
    Gets parameters (including defaults if not supplied)

    Parameters
    ----------
    model_type_sr : str
        Model type.
    cat_boost_dt : dict
        User-supplied parameters.

    Raises
    ------
    NotImplementedError
        Implemented values of model_type_sr are ['classification', 'regression']

    Returns
    -------
    general_defaults_dt : dict
        Parameters passed to CatBoost.

    '''
    general_defaults_dt = {
        'cat_features': [],
        'early_stopping_rounds': 10,
        'ignored_features': [],
        'iterations': 100,
        'monotone_constraints': {},
        'random_seed': 0,
        'task_type': 'CPU', 
        'use_best_model': True}
    implemented_model_types_lt = ['classification', 'regression']
    if model_type_sr == 'classification':
        model_defaults_dt = {
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'custom_metric': ['Logloss', 'PRAUC', 'Precision', 'Recall']}
        general_defaults_dt.update(model_defaults_dt)
    elif model_type_sr == 'regression':
        model_defaults_dt = {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'custom_metric': ['RMSE', 'R2', 'MAE', 'MAPE']}
        general_defaults_dt.update(model_defaults_dt)
    else:
        raise NotImplementedError(f'Implemented values of model_type_sr are {implemented_model_types_lt}')
    general_defaults_dt.update(cat_boost_dt)
    general_defaults_dt['verbose'] = general_defaults_dt['iterations'] // 10
    return general_defaults_dt
