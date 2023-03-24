# =============================================================================
# Load libraries
# =============================================================================

import numpy as np
import pandas as pd
from feature_engine import datetime as fedt
from feature_engine import encoding as feeg
from feature_engine import outliers as feos
from sklearn import linear_model as snlm
from sklearn import model_selection as snmos
from sklearn import pipeline as snpe
from skylar_toolbox import model_selection as stms

# =============================================================================
# DatetimeFeaturesTuner
# =============================================================================

class DatetimeFeaturesTuner:
    def __init__(
        self, 
        model_type_sr: str, 
        cv,
        date_part_type_sr: str,
        model = None):
        '''
        Tunes `features_to_extract` for fedt.DatetimeFeatures()

        Parameters
        ----------
        model_type_sr : str
            Model type.
        cv : TYPE
            Method of cross-validating.
        date_part_type_sr : str
            Date part type.
        model : TYPE, optional
            Model. The default is None.

        Raises
        ------
        NotImplementedError
            Implemented values of model_type_sr are ['classification', 'regression']
            Implemented values of date_part_type_sr are ['sub-year', 'sub-week']

        Returns
        -------
        None.

        '''
        implemented_model_types_lt = ['classification', 'regression']
        if model_type_sr not in implemented_model_types_lt:
            raise NotImplementedError(f'Implemented values of model_type_sr are {implemented_model_types_lt}')
        self.model_type_sr = model_type_sr
        self.cv = cv
        implemented_date_part_types_lt = ['sub-year', 'sub-week']
        if date_part_type_sr not in implemented_date_part_types_lt:
            raise NotImplementedError(f'Implemented values of date_part_type_sr are {implemented_date_part_types_lt}')
        self.date_part_type_sr = date_part_type_sr
        if model is None:
            self.model = snlm.LogisticRegression(penalty=None) if self.model_type_sr == 'classification' \
                else snlm.LinearRegression()
        else:
            self.model = model
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series):
        '''
        Tunes and stores metadata

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
        # Initialize pipeline
        pe = snpe.Pipeline(steps=[
            ('extract', fedt.DatetimeFeatures()),
            ('encode', feeg.OneHotEncoder(ignore_format=True)),
            ('model', self.model)])
        
        # Initialize tuner
        if self.date_part_type_sr == 'sub-year':
            features_to_extract_lt = [['semester'], ['quarter'], ['month'], ['week']]
        elif self.date_part_type_sr == 'sub-week':
            features_to_extract_lt = [['weekend'], ['day_of_week']]
        self.gscv = snmos.GridSearchCV(
            estimator=pe, 
            param_grid=dict(extract__features_to_extract=features_to_extract_lt),
            scoring='roc_auc' if self.model_type_sr == 'classification' else 'r2',
            cv=self.cv,
            refit=True, 
            verbose=3, 
            return_train_score=True)
    
        # Fit it
        self.gscv.fit(X=X, y=y)
        
        # Get CV results
        self.cv_results_df = pd.DataFrame(data=self.gscv.cv_results_)
        
        # Get params
        self.params_lt = list(self.gscv.param_grid.keys())
        self.columns_lt = [f'param_{param_sr}' for param_sr in self.params_lt]
        
        # Get plot data
        self.plot_df = self._get_plot_data()
        return self
    
    def plot_scores_v_param(self):
        '''
        Plots train and test scores against parameter values

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        ax = self.plot_df.plot(y='mean_train_score', yerr='se2_train_score')
        self.plot_df.plot(y='mean_test_score', yerr='se2_test_score', ax=ax)
        fig = ax.figure
        return fig
    
    def _get_plot_data(self):
        '''
        Gets plot data

        Returns
        -------
        plot_df : pd.DataFrame
            Plot data.

        '''
        plot_df = (
            self.cv_results_df
            .set_index(keys=self.columns_lt)
            .loc[:, ['mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score']]
            .assign(
                se2_train_score = lambda x: 2 * x['std_train_score'] / np.sqrt(self.gscv.n_splits_),
                se2_test_score = lambda x: 2 * x['std_test_score'] / np.sqrt(self.gscv.n_splits_)))
        return plot_df
    

# =============================================================================
# RareLabelEncoderTuner
# =============================================================================

class RareLabelEncoderTuner:
    def __init__(
        self, 
        model_type_sr: str, 
        cv,
        model = None,
        min_frequencies_lt: list = [0, 0.01, 0.05, 0.1]):
        '''
        Tunes `tol` for feeg.RareLabelEncoder()

        Parameters
        ----------
        model_type_sr : str
            Model type.
        cv : TYPE
            Method of cross-validating.
        model : TYPE, optional
            Model. The default is None.
        min_frequencies_lt : list, optional
            Minimum frequencies required not to be encoded. The default is [0, 0.01, 0.05, 0.1].

        Raises
        ------
        NotImplementedError
            Implemented values of model_type_sr are  ['classification', 'regression']

        Returns
        -------
        None.

        '''
        implemented_model_types_lt = ['classification', 'regression']
        if model_type_sr not in implemented_model_types_lt:
            raise NotImplementedError(f'Implemented values of model_type_sr are {implemented_model_types_lt}')
        self.model_type_sr = model_type_sr
        self.cv = cv
        if model is None:
            self.model = snlm.LogisticRegression(penalty=None) if self.model_type_sr == 'classification' \
                else snlm.LinearRegression()
        else:
            self.model = model
        self.min_frequencies_lt = min_frequencies_lt
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series):
        '''
        Tunes and stores metadata

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
        # Initialize pipeline
        pe = snpe.Pipeline(steps=[
            ('rare_label_encode', feeg.RareLabelEncoder(n_categories=1, ignore_format=True)),
            ('one_hot_encode', feeg.OneHotEncoder(ignore_format=True)),
            ('model', self.model)])
        
        # Initialize tuner
        self.cgscv = stms.CustomGridSearchCV(
            estimator=pe, 
            param_grid_dt=dict(rare_label_encode__tol=self.min_frequencies_lt),
            scoring='roc_auc' if self.model_type_sr == 'classification' else 'r2',
            cv=self.cv)
    
        # Fit it
        self.cgscv.fit(X=X, y=y)
        return self
    
    def plot_scores_v_param(self):
        '''
        Plots train and test scores against parameter values

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        fig = self.cgscv.plot_scores_v_param(column_sr=self.cgscv.columns_lt[0])
        return fig
    
# =============================================================================
# WinsorizerTuner
# =============================================================================

class WinsorizerTuner:
    def __init__(
        self, 
        model_type_sr: str, 
        cv,
        tail_sr: str,
        model = None,
        quantiles_lt: list = [1e-5, 0.01, 0.05, 0.1]):
        '''
        Tunes `fold` for feos.Winsorizer()

        Parameters
        ----------
        model_type_sr : str
            Model type.
        cv : TYPE
            Method of cross-validating.
        tail_sr : str
            Tail to winsorize.
        model : TYPE, optional
            Model. The default is None.
        quantiles_lt : list, optional
            Quantiles. The default is [1e-5, 0.01, 0.05, 0.1].

        Raises
        ------
        NotImplementedError
            Implemented values of model_type_sr are  ['classification', 'regression'].

        Returns
        -------
        None.

        '''
        implemented_model_types_lt = ['classification', 'regression']
        if model_type_sr not in implemented_model_types_lt:
            raise NotImplementedError(f'Implemented values of model_type_sr are {implemented_model_types_lt}')
        self.model_type_sr = model_type_sr
        self.cv = cv
        self.tail_sr = tail_sr
        if model is None:
            self.model = snlm.LogisticRegression(penalty=None) if self.model_type_sr == 'classification' \
                else snlm.LinearRegression()
        else:
            self.model = model
        self.quantiles_lt = quantiles_lt
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series):
        '''
        Tunes and stores metadata

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
        # Initialize pipeline
        pe = snpe.Pipeline(steps=[
            ('winsorize', feos.Winsorizer(capping_method='quantiles', tail=self.tail_sr, missing_values='ignore')),
            ('model', self.model)])
        
        # Initialize tuner
        self.cgscv = stms.CustomGridSearchCV(
            estimator=pe, 
            param_grid_dt=dict(winsorize__fold=self.quantiles_lt),
            scoring='roc_auc' if self.model_type_sr == 'classification' else 'r2',
            cv=self.cv)
    
        # Fit it
        try:
            self.cgscv.fit(X=X, y=y)
        except:
            pass
        return self
    
    def plot_scores_v_param(self):
        '''
        Plots train and test scores against parameter values

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        fig = self.cgscv.plot_scores_v_param(column_sr=self.cgscv.columns_lt[0])
        return fig