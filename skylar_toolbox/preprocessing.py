# =============================================================================
# Load libraries
# =============================================================================

import pandas as pd
from feature_engine import encoding as feeg
from feature_engine import outliers as feos
from sklearn import linear_model as snlm
from sklearn import pipeline as snpe
from skylar_toolbox import model_selection as stms

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