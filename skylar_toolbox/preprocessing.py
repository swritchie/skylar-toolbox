# =============================================================================
# Load libraries
# =============================================================================

import pandas as pd
from feature_engine import encoding as feeg
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
        min_frequencies_lt: list = [0, 0.01, 0.05, 0.1]):
        '''
        Tunes `tol` for feeg.RareLabelEncoder()

        Parameters
        ----------
        model_type_sr : str
            Model type.
        cv : TYPE
            Method of cross-validating.
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
            Features.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        # Initialize pipeline
        lr = snlm.LogisticRegression(penalty=None) if self.model_type_sr == 'classification' \
            else snlm.LinearRegression()
        pe = snpe.Pipeline(steps=[
            ('rare_label_encode', feeg.RareLabelEncoder(n_categories=1, ignore_format=True)),
            ('one_hot_encode', feeg.OneHotEncoder(ignore_format=True)),
            ('model', lr)])
        
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