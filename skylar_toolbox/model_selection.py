# =============================================================================
# Load libraries
# =============================================================================

import pandas as pd
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from sklearn import model_selection as snmos

# =============================================================================
# CustomGridSearchCV
# =============================================================================

class CustomGridSearchCV:
    def __init__(
        self, 
        estimator, 
        param_grid_dt: dict, 
        scoring, 
        cv):
        '''
        Wraps scikit-learn GridSearchCV to store metadata and provide plotting utilities

        Parameters
        ----------
        estimator : TYPE
            Estimator.
        param_grid_dt : dict
            Parameters.
        scoring : TYPE
            Method of scoring.
        cv : TYPE
            Method of cross-validating.

        Returns
        -------
        None.

        '''
        self.estimator = estimator
        self.param_grid_dt = param_grid_dt
        self.scoring = scoring
        self.cv = cv
    
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
        # Initialize and fit model
        self.gscv = snmos.GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid_dt,
            scoring=self.scoring,
            refit=True, 
            cv=self.cv, 
            verbose=3, 
            return_train_score=True)
        self.gscv.fit(X=X, y=y)
        
        # Get CV results
        self.cv_results_df = pd.DataFrame(data=self.gscv.cv_results_)
        
        # Get params and columns
        self.params_lt = list(self.param_grid_dt.keys())
        self.columns_lt = [f'param_{param_sr}' for param_sr in self.params_lt]
        
        # Get scores in long format
        self.scores_df = self._get_long_scores()
        return self
    
    def plot_scores_v_param(
            self,
            column_sr: str):
        '''
        Plots train and test scores against parameter values

        Parameters
        ----------
        column_sr : str
            Column name of parameter.

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        ax = sns.pointplot(data=self.scores_df, x=column_sr, y='train')
        sns.pointplot(data=self.scores_df, x=column_sr, y='test', ax=ax)
        ax.tick_params(axis='x', labelrotation=90)
        fig = ax.figure
        return fig
    
    def plot_scores_v_params(
            self, 
            column_sr: str, 
            column_sr2: str):
        '''
        Plots heatmap of mean test scores against values of two parameters

        Parameters
        ----------
        column_sr : str
            Column name of first parameter.
        column_sr2 : str
            Column name of second parameter.

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        agg_scores_df = (
            self.scores_df
            .groupby(by=[column_sr, column_sr2])['test']
            .mean()
            .unstack())
        ax = sns.heatmap(data=agg_scores_df, cmap=plt.cm.coolwarm)
        fig = ax.figure
        return fig
    
    def plot_pairs(self):
        '''
        Plots pair grid of train/test scores v. parameters

        Returns
        -------
        fig : plt.Figure
            Figure.

        '''
        pg = sns.pairplot(data=self.scores_df, x_vars=self.columns_lt, y_vars=['train', 'test'])
        fig = pg.figure
        return fig
    
    def _get_long_scores(self):
        '''
        Converts scores from wide to long form for plotting

        Returns
        -------
        scores_df : pd.DataFrame
            Scores.

        '''
        scores_df = (
            self.cv_results_df
            .set_index(keys=self.columns_lt)
            .filter(like='split')
            .stack()
            .to_frame(name='scores')
            .reset_index(names=self.columns_lt + ['columns'])
            .assign(
                splits = lambda x: x['columns'].str.split(pat='_').str.get(i=0),
                score_types = lambda x: x['columns'].str.split(pat='_').str.get(i=1))
            .drop(columns='columns')
            .set_index(keys=self.columns_lt + ['splits', 'score_types'])
            .squeeze()
            .unstack()
            .reset_index())
        return scores_df
    
    