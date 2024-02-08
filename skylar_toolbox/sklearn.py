# =============================================================================
# Load libraries
# =============================================================================

import numpy as np
import pandas as pd
import warnings
from sklearn import tree as snte

# =============================================================================
# RandomForestInspector
# =============================================================================

class RandomForestInspector:
    def __init__(
            self, 
            model,
            scorer):
        '''
        Inspects random forest

        Parameters
        ----------
        model : TYPE
            Random forest.
        scorer : callable
            Scoring function.

        Returns
        -------
        None.

        '''
        self.model = model
        self.scorer = scorer

    def fit(
            self, 
            X: pd.DataFrame, 
            y: pd.Series):
        '''
        Gets predictions from component estimators, aggregates them, and scores ensemble by number of estimators

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
        # Get predictions
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore')
            y_preds = self._get_predictions(X=X)

        # Aggregate them
        self.y_preds_agg = self._aggregate_predictions(y_preds=y_preds)

        # Get scores
        self.scores_ss = self._get_scores(y_true=y, y_preds=y_preds)
        return self        

    def plot_predictions(self):
        '''
        Plots prediction mean and standard error

        Returns
        -------
        ax : plt.Axes
            Axis.

        '''
        ax = (
            self.y_preds_agg
            .sort_values(by='mean')
            .reset_index()
            .plot(y='mean', yerr='sem'))
        return ax

    def plot_scores(self):
        '''
        Plots ensemble scores by number of estimators

        Returns
        -------
        ax : plt.Axes
            Axis.

        '''
        ax = self.scores_ss.plot(marker='.')
        return ax
    
    def delete_predictions(self):
        '''
        Deletes predictions

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        self.__delattr__('y_preds_agg')
        return self
    
    def _get_predictions(
            self, 
            X: pd.DataFrame):
        '''
        Gets predictions for all estimators in ensemble

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        y_preds : pd.DataFrame
            Predictions of shape (n_examples, n_estimators).

        '''
        def _get_prediction(
                estimator, 
                X: pd.DataFrame = X):
            '''
            Gets predictions for estimator in ensemble

            Parameters
            ----------
            estimator : TYPE
                Decision tree.
            X : pd.DataFrame, optional
                Feature matrix. The default is X.

            Returns
            -------
            np.ndarray
                Predictions.

            '''
            if isinstance(estimator, snte.DecisionTreeClassifier):
                return estimator.predict_proba(X=X)[:, 1]
            return estimator.predict(X=X)
        y_preds = pd.DataFrame(
            data=np.column_stack(tup=tuple(map(_get_prediction, self.model.estimators_))), 
            index=X.index)
        return y_preds

    def _aggregate_predictions(
            self, 
            y_preds: pd.DataFrame, 
            n_estimators_it: int = None):
        '''
        Aggregates predictions across estimators

        Parameters
        ----------
        y_preds : pd.DataFrame
            Predictions.
        n_estimators_it : int, optional
            Number of estimators to aggregate across. The default is None.

        Returns
        -------
        y_preds_agg : pd.DataFrame
            Aggregated predictions.

        '''
        y_preds_agg = (
            y_preds
            .iloc[:, :n_estimators_it]
            .agg(func=['count', 'mean', 'sem'], axis=1))
        return y_preds_agg

    def _get_scores(
            self, 
            y_true: pd.Series, 
            y_preds: pd.DataFrame):
        '''
        Gets scores of ensemble (subsets)

        Parameters
        ----------
        y_true : pd.Series
            Target vector.
        y_preds : pd.DataFrame
            Predictions.

        Returns
        -------
        scores_ss : pd.Series
            Scores.

        '''
        def _get_score(
                n_estimators_it: int, 
                y_true: pd.Series = y_true, 
                y_preds: pd.DataFrame = y_preds):
            '''
            Gets score of ensemble (subset)

            Parameters
            ----------
            n_estimators_it : int
                Number of estimators to score.
            y_true : pd.Series, optional
                Target vector. The default is y_true.
            y_preds : pd.DataFrame, optional
                Predictions. The default is y_preds.

            Returns
            -------
            score_ft : float
                Score.

            '''
            y_preds_agg = self._aggregate_predictions(
                y_preds=y_preds,
                n_estimators_it=n_estimators_it)
            score_ft = self.scorer(y_true, y_preds_agg['mean'])
            return score_ft
        scores_ss = pd.Series(
            data=map(_get_score, range(1, self.model.n_estimators + 1)), 
            name='scores')
        return scores_ss
