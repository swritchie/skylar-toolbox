# =============================================================================
# Load libraries
# =============================================================================

import pandas as pd
from sklearn import base as snbe
from sklearn import feature_selection as snfs

# =============================================================================
# DuplicateDropper
# =============================================================================

class DuplicateDropper(snbe.BaseEstimator, snbe.TransformerMixin):
    def __init__(
            self, 
            first_n_rows_it: int):
        '''
        Identifies and drops duplicate features

        Parameters
        ----------
        first_n_rows_it : int
            Number of rows to check (before full check).

        Returns
        -------
        None.

        '''
        self.first_n_rows_it = first_n_rows_it
    
    def fit(
            self, 
            X: pd.DataFrame, 
            y: pd.Series = None):
        '''
        Identifies duplicate features

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series, optional
            Target vector. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        self.duplicates_lt = []
        for index_it, column_sr in enumerate(iterable=X.columns):
            for column_sr2 in X.columns[index_it+1:]:
                if X[column_sr].iloc[:self.first_n_rows_it].equals(other=X[column_sr2].iloc[:self.first_n_rows_it]):
                    if X[column_sr].equals(other=X[column_sr2]):
                        self.duplicates_lt.append(column_sr2)
        return self
    
    def transform(
            self, 
            X: pd.DataFrame):
        '''
        Drops duplicate features

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with duplicates.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix without duplicates.

        '''
        X.drop(columns=self.duplicates_lt, inplace=True)
        return X
    
# =============================================================================
# GenericUnivariateSelector
# =============================================================================

class GenericUnivariateSelector(snfs.GenericUnivariateSelect):
    def transform(self, X):
        return pd.DataFrame(
            data=X.loc[:, self.get_support()],
            index=X.index, 
            columns=self.get_feature_names_out())
    
# =============================================================================
# ModelBasedSelector
# =============================================================================

class ModelBasedSelector(snfs.SelectFromModel):
    def transform(self, X):
        return pd.DataFrame(
            data=X.loc[:, self.get_support()], 
            index=X.index, 
            columns=self.get_feature_names_out())
    
# =============================================================================
# RFE
# =============================================================================
    
class RFE(snfs.RFE):
    def transform(self, X):
        return pd.DataFrame(
            data=X.loc[:, self.get_support()], 
            index=X.index, 
            columns=self.get_feature_names_out())
    
# =============================================================================
# RFECV
# =============================================================================

class RFECV(snfs.RFECV):
    def transform(self, X):
        return pd.DataFrame(
            data=X.loc[:, self.get_support()], 
            index=X.index, 
            columns=self.get_feature_names_out())

# =============================================================================
# SequentialFeatureSelector
# =============================================================================

class SequentialFeatureSelector(snfs.SequentialFeatureSelector):
    def transform(self, X):
        return pd.DataFrame(
            data=X.loc[:, self.get_support()], 
            index=X.index, 
            columns=self.get_feature_names_out())