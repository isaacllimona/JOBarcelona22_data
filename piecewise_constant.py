import numpy as np
from sklearn.base import DensityMixin
from sklearn.utils.validation import column_or_1d

class OuterInnerConstantDistribution(DensityMixin):
    """
    Models a 2-part piecewise constant distribution 
    where an inner interval lies within the outer interval.

    Must be given the bounds of both intervals.

    All values are assumed to lie within the outer bounds.

    Args:
        outer_lbound: scalar
            lower bound of the outer interval
        outer_ubound: scalar
            upper bound of the outer interval
        inner_lbound: scalar
            lower bound of the inner interval
        inner_ubound: scalar
            upper bound of the inner interval
    """
    def __init__(self, outer_lbound, outer_ubound, inner_lbound, inner_ubound):
        self.outer_lbound = outer_lbound
        self.outer_ubound = outer_ubound
        self.inner_lbound = inner_lbound
        self.inner_ubound = inner_ubound
    
    def fit(self, X, y=None):
        """
        Train the model on the given data.
        
        Args:
            X: array-like of shape (n_samples,)
                Data
            y: Ignored, present for API consistency.
        
        Returns:
            self: 
                The trained model
        """
        X = column_or_1d(X)

        in_mask, _ = self.__get_in_out_masks(X)
        total_count = len(X)
        in_count = np.count_nonzero(in_mask)
        out_count = total_count - in_count

        inner_width = self.inner_ubound - self.inner_lbound
        outer_width = self.outer_ubound - self.outer_lbound - inner_width

        self.in_proba = (in_count / total_count) / inner_width
        self.out_proba =  (out_count / total_count) / outer_width

        self.log_in_proba = np.log(self.in_proba)
        self.log_out_proba = np.log(self.out_proba)

        return self

    def predict_proba(self, X):
        """
        Predict the probability density of the given data.
        
        Args:
            X: array-like of shape (n_samples, n_features).
                Data.

        Returns: array of shape (n_samples, n_features)
            proba: The probability density of the given data
        """
        X = column_or_1d(X)

        in_mask, out_mask = self.__get_in_out_masks(X)
        proba = np.empty(len(X))
        proba[in_mask] = self.in_proba
        proba[out_mask] = self.out_proba
        return proba

    def predict_log_proba(self, X):
        """
        Predict the log probability density of the given data.
        
        Args:
            X: array-like of shape (n_samples, n_features).
                Data.

        Returns:
            log_proba: array of shape (n_samples, n_features)
                The log probability density of the given data
        """
        X = column_or_1d(X)

        in_mask, out_mask = self.__get_in_out_masks(X)
        log_proba = np.empty(len(X))
        log_proba[in_mask] = self.log_in_proba
        log_proba[out_mask] = self.log_out_proba
        return log_proba

    def __get_in_out_masks(self, X):
        in_mask = np.logical_and(X >= self.inner_lbound, X < self.inner_ubound)
        out_mask = np.logical_not(in_mask)
        return in_mask, out_mask
