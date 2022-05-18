import numpy as np
from sklearn.base import ClassifierMixin

class BayesClassifier(ClassifierMixin):
    """
    Multi-class classifier based on applying Bayes' rule.
    Uses a different user-defined model for the likelihood of each class.
    
    Args:
            likelihood_models: Sequence of instances of classes implementing the
                likelihood models of each class. They should implement the methods
                fit and either predict_log_proba or score_samples. Both should
                return the log likelihood of the given data.
    """
    def __init__(self, likelihood_models):
        self.__likelihood_models = likelihood_models
        self.__n_classes = len(likelihood_models)

    def fit(self, X, y):
        """
        Fits both the priors and all the likelihood models on the
        given data.

        Args:
            X: array-like of shape (n_samples, n_features)
                Data
            y: array-like of shape (n_samples,) 
                Class label of each training example. The labels should be 
                integers from 0 up to the number of classes.

        Returns:
            self: The fitted model
        """
        self.fit_priors(None, y)

        for class_, model in enumerate(self.__likelihood_models):
            mask = (y == class_)
            X_single_class = X[mask]
            model.fit(X_single_class)

        return self

    def fit_priors(self, X, y):
        """
        Fit the priors to the given data.
        
        Args:
            X: Ignored.
            y: array-like of shape (n_samples,) 
                Class label of each training example. The labels should be 
                integers from 0 up to the number of classes.
        
        Returns:
            self: The fitted model
        """
        priors = np.empty(self.__n_classes)
        total_count = len(y)
        for class_ in range(len(priors)):
            mask = (y == class_)
            class_count = np.count_nonzero(mask)
            class_prior = class_count / total_count
            priors[class_] = class_prior

        self.__priors = priors
        self.__log_priors = np.log(priors)

        return self

    def score_samples(self, X):
        """
        Gives a score to each class for every sample of the data.
        The score is the log of a value proportional to the posterior; thus, a higher score
        means a higher posterior.

        Args:
            X: array-like of shape (n_samples, n_features)
                Data

        Returns:
            score: Array of shape (n_samples, n_classes)
                The log of the posterior of every class under
                the given observations offset by a constant.
        """
        log_priors = self.__log_priors
        log_likelihoods = np.empty((len(X), self.__n_classes))

        for class_, model in enumerate(self.__likelihood_models):
            if hasattr(model, "predict_log_proba"):
                log_likelihood = model.predict_log_proba(X)
            elif hasattr(model, "score_samples"):
                log_likelihood = model.score_samples(X)
            else:
                raise AttributeError(f"Model {model} implements neither \"predict_log_proba\" "
                    "nor \"score_samples\".")
                    
            log_likelihoods[:, class_] = log_likelihood

        score = log_priors + log_likelihoods
        return score

    def predict(self, X):
        """
        Predicts the most likely class (class with the highest posterior)
        under the given observations.

        Args:
            X: array-like of shape (n_samples, n_features)
                Data

        Returns:
            predictions: Array of shape (n_samples,)
                The most likely class under the given observations
        """
        log_probs = self.score_samples(X)
        predictions = np.argmax(log_probs, axis=1)
        return predictions
