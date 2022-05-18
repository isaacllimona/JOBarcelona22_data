import numpy as np
from sklearn.base import DensityMixin

class JointDistribution(DensityMixin):
    """
    Models the joint distribution of various models, 
    assuming conditional independence of the models.
    """
    def __init__(self):
        self.__models = []
        self.__features = []
        self.__trainable = []

    def add_model(self, model, features, train=True):
        """
        Add a model to the joint distribution.
        Args:
            model: An instance of a class representing a probability distribution.

            features: Sequence
                Name of features the model should use for training
                and prediction.

            train: True of False
                Whether to train the model when calling fit(). If set to
                True, the model should implement the fit method.

        Returns:
            self
        """
        self.__models.append(model)
        self.__features.append(features)
        if train:
            self.__trainable.append(model)

        return self

    def fit(self, X, y=None):
        """
        Train all the trainable models on the given data.
        
        Args:
            X: Array-like of shape (n_samples, n_features)
                Data
            y: Ignored, present for API consistency.

        Returns:
            self: The fitted model
        """
        for model, features in zip(self.__trainable, self.__features):
            model.fit(X[features])
        
        return self

    def predict_proba(self, X):
        """
        Predict the probability density of the data under the joint model.
        All models should implement "predict_proba" to use this method.
        
        Args:
            X: array-like of shape (n_samples, n_features).
                Data.

        Returns:
            proba: array of shape (n_samples,).
                The probability density of the data under the joint model.
        """
        proba = np.ones(len(X))
        for model, features in zip(self.__models, self.__features):
            proba *= model.predict_proba(X[features])
        
        return proba

    def predict_log_proba(self, X):
        """
        Predict the log probability density of the observations under the joint model.
        All models should implement either "predict_log_proba" or "score_samples" to use
        this method. The latter should return the log probability density given the data.

        Args:
            X: array-like of shape (n_samples, n_features).
                Data.

        Returns:
            log_proba: array of shape (n_samples,).
                The log probability density of the observations under the joint model.
        """
        log_proba = np.zeros(len(X))
        for model, features in zip(self.__models, self.__features):
            if hasattr(model, "predict_log_proba"):
                model_log_proba = model.predict_log_proba(X[features])
            elif hasattr(model, "score_samples"):
                model_log_proba = model.score_samples(X[features])
            else:
                raise AttributeError(f"Model {model} implements neither \"predict_log_proba\" "
                    "nor \"score_samples\".")
            
            log_proba += model_log_proba
        
        return log_proba