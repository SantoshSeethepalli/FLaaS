"""
Abstract base class for all models.
Reference: Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, 12, 2825-2830.
"""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all models.
    Reference: Pedregosa et al. (2011) for API design inspiration.
    """
    
    def __init__(self, config):
        self.config = config
        self.is_trained = False
    
    @abstractmethod
    def train(self, X, y):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Predict probabilities."""
        pass 