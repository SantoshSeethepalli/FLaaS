from abc import ABC, abstractmethod


class BaseModel(ABC):
    ## interface implementation for all models.
    
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
    
    @abstractmethod
    def set_weights_from_flat(self, flat_weights):
        """Set model weights from a flat list."""
        pass
    
    @abstractmethod
    def get_flat_weights(self):
        """Get model weights as a flat list."""
        pass 