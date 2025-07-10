"""
Reference: Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent." Proceedings of COMPSTAT'2010, 177-186.
"""
import time
import numpy as np
from .models.mlp import MLPModel
from .models.logistic_regression import LogisticRegressionModel
from .data_loader import DataLoader


def create_model(config):
    """
    Factory function to create models based on type.
    """
    model_type = config.get("model_type", "mlp")
    
    if model_type == "mlp":
        return MLPModel(config)
    elif model_type == "logistic_regression":
        return LogisticRegressionModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class TrainingManager:
    """
    Manages model training and evaluation.
    Closely follows Bottou (2010) for SGD and evaluation.
    """
    def __init__(self, config):
        self.config = config
        self.model = None
        self.data_loader = None
    
    def setup(self):
        """
        Setup model and data loader.
        """
        self.model = create_model(self.config)
        self.data_loader = DataLoader(self.config)
        print("Setup complete")
    
    def train(self, data_path):
        """
        Train the model using SGD and evaluate accuracy (Bottou, 2010).
        """
        if not self.model or not self.data_loader:
            raise ValueError("Must setup first")
        
        print(f"Training on {data_path}")
        start_time = time.time()
        
        X, y = self.data_loader.load_data(data_path)
        X_train, X_test, y_train, y_test = self.data_loader.split_data(X, y)
        
        # Model training using SGD (Bottou, 2010)
        self.model.train(X_train, y_train)
        
        # Model evaluation
        predictions = self.model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        
        training_time = time.time() - start_time
        
        metadata = {
            "model_type": self.config["model_type"],
            "training_time": training_time,
            "accuracy": accuracy,
            "epochs": self.config["epochs"]
        }
        
        print(f"Training completed in {training_time:.2f}s, Accuracy: {accuracy:.4f}")
        return metadata
    
    def get_weights(self):
        """
        Get model weights for upload.
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model not trained")
        
        weights = {}
        for attr_name in dir(self.model):
            if attr_name.startswith('weights') or attr_name.startswith('bias') or attr_name.startswith('W_') or attr_name.startswith('b_') or attr_name.startswith('beta'):
                attr_value = getattr(self.model, attr_name)
                if hasattr(attr_value, 'tolist'):
                    weights[attr_name] = attr_value.tolist()
        
        return weights 