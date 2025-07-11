"""
Main federated learning client class.
"""

import json
import os
from .api import ServerAPI
from .training import TrainingManager
from .config import SERVER_URL, USERNAME, EMAIL, CLIENT_ID


class FederatedClient:
    def __init__(self, server_url=SERVER_URL):
        self.api = ServerAPI(server_url)
        self.training_manager = None
        self.config = None
        self.join_code = None
        self.user = {"username": USERNAME, "email": EMAIL}

    def join_round(self, join_code):
        """Join a federated learning round."""
        print(f"Joining round with code: {join_code}")
        data = self.api.join_round(join_code, self.user)
        self.config = data  # The server returns the contract directly
        self.join_code = join_code
        # Store join code in config for future use
        self.config["join_code"] = self.join_code
        # Store server weights if provided
        if "global_weights" in data:
            self.config["global_weights"] = data["global_weights"]
            print(f"Received global weights from server (round {data.get('round', 'unknown')})")
        # Copy training params to top level if present
        tp = self.config.get("training_params", {})
        for k in ["learning_rate", "epochs", "batch_size"]:
            if k in tp:
                self.config[k] = tp[k]
        # Save contract to file
        contract_path = os.path.join(os.path.dirname(__file__), "contract.json")
        with open(contract_path, "w") as f:
            json.dump(self.config, f, indent=2)
        print("Joined round successfully")
        return self.config
    
    def load_contract(self):
        """Load contract from file if it exists."""
        try:
            contract_path = os.path.join(os.path.dirname(__file__), "contract.json")
            with open(contract_path, "r") as f:
                self.config = json.load(f)
                self.join_code = self.config.get("join_code")
                return self.config
        except FileNotFoundError:
            return None
    
    def setup(self):
        """Setup training components."""
        # Always sync contract from server before setup
        if not self.config or not self.join_code:
            self.load_contract()
        if not self.join_code:
            raise ValueError("Must join round first or have join_code in contract.json")
        # Sync contract from server
        data = self.api.sync_contract(self.join_code)
        if "contract" in data:
            self.config = data["contract"]
        else:
            self.config = data
        # Always persist join_code
        self.config["join_code"] = self.join_code
        # Copy training params to top level if present
        tp = self.config.get("training_params", {})
        for k in ["learning_rate", "epochs", "batch_size"]:
            if k in tp:
                self.config[k] = tp[k]
        # Save latest contract to file (as cache)
        contract_path = os.path.join(os.path.dirname(__file__), "contract.json")
        with open(contract_path, "w") as f:
            json.dump(self.config, f, indent=2)
        self.training_manager = TrainingManager(self.config)
        self.training_manager.setup()
    
    def train(self, data_path):
        """Train the model."""
        # Always ensure config is up to date before training
        if not self.training_manager:
            self.setup()
        if not self.config:
            print("Error: Config not loaded. Please sync first.")
            return None
        if not self.training_manager:
            print("Error: Training manager not initialized.")
            return None
        metadata = self.training_manager.train(data_path)
        if metadata is None:
            return None
        model = self.training_manager.model
        if not model or not hasattr(model, 'get_flat_weights'):
            print("Error: Model not initialized or does not implement get_flat_weights.")
            return None
        model_update = model.get_flat_weights()
        training_metadata = {
            "training_time_sec": metadata.get("training_time"),
            "epochs_completed": metadata.get("epochs"),
            "local_accuracy": metadata.get("accuracy"),
            "local_loss": metadata.get("loss", None)
        }
        result = {
            "client_id": self.config.get("client_id", "client_1"),
            "round_id": self.config.get("round_id", 1),
            "model_update": model_update,
            "training_metadata": training_metadata
        }
        result_path = os.path.join(os.path.dirname(__file__), "result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print("Saved training result to result.json")
        return metadata

    def upload(self, metadata=None):
        """Upload model results to server in the new format."""
        # Read from result.json
        try:
            result_path = os.path.join(os.path.dirname(__file__), "result.json")
            with open(result_path, "r") as f:
                upload_data = json.load(f)
        except FileNotFoundError:
            print("Error: result.json not found. Please run train first.")
            return None
        print("Uploading model update...")
        try:
            upload_data["user"] = self.user  # Add user info to upload
            upload_data["client_id"] = CLIENT_ID  # Ensure correct client_id
            response = self.api.upload_model(upload_data)
            print(f"Server response: {response}")
        except Exception as e:
            print(f"Failed to upload to server: {e}")
        return upload_data
    
    def run_full_cycle(self, join_code, data_path):
        """Run complete federated learning cycle."""
        print("Starting federated learning cycle...")
        
        self.join_round(join_code)
        self.setup()
        if not self.training_manager:
            print("Error: Training manager not initialized.")
            return None
        metadata = self.training_manager.train(data_path)
        if metadata is None:
            return None
        model = self.training_manager.model
        if not model or not hasattr(model, 'get_flat_weights'):
            print("Error: Model not initialized or does not implement get_flat_weights.")
            return None
        model_update = model.get_flat_weights()
        training_metadata = {
            "training_time_sec": metadata.get("training_time"),
            "epochs_completed": metadata.get("epochs"),
            "local_accuracy": metadata.get("accuracy"),
            "local_loss": metadata.get("loss", None)
        }
        if not self.config:
            print("Error: Config not loaded.")
            return None
        result = {
            "client_id": self.config.get("client_id", "client_1"),
            "round_id": self.config.get("round_id", 1),
            "model_update": model_update,
            "training_metadata": training_metadata
        }
        result_path = os.path.join(os.path.dirname(__file__), "result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print("Saved training result to result.json")
        upload_data = self.upload(metadata)
        
        print("Full cycle completed!")
        return {"metadata": metadata, "upload": upload_data} 

    def sync_contract(self):
        """Pull the latest contract from the server and save it as contract.json."""
        if not self.join_code:
            self.load_contract()
        if not self.join_code:
            raise ValueError("Must join round first or have join_code in contract.json")
        print("Syncing contract from server...")
        data = self.api.sync_contract(self.join_code)
        if "contract" in data:
            self.config = data["contract"]
        else:
            self.config = data
        # Always persist join_code
        self.config["join_code"] = self.join_code
        # Copy training params to top level if present
        tp = self.config.get("training_params", {})
        for k in ["learning_rate", "epochs", "batch_size"]:
            if k in tp:
                self.config[k] = tp[k]
        contract_path = os.path.join(os.path.dirname(__file__), "contract.json")
        with open(contract_path, "w") as f:
            json.dump(self.config, f, indent=2)
        print("Contract synced and saved to flclient/contract.json") 