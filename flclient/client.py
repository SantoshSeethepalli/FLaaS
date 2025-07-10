"""
Main federated learning client class.
"""

import json
import os
from .api import ServerAPI
from .training import TrainingManager
from .config import SERVER_URL


class FederatedClient:
    def __init__(self, server_url=SERVER_URL):
        self.api = ServerAPI(server_url)
        self.training_manager = None
        self.config = None
        self.join_code = None
        self.user = self.load_user_config()

    def load_user_config(self):
        try:
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'user_config.json'), 'r') as f:
                user = json.load(f)
            if not user.get('username') or not user.get('email'):
                raise ValueError
            return user
        except Exception:
            print('Error: user_config.json missing or incomplete. Please create it with username and email.')
            exit(1)
    
    def join_round(self, join_code):
        """Join a federated learning round."""
        print(f"Joining round with code: {join_code}")
        
        data = self.api.join_round(join_code, self.user)
        self.config = data["contract"]
        self.join_code = join_code
        
        # Store join code in config for future use
        self.config["join_code"] = join_code
        
        # Save contract to file
        with open("contract.json", "w") as f:
            json.dump(self.config, f, indent=2)
        
        print("Joined round successfully")
        return self.config
    
    def sync(self):
        """Sync configuration from server."""
        # Load contract first if not already loaded
        if not self.config:
            self.load_contract()
        
        if not self.join_code:
            raise ValueError("Must join round first or have join_code in contract.json")
        
        print("Syncing configuration...")
        
        data = self.api.sync_contract(self.join_code)
        self.config = data["contract"]
        # Preserve the join_code in the updated config
        self.config["join_code"] = self.join_code
        print(f"Synced {self.config['model_type']} model config")
        return self.config
    
    def load_contract(self):
        """Load contract from file if it exists."""
        try:
            with open("contract.json", "r") as f:
                self.config = json.load(f)
                self.join_code = self.config.get("join_code")
                return self.config
        except FileNotFoundError:
            return None
    
    def setup(self):
        """Setup training components."""
        if not self.config:
            raise ValueError("Must join round first")
        
        self.training_manager = TrainingManager(self.config)
        self.training_manager.setup()
    
    def train(self, data_path):
        """Train the model."""
        if not self.training_manager:
            print("Error: Must setup first")
            return None
        
        if not self.config:
            print("Error: Config not loaded. Please join a round first.")
            return None
        
        metadata = self.training_manager.train(data_path)
        if metadata is None:
            return None
        
        # Save weights and metadata to result.json
        weights = self.training_manager.get_weights()
        result = {
            "model_type": self.config["model_type"],
            "metadata": metadata,
            "weights": weights,
            "join_code": self.join_code,
            "client_id": f"client_{self.join_code}",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        with open("result.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Saved training result to result.json")
        return metadata
    
    def upload(self, metadata=None):
        """Upload model results to server."""
        # Read from result.json
        try:
            with open("result.json", "r") as f:
                upload_data = json.load(f)
        except FileNotFoundError:
            print("Error: result.json not found. Please run train first.")
            return None
        # Add user info to upload
        upload_data["username"] = self.user["username"]
        upload_data["email"] = self.user["email"]
        print("Uploading model update...")
        try:
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
        metadata = self.train(data_path)
        upload_data = self.upload(metadata)
        
        print("Full cycle completed!")
        return {"metadata": metadata, "upload": upload_data} 