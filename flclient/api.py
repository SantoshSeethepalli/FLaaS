import requests
from .config import SERVER_URL


class ServerAPI:
    def __init__(self, server_url=SERVER_URL):
        self.server_url = server_url
    
    def _handle_response(self, response, operation):
        """Handle API response and raise exceptions on error."""
        if response.status_code != 200:
            raise Exception(f"Failed to {operation}: {response.json().get('error', 'Unknown error')}")
        return response.json()
    
    def join_round(self, join_code, user_info=None):
        """Join a federated learning round."""
        payload = {"join_code": join_code}
        if user_info:
            payload.update(user_info)
        response = requests.post(f"{self.server_url}/join", json=payload)
        return self._handle_response(response, "join round")
    
    def sync_contract(self, join_code):
        """Sync contract from server."""
        response = requests.post(f"{self.server_url}/sync", json={"join_code": join_code})
        return self._handle_response(response, "sync")
    
    def upload_model(self, upload_data):
        """Upload model weights and metadata."""
        response = requests.post(f"{self.server_url}/upload", json=upload_data)
        return self._handle_response(response, "upload") 