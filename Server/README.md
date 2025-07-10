# Federated Learning Server

Simple Flask server to receive model updates from federated learning clients.

## Usage

```bash
cd Server
pip install -r requirements.txt
python server.py
```

Server will start on port 3197 and accept POST requests at `/upload`. 