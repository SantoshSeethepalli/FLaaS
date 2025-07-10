# Federated Learning Client

A simple Python package for federated learning clients.

## Structure

```
flclient/
├── __main__.py            # CLI commands
├── client.py              # Main client class
├── api.py                 # Server communication
├── training.py            # Training workflow
├── config.py              # Configuration constants
├── data_loader.py         # CSV data loading
└── models/                # Model implementations
    ├── __init__.py        # Model factory
    ├── base.py            # Base model class
    ├── mlp.py             # MLP model
    └── logistic_regression.py # Logistic regression model

Server/
├── server.py              # Flask server
├── contract.json          # Static model configuration
├── requirements.txt       # Server dependencies
└── README.md             # Server documentation

data/                      # User data folder
├── README.md             # Data format instructions
└── sample_data.csv       # Example data file
```

## Usage

### Server Setup
```bash
cd Server
pip install -r requirements.txt
python server.py
```

### Client Usage

```bash
# Join a training round
python -m flclient join ABC123

# Sync configuration updates
python -m flclient sync

# Train model
python -m flclient train --data data/sample_data.csv --join-code ABC123

# Upload results
python -m flclient upload

# Full cycle (join + train + upload)
python -m flclient trainAndUpload --data data/sample_data.csv --join-code ABC123
```

## Data Folder

Place your CSV training data files in the `data/` folder. See `data/README.md` for format requirements.

## Models

The system supports multiple model types:
- **MLP** (Multi-Layer Perceptron) - Default
- **Logistic Regression** - Simple linear model

To use a different model, update the `model_type` in `Server/contract.json`.

## Valid Join Codes

- ABC123
- DEF456  
- GHI789

## Server Endpoints

- `POST /join` - Join round with code
- `POST /sync` - Sync contract updates
- `POST /upload` - Upload model weights
- `GET /codes` - List valid join codes

## Install

```bash
pip install -r requirements.txt
``` 