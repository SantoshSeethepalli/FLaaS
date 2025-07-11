import json
import os
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

VALID_JOIN_CODES = ["ABC123", "DEF456", "GHI789"]
global_state = {
    "round_id": 1,
    "global_weights": [],
    "model_type": None,
    "model_version": "v1.0",
    "aggregation": "FedAvg"
}
client_updates = {}


def load_contract():
    contract_file = os.path.join(os.path.dirname(__file__), 'contract.json')
    with open(contract_file, 'r') as f:
        contract = json.load(f)
    # Always include input_size and hidden_size if present in the file
    if "input_size" not in contract:
        contract["input_size"] = 10  # Default or set as needed
    if contract.get("model_type") == "mlp" and "hidden_size" not in contract:
        contract["hidden_size"] = 32  # Default or set as needed
    return contract

def generate_initial_weights(contract):
    model_type = contract["model_type"]
    if model_type == "logistic_regression":
        input_size = contract["input_size"] if "input_size" in contract else 10
        weights = np.random.uniform(-0.1, 0.1, input_size + 1)  # +1 for bias
        return weights.tolist()
    elif model_type == "mlp":
        input_size = contract.get("input_size", 10)
        hidden_size = contract.get("hidden_size", 32)
        # Flatten all weights and biases into a single list
        W_ih = np.random.uniform(-0.1, 0.1, (input_size, hidden_size)).flatten()
        b_h = np.zeros(hidden_size)
        W_ho = np.random.uniform(-0.1, 0.1, hidden_size).flatten()
        b_o = np.zeros(1)
        weights = np.concatenate([W_ih, b_h, W_ho, b_o])
        return weights.tolist()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def validate_join_code(join_code):
    if join_code not in VALID_JOIN_CODES:
        return jsonify({"error": "Invalid join code"}), 400
    return None

@app.route('/join', methods=['POST'])
def join_round():
    global global_state
    data = request.get_json()
    join_code = data.get('join_code')
    error_response = validate_join_code(join_code)
    if error_response:
        return error_response
    contract = load_contract()
    # Generate initial weights if not present
    if not global_state["global_weights"]:
        contract["initial_weights"] = generate_initial_weights(contract)
        global_state["global_weights"] = contract["initial_weights"]
        global_state["model_type"] = contract["model_type"]
        global_state["model_version"] = contract.get("model_version", "v1.0")
        global_state["aggregation"] = contract.get("aggregation", "FedAvg")
    else:
        contract["initial_weights"] = global_state["global_weights"]
    contract["round_id"] = global_state["round_id"]
    return jsonify(contract)

@app.route('/sync', methods=['POST'])
def sync_contract():
    global global_state
    data = request.get_json()
    join_code = data.get('join_code')
    error_response = validate_join_code(join_code)
    if error_response:
        return error_response
    contract = load_contract()
    contract["initial_weights"] = global_state["global_weights"]
    contract["round_id"] = global_state["round_id"]
    return jsonify(contract)

@app.route('/upload', methods=['POST'])
def upload():
    global global_state, client_updates
    data = request.get_json()
    client_id = data.get('client_id')
    user_info = data.get('user', {})  # Extract user info
    round_id = data.get('round_id')
    model_update = data.get('model_update')
    training_metadata = data.get('training_metadata', {})
    # Store client update
    client_updates[client_id] = {
        "model_update": model_update,
        "training_metadata": training_metadata,
        "round_id": round_id,
        "user": user_info  # Store user info
    }
    print(f"Received update from {client_id} ({user_info}) for round {round_id}")
    # Aggregate if enough clients (for demo, aggregate after 2)
    if len(client_updates) >= 2:
        updates = [np.array(update["model_update"]) for update in client_updates.values()]
        new_weights = np.mean(updates, axis=0)
        global_state["global_weights"] = new_weights.tolist()
        global_state["round_id"] += 1
        client_updates = {}
        print(f"Aggregated new global weights for round {global_state['round_id']}")
    return jsonify({"status": "received", "current_round": global_state["round_id"]})

@app.route('/codes', methods=['GET'])
def list_codes():
    return jsonify({"valid_codes": VALID_JOIN_CODES})

if __name__ == '__main__':
    app.run(port=3197) 