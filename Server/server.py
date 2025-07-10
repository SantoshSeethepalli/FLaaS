from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)

VALID_JOIN_CODES = ["ABC123", "DEF456", "GHI789"]


def load_contract():
    """Load static contract from file."""
    contract_file = os.path.join(os.path.dirname(__file__), 'contract.json')
    with open(contract_file, 'r') as f:
        return json.load(f)


def validate_join_code(join_code):
    """Validate join code and return error response if invalid."""
    if join_code not in VALID_JOIN_CODES:
        return jsonify({"error": "Invalid join code"}), 400
    return None


@app.route('/join', methods=['POST'])
def join_round():
    data = request.get_json()
    join_code = data.get('join_code')
    username = data.get('username', 'Unknown')
    email = data.get('email', 'Unknown')
    
    error_response = validate_join_code(join_code)
    if error_response:
        return error_response
    
    contract = load_contract()
    print(f"Client joined with code: {join_code} (User: {username}, Email: {email})")
    return jsonify({
        "contract": contract,
        "client_id": f"client_{join_code}"
    })


@app.route('/sync', methods=['POST'])
def sync_contract():
    data = request.get_json()
    join_code = data.get('join_code')
    
    error_response = validate_join_code(join_code)
    if error_response:
        return error_response
    
    contract = load_contract()
    print(f"Client synced with code: {join_code}")
    return jsonify({"contract": contract})


@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    join_code = data.get('join_code')
    client_id = data.get('client_id')
    username = data.get('username', 'Unknown')
    email = data.get('email', 'Unknown')
    
    error_response = validate_join_code(join_code)
    if error_response:
        return error_response
    
    print(f"Received upload from client {client_id} for round {join_code} (User: {username}, Email: {email}):")
    print(f"Model type: {data.get('model_type')}")
    print(f"Accuracy: {data.get('metadata', {}).get('accuracy', 'N/A')}")
    print(f"Training time: {data.get('metadata', {}).get('training_time', 'N/A')}s")
    
    return jsonify({"status": "received"})


@app.route('/codes', methods=['GET'])
def list_codes():
    return jsonify({"valid_codes": VALID_JOIN_CODES})


if __name__ == '__main__':
    app.run(port=3197) 