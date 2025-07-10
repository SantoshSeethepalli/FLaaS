"""
Command-line interface for federated learning client.
"""

import sys
import argparse
from .client import FederatedClient


def load_contract_and_join_code(client, join_code_arg):
    """Helper function to load contract and get join code."""
    client.load_contract()
    
    join_code = join_code_arg
    if not join_code and client.config and "join_code" in client.config:
        join_code = client.config["join_code"]
    elif not join_code:
        print("Error: Join code required. Use --join-code or join first.")
        sys.exit(1)
    
    return join_code


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    subparsers = parser.add_subparsers(dest="command")
    
    join_parser = subparsers.add_parser("join")
    join_parser.add_argument("join_code", help="Join code for the round")
    
    sync_parser = subparsers.add_parser("sync")
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--join-code", help="Join code (optional if already joined)")
    
    upload_parser = subparsers.add_parser("upload")
    
    full_parser = subparsers.add_parser("trainAndUpload")
    full_parser.add_argument("--data", required=True)
    full_parser.add_argument("--join-code", help="Join code (optional if already joined)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    client = FederatedClient()
    
    try:
        if args.command == "join":
            client.join_round(args.join_code)
            print("Join completed")
            
        elif args.command == "sync":
            client.load_contract()
            client.sync()
            
        elif args.command == "train":
            join_code = load_contract_and_join_code(client, args.join_code)
            client.join_round(join_code)
            client.setup()
            metadata = client.train(args.data)
            print("Training completed")
                
        elif args.command == "upload":
            client.load_contract()
            client.upload()
            print("Upload completed")
            
        elif args.command == "trainAndUpload":
            join_code = load_contract_and_join_code(client, args.join_code)
            client.join_round(join_code)
            client.setup()
            metadata = client.train(args.data)
            client.upload(metadata)
            print("Full cycle completed")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 