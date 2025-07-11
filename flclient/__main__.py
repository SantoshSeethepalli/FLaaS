"""
Command-line interface for federated learning client.
"""

import sys
import argparse
from .client import FederatedClient


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    subparsers = parser.add_subparsers(dest="command")

    join_parser = subparsers.add_parser("join", help="Join a federated learning round with a join code")
    join_parser.add_argument("--code", required=True, help="Join code provided by the server")
    sync_parser = subparsers.add_parser("sync", help="Pull latest FL contract from server")
    train_parser = subparsers.add_parser("train", help="Train locally using contract.json and user data")
    train_parser.add_argument("--data", required=True)
    upload_parser = subparsers.add_parser("upload", help="Upload model update and training metadata to server")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    client = FederatedClient()

    try:
        if args.command == "join":
            try:
                client.join_round(args.code)
                print(f"Joined round with code: {args.code}")
            except Exception as e:
                print(f"Join failed: {e}")
                sys.exit(1)
        elif args.command == "sync":
            try:
                client.sync_contract()
                print("Sync completed")
            except Exception as e:
                print(f"Sync failed: {e}")
                sys.exit(1)
        elif args.command == "train":
            try:
                client.load_contract()
                client.setup()
                metadata = client.train(args.data)
                if metadata is not None:
                    print("Training completed")
                else:
                    print("Training failed. Check your data file and contract.")
            except FileNotFoundError as e:
                print(f"Training failed: {e}. Make sure the contract is synced and the data file exists.")
                sys.exit(1)
            except Exception as e:
                print(f"Training failed: {e}")
                sys.exit(1)
        elif args.command == "upload":
            try:
                client.load_contract()
                result = client.upload()
                if result is not None:
                    print("Upload completed")
                else:
                    print("Upload failed. Run training first.")
            except Exception as e:
                print(f"Upload failed: {e}")
                sys.exit(1)
    except KeyboardInterrupt:
        print("Operation cancelled by user.")
        sys.exit(1)

if __name__ == "__main__":
    main() 