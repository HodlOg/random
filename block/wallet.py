import argparse
import requests

import ecdsa
import hashlib
import base58

class Wallet:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.session = requests.Session()
        self.crypto_handler = CryptoHandler()
        
        self.private_key = self.crypto_handler.sk
        self.public_key = self.crypto_handler.vk
        self.address = base58.b58encode(hashlib.sha256(self.public_key.to_string()).digest()).decode()
        print(f"Wallet address: {self.address}")
        
    def sign_transaction(self, transaction):
        return self.crypto_handler.sign_object(transaction)

    def send_transaction(self, recipient, amount):
        transaction = {'sender': self.public_key.to_string().hex(), 'recipient': recipient, 'amount': amount}
        signature = self.sign_transaction(transaction)
        full_tx = {'meta': None, 'type': 'tx', 'data': transaction, 'signature': signature}
        print(full_tx)
        self.send_tx(full_tx)

    def get_balance(self):
        balance = 0
        blockchain = self.get_chain()
        print(blockchain)
        for block in blockchain:
            for transaction in block['data']:
                if transaction['recipient'] == self.public_key.to_string().hex():
                    balance += transaction['amount']
                if transaction['sender'] == self.public_key.to_string().hex():
                    balance -= transaction['amount']
        return balance

    def get_chain(self):
        try:
            response = self.session.get(f"{self.base_url}/chain")
            response.raise_for_status()
            chain = response.json()['chain']
            return chain
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Failed to get chain from node: {e}")
            return []

    def send_tx(self, transaction):
        try:
            response = self.session.post(f"{self.base_url}/message", json=transaction)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to send transaction to node: {e}")


class CryptoHandler:
    def __init__(self):
        # Generate private and public key upon instantiation
        self.sk = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
        self.vk = self.sk.get_verifying_key()
        
    def get_private_key(self):
        return self.sk.to_string().hex()
    
    def get_public_key(self):
        return self.vk.to_string().hex()
    
    def sign_object(self, obj):
        obj_str = str(obj)
        signature = self.sk.sign(obj_str.encode())
        return signature.hex()
    
def verify_signature(obj, public_key_hex, signature):
    # Verify the signature of an object given the public key and signature
    obj_str = str(obj)
    signature_bytes = bytes.fromhex(signature)
    public_key_bytes = bytes.fromhex(public_key_hex)
    vk = ecdsa.VerifyingKey.from_string(public_key_bytes, curve=ecdsa.SECP256k1)
    
    try:
        return vk.verify(signature_bytes, obj_str.encode())
    except ecdsa.BadSignatureError:
        return False

def main():
    parser = argparse.ArgumentParser(description='Wallet CLI')
    parser.add_argument('--host', type=str, default='localhost', help='Node host')
    parser.add_argument('--port', type=int, default=5000, help='Node port')
    parser.add_argument('--recipient', type=str, help='Recipient address')
    parser.add_argument('--amount', type=int, help='Transaction amount')
    parser.add_argument('--balance', action='store_true', help='Get wallet balance')
    args = parser.parse_args()

    wallet = Wallet(args.host, args.port)
    print(f"Connected to node at {args.host}:{args.port}")

    if args.balance:
        balance = wallet.get_balance()
        print(f"Wallet balance: {balance}")
    elif args.recipient and args.amount:
        wallet.send_transaction(args.recipient, args.amount)
        print(f"Transaction sent: {args.amount} coins to {args.recipient}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()