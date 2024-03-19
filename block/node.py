# node.py
import argparse
from flask import Flask, request, jsonify
import threading
import random
import time
import requests
from block import Block
from blockchain import Blockchain
from routing_table import RoutingTable
import ecdsa
import hashlib
import base58

from wallet import verify_signature

class Node:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.peers = set()
        self.routing_table = RoutingTable()
        self.blockchain = Blockchain()
        self.app = Flask(__name__)
        self.session = requests.Session()
        self.node_id = f"{host}:{port}"

        @self.app.route('/peers', methods=['GET'])
        def get_peers():
            return jsonify(list(self.peers))

        @self.app.route('/chain', methods=['GET'])
        def get_chain():
            chain = self.blockchain.to_dict()
            return jsonify({'chain': chain})

        @self.app.route('/message', methods=['POST'])
        def message():
            if not request.is_json:
                return jsonify({'message': 'Invalid request format'}), 400
            message = request.get_json()
            
            # print message with color
            print(f"\n\033[1;32;40mReceived message: {time.time()} \033[m")
            print(f"\033[1;32;40m{message}\033[m")
            print("\n")
            
            meta = message['meta']
            data = message['data']
            if message['type'] == 'tx':
                if self.is_valid_transaction(data, message['signature']):
                    self.blockchain.add_transaction(data)
                    self.send_message(data, 'message', exclude=meta)
                    return jsonify({'message': 'Transaction added successfully'})
                else:
                    return jsonify({'message': 'Invalid transaction'}), 400
            elif message['type'] == 'block':
                block = Block.from_dict(data)
                self.blockchain.add_block(block)
                print(f"\033[1;32;40mBlock added successfully {self.blockchain.to_dict()} {time.time()}\033[m")
                self.check_longest_chain()
                return jsonify({'message': 'Block added successfully'})
            elif message['type'] == 'connect':
                self.add_peer(data['host'], data['port'])
                return jsonify({'message': 'Connected to peer'})
            else:
                return jsonify({'message': 'Invalid message type'}), 400
            
        

    def start(self):
        threading.Thread(target=self.mine_blocks).start()
        self.app.run(host=self.host, port=self.port)
        
    def connect_to_peer(self, host, port):
        if (host, port) in self.peers or (host, port) == (self.host, self.port):
            return

        self.peers.add((host, port))
        print(f"Connecting to peer {host}:{port}")
        message = {
            'meta': f"{self.host}:{self.port}",
            'type': 'connect',
            'data': {
                'host': self.host,
                'port': self.port
            }
        }
        response = requests.post(f'http://{host}:{port}/message', json=message)

        if response.status_code == 200:
            peer_peers = requests.get(f'http://{host}:{port}/peers').json()
            print(f"peer_peers: {peer_peers}")
            for peer in peer_peers:
                self.connect_to_peer(peer[0], peer[1])
        else:
            self.peers.remove((host, port))
            print(f"Failed to connect to peer {host}:{port}")
    
    def mine_blocks(self):
        while True:
            block = self.blockchain.mine_block()
            if block:
                print(f"\n\033[1;31;40mMined new block: {block.to_dict()} {time.time()}\033[m")
                self.send_block(block)
                self.blockchain.add_block(block)
            time.sleep(random.uniform(1, 5))
            
    def send_block(self, block):
        for peer in self.peers:
            # print(f"Sending block to peer: {peer}")
            print(f"\n\033[1;32;40mSending block to peers: {time.time()} \033[m")
            try:
                url = f"http://{peer[0]}:{peer[1]}/message"
                self.session.post(url, json={'meta': f"{self.host}:{self.port}", 'type': 'block', 'data': block.to_dict()})
            except requests.exceptions.RequestException:
                pass

    def send_message(self, data, endpoint, exclude=None):
        print(f"\n\033[1;32;40mSending message to peers: \033[m")
        req_data = {
            'meta': f"{self.host}:{self.port}",
            'type': 'tx',
            'data': data
        }
        print(f"\033[1;32;40m{req_data}\033[m")
        print(f"\033[1;32;40mPeers: {self.peers}\033[m")
        print(f"\033[1;32;40mExclude: {exclude}\033[m")
        print("\n")
        for peer in self.peers:
            if f"{peer[0]}:{peer[1]}" == exclude:
                continue
            print(f"peer: {peer}")
            try:
                url = f"http://{peer[0]}:{peer[1]}/{endpoint}"
                self.session.post(url, json=req_data)
            except requests.exceptions.RequestException:
                pass
            
    def add_peer(self, peer_host, peer_port):
        self.peers.add((peer_host, peer_port))
        print({'message': 'Peer added successfully'})

    def check_longest_chain(self):
        print(f"\033[1;32;40mChecking longest chain\033[m")
        longest_chain = None
        max_length = len(self.blockchain.chain)

        for peer in self.peers:
            try:
                url = f"http://{peer[0]}:{peer[1]}/chain"
                response = self.session.get(url)
                response.raise_for_status()
                chain_data = response.json()['chain']
                chain = [Block.from_dict(block) for block in chain_data]

                if len(chain) > max_length and self.blockchain.is_valid_chain(chain):
                    longest_chain = chain
                    max_length = len(chain)
            except (requests.exceptions.RequestException, ValueError):
                pass

        if longest_chain:
            self.blockchain.chain = longest_chain
            print(f"\033[1;32;40mSwitched to the longest chain\033[m")

    def is_valid_transaction(self, transaction, signature):
        # Decode the Bitcoin address
        address = transaction['sender']
        is_valid = verify_signature(transaction, address, signature)
        if not is_valid:
            print(f"Invalid signature: {signature}")
            return False
        # Check if the sender has sufficient balance, commented out because we need fake balance for testing
        # sender_balance = self.blockchain.get_balance(transaction['sender'])
        # if sender_balance < transaction['amount']:
        #     print(f"Insufficient balance: {sender_balance}")
        #     return False
        
        return True
# cli for node
def main():
    parser = argparse.ArgumentParser(description='Node CLI')
    parser.add_argument('--host', type=str, default='localhost', help='Node host')
    parser.add_argument('--port', type=int, default=5000, help='Node port')
    args = parser.parse_args()
    node = Node(args.host, args.port)
    threading.Thread(target=node.start).start()
    while True:
        print("Enter command: ")
        command = input("> ")
        if command == 'add_peer':
            peer_host = input("Enter peer host: ")
            peer_port = int(input("Enter peer port: "))
            node.add_peer(peer_host, peer_port)
            
        elif command == 'get_peers':
            print(list(node.peers))
            
        elif command == 'connect_peer':
            peer_host = input("Enter peer host: ")
            peer_port = int(input("Enter peer port: "))
            node.connect_to_peer(peer_host, peer_port)
            print({'message': 'Connected to peer'})
            
        elif command == 'get_chain':
            chain = node.blockchain.to_dict()
            print({'chain': chain})
            
        elif command == 'add_transaction':
            sender = input("Enter sender: ")
            recipient = input("Enter recipient: ")
            amount = int(input("Enter amount: "))
            transaction = {
                'sender': sender,
                'recipient': recipient,
                'amount': amount
            }
            node.blockchain.add_transaction(transaction)
            node.send_message(transaction, 'tx')
            print({'message': 'Transaction added successfully'})
            
        elif command == 'add_block':
            index = int(input("Enter index: "))
            timestamp = float(input("Enter timestamp: "))
            data = input("Enter data: ")
            previous_hash = input("Enter previous hash: ")
            nonce = int(input("Enter nonce: "))
            block_data = {
                'index': index,
                'timestamp': timestamp,
                'data': data,
                'previous_hash': previous_hash,
                'nonce': nonce
            }
            block = Block.from_dict(block_data)
            node.blockchain.add_block(block)
            node.check_longest_chain()
            print({'message': 'Block added successfully'})
            
        else:
            print("Invalid command")
    
if __name__ == '__main__':
    main()