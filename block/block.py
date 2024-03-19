import hashlib
import pickle

class Block:
    def __init__(self, timestamp, data, previous_hash):
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self._hash = None

    @property
    def hash(self):
        if self._hash is None: self._hash = self.hash_func()
        return self._hash

    def hash_func(self):
        return hashlib.sha256(self.serialize()).digest()

    def serialize(self):
        return b''.join([
            int(self.timestamp).to_bytes(8, byteorder='little'),
            pickle.dumps(self.data),
            bytes.fromhex(self.previous_hash),
            self.nonce.to_bytes(4, byteorder='little')
        ])

    def mine(self, difficulty):
        target = 2 ** (256 - difficulty)
        while int.from_bytes(self.hash, byteorder='big') >= target:
            self.nonce += 1
            self._hash = None

    def is_valid(self, difficulty):
        target = 2 ** (256 - difficulty)
        return int.from_bytes(self.hash, byteorder='big') < target

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash.hex()
        }

    @staticmethod
    def from_dict(block_data):
        block = Block(
            block_data['timestamp'],
            block_data['data'],
            block_data['previous_hash']
        )
        block.nonce = block_data['nonce']
        block._hash = bytes.fromhex(block_data['hash'])
        return block

    def __eq__(self, other):
        if isinstance(other, Block): return self.hash == other.hash
        return False

    def __hash__(self):
        return hash(self.hash)
