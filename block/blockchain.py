import time
from block import Block

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.difficulty = 9

    @property
    def last_block(self):
        return self.chain[-1]

    def create_genesis_block(self):
        return Block(0, [], '0' * 64)

    def add_block(self, block):
        if self.is_valid_block(block):
            self.chain.append(block)
            self.pending_transactions = []

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine_block(self):
        if not self.pending_transactions:
            return None
        new_block = Block(
            time.time(),
            self.pending_transactions,
            self.last_block.hash.hex()
        )
        new_block.mine(self.difficulty)
        return new_block

    def replace_chain(self, chain):
        if self.is_valid_chain(chain) and self.get_total_work(chain) > self.get_total_work(self.chain):
            self.chain = chain
            self.pending_transactions = []
            print("Chain replaced with a longer valid chain")

    def is_valid_chain(self, chain):
        if not self.is_valid_genesis_block(chain[0]):
            return False
        for i in range(1, len(chain)):
            if not self.is_valid_block(chain[i]):
                return False
        return True

    def is_valid_genesis_block(self, block):
        return block == self.create_genesis_block()

    def is_valid_block(self, block):
        if block.previous_hash != self.last_block.hash.hex():
            return False
        if not block.is_valid(self.difficulty):
            return False
        return True

    def get_total_work(self, chain):
        return sum(2 ** block.difficulty for block in chain)

    def prune_transactions(self, max_transactions):
        if len(self.chain) > max_transactions:
            self.chain = self.chain[-max_transactions:]

    def get_balance(self, address):
        balance = 0
        for block in self.chain:
            for transaction in block.data:
                if transaction['recipient'] == address:
                    balance += transaction['amount']
                if transaction['sender'] == address:
                    balance -= transaction['amount']
        return balance

    def to_dict(self):
        return [block.to_dict() for block in self.chain]