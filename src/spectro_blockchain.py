import hashlib
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import threading
import random

class Block:
    """Đại diện cho một block trong blockchain"""
    def __init__(self, index: int, timestamp: float, data: Dict, previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Tính toán hash của block"""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class Transaction:
    """Đại diện cho một giao dịch"""
    def __init__(self, tx_type: str, sender: str, receiver: str, material_id: str, 
                 spectral_hash: str = None, metadata: Dict = None):
        self.tx_type = tx_type  # registerMaterial, transferOwnership, verifyMaterial
        self.sender = sender
        self.receiver = receiver
        self.material_id = material_id
        self.spectral_hash = spectral_hash
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.tx_id = self.generate_tx_id()
    
    def generate_tx_id(self) -> str:
        """Tạo ID duy nhất cho transaction"""
        tx_string = f"{self.tx_type}{self.sender}{self.receiver}{self.material_id}{self.timestamp}"
        return hashlib.sha256(tx_string.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        return {
            "tx_id": self.tx_id,
            "tx_type": self.tx_type,
            "sender": self.sender,
            "receiver": self.receiver,
            "material_id": self.material_id,
            "spectral_hash": self.spectral_hash,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

class SpectroChain:
    """Blockchain chính cho SpectroChain-Dental"""
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.material_registry = {}  # material_id -> spectral_hash
        self.ownership_history = {}  # material_id -> [owner_history]
        self.mining_reward = 0
        self.difficulty = 2  # Độ khó đơn giản cho demo
    
    def create_genesis_block(self) -> Block:
        """Tạo block đầu tiên"""
        return Block(0, time.time(), {"type": "genesis"}, "0")
    
    def get_latest_block(self) -> Block:
        return self.chain[-1]
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """Thêm giao dịch vào pending pool"""
        # Validate transaction
        if self.validate_transaction(transaction):
            self.pending_transactions.append(transaction)
            return True
        return False
    
    def validate_transaction(self, transaction: Transaction) -> bool:
        """Validate giao dịch"""
        if transaction.tx_type == "registerMaterial":
            # Chỉ manufacturer mới có thể đăng ký
            return transaction.sender.startswith("manufacturer")
        elif transaction.tx_type == "transferOwnership":
            # Phải có quyền sở hữu hiện tại
            current_owner = self.get_current_owner(transaction.material_id)
            return current_owner == transaction.sender
        elif transaction.tx_type == "verifyMaterial":
            # Material phải tồn tại trong registry
            return transaction.material_id in self.material_registry
        return True
    
    def mine_pending_transactions(self, mining_reward_address: str) -> bool:
        """Mine các giao dịch pending"""
        if not self.pending_transactions:
            return False
        
        # Tạo block mới
        block_data = {
            "transactions": [tx.to_dict() for tx in self.pending_transactions],
            "reward_address": mining_reward_address
        }
        
        new_block = Block(
            len(self.chain),
            time.time(),
            block_data,
            self.get_latest_block().hash
        )
        
        # Mine block (đơn giản hóa)
        new_block.hash = new_block.calculate_hash()
        
        # Thêm vào chain
        self.chain.append(new_block)
        
        # Xử lý các giao dịch
        for tx in self.pending_transactions:
            self.process_transaction(tx)
        
        # Clear pending transactions
        self.pending_transactions = []
        return True
    
    def process_transaction(self, transaction: Transaction):
        """Xử lý giao dịch sau khi mine"""
        if transaction.tx_type == "registerMaterial":
            self.material_registry[transaction.material_id] = transaction.spectral_hash
            self.ownership_history[transaction.material_id] = [transaction.sender]
        
        elif transaction.tx_type == "transferOwnership":
            if transaction.material_id in self.ownership_history:
                self.ownership_history[transaction.material_id].append(transaction.receiver)
    
    def get_current_owner(self, material_id: str) -> Optional[str]:
        """Lấy chủ sở hữu hiện tại của material"""
        if material_id in self.ownership_history:
            return self.ownership_history[material_id][-1]
        return None
    
    def verify_material(self, material_id: str, spectral_hash: str) -> bool:
        """Xác minh material bằng spectral hash"""
        if material_id not in self.material_registry:
            return False
        return self.material_registry[material_id] == spectral_hash
    
    def get_material_history(self, material_id: str) -> List[str]:
        """Lấy lịch sử sở hữu của material"""
        return self.ownership_history.get(material_id, [])
    
    def is_chain_valid(self) -> bool:
        """Kiểm tra tính hợp lệ của blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            if current_block.hash != current_block.calculate_hash():
                return False
            
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True 