import json
from web3 import Web3

# --- Configuration ---
# IMPORTANT: You must update these values after deploying the contract.
# 1. GANACHE_URL: The URL of your Ganache instance (usually http://127.0.0.1:7545)
# 2. CONTRACT_ADDRESS: The address of the deployed SpectroChain contract.
# 3. CONTRACT_ABI: The ABI from the generated JSON file.
# You can find the ABI in: ../../blockchain/build/contracts/SpectroChain.json

GANACHE_URL = "http://127.0.0.1:7545"
CONTRACT_ADDRESS = "test"  # <-- IMPORTANT: UPDATE THIS
CONTRACT_ABI = []  # <-- IMPORTANT: UPDATE THIS

# --- Web3 Setup ---
w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
# Set a default account (the first account from Ganache) for sending transactions
if w3.is_connected():
    w3.eth.default_account = w3.eth.accounts[0]
else:
    print("Warning: Not connected to Ganache. Please ensure Ganache is running.")

def get_contract_instance():
    """
    Returns an instance of the SpectroChain smart contract.
    Returns None if the contract address or ABI is not set.
    """
    if CONTRACT_ADDRESS == "YOUR_CONTRACT_ADDRESS" or not CONTRACT_ABI:
        print("Error: Contract address or ABI is not configured in blockchain.py")
        return None
    
    contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
    return contract

# --- Smart Contract Interaction Functions ---

def register_material_on_chain(product_id: str, batch_id: str, data_hash: str):
    """
    Calls the registerMaterial function on the smart contract.
    """
    contract = get_contract_instance()
    if not contract:
        raise Exception("Contract not initialized. Please check configuration.")

    try:
        # Execute the transaction
        tx_hash = contract.functions.registerMaterial(
            product_id, batch_id, data_hash
        ).transact()
        
        # Wait for the transaction to be mined
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        return {"status": "success", "tx_hash": tx_hash.hex(), "block_number": receipt.blockNumber}
    except Exception as e:
        print(f"Error registering material on-chain: {e}")
        return {"status": "error", "message": str(e)}


def get_hash_from_chain(product_id: str) -> str:
    """
    Calls the getMaterialHash view function on the smart contract.
    """
    contract = get_contract_instance()
    if not contract:
        raise Exception("Contract not initialized. Please check configuration.")
    
    try:
        stored_hash = contract.functions.getMaterialHash(product_id).call()
        return stored_hash
    except Exception as e:
        print(f"Error getting hash from chain: {e}")
        return ""

def load_contract_abi():
    """
    Loads the contract ABI from the JSON file.
    This function should be called at server startup.
    """
    global CONTRACT_ABI
    try:
        # Path relative to the backend/ directory where FastAPI runs
        with open("../blockchain/build/contracts/SpectroChain.json", "r") as f:
            contract_json = json.load(f)
            CONTRACT_ABI = contract_json["abi"]
            print("Successfully loaded contract ABI.")
    except FileNotFoundError:
        print("ABI file not found. Please compile and deploy the contract first by running 'truffle migrate'.")
    except Exception as e:
        print(f"An error occurred while loading the ABI: {e}") 