# SpectroChain Dental - MVP

This is the MVP (Minimum Viable Product) version of a dental material traceability system using Raman spectroscopy, RFID (simulated), and Blockchain. The system is built with a 4-layer architecture, using FastAPI for the backend and Ethereum/Ganache for the blockchain layer.

## System Architecture

1.  **Physical Layer**: Raman spectroscopy data is simulated using `.csv` files. RFID information (`productID`, `batchID`) is entered manually by users.
2.  **Connectivity Layer**: The web interface sends data to the backend server via RESTful API.
3.  **Blockchain Layer**: Smart contract `SpectroChain.sol` (written in Solidity) is deployed on a local Ganache network. It stores SHA-256 hashes of spectroscopy data.
4.  **Application Layer**: A simple web interface (HTML/Bootstrap/JS) served by FastAPI, supporting two roles:
    - **Manufacturer**: Register new materials.
    - **Clinician**: Verify material authenticity.

## Technologies Used

- **Backend**: Python 3.8+ with FastAPI
- **Blockchain**: Ethereum (simulated by Ganache)
- **Smart Contract**: Solidity v0.8.9
- **Blockchain Tools**: Truffle Suite, Web3.py
- **Frontend**: HTML, Bootstrap 5, JavaScript
- **Data**: `.csv` files in the `data/` directory.

---

## Installation and Setup Guide

### 1. Prerequisites

Before starting, make sure you have installed the following tools on your machine:

- **Node.js and npm**: Required to install Truffle. [Download here](https://nodejs.org/)
- **Truffle Suite**: Ethereum development tool. Install globally using npm:
  ```bash
  npm install -g truffle
  ```
- **Ganache**: Personal blockchain environment for testing. We recommend the graphical user interface (UI) version for easy transaction monitoring. [Download here](https://trufflesuite.com/ganache/)
- **Python 3.8+ and pip**: Environment to run the FastAPI backend. [Download here](https://www.python.org/)

### 2. Project Setup

**a. Clone Repository (Skip if you already have the source code)**

```bash
git clone <YOUR_REPO_URL>
cd SpectroChain-Dental
```

**b. Install Python Libraries**

Create a virtual environment (recommended) and install the required packages from the `requirements.txt` file.

```bash
# Create and activate virtual environment (example for Windows)
python -m venv venv
venv\Scripts\activate

# Install libraries
pip install -r requirements.txt
```

### 3. Run Blockchain

**a. Start Ganache**

Open the Ganache application you installed. Create a new "Quickstart" workspace. Ganache will initialize a local blockchain network, usually running at `HTTP://127.0.0.1:7545`, with 10 accounts each having 100 ETH.

**b. Compile and Deploy Smart Contract**

Open a new terminal, navigate to the `blockchain` directory and run the `truffle migrate` command.

```bash
cd blockchain
truffle migrate --reset
```

This command will:

1.  Compile the `SpectroChain.sol` contract into ABI and bytecode.
2.  Deploy the contract to the running Ganache network.
3.  Save the ABI and contract address information to the `blockchain/build/contracts/` directory.

After successful execution, you will see output similar to:

```
...
2_deploy_contracts.js
=====================

   Deploying 'SpectroChain'
   ------------------------
   > transaction hash:    0x...
   > Blocks: 2            Seconds: 0
   > contract address:    0xAbCdEf1234567890...  <-- THIS IS THE IMPORTANT ADDRESS
   > block number:        3
   ...
```

### 4. Configure Backend

Now you need to update the backend so it knows the address of the deployed smart contract.

**a. Get contract address:**

Copy the `contract address` value from the `truffle migrate` command result.

**b. Update configuration file:**

Open file `SpectroChain-Dental/backend/app/blockchain.py` and find the following line:

```python
CONTRACT_ADDRESS = "YOUR_CONTRACT_ADDRESS"  # <-- IMPORTANT: UPDATE THIS
```

Replace `"YOUR_CONTRACT_ADDRESS"` with the contract address you just copied. For example:

```python
CONTRACT_ADDRESS = "0xAbCdEf1234567890..."
```

Save the file. This file also automatically loads the ABI from `blockchain/build/contracts/SpectroChain.json`, so you don't need to manually copy the ABI.

### 5. Run Backend Server

Return to the main terminal (where you activated the Python virtual environment), navigate to the `backend` directory and start the FastAPI server using `uvicorn`.

```bash
cd backend
uvicorn app.main:app --reload
```

The server will start and usually run at `http://127.0.0.1:8000`. You will see startup logs, including notifications that the ABI has been loaded and successful connection to Ganache.

### 6. Use the Application

**a. Open browser:**

Access `http://127.0.0.1:8000`. You will see the SpectroChain web interface.

**b. Register Material (Manufacturer)**

1.  In the "Manufacturer" section, enter `Product ID` and `Batch ID`. For example:
    - Product ID: `PRODUCT-A`
    - Batch ID: `BATCH-001`
2.  Click "Choose File" and select one of the simulated data files from the `data/` directory (e.g., `product_A.csv`).
3.  Click the "Register on Blockchain" button.
4.  Wait a moment for the transaction to be processed. The result will be displayed in the "Result" box, including the data hash and transaction hash (Tx Hash).

**c. Verify Material (Clinician)**

1.  In the "Clinician" section, enter the `Product ID` you just registered (e.g., `PRODUCT-A`).
2.  Click "Choose File" and select the **correct** file `product_A.csv` again.
3.  Click the "Verify Authenticity" button.
    - **Expected result**: "Verified" status in green, because the hash of the uploaded file matches the hash stored on the blockchain.
4.  Now, try verifying again with a different file (e.g., `product_B.csv`) for the same `PRODUCT-A`.
    - **Expected result**: "Failed" status in yellow/orange, because the hash of `product_B.csv` doesn't match the stored hash of `PRODUCT-A`.

Congratulations! You have successfully installed and tested the MVP system!
