from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Import utility and blockchain functions
try:
    # Relative imports for when running with uvicorn
    from . import utils
    from . import blockchain
except ImportError:
    # Absolute imports for when running directly
    import utils
    import blockchain

# --- FastAPI App Initialization ---
app = FastAPI(title="SpectroChain Dental MVP")

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Mount static files directory
# This allows serving CSS, JS files
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR.parent / "static"),
    name="static",
)

# Setup Jinja2 templates
templates = Jinja2Templates(directory=BASE_DIR.parent / "templates")

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """
    On startup, load the smart contract ABI.
    """
    print("Application startup...")
    blockchain.load_contract_abi()
    # A simple check to see if we are connected to Ganache
    if not blockchain.w3.is_connected():
        print("FATAL: Cannot connect to Ganache. Please ensure it is running and accessible.")
    else:
        print(f"Connected to blockchain. Using account: {blockchain.w3.eth.default_account}")


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/register")
async def register_material(
    productID: str = Form(...),
    batchID: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Endpoint to register a new material.
    1. Reads the uploaded .csv file.
    2. Calculates its SHA-256 hash.
    3. Calls the smart contract to store the hash.
    """
    # Read file content
    file_content = await file.read()
    
    # Calculate hash
    data_hash = utils.calculate_hash(file_content)
    
    print(f"Registering: ProductID={productID}, BatchID={batchID}, Hash={data_hash}")
    
    # Send to blockchain
    try:
        result = blockchain.register_material_on_chain(productID, batchID, data_hash)
        if result.get("status") == "success":
            return {
                "message": "Material registered successfully!",
                "productID": productID,
                "batchID": batchID,
                "hash": data_hash,
                "transaction_hash": result.get("tx_hash")
            }
        else:
            raise HTTPException(status_code=500, detail=f"Blockchain transaction failed: {result.get('message')}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify")
async def verify_material(
    productID: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Endpoint to verify a material.
    1. Reads the uploaded .csv file to get a 'live' hash.
    2. Retrieves the stored hash from the blockchain for the given productID.
    3. Compares the two hashes.
    """
    # Read file content and calculate "live" hash
    file_content = await file.read()
    live_hash = utils.calculate_hash(file_content)
    
    print(f"Verifying: ProductID={productID}, Live Hash={live_hash}")
    
    try:
        # Get stored hash from blockchain
        stored_hash = blockchain.get_hash_from_chain(productID)
        
        if not stored_hash:
            return {
                "status": "Failed",
                "message": f"No material found on the blockchain for Product ID: {productID}",
                "productID": productID,
                "liveHash": live_hash,
                "storedHash": None
            }
            
        # Compare hashes
        if live_hash == stored_hash:
            return {
                "status": "Verified",
                "message": "The material is authentic. Live hash matches the stored hash.",
                "productID": productID,
                "liveHash": live_hash,
                "storedHash": stored_hash
            }
        else:
            return {
                "status": "Failed",
                "message": "Verification failed. The material's data does not match the record on the blockchain.",
                "productID": productID,
                "liveHash": live_hash,
                "storedHash": stored_hash
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

# --- Main block for direct execution ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8080, reload=True)