// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

/**
 * @title SpectroChain
 * @dev A simple smart contract to store and verify dental material authenticity
 * based on the hash of their Raman spectroscopy data.
 */
contract SpectroChain {

    // Event to be emitted on new material registration
    event MaterialRegistered(
        string indexed productID,
        string batchID,
        string dataHash,
        address registrant,
        uint256 timestamp
    );

    // Mapping from a product ID to its data hash
    // We use productID as the unique identifier for this MVP.
    mapping(string => string) private materialHashes;

    /**
     * @dev Registers a new material by storing the hash of its spectral data.
     * The combination of productID should be unique.
     * @param _productID Unique identifier for the product (e.g., SKU, model number).
     * @param _batchID The batch number of the product.
     * @param _dataHash The SHA-256 hash of the Raman spectroscopy data file.
     */
    function registerMaterial(
        string memory _productID,
        string memory _batchID,
        string memory _dataHash
    ) public {
        // For simplicity, we allow overwriting. In a real-world scenario, you might want
        // to add a require() check to prevent this, e.g.,
        // require(bytes(materialHashes[_productID]).length == 0, "Product already registered.");

        materialHashes[_productID] = _dataHash;

        emit MaterialRegistered(
            _productID,
            _batchID,
            _dataHash,
            msg.sender,
            block.timestamp
        );
    }

    /**
     * @dev Retrieves the stored data hash for a given product ID.
     * The verification logic (comparing this stored hash with a live hash)
     * will be done off-chain by the backend.
     * @param _productID The product ID to query.
     * @return The stored data hash.
     */
    function getMaterialHash(string memory _productID) public view returns (string memory) {
        return materialHashes[_productID];
    }
} 