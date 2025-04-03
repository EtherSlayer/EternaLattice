"""
Cryptographic functionality for the EternaLattice blockchain.
Implements hybrid cryptography combining SHA-3 and Blake2 hashing with
Kyber-768 post-quantum signatures.
"""
import hashlib
import hmac
import os
import json
import logging
from typing import Dict, Any, Tuple
import random

# In a production environment, we would use actual post-quantum libraries
# For this prototype, we'll simulate the behavior of these algorithms

logger = logging.getLogger(__name__)

# Dictionary to store key pairs for the simulation
# In real implementation, these would be securely stored
key_pairs: Dict[str, Dict[str, Any]] = {}

def generate_key_pair(node_id: str) -> Tuple[str, str]:
    """
    Generate a simulated quantum-resistant key pair.
    
    Args:
        node_id: ID of the node
        
    Returns:
        Tuple[str, str]: (public_key, private_key)
    """
    # In a real implementation, this would use a post-quantum algorithm like Kyber
    # For the prototype, we'll generate random strings as keys
    private_key = hashlib.sha3_256(os.urandom(32)).hexdigest()
    public_key = hashlib.sha3_256((private_key + node_id).encode()).hexdigest()
    
    # Store the key pair
    key_pairs[node_id] = {
        "public_key": public_key,
        "private_key": private_key
    }
    
    return public_key, private_key

def get_public_key(node_id: str) -> str:
    """
    Get the public key for a node.
    
    Args:
        node_id: ID of the node
        
    Returns:
        str: Public key
    """
    if node_id not in key_pairs:
        # For the prototype, generate a key pair if it doesn't exist
        generate_key_pair(node_id)
    
    return key_pairs[node_id]["public_key"]

def generate_hash(data: Any) -> str:
    """
    Generate a hybrid hash using SHA3-256 and BLAKE2b.
    
    Args:
        data: Data to hash (can be any type that can be converted to string)
        
    Returns:
        str: Hexadecimal hash string
    """
    # Ensure data is a string
    if not isinstance(data, str):
        data = str(data)
    
    # Primary hash with SHA3-256
    primary = hashlib.sha3_256(data.encode('utf-8', errors='replace')).digest()
    
    # Secondary hash with BLAKE2b
    secondary = hashlib.blake2b(primary).hexdigest()
    
    return secondary

def hash_data(data: Any) -> str:
    """
    Alias for generate_hash to maintain compatibility with other modules.
    
    Args:
        data: Data to hash (can be any type that can be converted to string)
        
    Returns:
        str: Hexadecimal hash string
    """
    return generate_hash(data)

def verify_hash(data: Any, hash_value: str) -> bool:
    """
    Verify that a hash matches the data.
    
    Args:
        data: Data to verify (can be any type that can be converted to string)
        hash_value: Hash to check against
        
    Returns:
        bool: True if hash matches data, False otherwise
    """
    computed_hash = generate_hash(data)
    return computed_hash == hash_value

def sign_data(data: Any, node_id: str) -> str:
    """
    Sign data using the node's private key.
    
    Args:
        data: Data to sign (can be any type that can be converted to string)
        node_id: ID of the signing node
        
    Returns:
        str: Signature string
    """
    if node_id not in key_pairs:
        # For the prototype, generate a key pair if it doesn't exist
        generate_key_pair(node_id)
    
    private_key = key_pairs[node_id]["private_key"]
    
    # Ensure data is a string
    if not isinstance(data, str):
        data = str(data)
    
    # In a real implementation, this would use a post-quantum signature algorithm
    # For the prototype, we'll use HMAC with SHA3-256
    signature = hmac.new(
        private_key.encode(),
        data.encode('utf-8', errors='replace'),
        hashlib.sha3_256
    ).hexdigest()
    
    return signature

def verify_signature(data: Any, signature: str, node_id: str) -> bool:
    """
    Verify a signature using the node's public key.
    
    Args:
        data: Data that was signed (can be any type that can be converted to string)
        signature: Signature to verify
        node_id: ID of the node
        
    Returns:
        bool: True if signature is valid, False otherwise
    """
    # Special case for genesis block
    if node_id == "EternaLattice_Genesis":
        # For the prototype, we'll accept any signature for the genesis block
        return True
    
    if node_id not in key_pairs:
        logger.warning(f"No key pair found for node {node_id}")
        return False
    
    private_key = key_pairs[node_id]["private_key"]
    
    # Ensure data is a string
    if not isinstance(data, str):
        data = str(data)
    
    # Calculate expected signature
    expected_signature = hmac.new(
        private_key.encode(),
        data.encode('utf-8', errors='replace'),
        hashlib.sha3_256
    ).hexdigest()
    
    return signature == expected_signature

def encrypt_data(data: Any, recipient_id: str) -> str:
    """
    Encrypt data for a specific recipient.
    
    Args:
        data: Data to encrypt (can be any type that can be converted to string)
        recipient_id: ID of the recipient
        
    Returns:
        str: Encrypted data
    """
    # In a real implementation, this would use hybrid encryption with a post-quantum algorithm
    # For the prototype, we'll simulate encryption with a simple transformation
    
    if recipient_id not in key_pairs:
        generate_key_pair(recipient_id)
    
    public_key = key_pairs[recipient_id]["public_key"]
    
    # Ensure data is a string
    if not isinstance(data, str):
        data = str(data)
    
    # Simple XOR "encryption" for demonstration
    # This is NOT secure and only for simulation purposes
    key_bytes = bytes.fromhex(public_key[:32])
    data_bytes = data.encode('utf-8', errors='replace')
    encrypted = bytearray()
    
    for i in range(len(data_bytes)):
        encrypted.append(data_bytes[i] ^ key_bytes[i % len(key_bytes)])
    
    return encrypted.hex()

def decrypt_data(encrypted_data: str, node_id: str) -> str:
    """
    Decrypt data using the node's private key.
    
    Args:
        encrypted_data: Data to decrypt
        node_id: ID of the node
        
    Returns:
        str: Decrypted data
    """
    # In a real implementation, this would use hybrid decryption with a post-quantum algorithm
    # For the prototype, we'll simulate decryption with a simple transformation
    
    if node_id not in key_pairs:
        logger.warning(f"No key pair found for node {node_id}")
        return ""
    
    # Validate encrypted_data is a properly formatted hex string
    try:
        encrypted_bytes = bytes.fromhex(encrypted_data)
    except ValueError:
        logger.warning(f"Invalid encrypted data format: {encrypted_data}")
        return f"[Error: Could not decrypt data - invalid format]"
    
    public_key = key_pairs[node_id]["public_key"]
    
    # Simple XOR "decryption" for demonstration
    # This is NOT secure and only for simulation purposes
    key_bytes = bytes.fromhex(public_key[:32])
    decrypted = bytearray()
    
    for i in range(len(encrypted_bytes)):
        decrypted.append(encrypted_bytes[i] ^ key_bytes[i % len(key_bytes)])
    
    # Handle potential decoding errors
    try:
        return decrypted.decode('utf-8', errors='replace')
    except Exception as e:
        logger.warning(f"Error decoding decrypted data: {e}")
        return f"[Error: Data corrupted during decryption]"
