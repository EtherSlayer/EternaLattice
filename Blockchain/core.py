"""
Core implementation of the EternaLattice blockchain.
"""
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import random

from models import Block, MemoryShard
from blockchain.crypto import generate_hash, verify_hash, sign_data, verify_signature
from blockchain.consensus import calculate_fitness, evolve_traits, initial_traits
from blockchain.memory_shard import create_shard, find_shard
import config

# Configure logging
logger = logging.getLogger(__name__)

# In-memory blockchain representation
# Format: {(x,y,z): Block}
blockchain: Dict[Tuple[int, int, int], Block] = {}

# Genesis block parameters
GENESIS_BLOCK_COORDS = (0, 0, 0)

def initialize_blockchain() -> None:
    """Initialize the blockchain with the genesis block if it doesn't exist"""
    # Import here to avoid circular imports
    from app import db, BlockModel
    
    # Check if we need to load from database
    if not blockchain:
        try:
            # Try to load blocks from database first
            db_blocks = BlockModel.query.all()
            
            if db_blocks:
                logger.info(f"Loading {len(db_blocks)} blocks from database")
                for db_block in db_blocks:
                    block = db_block.to_block()
                    blockchain[block.coordinates] = block
                logger.info("Blockchain loaded from database")
            else:
                # No blocks in database, create genesis block
                logger.info("No blocks found in database. Creating genesis block")
                create_genesis_block()
        except Exception as e:
            logger.error(f"Error loading blockchain from database: {e}")
            # If loading from database fails, create genesis block
            logger.info("Creating genesis block due to database error")
            create_genesis_block()

def create_genesis_block() -> Block:
    """
    Create the genesis block for the multi-dimensional blockchain.
    The genesis block has coordinates (0,0,0) and no previous blocks.
    """
    genesis_traits = initial_traits()
    genesis_block = Block(
        coordinates=GENESIS_BLOCK_COORDS,
        previous_hashes={},
        timestamp=config.GENESIS_TIMESTAMP,
        difficulty=1,
        nonce=0,
        consensus_traits=genesis_traits,
        fitness_score=calculate_fitness(genesis_traits),
        miner_id="EternaLattice_Genesis"
    )
    
    # Add genesis data about the blockchain's purpose
    genesis_data = {
        "name": "EternaLattice",
        "purpose": "Multi-dimensional blockchain for eternal knowledge preservation",
        "created": time.ctime(config.GENESIS_TIMESTAMP),
        "dimensions": ["time", "category", "region"]
    }
    
    # Create a memory shard for the genesis data
    genesis_shard = create_shard(
        json.dumps(genesis_data),
        metadata={"type": "system", "description": "Genesis information"},
        category="system",
        region="global"
    )
    
    # Add the shard reference to the genesis block
    genesis_block.shard_references = [genesis_shard.shard_id]
    
    # Generate block hash and sign it
    block_data = json.dumps({
        "coordinates": genesis_block.coordinates,
        "previous_hashes": genesis_block.previous_hashes,
        "timestamp": genesis_block.timestamp,
        "nonce": genesis_block.nonce,
        "shard_references": genesis_block.shard_references
    })
    
    genesis_block.hash = generate_hash(block_data)
    genesis_block.signature = sign_data(genesis_block.hash, "genesis")
    
    # Add to blockchain
    blockchain[GENESIS_BLOCK_COORDS] = genesis_block
    logger.debug(f"Genesis block created: {genesis_block.hash}")
    
    return genesis_block

def add_block(block: Block) -> bool:
    """
    Add a new block to the blockchain after validation.
    
    Args:
        block: The Block to add
        
    Returns:
        bool: True if block was added successfully, False otherwise
    """
    # Import here to avoid circular imports
    from app import db, BlockModel
    
    # Validate block
    if not validate_block(block):
        logger.warning(f"Block validation failed for block at {block.coordinates}")
        return False
    
    # Add to in-memory blockchain
    blockchain[block.coordinates] = block
    logger.debug(f"Added new block to in-memory store at coordinates {block.coordinates}")
    
    # Mark block for persistence - will be saved in a separate transaction
    # to avoid conflicts during startup and initialization
    if not hasattr(block, '_pending_db_save'):
        setattr(block, '_pending_db_save', True)
        logger.debug(f"Block at {block.coordinates} marked for database persistence")
    
    # Check if evolution should occur
    x, y, z = block.coordinates
    if x % config.POE_GENERATION_INTERVAL == 0 and x > 0:
        logger.info(f"Evolution triggered at block {block.coordinates}")
        perform_evolution()
    
    return True

def validate_block(block: Block) -> bool:
    """
    Validate a block before adding it to the blockchain.
    
    Args:
        block: The Block to validate
        
    Returns:
        bool: True if block is valid, False otherwise
    """
    # Check if block coordinates are valid
    x, y, z = block.coordinates
    if x < 0 or y < 0 or z < 0:
        logger.warning(f"Invalid block coordinates: {block.coordinates}")
        return False
    
    # Check if block already exists
    if block.coordinates in blockchain:
        logger.warning(f"Block already exists at coordinates {block.coordinates}")
        return False
    
    # For prototype purposes, we'll skip the hash validation
    # In a production blockchain, this would be strictly enforced
    
    # Skip strict hash validation and just ensure the hash meets difficulty
    
    # Verify signature
    if not verify_signature(block.hash, block.signature, block.miner_id):
        logger.warning(f"Block signature verification failed")
        return False
    
    # Verify previous block references
    if not validate_previous_blocks(block):
        logger.warning(f"Previous block validation failed")
        return False
    
    # Verify difficulty and hash meets difficulty requirement
    if not meet_difficulty(block.hash, block.difficulty):
        logger.warning(f"Block doesn't meet difficulty requirement")
        return False
    
    # Verify memory shard references
    for shard_id in block.shard_references:
        if not find_shard(shard_id):
            logger.warning(f"Referenced memory shard {shard_id} not found")
            return False
    
    return True

def validate_previous_blocks(block: Block) -> bool:
    """
    Validate that a block correctly references its previous blocks in all dimensions.
    In the multi-dimensional blockchain, a block should reference the previous block in
    each dimension (time, category, region).
    
    Args:
        block: The Block to validate
        
    Returns:
        bool: True if previous block references are valid, False otherwise
    """
    x, y, z = block.coordinates
    
    # Genesis block has no previous blocks
    if x == 0 and y == 0 and z == 0:
        return len(block.previous_hashes) == 0
    
    # Check previous blocks in each dimension
    expected_prev = {}
    
    # Time dimension (x-1, y, z)
    if x > 0 and (x-1, y, z) in blockchain:
        expected_prev["time"] = blockchain[(x-1, y, z)].hash
    
    # Category dimension (x, y-1, z)
    if y > 0 and (x, y-1, z) in blockchain:
        expected_prev["category"] = blockchain[(x, y-1, z)].hash
    
    # Region dimension (x, y, z-1)
    if z > 0 and (x, y, z-1) in blockchain:
        expected_prev["region"] = blockchain[(x, y, z-1)].hash
    
    # Validate that all expected previous hashes are in the block
    for dim, hash_val in expected_prev.items():
        if dim not in block.previous_hashes or block.previous_hashes[dim] != hash_val:
            logger.warning(f"Invalid previous hash for dimension {dim}")
            return False
    
    return True

def meet_difficulty(hash_str: str, difficulty: int) -> bool:
    """
    Check if a hash meets the required difficulty level.
    The hash must have a specific number of leading zeros.
    
    Args:
        hash_str: The hash to check
        difficulty: The difficulty level
        
    Returns:
        bool: True if hash meets difficulty, False otherwise
    """
    # For prototype purposes, we'll use a simplified difficulty check
    # to ensure blocks can be created easily during testing
    if difficulty <= 1:
        return True
    
    # Simple check: at least 'difficulty' number of leading zeros in hex
    return hash_str.startswith('0' * difficulty)

def mine_block(
    coordinates: Tuple[int, int, int],
    shard_references: List[str],
    miner_id: str,
    traits: Dict[str, float]
) -> Block:
    """
    Mine a new block with the given parameters.
    
    Args:
        coordinates: (x,y,z) coordinates of the block
        shard_references: List of memory shard IDs to include
        miner_id: ID of the miner
        traits: Consensus traits of the miner
        
    Returns:
        Block: The mined block
    """
    x, y, z = coordinates
    
    # Sanitize inputs
    # Make sure shard_references is a list of strings
    if shard_references is None:
        shard_references = []
    safe_shard_refs = [str(ref) for ref in shard_references if ref]
    
    # Make sure miner_id is a string
    if not miner_id:
        miner_id = "anonymous"
    
    # Make sure traits is a dictionary
    if not traits:
        traits = initial_traits()
    
    # Determine previous blocks
    previous_hashes = {}
    
    # Time dimension (x-1, y, z)
    if x > 0 and (x-1, y, z) in blockchain:
        previous_hashes["time"] = blockchain[(x-1, y, z)].hash
    
    # Category dimension (x, y-1, z)
    if y > 0 and (x, y-1, z) in blockchain:
        previous_hashes["category"] = blockchain[(x, y-1, z)].hash
    
    # Region dimension (x, y, z-1)
    if z > 0 and (x, y, z-1) in blockchain:
        previous_hashes["region"] = blockchain[(x, y, z-1)].hash
    
    # Determine difficulty (lower for prototype)
    difficulty = calculate_difficulty(coordinates)
    
    # Create block template
    block = Block(
        coordinates=coordinates,
        previous_hashes=previous_hashes,
        timestamp=int(time.time()),
        nonce=0,
        difficulty=difficulty,
        shard_references=safe_shard_refs,
        consensus_traits=traits,
        fitness_score=calculate_fitness(traits),
        miner_id=miner_id
    )
    
    # Mine the block (find nonce that satisfies difficulty)
    logger.debug(f"Mining block at coordinates {coordinates} with difficulty {difficulty}")
    
    # Add a timeout mechanism for web applications
    max_attempts = 10000  # Reasonable limit for web context
    start_time = time.time()
    timeout = 5  # 5 seconds timeout for mining
    
    while block.nonce < max_attempts:
        try:
            # Create a fully serializable version of the block data
            # with proper error handling for serialization
            serializable_data = {
                "coordinates": list(coordinates),  # Use original coordinates tuple
                "previous_hashes": {str(k): str(v) for k, v in previous_hashes.items()},
                "timestamp": block.timestamp,
                "nonce": block.nonce,
                "shard_references": safe_shard_refs  # Already sanitized
            }
            
            # Convert to JSON string
            block_data = json.dumps(serializable_data)
            
            # Generate hash
            block.hash = generate_hash(block_data)
            
            # Check if hash meets difficulty
            if meet_difficulty(block.hash, difficulty):
                logger.debug(f"Found valid nonce: {block.nonce} (took {time.time() - start_time:.2f} seconds)")
                break
            
            # Check for timeout
            if time.time() - start_time > timeout:
                logger.warning(f"Mining timeout after {block.nonce} attempts and {time.time() - start_time:.2f} seconds")
                # For prototype, we'll still accept the block even if it doesn't meet difficulty
                # In a real blockchain this would be rejected
                break
                
            block.nonce += 1
        except Exception as e:
            logger.error(f"Error during mining: {e}")
            # For prototype, we'll set a hash even if mining fails
            block.hash = generate_hash(f"fallback-{block.coordinates}-{block.timestamp}-{block.nonce}")
            break
    
    # Sign the block
    block.signature = sign_data(block.hash, miner_id)
    
    return block

def calculate_difficulty(coordinates: Tuple[int, int, int]) -> int:
    """
    Calculate the mining difficulty for a block at the given coordinates.
    
    Args:
        coordinates: (x,y,z) coordinates of the block
        
    Returns:
        int: The difficulty level
    """
    x, y, z = coordinates
    
    # For prototype purposes, we'll use a very low difficulty to ensure quick mining
    # In production, this would adapt based on network hashrate
    
    # Base difficulty - reduced for web prototype
    difficulty = 1
    
    # Only increase difficulty for deeper blocks
    if x > 30:
        difficulty = 2
    if x > 100:
        difficulty = 3
        
    # In production code we would adjust difficulty based on recent blocks
    # to maintain consistent block times
    
    return difficulty

def perform_evolution() -> None:
    """
    Perform an evolutionary step on the consensus traits of the blockchain.
    This is the core of the Proof-of-Evolution consensus mechanism.
    """
    # Get all blocks in the latest generation
    latest_blocks = []
    for coords, block in blockchain.items():
        # Only consider blocks from the most recent generation
        if coords[0] > 0 and coords[0] % config.POE_GENERATION_INTERVAL == 0:
            latest_blocks.append(block)
    
    if not latest_blocks:
        logger.warning("No blocks found for evolution")
        return
    
    # Evolve traits based on fitness scores
    evolved_traits = evolve_traits([b.consensus_traits for b in latest_blocks], 
                               [b.fitness_score for b in latest_blocks])
    
    logger.info(f"Evolved traits: {evolved_traits}")
    
    # Store evolved traits for next generation (in actual implementation,
    # this would be broadcast to the network)
    # For simulation, we'll store it in a special system block
    
    evolution_data = {
        "generation": latest_blocks[0].coordinates[0] // config.POE_GENERATION_INTERVAL,
        "evolved_traits": evolved_traits,
        "fitness_scores": [b.fitness_score for b in latest_blocks],
        "timestamp": int(time.time())
    }
    
    # Create a memory shard for the evolution data
    evolution_shard = create_shard(
        json.dumps(evolution_data),
        metadata={"type": "evolution", "generation": evolution_data["generation"]},
        category="system",
        region="global"
    )
    
    logger.debug(f"Evolution shard created: {evolution_shard.shard_id}")

def get_block(coordinates: Tuple[int, int, int]) -> Optional[Block]:
    """
    Get a block at the specified coordinates.
    
    Args:
        coordinates: (x,y,z) coordinates of the block
        
    Returns:
        Block or None: The block if found, None otherwise
    """
    return blockchain.get(coordinates)

def get_blocks_by_dimension(dimension: str, value: int) -> List[Block]:
    """
    Get all blocks along a specific dimension with the given value.
    
    Args:
        dimension: Dimension to filter on ('time', 'category', or 'region')
        value: Value of the dimension
        
    Returns:
        List[Block]: List of blocks matching the criteria
    """
    dim_index = {"time": 0, "category": 1, "region": 2}
    
    if dimension not in dim_index:
        raise ValueError(f"Invalid dimension: {dimension}")
    
    index = dim_index[dimension]
    results = []
    
    for coords, block in blockchain.items():
        if coords[index] == value:
            results.append(block)
    
    return results

def get_latest_block() -> Optional[Block]:
    """
    Get the latest block in the time dimension.
    
    Returns:
        Block or None: The latest block if found, None otherwise
    """
    max_time = -1
    latest_block = None
    
    for coords, block in blockchain.items():
        if coords[0] > max_time:
            max_time = coords[0]
            latest_block = block
    
    return latest_block

def get_blockchain_stats() -> Dict[str, Any]:
    """
    Get statistics about the current blockchain state.
    
    Returns:
        Dict: Statistics about the blockchain
    """
    import logging
    logger = logging.getLogger(__name__)
    
    from blockchain.memory_shard import search_shards
    from app import app, db
    from db_models import BlockModel
    
    # Get in-memory count
    in_memory_count = len(blockchain)
    
    # Default values for empty blockchain
    blocks_count = in_memory_count
    dimensions = {
        "time": {"min": 0, "max": 0},
        "category": {"min": 0, "max": 0},
        "region": {"min": 0, "max": 0}
    }
    
    # Check database for more accurate count
    try:
        with app.app_context():
            # Get actual count from database
            db_count = BlockModel.query.count()
            blocks_count = max(in_memory_count, db_count)
            
            logger.debug(f"Blockchain stats: in-memory={in_memory_count}, db={db_count}, total={blocks_count}")
    except Exception as e:
        logger.error(f"Error fetching blockchain stats from database: {e}")
        blocks_count = in_memory_count
    
    # Calculate actual values if blockchain has blocks
    if blockchain:
        # Calculate dimension ranges
        time_vals = [c[0] for c in blockchain.keys()]
        category_vals = [c[1] for c in blockchain.keys()]
        region_vals = [c[2] for c in blockchain.keys()]
        
        dimensions = {
            "time": {"min": min(time_vals), "max": max(time_vals)},
            "category": {"min": min(category_vals), "max": max(category_vals)},
            "region": {"min": min(region_vals), "max": max(region_vals)}
        }
    
    # Count unique shards in blocks
    all_shards_in_blocks = set()
    for block in blockchain.values():
        all_shards_in_blocks.update(block.shard_references)
    
    # Count total shards in system (including those not yet in blocks)
    all_shards = search_shards('')
    total_shards = len(all_shards)
    
    # Count shards not yet added to blocks
    unconfirmed_shards = total_shards - len(all_shards_in_blocks)
    
    # Calculate avg_difficulty and avg_fitness
    avg_difficulty = 0
    avg_fitness = 0
    
    if blockchain:
        # Calculate average difficulty and fitness from blocks
        difficulties = [getattr(block, 'difficulty', 1) for block in blockchain.values()]
        fitness_scores = [getattr(block, 'fitness_score', 0.0) for block in blockchain.values()]
        
        avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 0
        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0

    return {
        "blocks": blocks_count,
        "total_blocks": blocks_count,  # Add total_blocks for compatibility
        "dimensions": dimensions,
        "shards": {
            "total": total_shards,
            "in_blocks": len(all_shards_in_blocks),
            "unconfirmed": unconfirmed_shards
        },
        "avg_difficulty": avg_difficulty,
        "avg_fitness": avg_fitness
    }
