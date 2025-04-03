"""
Configuration settings for EternaLattice genesis node
"""

import os
import logging

# Basic configuration
DEBUG = False
NODE_VERSION = "1.0.0"

# Data storage paths
HOME_DIR = os.path.expanduser("~")
DATA_DIR = r"C:\Users\lukeh\.eternalattice"
SQLITE_DB_PATH = os.path.join(DATA_DIR, "eternalattice.db")
KEY_STORAGE_PATH = os.path.join(DATA_DIR, "keys")
LOGS_PATH = os.path.join(DATA_DIR, "logs")

# Network configuration
DEFAULT_PORT = 5001  # Default P2P network port
P2P_PROTOCOL = "tcp"
MAX_PEERS = 50
HEARTBEAT_INTERVAL = 30  # seconds
CONNECTION_TIMEOUT = 10  # seconds

# Genesis node specific network configuration
PORT = 5000  # Port for the web interface
NODE_ID = "genesis_node"
IP_ADDRESS = "0.0.0.0"
INITIAL_PEERS = []  # Empty for the genesis node


# Consensus parameters
INITIAL_DIFFICULTY = 1
DIFFICULTY_ADJUSTMENT_FACTOR = 1.1
TARGET_BLOCK_TIME = 60  # seconds
MAX_BLOCK_SIZE = 1024 * 1024  # 1 MB

# Initial consensus traits - base values for the decentralized network
CONSENSUS_TRAITS = {
    "mutation_rate": 0.05,
    "crossover_rate": 0.7,
    "selection_pressure": 0.8,
    "novelty_preference": 0.3,
    "cooperation_factor": 0.7,
    "convergence_threshold": 0.1,
    "adaptation_rate": 0.2,
    "diversity_weight": 0.5,
    "resilience_factor": 0.6
}

# Consensus traits bounds
TRAIT_BOUNDS = {
    "mutation_rate": (0.01, 0.2),
    "crossover_rate": (0.5, 0.9),
    "selection_pressure": (0.2, 1.0),
    "novelty_preference": (0.1, 0.9),
    "cooperation_factor": (0.2, 1.0),
    "convergence_threshold": (0.05, 0.5),
    "adaptation_rate": (0.05, 0.5),
    "diversity_weight": (0.2, 0.8),
    "resilience_factor": (0.3, 0.9)
}

# Memory shard replication settings
SHARD_REPLICATION_FACTOR = 5  # Number of nodes that should keep a copy of each shard
SHARD_REPLICATION_INTERVAL = 3600  # 1 hour between replication attempts

# Security settings
HASH_ALGORITHM = "sha3_256"  # Quantum-resistant
SIGNATURE_ALGORITHM = "ed25519"  # Fast and secure for signing
KEY_SIZE = 256  # bits

# Reputation system parameters
POINTS_FOR_MINING = 20
POINTS_FOR_SHARD_CREATION = 10
POINTS_FOR_SHARD_PRESERVATION = 5
POINTS_FOR_NETWORK_CONTRIBUTION = 2
LEVEL_MULTIPLIER = 100  # Points needed for level N = N² * LEVEL_MULTIPLIER

# Seed peers - hardcoded list of reliable nodes to bootstrap the network
SEED_PEERS = [
    "0.0.0.0:5000",  # Primary seed node
    "0.0.0.0:5001",  # Backup seed node
    "0.0.0.0:5002"   # Backup seed node
]

# Mining configuration
BLOCK_REWARD = 50  # Initial block reward
REWARD_HALVING_INTERVAL = 210000  # Number of blocks between reward halving
INITIAL_MINING_DIFFICULTY = 4  # Number of leading zeros required
DIFFICULTY_ADJUSTMENT_INTERVAL = 2016  # Blocks between difficulty adjustments
TARGET_BLOCK_TIME = 600  # Target time between blocks (10 minutes)

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(KEY_STORAGE_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_PATH, "eternalattice.log"))
    ]
)
