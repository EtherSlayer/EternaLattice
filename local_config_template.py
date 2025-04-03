
"""
Configuration settings for EternaLattice in standalone node mode
Template version - Copy to local_config.py and customize
"""

import os
import logging

# Basic configuration
DEBUG = False
NODE_VERSION = "1.0.0"

# Data storage paths
HOME_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME_DIR, ".eternalattice")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "eternalattice.db")
KEY_STORAGE_PATH = os.path.join(DATA_DIR, "keys")
LOGS_PATH = os.path.join(DATA_DIR, "logs")

# Network configuration
DEFAULT_PORT = 4444  # Default P2P network port
P2P_PROTOCOL = "tcp"
MAX_PEERS = 50
HEARTBEAT_INTERVAL = 30  # seconds
CONNECTION_TIMEOUT = 10  # seconds

# Add your seed peers here
SEED_PEERS = [
    # Format: "ip:port"
    # These would be reliable nodes maintained by the project
]

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
