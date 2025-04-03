"""
EternaLattice blockchain module.
"""
from .models import Block, MemoryShard
from . import core, crypto, consensus, memory_shard, network, visualization

# Export key functions for easier access
from blockchain.core import (
    initialize_blockchain, create_genesis_block, add_block, mine_block,
    get_block, get_blocks_by_dimension, get_latest_block, get_blockchain_stats
)
from blockchain.crypto import (
    generate_key_pair, generate_hash, hash_data, sign_data, verify_signature,
    encrypt_data, decrypt_data
)
from blockchain.consensus import (
    initial_traits, calculate_fitness, evolve_traits, get_consensus_state
)
from blockchain.memory_shard import (
    create_shard, find_shard, get_shard_data, search_shards,
    get_categories, get_regions, get_shard_stats
)
from blockchain.network import (
    initialize_network, start_network, stop_network, broadcast_block,
    broadcast_shard, get_network_stats
)
from blockchain.visualization import (
    generate_3d_lattice_data, generate_2d_projection,
    generate_consensus_evolution_chart, generate_fitness_landscape,
    generate_blockchain_stats_chart
)
