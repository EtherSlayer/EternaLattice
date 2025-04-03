"""
Network implementation for the EternaLattice blockchain.
Handles peer discovery, communication, and synchronization.
"""
import logging
import time
import json
import random
from typing import Dict, List, Any, Optional
import threading
import uuid

from models import Node, Block, MemoryShard
from blockchain.crypto import (
    generate_key_pair, sign_data, verify_signature, 
    encrypt_data, decrypt_data
)
import config

logger = logging.getLogger(__name__)

# In-memory storage for nodes
nodes: Dict[str, Node] = {}

# This node's ID
local_node_id = str(uuid.uuid4())

# Flag to indicate if the network is running
network_running = False

# Network start time
network_start_time = None

def initialize_network() -> str:
    """
    Initialize the network and generate a key pair for this node.
    
    Returns:
        str: This node's ID
    """
    global local_node_id
    
    # Generate key pair
    public_key, _ = generate_key_pair(local_node_id)
    
    # Create node object
    node = Node(
        node_id=local_node_id,
        public_key=public_key,
        address="127.0.0.1",  # For simulation
        port=8000,  # For simulation
        traits={}  # Will be populated later
    )
    
    # Add to nodes dictionary
    nodes[local_node_id] = node
    logger.info(f"Initialized local node with ID: {local_node_id}")
    
    return local_node_id

def add_node(node: Node) -> bool:
    """
    Add a node to the network.
    
    Args:
        node: The node to add
        
    Returns:
        bool: True if the node was added, False otherwise
    """
    if node.node_id in nodes:
        # Update existing node
        nodes[node.node_id] = node
        logger.debug(f"Updated node: {node.node_id}")
        return True
    
    # Add new node
    nodes[node.node_id] = node
    logger.info(f"Added new node: {node.node_id}")
    return True

def remove_node(node_id: str) -> bool:
    """
    Remove a node from the network.
    
    Args:
        node_id: The ID of the node to remove
        
    Returns:
        bool: True if the node was removed, False otherwise
    """
    if node_id in nodes:
        del nodes[node_id]
        logger.info(f"Removed node: {node_id}")
        return True
    
    logger.warning(f"Attempted to remove non-existent node: {node_id}")
    return False

def get_node(node_id: str) -> Optional[Node]:
    """
    Get a node by its ID.
    
    Args:
        node_id: The ID of the node
        
    Returns:
        Node or None: The node if found, None otherwise
    """
    return nodes.get(node_id)

def get_all_nodes() -> List[Node]:
    """
    Get all nodes in the network.
    
    Returns:
        List[Node]: List of all nodes
    """
    return list(nodes.values())

def broadcast_block(block: Block) -> bool:
    """
    Broadcast a block to all nodes in the network.
    
    Args:
        block: The block to broadcast
        
    Returns:
        bool: True if the broadcast succeeded, False otherwise
    """
    # In a real implementation, this would send the block to all peers
    # For this prototype, we'll simulate broadcasting
    
    logger.info(f"Broadcasting block: {block.hash}")
    
    # Simulate network delay
    time.sleep(0.1)
    
    # Simulate random failures
    if random.random() < 0.05:
        logger.warning("Simulated network failure during broadcast")
        return False
    
    # In a real implementation, we would sign the message and send it to peers
    # For the prototype, we'll just log that it was "sent"
    for node_id, node in nodes.items():
        if node_id != local_node_id:
            logger.debug(f"Simulating send of block {block.hash} to node {node_id}")
    
    return True

def broadcast_shard(shard: MemoryShard) -> bool:
    """
    Broadcast a memory shard to all nodes in the network.
    
    Args:
        shard: The shard to broadcast
        
    Returns:
        bool: True if the broadcast succeeded, False otherwise
    """
    # In a real implementation, this would send the shard to all peers
    # For this prototype, we'll simulate broadcasting
    
    logger.info(f"Broadcasting memory shard: {shard.shard_id}")
    
    # Simulate network delay
    time.sleep(0.1)
    
    # Simulate random failures
    if random.random() < 0.05:
        logger.warning("Simulated network failure during broadcast")
        return False
    
    # In a real implementation, we would sign the message and send it to peers
    # For the prototype, we'll just log that it was "sent"
    for node_id, node in nodes.items():
        if node_id != local_node_id:
            logger.debug(f"Simulating send of shard {shard.shard_id} to node {node_id}")
    
    return True

def request_block(block_hash: str) -> Optional[Block]:
    """
    Request a block from the network.
    
    Args:
        block_hash: The hash of the block to request
        
    Returns:
        Block or None: The requested block if found, None otherwise
    """
    # In a real implementation, this would send a request to peers
    # For this prototype, we'll simulate the request
    
    logger.info(f"Requesting block: {block_hash}")
    
    # Simulate network delay
    time.sleep(0.2)
    
    # Simulate random failures
    if random.random() < 0.1:
        logger.warning("Simulated network failure during block request")
        return None
    
    # Simulate that the block was not found
    logger.debug(f"Block {block_hash} not found in the network")
    return None

def request_shard(shard_id: str) -> Optional[MemoryShard]:
    """
    Request a memory shard from the network.
    
    Args:
        shard_id: The ID of the shard to request
        
    Returns:
        MemoryShard or None: The requested shard if found, None otherwise
    """
    # In a real implementation, this would send a request to peers
    # For this prototype, we'll simulate the request
    
    logger.info(f"Requesting memory shard: {shard_id}")
    
    # Simulate network delay
    time.sleep(0.2)
    
    # Simulate random failures
    if random.random() < 0.1:
        logger.warning("Simulated network failure during shard request")
        return None
    
    # Simulate that the shard was not found
    logger.debug(f"Shard {shard_id} not found in the network")
    return None

def heartbeat() -> None:
    """
    Send a heartbeat to all nodes to indicate that this node is still alive.
    """
    global network_running
    
    while network_running:
        try:
            for node_id, node in list(nodes.items()):
                # Skip self
                if node_id == local_node_id:
                    continue
                
                # Simulate heartbeat
                logger.debug(f"Sending heartbeat to node {node_id}")
                
                # Simulate random failures
                if random.random() < 0.1:
                    logger.warning(f"Simulated heartbeat failure to node {node_id}")
                    
                    # Check if the node has been offline for too long
                    if time.time() - node.last_seen > config.HEARTBEAT_INTERVAL * 3:
                        # Remove the node
                        logger.warning(f"Node {node_id} has been offline for too long, removing")
                        remove_node(node_id)
                    continue
                
                # Update last seen
                node.last_seen = int(time.time())
                nodes[node_id] = node
            
            # Wait for next heartbeat
            time.sleep(config.HEARTBEAT_INTERVAL)
        except Exception as e:
            logger.error(f"Error in heartbeat: {e}")
            time.sleep(config.HEARTBEAT_INTERVAL)

def start_network() -> None:
    """
    Start the network services.
    """
    global network_running, network_start_time
    
    if network_running:
        logger.warning("Network already running")
        return
    
    # Initialize the network
    initialize_network()
    
    # Start the heartbeat thread
    network_running = True
    network_start_time = time.time()  # Set the network start time
    heartbeat_thread = threading.Thread(target=heartbeat)
    heartbeat_thread.daemon = True
    heartbeat_thread.start()
    
    logger.info("Network started")

def stop_network() -> None:
    """
    Stop the network services.
    """
    global network_running, network_start_time
    
    if not network_running:
        logger.warning("Network already stopped")
        return
    
    # Stop the heartbeat thread
    network_running = False
    network_start_time = None  # Reset the network start time
    
    logger.info("Network stopped")

def get_network_stats() -> Dict[str, Any]:
    """
    Get statistics about the current network state.
    
    Returns:
        Dict: Statistics about the network
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Get current time for activity check
    current_time = time.time()
    
    # A node is considered active if it was seen within 3x the heartbeat interval
    active_time_threshold = current_time - (config.HEARTBEAT_INTERVAL * 3)
    
    # Filter active nodes based on last_seen timestamp
    active_nodes = [node for node in nodes.values() 
                   if node.last_seen > active_time_threshold]
    
    # Get node roles if available 
    node_roles = {}
    for node_id, node in nodes.items():
        if hasattr(node, 'traits') and node.traits:
            role = "Unknown"
            # Determine role based on traits
            adaptability = node.traits.get('adaptability', 0.5)
            specialization = node.traits.get('specialization', 0.5)
            
            if adaptability > 0.7:
                if specialization > 0.7:
                    role = "Explorer"
                else:
                    role = "Connector"
            else:
                if specialization > 0.7:
                    role = "Guardian"
                else:
                    role = "Validator"
                    
            node_roles[node_id] = role
    
    # Make sure the local node is always considered active
    # This ensures we always report at least 1 active node
    has_local_node = any(getattr(node, 'node_id', '') == local_node_id for node in active_nodes)
    
    # If local node isn't in active nodes list and exists, adjust the count
    if not has_local_node and local_node_id:
        active_node_count = len(active_nodes) + 1
    else:
        active_node_count = len(active_nodes)
    
    # Create enhanced stats - total_nodes should be at least 1 (the local node)
    stats = {
        "total_nodes": max(1, len(nodes)),
        "active_nodes": active_node_count,
        "local_node_id": local_node_id,
        "network_running": network_running,
        "heartbeat_interval": config.HEARTBEAT_INTERVAL,
        "max_peers": config.MAX_PEERS,
        "network_uptime": current_time - network_start_time if network_start_time else 0,
        "roles_distribution": node_roles
    }
    
    # Log network status
    logger.debug(f"Network stats: {active_node_count}/{max(1, len(nodes))} nodes active")
    
    return stats
