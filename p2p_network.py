"""
P2P Networking module for EternaLattice in decentralized mode.
This module handles direct node-to-node communications without a central server.
"""

import asyncio
import json
import logging
import random
import socket
import struct
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from uuid import uuid4

from local_config import (
    DEFAULT_PORT, P2P_PROTOCOL, MAX_PEERS, 
    HEARTBEAT_INTERVAL, CONNECTION_TIMEOUT, SEED_PEERS
)

logger = logging.getLogger('eternalattice.p2p')

# Node information
local_node_id = str(uuid4())
known_peers = {}  # node_id -> {node_id, address, port, last_seen, traits, ...}
connections = {}  # connection_id -> {socket, reader, writer, node_id, ...}
message_handlers = {}  # message_type -> handler_function
running = False
server = None

# Peer management and security tracking
banned_peers = {}  # node_id -> {reason, timestamp}
peer_reliability = {}  # node_id -> {success_count, failure_count, last_failure, reputation}
peer_gossip_cache = {}  # Store recently gossiped peer information to prevent flooding
malicious_activity = {}  # Track potentially malicious activities by peers

# Message types
MESSAGE_TYPES = {
    'HELLO': 0,
    'HEARTBEAT': 1,
    'PEER_LIST': 2,
    'BLOCK': 3,
    'SHARD': 4,
    'BLOCK_REQUEST': 5,
    'SHARD_REQUEST': 6,
    'SEARCH_QUERY': 7,
    'SEARCH_RESULT': 8,
    'CONSENSUS_UPDATE': 9,
    'REPUTATION_UPDATE': 10,
    'PEER_GOSSIP': 11,   # Gossiping about other peers (share information)
    'PEER_RECOMMEND': 12, # Peer recommendation for reliable nodes
    'MALICIOUS_REPORT': 13, # Report a potentially malicious peer
    'NETWORK_STATUS': 14, # Network-wide status information
    'GOODBYE': 99
}

# Message header format (binary)
# | Magic (4 bytes) | Message Type (1 byte) | Payload Length (4 bytes) | Node ID (36 bytes) |
HEADER_FORMAT = '!4sBI36s'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAGIC = b'ETRN'  # EteRNalattice magic bytes

class P2PError(Exception):
    """Base exception for P2P networking errors."""
    pass

class ConnectionError(P2PError):
    """Exception raised for connection errors."""
    pass

class MessageError(P2PError):
    """Exception raised for message formatting errors."""
    pass

async def handle_connection(reader, writer):
    """Handle a new incoming connection."""
    addr = writer.get_extra_info('peername')
    logger.info(f"New connection from {addr}")
    
    connection_id = str(uuid4())
    connections[connection_id] = {
        'reader': reader,
        'writer': writer,
        'address': addr[0],
        'port': addr[1],
        'node_id': None,
        'last_activity': time.time()
    }
    
    try:
        # Read and process messages until connection closes
        while True:
            header_data = await reader.readexactly(HEADER_SIZE)
            magic, msg_type, payload_len, node_id = struct.unpack(HEADER_FORMAT, header_data)
            
            if magic != MAGIC:
                logger.warning(f"Invalid magic bytes from {addr}")
                break
            
            node_id = node_id.decode('utf-8')
            connections[connection_id]['node_id'] = node_id
            connections[connection_id]['last_activity'] = time.time()
            
            if payload_len > 0:
                payload = await reader.readexactly(payload_len)
                payload = json.loads(payload.decode('utf-8'))
            else:
                payload = {}
            
            # Handle the message based on its type
            msg_type_name = next((name for name, code in MESSAGE_TYPES.items() if code == msg_type), 'UNKNOWN')
            logger.debug(f"Received {msg_type_name} message from {node_id[:8]}...")
            
            if msg_type in message_handlers:
                await message_handlers[msg_type](node_id, payload, connection_id)
            else:
                logger.warning(f"No handler for message type {msg_type}")
            
            # Update peer information
            if node_id in known_peers:
                known_peers[node_id]['last_seen'] = time.time()
            
            # Handle GOODBYE messages
            if msg_type == MESSAGE_TYPES['GOODBYE']:
                logger.info(f"Peer {node_id[:8]}... has disconnected")
                break
                
    except asyncio.IncompleteReadError:
        logger.info(f"Connection closed by peer {addr}")
    except Exception as e:
        logger.error(f"Error handling connection: {e}")
    finally:
        writer.close()
        if connection_id in connections:
            del connections[connection_id]

async def send_message(node_id, msg_type, payload=None, connection_id=None):
    """Send a message to a specific node."""
    if payload is None:
        payload = {}
    
    # Find the connection
    if connection_id and connection_id in connections:
        conn = connections[connection_id]
    else:
        # Find a connection for this node_id
        conn = None
        for cid, c in connections.items():
            if c.get('node_id') == node_id:
                conn = c
                break
        
        if not conn:
            # No existing connection, try to establish one
            if node_id in known_peers:
                peer = known_peers[node_id]
                address = peer['address']
                port = peer['port']
                
                try:
                    reader, writer = await asyncio.open_connection(address, port)
                    cid = str(uuid4())
                    conn = {
                        'reader': reader,
                        'writer': writer,
                        'address': address,
                        'port': port,
                        'node_id': node_id,
                        'last_activity': time.time()
                    }
                    connections[cid] = conn
                    
                    # Start a task to handle incoming messages on this connection
                    asyncio.create_task(handle_connection(reader, writer))
                except Exception as e:
                    logger.error(f"Failed to connect to peer {node_id[:8]}...: {e}")
                    return False
            else:
                logger.error(f"No known peer with ID {node_id[:8]}...")
                return False
    
    try:
        # Serialize payload
        payload_bytes = json.dumps(payload).encode('utf-8')
        
        # Create header
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            msg_type,
            len(payload_bytes),
            local_node_id.encode('utf-8')
        )
        
        # Send header + payload
        conn['writer'].write(header + payload_bytes)
        await conn['writer'].drain()
        
        conn['last_activity'] = time.time()
        return True
    except Exception as e:
        logger.error(f"Error sending message to {node_id[:8]}...: {e}")
        return False

async def broadcast_message(msg_type, payload=None, exclude_nodes=None):
    """Broadcast a message to all connected peers."""
    if payload is None:
        payload = {}
    
    if exclude_nodes is None:
        exclude_nodes = []
    
    success_count = 0
    
    # Send to each connected peer
    for node_id in list(known_peers.keys()):
        if node_id not in exclude_nodes:
            if await send_message(node_id, msg_type, payload):
                success_count += 1
    
    return success_count

async def handle_hello(node_id, payload, connection_id):
    """Handle a HELLO message from a peer."""
    address = payload.get('address')
    port = payload.get('port')
    traits = payload.get('traits', {})
    
    # Update known_peers with this node's information
    known_peers[node_id] = {
        'node_id': node_id,
        'address': address,
        'port': port,
        'last_seen': time.time(),
        'traits': traits,
        'connection_id': connection_id
    }
    
    logger.info(f"Added peer {node_id[:8]}... at {address}:{port}")
    
    # Respond with our own HELLO
    await send_message(
        node_id, 
        MESSAGE_TYPES['HELLO'],
        {
            'address': get_local_ip(),
            'port': DEFAULT_PORT,
            'traits': get_local_traits()
        },
        connection_id
    )
    
    # Also send our peer list
    peer_list = get_peer_list(exclude_node=node_id)
    await send_message(
        node_id,
        MESSAGE_TYPES['PEER_LIST'],
        {'peers': peer_list},
        connection_id
    )

async def handle_heartbeat(node_id, payload, connection_id):
    """Handle a HEARTBEAT message from a peer."""
    # Just update last_seen
    if node_id in known_peers:
        known_peers[node_id]['last_seen'] = time.time()
    
    # Optionally update peer's traits if included
    if 'traits' in payload and node_id in known_peers:
        known_peers[node_id]['traits'] = payload['traits']

async def handle_peer_list(node_id, payload, connection_id):
    """Handle a PEER_LIST message containing other peers."""
    peers = payload.get('peers', [])
    new_peers = 0
    
    for peer in peers:
        peer_id = peer.get('node_id')
        if peer_id and peer_id != local_node_id and peer_id not in known_peers:
            # Add this new peer
            known_peers[peer_id] = {
                'node_id': peer_id,
                'address': peer.get('address'),
                'port': peer.get('port'),
                'last_seen': 0,  # We haven't seen this peer directly yet
                'traits': peer.get('traits', {})
            }
            new_peers += 1
    
    if new_peers > 0:
        logger.info(f"Added {new_peers} new peers from peer {node_id[:8]}...")

async def handle_block(node_id, payload, connection_id):
    """Handle a BLOCK message containing a new block."""
    # This function would be implemented to integrate with the blockchain module
    # It would validate and add the block to the local blockchain
    pass

async def handle_shard(node_id, payload, connection_id):
    """Handle a SHARD message containing a memory shard."""
    # This function would be implemented to integrate with the memory shard module
    # It would validate and add the shard to local storage
    pass

async def heartbeat_task():
    """Periodically send heartbeats to all connected peers."""
    while running:
        try:
            await broadcast_message(
                MESSAGE_TYPES['HEARTBEAT'],
                {'traits': get_local_traits()}
            )
            
            # Prune stale peers
            now = time.time()
            stale_peers = []
            for peer_id, peer in list(known_peers.items()):
                if now - peer['last_seen'] > HEARTBEAT_INTERVAL * 3:
                    stale_peers.append(peer_id)
            
            for peer_id in stale_peers:
                logger.info(f"Removing stale peer {peer_id[:8]}...")
                del known_peers[peer_id]
                
            await asyncio.sleep(HEARTBEAT_INTERVAL)
        except Exception as e:
            logger.error(f"Error in heartbeat task: {e}")
            await asyncio.sleep(5)  # Wait a bit before retrying

async def discovery_task():
    """Periodically try to discover new peers."""
    while running:
        try:
            # If we have too few peers, try to connect to more
            if len(known_peers) < MAX_PEERS / 2:
                # Try seed peers first
                for seed in SEED_PEERS:
                    if len(known_peers) >= MAX_PEERS:
                        break
                    
                    try:
                        address, port = seed.split(':')
                        port = int(port)
                        await connect_to_peer_async(address, port)
                    except Exception as e:
                        logger.error(f"Failed to connect to seed peer {seed}: {e}")
                
                # Try to connect to peers-of-peers
                for peer_id, peer in list(known_peers.items()):
                    if len(known_peers) >= MAX_PEERS:
                        break
                    
                    # Ask this peer for its peer list
                    await send_message(peer_id, MESSAGE_TYPES['PEER_LIST'], {})
            
            await asyncio.sleep(60)  # Run discovery every minute
        except Exception as e:
            logger.error(f"Error in discovery task: {e}")
            await asyncio.sleep(5)  # Wait a bit before retrying

async def server_main():
    """Main server function to handle incoming connections."""
    global server
    
    try:
        server = await asyncio.start_server(
            handle_connection,
            '0.0.0.0',  # Listen on all interfaces
            DEFAULT_PORT
        )
        
        addr = server.sockets[0].getsockname()
        logger.info(f'P2P server started on {addr}')
        
        # Start background tasks
        asyncio.create_task(heartbeat_task())
        asyncio.create_task(discovery_task())
        asyncio.create_task(gossip_task())  # Start the gossip protocol task
        
        # Periodically check for and report malicious peers
        asyncio.create_task(malicious_detection_task())
        
        # Periodically share network status
        asyncio.create_task(network_status_task())
        
        async with server:
            await server.serve_forever()
    except Exception as e:
        logger.error(f"Server error: {e}")
        running = False

async def malicious_detection_task():
    """Task to periodically check for potential malicious peers based on behavior patterns."""
    while running:
        try:
            # Wait a while between checks to gather enough data
            await asyncio.sleep(300)  # 5 minutes
            
            # Check for suspicious patterns in all peers
            for peer_id, peer in list(known_peers.items()):
                # Skip peers we already know are reliable
                if get_peer_reliability(peer_id) > 0.8:
                    continue
                
                # Look for possible DoS attempt (excessive messages)
                if peer_id in peer_reliability:
                    recent_interactions = peer_reliability[peer_id].get('interactions', [])
                    if len(recent_interactions) >= 50:  # Arbitrary threshold
                        # Check if interactions happened in a very short time
                        if recent_interactions:
                            newest = max(i.get('timestamp', 0) for i in recent_interactions)
                            oldest = min(i.get('timestamp', 0) for i in recent_interactions)
                            time_span = newest - oldest
                            
                            # If many interactions in a short time, might be DoS
                            if time_span < 60 and len(recent_interactions) > 30:  # More than 30 msgs in less than a minute
                                reason = f"Possible DoS attempt: {len(recent_interactions)} messages in {time_span:.1f} seconds"
                                logger.warning(f"Detecting suspicious behavior from {peer_id[:8]}...: {reason}")
                                
                                # Don't ban immediately, but report to other peers
                                evidence = {
                                    'message_count': len(recent_interactions),
                                    'time_span': time_span,
                                    'reliable_score': get_peer_reliability(peer_id)
                                }
                                
                                # Only report if we're confident
                                if time_span < 20 and len(recent_interactions) > 50:
                                    await broadcast_message(
                                        MESSAGE_TYPES['MALICIOUS_REPORT'],
                                        {
                                            'peer_id': peer_id,
                                            'reason': reason,
                                            'evidence': evidence
                                        }
                                    )
        except Exception as e:
            logger.error(f"Error in malicious detection task: {e}")
            await asyncio.sleep(30)

async def network_status_task():
    """Periodically share network status information with peers."""
    while running:
        try:
            # Only share if we have peers
            if len(known_peers) > 0:
                # Get local blockchain and shard stats (placeholder)
                from local_db_adapter import LocalDatabaseAdapter
                db_adapter = LocalDatabaseAdapter()
                
                try:
                    # Get blockchain stats
                    stats = db_adapter.get_blockchain_stats()
                    
                    # Create status payload
                    status_payload = {
                        'stats': {
                            'peers_count': len(known_peers),
                            'blocks_count': stats.get('blocks', 0),
                            'shards_count': stats.get('shards', {}).get('total', 0),
                            'timestamp': time.time()
                        }
                    }
                    
                    # Share with a few random peers
                    target_count = min(2, len(known_peers))
                    if target_count > 0:
                        targets = random.sample(list(known_peers.keys()), target_count)
                        for target_id in targets:
                            await send_message(target_id, MESSAGE_TYPES['NETWORK_STATUS'], status_payload)
                except Exception as stats_err:
                    logger.error(f"Error getting stats for network status: {stats_err}")
            
            # Run status updates periodically, less frequently than heartbeats
            await asyncio.sleep(180)  # 3 minutes
        except Exception as e:
            logger.error(f"Error in network status task: {e}")
            await asyncio.sleep(30)

def start_server():
    """Start the P2P server in a background thread."""
    global running
    
    if running:
        logger.warning("P2P server already running")
        return
    
    running = True
    
    # Register message handlers
    register_message_handlers()
    
    # Start the asyncio event loop in a separate thread
    def run_server():
        asyncio.run(server_main())
    
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    logger.info(f"P2P network started with node ID: {local_node_id}")
    
    return server_thread

def stop_server():
    """Stop the P2P server."""
    global running, server
    
    if not running:
        return
    
    running = False
    
    # Close all connections
    for conn_id, conn in list(connections.items()):
        try:
            conn['writer'].close()
        except Exception:
            pass
    
    connections.clear()
    
    # Stop server
    if server:
        server.close()
    
    logger.info("P2P server stopped")

def connect_to_peer(address, port=DEFAULT_PORT):
    """Connect to a peer (synchronous wrapper)."""
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(connect_to_peer_async(address, port))
        return result
    finally:
        loop.close()

async def connect_to_peer_async(address, port=DEFAULT_PORT):
    """Connect to a peer and perform handshake (async version)."""
    try:
        logger.info(f"Connecting to peer at {address}:{port}")
        
        # Open connection
        reader, writer = await asyncio.open_connection(address, port)
        
        # Create a connection ID and add to connections
        connection_id = str(uuid4())
        connections[connection_id] = {
            'reader': reader,
            'writer': writer,
            'address': address,
            'port': port,
            'node_id': None,  # We don't know this yet
            'last_activity': time.time()
        }
        
        # Send HELLO message
        payload = {
            'address': get_local_ip(),
            'port': DEFAULT_PORT,
            'traits': get_local_traits()
        }
        
        payload_bytes = json.dumps(payload).encode('utf-8')
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            MESSAGE_TYPES['HELLO'],
            len(payload_bytes),
            local_node_id.encode('utf-8')
        )
        
        writer.write(header + payload_bytes)
        await writer.drain()
        
        # Start a task to handle incoming messages
        asyncio.create_task(handle_connection(reader, writer))
        
        logger.info(f"Connected to peer at {address}:{port}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to peer at {address}:{port}: {e}")
        return False

def broadcast_block(block):
    """Broadcast a block to all peers."""
    loop = asyncio.new_event_loop()
    try:
        payload = {'block': block.to_dict()}
        result = loop.run_until_complete(broadcast_message(MESSAGE_TYPES['BLOCK'], payload))
        logger.info(f"Broadcasted block to {result} peers")
        return result
    finally:
        loop.close()

def broadcast_shard(shard):
    """Broadcast a memory shard to all peers."""
    loop = asyncio.new_event_loop()
    try:
        payload = {'shard': shard.to_dict()}
        result = loop.run_until_complete(broadcast_message(MESSAGE_TYPES['SHARD'], payload))
        logger.info(f"Broadcasted shard to {result} peers")
        return result
    finally:
        loop.close()

def get_peer_list(exclude_node=None):
    """Get a list of known peers for sharing."""
    peers = []
    
    for peer_id, peer in known_peers.items():
        if exclude_node and peer_id == exclude_node:
            continue
            
        peers.append({
            'node_id': peer_id,
            'address': peer['address'],
            'port': peer['port'],
            'traits': peer.get('traits', {})
        })
    
    return peers

def get_local_ip():
    """Get the local IP address."""
    try:
        # This doesn't actually establish a connection, just creates a socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))  # Connect to a public DNS server
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback to localhost
        return '127.0.0.1'

def get_local_traits():
    """Get the local node's traits for sharing with peers."""
    # In a full implementation, this would get the latest traits from consensus
    return {
        'mutation_rate': 0.05,
        'crossover_rate': 0.7,
        'selection_pressure': 0.8,
        'novelty_preference': 0.3,
        'cooperation_factor': 0.7,
    }

# New message handlers for enhanced P2P networking

async def handle_peer_gossip(node_id, payload, connection_id):
    """Handle gossip messages about other peers."""
    gossip_peers = payload.get('peers', [])
    gossip_time = payload.get('timestamp', time.time())
    gossip_ttl = payload.get('ttl', 2)  # Time to live - how many more hops this gossip should travel
    
    # Process each peer in the gossip
    for peer_info in gossip_peers:
        peer_id = peer_info.get('node_id')
        
        # Skip if we're the subject of the gossip, or if it's about the sender
        if peer_id == local_node_id or peer_id == node_id:
            continue
            
        # Check the timestamp to avoid processing old gossip
        if peer_id in peer_gossip_cache:
            if gossip_time <= peer_gossip_cache[peer_id].get('timestamp', 0):
                continue  # Skip this peer, we have newer information
        
        # Add to gossip cache to prevent re-processing
        peer_gossip_cache[peer_id] = {
            'timestamp': gossip_time,
            'source': node_id,
            'info': peer_info
        }
        
        # Update our known_peers if it's new or has newer information
        if peer_id not in known_peers:
            known_peers[peer_id] = {
                'node_id': peer_id,
                'address': peer_info.get('address'),
                'port': peer_info.get('port'),
                'last_seen': 0,  # We haven't directly seen this peer
                'traits': peer_info.get('traits', {}),
                'reliability': peer_info.get('reliability', 0.5)  # Default middle reliability
            }
            logger.info(f"Added new peer {peer_id[:8]}... from gossip")
    
    # Propagate gossip if TTL allows
    if gossip_ttl > 0:
        # Decrease TTL
        payload['ttl'] = gossip_ttl - 1
        
        # Forward to a subset of peers (to prevent flooding)
        forward_count = min(3, len(known_peers) // 2)  # Forward to at most 3 peers or half our peers
        forward_peers = random.sample(list(known_peers.keys()), min(forward_count, len(known_peers)))
        
        for forward_id in forward_peers:
            if forward_id != node_id:  # Don't send back to source
                await send_message(forward_id, MESSAGE_TYPES['PEER_GOSSIP'], payload)

async def handle_peer_recommend(node_id, payload, connection_id):
    """Handle peer recommendations for reliable nodes."""
    recommendations = payload.get('recommendations', [])
    
    for rec in recommendations:
        peer_id = rec.get('node_id')
        reliability = rec.get('reliability', 0.0)
        
        # Skip self-recommendations
        if peer_id == node_id:
            continue
            
        # Update peer reliability based on recommendation
        if peer_id in known_peers:
            # Weight the recommendation based on the recommender's own reliability
            recommender_reliability = get_peer_reliability(node_id)
            weighted_reliability = reliability * recommender_reliability
            
            # Combine with existing reliability (with dampening to prevent manipulation)
            current_reliability = known_peers[peer_id].get('reliability', 0.5)
            new_reliability = (current_reliability * 0.7) + (weighted_reliability * 0.3)
            
            # Update peer's reliability rating
            known_peers[peer_id]['reliability'] = new_reliability
            
            logger.debug(f"Updated reliability for peer {peer_id[:8]}... to {new_reliability:.2f}")

async def handle_malicious_report(node_id, payload, connection_id):
    """Handle reports of potentially malicious nodes."""
    reported_id = payload.get('peer_id')
    reason = payload.get('reason', 'Unknown')
    evidence = payload.get('evidence', {})
    
    if reported_id == local_node_id:
        # We're being reported! Log this but don't take action on ourselves
        logger.warning(f"We've been reported as malicious by {node_id[:8]}... for reason: {reason}")
        return
    
    # Track the report in malicious activity log
    if reported_id not in malicious_activity:
        malicious_activity[reported_id] = {
            'reports': [],
            'total_reports': 0,
            'unique_reporters': set()
        }
    
    # Add this report
    malicious_activity[reported_id]['reports'].append({
        'reporter': node_id,
        'reason': reason,
        'evidence': evidence,
        'timestamp': time.time()
    })
    malicious_activity[reported_id]['total_reports'] += 1
    malicious_activity[reported_id]['unique_reporters'].add(node_id)
    
    # Check if we should ban this peer
    if should_ban_peer(reported_id):
        ban_peer(reported_id, f"Reported as malicious: {reason}")
        
        # Inform others about this banned peer
        await broadcast_message(
            MESSAGE_TYPES['MALICIOUS_REPORT'],
            {
                'peer_id': reported_id,
                'reason': reason,
                'evidence': evidence
            },
            exclude_nodes=[node_id]  # Don't send back to the reporter
        )

async def handle_network_status(node_id, payload, connection_id):
    """Handle network-wide status information."""
    network_stats = payload.get('stats', {})
    peers_count = network_stats.get('peers_count', 0)
    known_blocks = network_stats.get('blocks_count', 0)
    known_shards = network_stats.get('shards_count', 0)
    
    # Log the network stats
    logger.info(f"Network status from {node_id[:8]}... - Peers: {peers_count}, Blocks: {known_blocks}, Shards: {known_shards}")
    
    # Could update local view of network size here if needed

# Enhanced peer management functions
def get_peer_reliability(node_id):
    """Get the reliability rating for a peer."""
    if node_id not in peer_reliability:
        return 0.5  # Default middle reliability
    
    # Calculate reliability score from success/failure ratio
    peer = peer_reliability[node_id]
    total_interactions = peer.get('success_count', 0) + peer.get('failure_count', 0)
    
    if total_interactions == 0:
        return 0.5
    
    # Calculate base reliability from success rate
    reliability = peer.get('success_count', 0) / total_interactions
    
    # Apply time decay - reduce reliability of peers we haven't interacted with recently
    last_interaction = peer.get('last_interaction', 0)
    time_since_last = time.time() - last_interaction
    
    # Apply a gentle decay for peers we haven't seen in a while
    if time_since_last > 3600:  # More than an hour
        time_factor = max(0.5, 1.0 - (time_since_last / (24 * 3600)))  # Decay to 0.5 over 24 hours
        reliability = reliability * time_factor
    
    return reliability

def update_peer_reliability(node_id, success=True, interaction_type='message'):
    """Update a peer's reliability based on interactions."""
    if node_id not in peer_reliability:
        peer_reliability[node_id] = {
            'success_count': 0,
            'failure_count': 0,
            'last_interaction': time.time(),
            'interactions': []
        }
    
    # Update counters
    if success:
        peer_reliability[node_id]['success_count'] += 1
    else:
        peer_reliability[node_id]['failure_count'] += 1
    
    # Update last interaction time
    peer_reliability[node_id]['last_interaction'] = time.time()
    
    # Record this interaction
    peer_reliability[node_id]['interactions'].append({
        'type': interaction_type,
        'success': success,
        'timestamp': time.time()
    })
    
    # Limit the interaction history to prevent unbounded growth
    if len(peer_reliability[node_id]['interactions']) > 100:
        peer_reliability[node_id]['interactions'] = peer_reliability[node_id]['interactions'][-100:]

def should_ban_peer(node_id):
    """Determine if a peer should be banned based on malicious activity."""
    if node_id not in malicious_activity:
        return False
    
    peer_activity = malicious_activity[node_id]
    
    # Criteria for banning:
    # 1. Multiple unique reporters
    if len(peer_activity['unique_reporters']) >= 3:
        return True
    
    # 2. Many reports over time
    if peer_activity['total_reports'] >= 10:
        return True
    
    # 3. Very low reliability
    if get_peer_reliability(node_id) < 0.2:
        return True
    
    return False

def ban_peer(node_id, reason):
    """Ban a peer from the network."""
    if node_id in banned_peers:
        return  # Already banned
    
    # Record the ban
    banned_peers[node_id] = {
        'reason': reason,
        'timestamp': time.time()
    }
    
    # Remove from known_peers
    if node_id in known_peers:
        logger.warning(f"Banning peer {node_id[:8]}... - Reason: {reason}")
        del known_peers[node_id]
    
    # Close any active connections
    for conn_id, conn in list(connections.items()):
        if conn.get('node_id') == node_id:
            try:
                conn['writer'].close()
            except Exception:
                pass
            del connections[conn_id]

# Gossip protocol
async def gossip_task():
    """Periodically share peer information via gossip protocol."""
    while running:
        try:
            # Only gossip if we have peers
            if len(known_peers) > 0:
                # Select a subset of our best peers to gossip about
                peers_to_gossip = []
                for peer_id, peer in known_peers.items():
                    # Include reliability in the gossip
                    peer_info = {
                        'node_id': peer_id,
                        'address': peer.get('address'),
                        'port': peer.get('port'),
                        'reliability': get_peer_reliability(peer_id),
                        'traits': peer.get('traits', {})
                    }
                    peers_to_gossip.append(peer_info)
                
                # Only share our top 5 most reliable peers
                if len(peers_to_gossip) > 5:
                    peers_to_gossip.sort(key=lambda p: p.get('reliability', 0), reverse=True)
                    peers_to_gossip = peers_to_gossip[:5]
                
                # Create gossip payload
                gossip_payload = {
                    'peers': peers_to_gossip,
                    'timestamp': time.time(),
                    'ttl': 2  # Allow the gossip to travel 2 hops
                }
                
                # Send to a random subset of peers
                target_count = min(3, len(known_peers))  # At most 3 targets
                if target_count > 0:
                    targets = random.sample(list(known_peers.keys()), target_count)
                    for target_id in targets:
                        await send_message(target_id, MESSAGE_TYPES['PEER_GOSSIP'], gossip_payload)
            
            # Run gossip relatively frequently in small networks, less often in larger ones
            await asyncio.sleep(60 + len(known_peers) * 2)  # 1 minute + 2 seconds per peer
        except Exception as e:
            logger.error(f"Error in gossip task: {e}")
            await asyncio.sleep(10)  # Wait a bit before retrying

def register_message_handlers():
    """Register all message handlers."""
    message_handlers[MESSAGE_TYPES['HELLO']] = handle_hello
    message_handlers[MESSAGE_TYPES['HEARTBEAT']] = handle_heartbeat
    message_handlers[MESSAGE_TYPES['PEER_LIST']] = handle_peer_list
    message_handlers[MESSAGE_TYPES['BLOCK']] = handle_block
    message_handlers[MESSAGE_TYPES['SHARD']] = handle_shard
    message_handlers[MESSAGE_TYPES['PEER_GOSSIP']] = handle_peer_gossip
    message_handlers[MESSAGE_TYPES['PEER_RECOMMEND']] = handle_peer_recommend
    message_handlers[MESSAGE_TYPES['MALICIOUS_REPORT']] = handle_malicious_report
    message_handlers[MESSAGE_TYPES['NETWORK_STATUS']] = handle_network_status
    # Other handlers would be registered here as they are implemented