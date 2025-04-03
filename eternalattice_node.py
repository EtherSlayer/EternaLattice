#!/usr/bin/env python3
"""
EternaLattice Node - Decentralized Knowledge Preservation Network

This script runs a standalone node in the EternaLattice network.
It combines core blockchain functionality, consensus, and networking
in a single executable with a command-line interface.

Usage:
  python eternalattice_node.py [command] [options]

Commands:
  start                Start the node and connect to the network
  add_shard            Add a new memory shard to the network
  mine                 Mine a new block in the lattice
  status               Display node status and statistics
  explore              Explore the blockchain and memory shards
  peers                List connected peer nodes
  help                 Display this help message
"""

import argparse
import json
import logging
import os
import sys
import time
import webbrowser
from datetime import datetime
from threading import Thread
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('eternalattice.log')
    ]
)

logger = logging.getLogger('eternalattice')

# Import core modules from blockchain directory
try:
    from blockchain.core import blockchain, create_genesis_block, get_blockchain_stats
    from blockchain.consensus import evolve_traits, initial_traits, calculate_fitness
    from blockchain.crypto import generate_key_pair, sign_data, verify_signature, hash_data
    from blockchain.memory_shard import memory_shards, MemoryShard
    from blockchain.network import local_node_id, start_network, connect_to_peer, get_peer_list, broadcast_shard, broadcast_block
    from models import Block, UserReputation, Node
    import app
    from config import SQLITE_DB_PATH
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Initialize SQLite database
def initialize_database(db_path=None):
    """Initialize a local SQLite database for the node."""
    if db_path is None:
        # Default to a local SQLite database in the user's home directory
        home_dir = os.path.expanduser("~")
        data_dir = os.path.join(home_dir, ".eternalattice")
        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, "eternalattice.db")
    
    # Override the database URL to use SQLite
    if hasattr(app, 'app'):
        app.app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
        
        # Initialize database tables
        with app.app.app_context():
            app.db.create_all()
            logger.info(f"Database initialized at {db_path}")
    else:
        logger.error("Flask app not available, database initialization skipped")
    
    return db_path

# Node commands implementation
def start_node(args):
    """Start the EternaLattice node and connect to peers."""
    logger.info("Starting EternaLattice node...")
    
    # Initialize database
    db_path = initialize_database()
    
    # Start network services
    start_network()
    
    # Connect to seed peers if provided
    if args.peers:
        for peer in args.peers.split(','):
            connect_to_peer(peer)
    
    logger.info(f"Node started with ID: {local_node_id}")
    logger.info(f"Database location: {db_path}")
    
    # Start local web interface if requested
    if args.web:
        start_web_interface(args.port)
    
    return True

def add_shard_command(args):
    """Add a new memory shard to the network."""
    logger.info(f"Adding new memory shard: {args.data[:30]}...")
    
    # Create metadata
    metadata = {
        "creator": local_node_id,
        "timestamp": int(time.time()),
        "category": args.category,
        "region": args.region,
        "type": args.type,
        "source": args.source
    }
    
    # Create the memory shard
    shard = MemoryShard(
        data=args.data,
        metadata=metadata,
        category=args.category,
        region=args.region
    )
    
    # Sign the shard
    privkey, pubkey = generate_key_pair()  # This should come from stored keys
    shard.signature = sign_data(shard.to_json(), privkey)
    
    # Calculate hash
    shard.hash = hash_data(shard.to_json())
    
    # Add to local storage
    memory_shards[shard.shard_id] = shard
    
    # Broadcast to network
    broadcast_shard(shard)
    
    # Update user reputation if available
    update_user_reputation('create_shard', 10, f"Created shard: {shard.shard_id[:8]}...")
    
    logger.info(f"Memory shard created with ID: {shard.shard_id}")
    print(f"Memory shard created with ID: {shard.shard_id}")
    
    return shard.shard_id

def mine_block_command(args):
    """Mine a new block in the lattice."""
    logger.info("Mining a new block...")
    
    # Get coordinates for new block
    time_coord = int(args.x) if args.x is not None else len([b for b in blockchain.values() if b.coordinates[1] == 0 and b.coordinates[2] == 0])
    category_coord = int(args.y) if args.y is not None else 0
    region_coord = int(args.z) if args.z is not None else 0
    
    coordinates = (time_coord, category_coord, region_coord)
    
    # Check if coordinates are already taken
    if coordinates in blockchain:
        logger.error(f"Coordinates {coordinates} already occupied in the blockchain")
        print(f"Error: Coordinates {coordinates} already occupied in the blockchain")
        return False
    
    # Get previous hashes for multi-dimensional connections
    previous_hashes = {}
    if time_coord > 0:
        time_prev = (time_coord - 1, category_coord, region_coord)
        if time_prev in blockchain:
            previous_hashes['time'] = blockchain[time_prev].hash
    
    if category_coord > 0:
        category_prev = (time_coord, category_coord - 1, region_coord)
        if category_prev in blockchain:
            previous_hashes['category'] = blockchain[category_prev].hash
    
    if region_coord > 0:
        region_prev = (time_coord, category_coord, region_coord - 1)
        if region_prev in blockchain:
            previous_hashes['region'] = blockchain[region_prev].hash
    
    # Create a new block
    new_block = Block(
        coordinates=coordinates,
        previous_hashes=previous_hashes,
        timestamp=int(time.time()),
        miner_id=local_node_id
    )
    
    # Add shards to the block (max 10 unconfirmed shards)
    unconfirmed_shards = [s for s in memory_shards.values() if not any(s.shard_id in block.shard_references for block in blockchain.values())]
    for shard in unconfirmed_shards[:10]:
        new_block.shard_references.append(shard.shard_id)
    
    # Apply PoE consensus mechanism
    if len(blockchain) > 0:
        # Get parent blocks for trait evolution
        parent_blocks = [block for block in blockchain.values() 
                         if block.coordinates[0] == time_coord - 1]
        
        if parent_blocks:
            # Evolve traits from parent population
            population = [block.consensus_traits for block in parent_blocks]
            fitness_scores = [block.fitness_score for block in parent_blocks]
            new_block.consensus_traits = evolve_traits(population, fitness_scores)
        else:
            new_block.consensus_traits = initial_traits()
    else:
        # Genesis block traits
        new_block.consensus_traits = initial_traits()
    
    # Calculate fitness score
    new_block.fitness_score = calculate_fitness(new_block.consensus_traits)
    
    # Mining process (simple PoW for demonstration)
    difficulty = 1
    new_block.difficulty = difficulty
    
    found = False
    attempts = 0
    max_attempts = 1000  # Limit for demonstration
    
    while not found and attempts < max_attempts:
        new_block.nonce = attempts
        block_hash = hash_data(new_block.to_json())
        if block_hash.startswith('0' * difficulty):
            found = True
            new_block.hash = block_hash
        attempts += 1
    
    if not found:
        logger.error("Failed to mine block within attempt limit")
        print("Error: Failed to mine block within attempt limit")
        return False
    
    # Sign the block
    privkey, pubkey = generate_key_pair()  # This should come from stored keys
    new_block.signature = sign_data(new_block.to_json(), privkey)
    
    # Add to blockchain
    blockchain[coordinates] = new_block
    
    # Broadcast to network
    broadcast_block(new_block)
    
    # Update user reputation if available
    update_user_reputation('mine_block', 20, f"Mined block at coordinates {coordinates}")
    
    logger.info(f"Block successfully mined at coordinates {coordinates}")
    print(f"Block successfully mined at coordinates {coordinates}")
    
    return True

def status_command(args):
    """Display node status and statistics."""
    print("\n===== ETERNALATTICE NODE STATUS =====")
    print(f"Node ID: {local_node_id[:8]}...")
    print(f"Started: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Blockchain stats
    stats = get_blockchain_stats()
    print("\nBlockchain Status:")
    print(f"  Total Blocks: {stats.get('blocks', 0)}")
    print(f"  Latest Block: {max([b.coordinates[0] for b in blockchain.values()]) if blockchain else 0}")
    print(f"  Dimensions: {stats.get('dimensions', {})}")
    
    # Shard stats
    print("\nMemory Shards:")
    shards_in_blocks = sum(1 for block in blockchain.values() for _ in block.shard_references)
    unconfirmed = len(memory_shards) - shards_in_blocks
    print(f"  Total Shards: {len(memory_shards)}")
    print(f"  In Blocks: {shards_in_blocks}")
    print(f"  Unconfirmed: {unconfirmed}")
    
    # Network stats
    peers = get_peer_list()
    print("\nNetwork Status:")
    print(f"  Connected Peers: {len(peers)}")
    
    # User reputation
    user_rep = get_user_reputation()
    if user_rep:
        print("\nUser Reputation:")
        print(f"  Username: {user_rep.username}")
        print(f"  Level: {user_rep.level}")
        print(f"  Total Points: {user_rep.total_points}")
        print(f"  Blocks Mined: {user_rep.mined_blocks}")
        print(f"  Shards Created: {user_rep.created_shards}")
        print(f"  Badges: {', '.join(user_rep.badges) if user_rep.badges else 'None'}")
    
    return True

def explore_command(args):
    """Explore the blockchain and memory shards."""
    if args.block:
        # Parse coordinates from string like "0,0,0"
        try:
            coordinates = tuple(map(int, args.block.split(',')))
            if coordinates in blockchain:
                block = blockchain[coordinates]
                print(f"\nBlock at coordinates {coordinates}:")
                print(f"  Hash: {block.hash[:16]}...")
                print(f"  Timestamp: {datetime.fromtimestamp(block.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Difficulty: {block.difficulty}")
                print(f"  Fitness Score: {block.fitness_score:.4f}")
                print(f"  Miner: {block.miner_id[:8]}...")
                print(f"  Shards: {len(block.shard_references)}")
                
                if args.verbose:
                    print("\n  Consensus Traits:")
                    for trait, value in block.consensus_traits.items():
                        print(f"    {trait}: {value:.4f}")
                    
                    print("\n  Previous Hashes:")
                    for dim, prev_hash in block.previous_hashes.items():
                        print(f"    {dim}: {prev_hash[:16]}...")
                    
                    print("\n  Shard References:")
                    for i, shard_id in enumerate(block.shard_references):
                        print(f"    {i+1}. {shard_id}")
            else:
                print(f"Block at coordinates {coordinates} not found")
        except ValueError:
            print("Invalid coordinates format. Use x,y,z (e.g., 0,0,0)")
    
    elif args.shard:
        shard_id = args.shard
        if shard_id in memory_shards:
            shard = memory_shards[shard_id]
            print(f"\nMemory Shard {shard_id}:")
            print(f"  Category: {shard.category}")
            print(f"  Region: {shard.region}")
            print(f"  Creation Time: {datetime.fromtimestamp(shard.creation_time).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Replication Count: {shard.replication_count}")
            
            if args.verbose:
                print("\n  Metadata:")
                for key, value in shard.metadata.items():
                    print(f"    {key}: {value}")
                
                print("\n  Data Preview:")
                data_preview = shard.data[:100] + "..." if len(shard.data) > 100 else shard.data
                print(f"    {data_preview}")
        else:
            print(f"Shard with ID {shard_id} not found")
    
    else:
        # List all blocks and shards with brief info
        print("\n===== BLOCKCHAIN EXPLORER =====")
        
        print("\nBlocks:")
        for coords, block in sorted(blockchain.items()):
            print(f"  ({coords[0]},{coords[1]},{coords[2]}) - {len(block.shard_references)} shards, mined by {block.miner_id[:8]}...")
        
        print("\nRecent Memory Shards:")
        sorted_shards = sorted(memory_shards.values(), key=lambda x: x.creation_time, reverse=True)
        for i, shard in enumerate(sorted_shards[:10]):
            print(f"  {i+1}. {shard.shard_id[:8]}... - {shard.category}/{shard.region} - {datetime.fromtimestamp(shard.creation_time).strftime('%Y-%m-%d')}")
    
    return True

def peers_command(args):
    """List connected peer nodes."""
    peers = get_peer_list()
    
    print("\n===== CONNECTED PEERS =====")
    if not peers:
        print("No peers connected")
    else:
        for i, peer in enumerate(peers):
            print(f"{i+1}. {peer['node_id'][:8]}... - {peer['address']}:{peer['port']}")
            if args.verbose:
                last_seen = datetime.fromtimestamp(peer['last_seen']).strftime('%Y-%m-%d %H:%M:%S')
                print(f"   Last seen: {last_seen}")
                print(f"   Traits: {json.dumps(peer['traits'], indent=2)}")
    
    return True

def help_command(args):
    """Display help information."""
    parser.print_help()
    return True

# Helper functions
def update_user_reputation(activity, points, description=""):
    """Update the user's reputation based on activities."""
    # In a local node setup, we need to maintain user reputation locally
    # This is a simplified version of the web app's reputation system
    user_rep = get_user_reputation()
    
    if user_rep:
        user_rep.add_points(points, activity, description)
        # Store updated reputation (would save to database in full implementation)
        
        logger.info(f"User reputation updated: +{points} points for {activity}")
        return True
    
    return False

def get_user_reputation():
    """Get the current user's reputation object."""
    # In a proper implementation, this would load from database
    # For this prototype, return a dummy reputation
    user_rep = UserReputation(
        user_id=local_node_id,
        username="local_user",
        display_name="Local User"
    )
    return user_rep

def start_web_interface(port=5000):
    """Start the local web interface."""
    try:
        # We reuse the existing Flask app, but bind it to localhost only
        from threading import Thread
        from flask import Flask
        
        # Create a Flask app specifically for the standalone interface
        # or use the existing app with our routes
        if hasattr(app, 'app'):
            flask_app = app.app
            
            # Register the standalone routes with the existing app
            try:
                from standalone_routes import standalone_bp
                flask_app.register_blueprint(standalone_bp)
                logger.info("Registered standalone routes with existing Flask app")
            except ImportError as e:
                logger.warning(f"Failed to import standalone routes: {e}")
        else:
            # Create a new Flask app if one doesn't exist
            flask_app = Flask(__name__)
            
            # Register the standalone routes
            try:
                from standalone_routes import standalone_bp
                flask_app.register_blueprint(standalone_bp)
                logger.info("Created new Flask app with standalone routes")
            except ImportError as e:
                logger.warning(f"Failed to import standalone routes: {e}")
        
        def run_server():
            # Use 0.0.0.0 to allow access from other devices on the network
            flask_app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
        server_thread = Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Open browser after a short delay
        time.sleep(1)
        webbrowser.open(f"http://127.0.0.1:{port}")
        
        logger.info(f"Web interface started at http://127.0.0.1:{port}")
        print(f"Web interface started at http://127.0.0.1:{port}")
        print(f"Network access available at http://{get_local_ip()}:{port}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")
        return False

def get_local_ip():
    """Get the local IP address for network access."""
    import socket
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

# Command-line argument parsing
parser = argparse.ArgumentParser(description="EternaLattice Node - Decentralized Knowledge Preservation Network")
subparsers = parser.add_subparsers(dest='command', help='Command to run')

# Start node command
start_parser = subparsers.add_parser('start', help='Start the node and connect to the network')
start_parser.add_argument('--peers', help='Comma-separated list of peer addresses (ip:port)')
start_parser.add_argument('--web', action='store_true', help='Start local web interface')
start_parser.add_argument('--port', type=int, default=5000, help='Port for web interface (default: 5000)')

# Add shard command
add_shard_parser = subparsers.add_parser('add_shard', help='Add a new memory shard to the network')
add_shard_parser.add_argument('data', help='Content for the memory shard')
add_shard_parser.add_argument('--category', default='general', help='Category for the shard (default: general)')
add_shard_parser.add_argument('--region', default='global', help='Region for the shard (default: global)')
add_shard_parser.add_argument('--type', default='text', help='Type of data (text, json, etc.)')
add_shard_parser.add_argument('--source', default='', help='Source of the information')
add_shard_parser.add_argument('--tags', help='Comma-separated list of tags for better searchability')

# Mine block command
mine_parser = subparsers.add_parser('mine', help='Mine a new block in the lattice')
mine_parser.add_argument('--x', type=int, help='X coordinate (time dimension)')
mine_parser.add_argument('--y', type=int, help='Y coordinate (category dimension)')
mine_parser.add_argument('--z', type=int, help='Z coordinate (region dimension)')
mine_parser.add_argument('--auto', action='store_true', help='Automatically mine at optimal coordinates')

# Status command
status_parser = subparsers.add_parser('status', help='Display node status and statistics')
status_parser.add_argument('--json', action='store_true', help='Output in JSON format for scripting')

# Explore command
explore_parser = subparsers.add_parser('explore', help='Explore the blockchain and memory shards')
explore_parser.add_argument('--block', help='Block coordinates to explore (format: x,y,z)')
explore_parser.add_argument('--shard', help='Shard ID to explore')
explore_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
explore_parser.add_argument('--all', action='store_true', help='Show all blocks and shards')

# Search shard command
search_parser = subparsers.add_parser('search_shards', help='Search for memory shards by content or metadata')
search_parser.add_argument('query', help='Search query string')
search_parser.add_argument('--category', help='Filter by category')
search_parser.add_argument('--region', help='Filter by region')
search_parser.add_argument('--tags', help='Filter by comma-separated list of tags')
search_parser.add_argument('--created-after', help='Filter by creation date (format: YYYY-MM-DD)')
search_parser.add_argument('--created-by', help='Filter by creator ID')

# Peers command
peers_parser = subparsers.add_parser('peers', help='List connected peer nodes')
peers_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed peer information')
peers_parser.add_argument('--connect', help='Connect to a new peer (format: address:port)')
peers_parser.add_argument('--disconnect', help='Disconnect from a peer by ID')

# Leaderboard command
leaderboard_parser = subparsers.add_parser('leaderboard', help='Display leaderboard of top contributors')
leaderboard_parser.add_argument('--limit', type=int, default=10, help='Number of users to show (default: 10)')
leaderboard_parser.add_argument('--category', help='Filter by contribution category (mine_block, create_shard, preserve_shard)')

# User profile command
profile_parser = subparsers.add_parser('profile', help='Display or update user profile')
profile_parser.add_argument('--user-id', help='User ID to look up (default: local user)')
profile_parser.add_argument('--set-username', help='Set a new username')
profile_parser.add_argument('--set-display-name', help='Set a new display name')

# Import/export command
import_export_parser = subparsers.add_parser('backup', help='Backup or restore blockchain data')
import_export_parser.add_argument('--export', help='Export blockchain to file')
import_export_parser.add_argument('--import', dest='import_file', help='Import blockchain from file')
import_export_parser.add_argument('--shards-only', action='store_true', help='Only backup/restore shards')
import_export_parser.add_argument('--blocks-only', action='store_true', help='Only backup/restore blocks')

# Network diagnostics command
diagnostics_parser = subparsers.add_parser('diagnostics', help='Run network diagnostics')
diagnostics_parser.add_argument('--deep', action='store_true', help='Run deep diagnostics (slower)')
diagnostics_parser.add_argument('--repair', action='store_true', help='Attempt to repair issues')

# Help command
help_parser = subparsers.add_parser('help', help='Display help information')

# Command implementations for new CLI features
def search_shards_command(args):
    """Search for memory shards based on query and filters."""
    logger.info(f"Searching shards with query: {args.query}")
    
    # Basic search using the adapter
    from local_db_adapter import LocalDatabaseAdapter
    db_adapter = LocalDatabaseAdapter()
    
    # Perform the search
    results = db_adapter.search_shards(args.query, category=args.category, region=args.region)
    
    # Apply additional filters
    if args.tags:
        tags_list = [tag.strip() for tag in args.tags.split(',')]
        results = [shard for shard in results if 
                  any(tag in shard.metadata.get('tags', []) for tag in tags_list)]
    
    if args.created_after:
        try:
            from datetime import datetime
            date_filter = datetime.strptime(args.created_after, "%Y-%m-%d").timestamp()
            results = [shard for shard in results if shard.creation_time >= date_filter]
        except ValueError:
            print(f"Error: Invalid date format. Use YYYY-MM-DD")
    
    if args.created_by:
        results = [shard for shard in results if 
                  shard.metadata.get('creator') == args.created_by]
    
    # Display results
    if not results:
        print("No matching shards found.")
        return []
    
    print(f"\nFound {len(results)} matching shards:")
    for i, shard in enumerate(results, 1):
        print(f"  {i}. ID: {shard.shard_id[:8]}...")
        print(f"     Category: {shard.category}, Region: {shard.region}")
        print(f"     Created: {datetime.fromtimestamp(shard.creation_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"     Data Preview: {shard.data[:50]}..." if len(shard.data) > 50 else f"     Data: {shard.data}")
        print()
    
    return results

def leaderboard_command(args):
    """Display a leaderboard of top contributors."""
    logger.info(f"Generating leaderboard of top {args.limit} contributors")
    
    # Get user reputation data from the database
    from local_db_adapter import LocalDatabaseAdapter
    db_adapter = LocalDatabaseAdapter()
    all_users = db_adapter.load_all_user_reputations()
    
    # Apply category filter if specified
    if args.category:
        category = args.category.lower()
        filtered_users = []
        
        for user in all_users:
            relevant_contributions = [
                c for c in user.contribution_history
                if c.get('activity') == category
            ]
            if relevant_contributions:
                # Calculate category-specific points
                category_points = sum(c.get('points', 0) for c in relevant_contributions)
                user_copy = UserReputation(
                    user_id=user.user_id,
                    username=user.username,
                    display_name=user.display_name,
                    total_points=category_points
                )
                filtered_users.append(user_copy)
        
        # Sort by category-specific points
        filtered_users.sort(key=lambda u: u.total_points, reverse=True)
        users_to_show = filtered_users[:args.limit]
    else:
        # Sort by total points
        all_users.sort(key=lambda u: u.total_points, reverse=True)
        users_to_show = all_users[:args.limit]
    
    # Display the leaderboard
    if not users_to_show:
        print("No users found for the leaderboard.")
        return
    
    print("\n===== ETERNALATTICE LEADERBOARD =====")
    if args.category:
        print(f"Category: {args.category}")
    
    print("\nRank  Username             Level  Points  Blocks  Shards  Badges")
    print("-" * 70)
    
    for i, user in enumerate(users_to_show, 1):
        badges_preview = ", ".join(list(user.badges)[:2])
        if len(user.badges) > 2:
            badges_preview += f" (+{len(user.badges) - 2} more)"
        
        username_display = user.username[:20].ljust(20)
        print(f"{i:3}   {username_display}  {user.level:5}  {user.total_points:6}  {user.mined_blocks:6}  {user.created_shards:6}  {badges_preview}")
    
    return users_to_show

def profile_command(args):
    """Display or update a user profile."""
    from local_db_adapter import LocalDatabaseAdapter
    db_adapter = LocalDatabaseAdapter()
    
    # Get user ID (local or specified)
    user_id = args.user_id or local_node_id
    
    # Load user data
    user = db_adapter.load_user_reputation(user_id)
    
    if not user:
        if user_id == local_node_id:
            # Create a new user record for local node
            user = UserReputation(
                user_id=local_node_id,
                username="local_user",
                display_name="Local User"
            )
            db_adapter.save_user_reputation(user)
            print("Created new user profile for local node")
        else:
            print(f"Error: User with ID {user_id[:8]}... not found")
            return False
    
    # Handle profile updates
    if args.set_username:
        user.username = args.set_username
        db_adapter.save_user_reputation(user)
        print(f"Username updated to: {user.username}")
    
    if args.set_display_name:
        user.display_name = args.set_display_name
        db_adapter.save_user_reputation(user)
        print(f"Display name updated to: {user.display_name}")
    
    # Display user profile
    print(f"\n===== USER PROFILE: {user.display_name} =====")
    print(f"User ID: {user.user_id[:8]}...")
    print(f"Username: {user.username}")
    print(f"Level: {user.level} ({user.total_points} points)")
    print(f"Creation Date: {datetime.fromtimestamp(user.created_at).strftime('%Y-%m-%d')}")
    print(f"Last Activity: {datetime.fromtimestamp(user.last_contribution).strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nContributions:")
    print(f"  Blocks Mined: {user.mined_blocks}")
    print(f"  Shards Created: {user.created_shards}")
    print(f"  Shards Preserved: {user.preserved_shards}")
    
    print("\nBadges:")
    if not user.badges:
        print("  No badges earned yet")
    else:
        for badge in sorted(user.badges):
            print(f"  • {badge}")
    
    print("\nRecent Activity:")
    recent_history = sorted(user.contribution_history, key=lambda c: c.get('timestamp', 0), reverse=True)[:5]
    if not recent_history:
        print("  No recent activity")
    else:
        for activity in recent_history:
            activity_time = datetime.fromtimestamp(activity.get('timestamp', 0)).strftime('%Y-%m-%d')
            action = activity.get('activity', 'unknown')
            points = activity.get('points', 0)
            description = activity.get('description', '')
            print(f"  • {activity_time}: {action} (+{points} pts) - {description}")
    
    return user

def backup_command(args):
    """Backup or restore blockchain data."""
    import json
    from local_db_adapter import LocalDatabaseAdapter
    db_adapter = LocalDatabaseAdapter()
    
    # Handle export
    if args.export:
        export_path = args.export
        data = {}
        
        # Export blocks if requested
        if not args.shards_only:
            blocks_dict = {}
            blocks = db_adapter.load_blocks()
            for coords, block in blocks.items():
                blocks_dict[str(coords)] = block.to_dict()
            data['blocks'] = blocks_dict
            print(f"Exported {len(blocks)} blocks")
        
        # Export shards if requested
        if not args.blocks_only:
            shards_dict = {}
            shards = db_adapter.load_shards()
            for shard_id, shard in shards.items():
                shards_dict[shard_id] = shard.to_dict()
            data['shards'] = shards_dict
            print(f"Exported {len(shards)} shards")
        
        # Write to file
        try:
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Blockchain data exported successfully to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            print(f"Error: Failed to write to {export_path}")
            return False
    
    # Handle import
    elif args.import_file:
        import_path = args.import_file
        try:
            with open(import_path, 'r') as f:
                data = json.load(f)
            
            # Import blocks if requested
            if not args.shards_only and 'blocks' in data:
                blocks_dict = data['blocks']
                import_count = 0
                for coords_str, block_dict in blocks_dict.items():
                    block = Block.from_dict(block_dict)
                    db_adapter.save_block(block)
                    import_count += 1
                print(f"Imported {import_count} blocks")
            
            # Import shards if requested
            if not args.blocks_only and 'shards' in data:
                shards_dict = data['shards']
                import_count = 0
                for shard_id, shard_dict in shards_dict.items():
                    shard = MemoryShard.from_dict(shard_dict)
                    db_adapter.save_shard(shard)
                    import_count += 1
                print(f"Imported {import_count} shards")
            
            print(f"Blockchain data imported successfully from {import_path}")
            return True
        except Exception as e:
            logger.error(f"Error importing data: {e}")
            print(f"Error: Failed to import from {import_path}")
            return False
    
    else:
        print("Error: Must specify either --export or --import")
        return False

def diagnostics_command(args):
    """Run network diagnostics."""
    print("\n===== ETERNALATTICE NETWORK DIAGNOSTICS =====")
    
    # Check database
    from local_db_adapter import LocalDatabaseAdapter
    db_adapter = LocalDatabaseAdapter()
    
    print("\nDatabase Check:")
    try:
        blocks = db_adapter.load_blocks()
        shards = db_adapter.load_shards()
        users = db_adapter.load_all_user_reputations()
        print(f"  ✓ Database accessible")
        print(f"  ✓ Blocks table: {len(blocks)} records")
        print(f"  ✓ Shards table: {len(shards)} records")
        print(f"  ✓ Users table: {len(users)} records")
    except Exception as e:
        print(f"  ✗ Database error: {e}")
        if args.repair:
            print("  Attempting repair...")
            try:
                db_adapter._ensure_tables()
                print("  ✓ Database tables recreated")
            except Exception as repair_e:
                print(f"  ✗ Repair failed: {repair_e}")
    
    # Check blockchain integrity
    print("\nBlockchain Integrity Check:")
    try:
        blocks = db_adapter.load_blocks()
        if not blocks:
            print("  ! No blocks found, blockchain may be empty")
        else:
            # Verify block connections
            orphaned_blocks = 0
            for coords, block in blocks.items():
                if coords[0] > 0:  # Not genesis block
                    if 'time' not in block.previous_hashes:
                        orphaned_blocks += 1
            
            if orphaned_blocks > 0:
                print(f"  ! Found {orphaned_blocks} orphaned blocks")
                if args.repair:
                    print("  Repair not implemented for orphaned blocks")
            else:
                print("  ✓ All blocks properly connected")
            
            # Verify block hashes
            from blockchain.crypto import hash_data
            invalid_blocks = 0
            for coords, block in blocks.items():
                # Test data for hash calculation
                test_json = block.to_json()
                test_hash = hash_data(test_json)
                if test_hash != block.hash:
                    invalid_blocks += 1
            
            if invalid_blocks > 0:
                print(f"  ! Found {invalid_blocks} blocks with invalid hashes")
            else:
                print("  ✓ All block hashes valid")
    except Exception as e:
        print(f"  ✗ Blockchain check error: {e}")
    
    # Network connectivity check
    print("\nNetwork Connectivity Check:")
    peers = get_peer_list()
    if not peers:
        print("  ! No connected peers")
    else:
        print(f"  ✓ Connected to {len(peers)} peers")
    
    # Deeper diagnostics if requested
    if args.deep:
        print("\nDeep Diagnostics:")
        
        # Check shard integrity
        print("  Checking shard integrity...")
        from blockchain.crypto import hash_data
        shards = db_adapter.load_shards()
        invalid_shards = 0
        for shard_id, shard in shards.items():
            test_json = shard.to_json()
            test_hash = hash_data(test_json)
            if test_hash != shard.hash:
                invalid_shards += 1
        
        if invalid_shards > 0:
            print(f"  ! Found {invalid_shards} shards with invalid hashes")
        else:
            print(f"  ✓ All {len(shards)} shards have valid hashes")
        
        # Check shard references in blocks
        print("  Checking shard references...")
        blocks = db_adapter.load_blocks()
        missing_shard_refs = 0
        for coords, block in blocks.items():
            for shard_id in block.shard_references:
                if shard_id not in shards:
                    missing_shard_refs += 1
        
        if missing_shard_refs > 0:
            print(f"  ! Found {missing_shard_refs} references to missing shards")
        else:
            print("  ✓ All shard references are valid")
    
    print("\nDiagnostics completed.")
    return True

# Main function
def main():
    """Main entry point for the EternaLattice node."""
    args = parser.parse_args()
    
    # Set default command if none provided
    if not args.command:
        parser.print_help()
        return
    
    # Execute the requested command
    commands = {
        'start': start_node,
        'add_shard': add_shard_command,
        'mine': mine_block_command,
        'status': status_command,
        'explore': explore_command,
        'search_shards': search_shards_command,
        'peers': peers_command,
        'leaderboard': leaderboard_command,
        'profile': profile_command,
        'backup': backup_command,
        'diagnostics': diagnostics_command,
        'help': help_command
    }
    
    if args.command in commands:
        result = commands[args.command](args)
        
        # Keep the program running if start command was executed
        if args.command == 'start':
            try:
                print("\nNode is running. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping node...")
                # Perform cleanup here if needed
                sys.exit(0)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main()