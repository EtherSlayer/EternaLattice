"""
Routes for the standalone node web interface.
These routes provide a web UI for interacting with an EternaLattice node in standalone mode.
"""

import os
import time
import json
import logging
import datetime
from typing import List, Dict, Any, Optional

try:
    from flask import Blueprint, render_template, jsonify, request, redirect, url_for
    from flask import flash, send_file, make_response
except ImportError:
    # For better error messages on any Flask imports
    logging.critical("Flask is not installed. Please install with 'pip install flask'")

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError:
    # For better error messages on any matplotlib imports
    logging.critical("Matplotlib is not installed. Please install with 'pip install matplotlib'")

import io
import base64

try:
    import networkit as nk
except ImportError:
    # For better error messages on any networkit imports
    logging.critical("NetworkIt is not installed. Please install with 'pip install networkit'")
    nk = None  # Set to None to avoid runtime errors

try:
    import numpy as np
except ImportError:
    # For better error messages on any numpy imports
    logging.critical("NumPy is not installed. Please install with 'pip install numpy'")
    np = None  # Set to None to avoid runtime errors

# Import EternaLattice modules
from blockchain.crypto import hash_data, generate_key_pair
from blockchain.core import get_blockchain_stats, get_block, get_latest_block, mine_block
from models import Block, MemoryShard, UserReputation
from local_db_adapter import LocalDatabaseAdapter
from improved_charts_update import enhanced_consensus_evolution_chart, enhanced_fitness_landscape, enhanced_blockchain_stats_chart

# Set up logging
logger = logging.getLogger(__name__)

# Create blueprint for standalone routes
standalone_bp = Blueprint('standalone', __name__, template_folder='templates')

# Database adapter
db_adapter = LocalDatabaseAdapter()

# Global variables
local_node_id = os.environ.get('ETERNALATTICE_NODE_ID', hash_data(str(time.time()))[:12])

def format_timestamp(timestamp):
    """Format a Unix timestamp as a human-readable date/time."""
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def format_duration(seconds):
    """Format seconds as a human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    else:
        return f"{seconds / 86400:.1f} days"

def get_recent_logs(count=20):
    """Get recent log entries for display in the UI."""
    try:
        log_file = 'eternalattice.log'
        if not os.path.exists(log_file):
            return ["No log file found."]

        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Get the last 'count' non-empty lines
        logs = [line.strip() for line in lines if line.strip()]
        return logs[-count:]
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return ["Error reading logs."]

def get_node_stats():
    """Get statistics about the node for display in the UI."""
    try:
        # Get blockchain stats
        blockchain_stats = db_adapter.get_blockchain_stats()
        
        # Get current timestamp
        current_time = time.time()
        
        # Calculate uptime from log file creation time or modification time
        log_file = 'eternalattice.log'
        if os.path.exists(log_file):
            log_mtime = os.path.getmtime(log_file)
            uptime = current_time - log_mtime
        else:
            uptime = 0
        
        # Get memory shard count
        memory_shards = db_adapter.load_shards()
        shard_count = len(memory_shards)
        
        # Get user count
        users = db_adapter.load_all_user_reputations()
        user_count = len(users)
        
        # Get latest block
        blocks = db_adapter.load_blocks()
        block_count = len(blocks)
        
        # Get last activity time
        last_activity_time = max([b.timestamp for b in blocks.values()]) if blocks else 0
        
        # Return stats
        return {
            "node_id": local_node_id,
            "uptime": format_duration(uptime),
            "uptime_seconds": uptime,
            "block_count": block_count,
            "shard_count": shard_count,
            "user_count": user_count,
            "last_activity": format_timestamp(last_activity_time) if last_activity_time > 0 else "Never",
            "blockchain_health": blockchain_stats.get("health_score", 0),
            "consensus_stability": blockchain_stats.get("consensus_stability", 0),
            "total_fitness": blockchain_stats.get("total_fitness", 0),
        }
    except Exception as e:
        logger.error(f"Error getting node stats: {e}")
        return {
            "node_id": local_node_id,
            "uptime": "Unknown",
            "uptime_seconds": 0,
            "block_count": 0,
            "shard_count": 0,
            "user_count": 0,
            "last_activity": "Never",
            "blockchain_health": 0,
            "consensus_stability": 0,
            "total_fitness": 0,
            "error": str(e)
        }

@standalone_bp.route('/')
def index():
    """Render the main standalone node dashboard."""
    try:
        # Get node stats
        stats = get_node_stats()
        
        # Get recent logs
        logs = get_recent_logs()
        
        # Get blockchain data for visualization
        blocks = db_adapter.load_blocks()
        
        # Generate blockchain visualization
        if blocks:
            time_blocks = sorted(blocks.values(), key=lambda b: b.timestamp)
            
            # Create a base64 encoded PNG for each visualization
            consensus_chart = enhanced_consensus_evolution_chart(blocks, time_blocks)
            fitness_chart = enhanced_fitness_landscape(blocks, time_blocks)
            stats_chart = enhanced_blockchain_stats_chart(blocks, db_adapter.get_blockchain_stats())
        else:
            consensus_chart = None
            fitness_chart = None
            stats_chart = None
        
        # Get recent shards
        shards = list(db_adapter.load_shards().values())
        recent_shards = sorted(shards, key=lambda s: s.creation_time, reverse=True)[:5]
        
        # Get leaderboard data
        users = db_adapter.load_all_user_reputations()
        top_users = sorted(users, key=lambda u: u.total_points, reverse=True)[:5]
        
        # Render the dashboard template
        return render_template(
            'standalone_dashboard.html',
            stats=stats,
            logs=logs,
            recent_shards=recent_shards,
            top_users=top_users,
            consensus_chart=consensus_chart,
            fitness_chart=fitness_chart,
            stats_chart=stats_chart,
            format_timestamp=format_timestamp
        )
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return f"Error rendering dashboard: {e}", 500

@standalone_bp.route('/api/mine_block', methods=['POST'])
def api_mine_block():
    """API endpoint to mine a new block."""
    try:
        # Get parameters from form
        x = request.form.get('x', type=int)
        y = request.form.get('y', type=int)
        z = request.form.get('z', type=int)
        auto = request.form.get('auto', 'false').lower() == 'true'
        
        # For auto mode, we need to determine reasonable coordinates
        # In a real implementation, mine_block would handle this with auto_position parameter
        coordinates = None
        
        if auto:
            # Let the blockchain core algorithm choose the best coordinates
            # If mine_block doesn't support auto_position, we can calculate ourselves
            try:
                # Use simple approach - get max x and increment
                blocks = db_adapter.load_blocks()
                if blocks:
                    max_x = max(b.x_coord for b in blocks.values())
                    coordinates = (max_x + 1, 0, 0)  # Next block in sequence
                else:
                    coordinates = (1, 0, 0)  # First non-genesis block
            except Exception as coord_err:
                logger.warning(f"Auto coordinate calculation failed: {coord_err}, using default")
                coordinates = (1, 0, 0)  # Default if calculation fails
                
        elif x is not None and y is not None and z is not None:
            # Use the coordinates provided by the user
            coordinates = (x, y, z)
        else:
            # Invalid parameters
            return jsonify({"success": False, "message": "Invalid parameters"})
        
        # Attempt to mine the block
        try:
            # Call mine_block with the appropriate parameters for your implementation
            # This might need adjustment based on your actual mine_block signature
            success, block = mine_block(
                coordinates=coordinates,
                miner_id=local_node_id,
                shard_references=[],  # No shards referenced by default
                traits=None  # Use default traits
            )
        except TypeError as e:
            # Fall back to simplified approach if parameters don't match
            logger.warning(f"Mine block parameter mismatch: {e}, trying alternative approach")
            success, block = mine_block(miner_id=local_node_id)
        
        if success:
            # Save the block to the database
            db_adapter.save_block(block)
            
            # Update user reputation
            user = db_adapter.load_user_reputation(local_node_id)
            if not user:
                user = UserReputation(
                    user_id=local_node_id,
                    username="local_user",
                    display_name="Local User"
                )
            
            user.mined_blocks += 1
            user.total_points += 10
            user.last_contribution = int(time.time())
            
            # Add to contribution history
            history = user.contribution_history
            history.append({
                "timestamp": int(time.time()),
                "activity": "mine_block",
                "points": 10,
                "description": f"Mined block at ({block.x_coord}, {block.y_coord}, {block.z_coord})"
            })
            user.contribution_history = history
            
            # Check for and award badges
            badges = set(user.badges)
            
            # First block badge
            if user.mined_blocks == 1:
                badges.add("blockchain_pioneer")
            
            # Experienced miner badge (5+ blocks)
            if user.mined_blocks >= 5:
                badges.add("lattice_architect")
                
            # Master miner badge (10+ blocks)
            if user.mined_blocks >= 10:
                badges.add("network_guardian")
                
            # Update badges
            user.badges = badges
            
            # Save user reputation
            db_adapter.save_user_reputation(user)
            
            # Include detailed block info in response
            return jsonify({
                "success": True, 
                "message": f"Block successfully mined at ({block.x_coord}, {block.y_coord}, {block.z_coord})",
                "block": {
                    "coords": [block.x_coord, block.y_coord, block.z_coord],
                    "hash": block.hash,
                    "timestamp": block.timestamp,
                    "fitness_score": block.fitness_score,
                    "badges_earned": list(badges - set(user.badges))  # New badges earned
                }
            })
        else:
            return jsonify({
                "success": False, 
                "message": "Failed to mine block. Another miner may have already mined at these coordinates."
            })
    except Exception as e:
        logger.error(f"Error mining block: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@standalone_bp.route('/api/add_shard', methods=['POST'])
def api_add_shard():
    """API endpoint to add a new memory shard."""
    try:
        # Get parameters from form
        content_type = request.form.get('type', 'text')
        category = request.form.get('category', 'general')
        region = request.form.get('region', 'global')
        tags = request.form.get('tags', '')
        
        # Process content based on type
        data = ""
        metadata = {
            "creator": local_node_id,
            "source": "standalone_node",
            "tags": [tag.strip() for tag in tags.split(',')] if tags else []
        }
        
        if content_type == 'text':
            data = request.form.get('data', '')
            if not data:
                return jsonify({"success": False, "message": "Shard data cannot be empty"})
            metadata["type"] = "text"
            
        elif content_type == 'image':
            if 'file' not in request.files:
                return jsonify({"success": False, "message": "No file uploaded"})
                
            file = request.files['file']
            if file.filename == '':
                return jsonify({"success": False, "message": "No file selected"})
                
            if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                return jsonify({"success": False, "message": "Only JPEG and PNG images are supported"})
            
            # Read file as base64 for storage
            file_data = file.read()
            if len(file_data) > 25 * 1024 * 1024:  # 25MB limit
                return jsonify({"success": False, "message": "File exceeds 25MB limit"})
                
            # Convert to base64 string for storage
            import base64
            data = base64.b64encode(file_data).decode('utf-8')
            
            # Set metadata
            metadata["type"] = file.content_type
            metadata["mime_type"] = file.content_type
            metadata["filename"] = file.filename
            metadata["size"] = len(file_data)
            
        elif content_type == 'pdf':
            if 'file' not in request.files:
                return jsonify({"success": False, "message": "No file uploaded"})
                
            file = request.files['file']
            if file.filename == '':
                return jsonify({"success": False, "message": "No file selected"})
                
            if not file.filename.lower().endswith('.pdf'):
                return jsonify({"success": False, "message": "Only PDF documents are supported"})
            
            # Read file as base64 for storage
            file_data = file.read()
            if len(file_data) > 25 * 1024 * 1024:  # 25MB limit
                return jsonify({"success": False, "message": "File exceeds 25MB limit"})
                
            # Convert to base64 string for storage
            import base64
            data = base64.b64encode(file_data).decode('utf-8')
            
            # Set metadata
            metadata["type"] = "application/pdf"
            metadata["mime_type"] = "application/pdf"
            metadata["filename"] = file.filename
            metadata["size"] = len(file_data)
        else:
            return jsonify({"success": False, "message": "Invalid content type"})
        
        # Create a new shard
        shard = MemoryShard(
            data=data,
            category=category,
            region=region,
            metadata=metadata
        )
        
        # Sign the shard with the local node ID
        # We need to create a signature for authenticity
        from blockchain.crypto import sign_data
        signature = sign_data(data, local_node_id)
        shard.signature = signature
        shard.hash = hash_data(data)
        
        # Save the shard to the database
        success = db_adapter.save_shard(shard)
        
        if success:
            # Update user reputation and award badges as appropriate
            user = db_adapter.load_user_reputation(local_node_id)
            if not user:
                user = UserReputation(
                    user_id=local_node_id,
                    username="local_user",
                    display_name="Local User"
                )
            
            user.created_shards += 1
            user.total_points += 5
            user.last_contribution = int(time.time())
            
            # Add to contribution history
            history = user.contribution_history
            
            # Format description based on content type
            if content_type == 'text':
                description = f"Created text shard: {data[:30]}..." if len(data) > 30 else f"Created text shard: {data}"
            elif content_type == 'image':
                description = f"Created image shard: {metadata.get('filename', 'Unnamed image')}"
            elif content_type == 'pdf':
                description = f"Created PDF shard: {metadata.get('filename', 'Unnamed document')}"
            else:
                description = "Created new memory shard"
                
            history.append({
                "timestamp": int(time.time()),
                "activity": "create_shard",
                "points": 5,
                "description": description
            })
            user.contribution_history = history
            
            # Check for and award badges
            badges = set(user.badges)
            
            # First shard badge
            if user.created_shards == 1:
                badges.add("first_shard")
            
            # Dedicated contributor badge (5+ shards)
            if user.created_shards >= 5:
                badges.add("dedicated_contributor")
                
            # Knowledge keeper badge (10+ shards)
            if user.created_shards >= 10:
                badges.add("knowledge_keeper")
                
            # Different content types badge
            shard_types = set()
            for prev_shard in db_adapter.load_shards().values():
                if prev_shard.metadata.get("creator") == local_node_id:
                    shard_types.add(prev_shard.metadata.get("type", "text"))
            
            if len(shard_types) >= 3:
                badges.add("diverse_curator")
                
            # Update badges
            user.badges = badges
            
            # Save user reputation
            db_adapter.save_user_reputation(user)
            
            return jsonify({
                "success": True, 
                "message": "Memory shard successfully added to the network",
                "shard": {
                    "id": shard.shard_id,
                    "category": shard.category,
                    "region": shard.region,
                    "creation_time": shard.creation_time,
                    "type": metadata["type"]
                }
            })
        else:
            return jsonify({"success": False, "message": "Failed to save memory shard"})
    except Exception as e:
        logger.error(f"Error adding shard: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@standalone_bp.route('/shards')
def shards():
    """Show all memory shards."""
    try:
        # Get all shards
        all_shards = db_adapter.load_shards()
        shards_list = list(all_shards.values())
        
        # Process shards to ensure metadata is correct
        import json
        for shard in shards_list:
            # Ensure the metadata is properly handled
            if isinstance(shard.metadata, str):
                try:
                    # If metadata is stored as a string, parse it
                    shard.metadata = json.loads(shard.metadata)
                    logger.info(f"Converted metadata from string to dictionary for shard {shard.shard_id}")
                except json.JSONDecodeError:
                    logger.error(f"Invalid metadata JSON for shard {shard.shard_id}: {shard.metadata}")
                    # Provide a default metadata dictionary to avoid errors
                    shard.metadata = {
                        "type": "text",
                        "creator": "unknown",
                        "tags": []
                    }
        
        # Sort by creation time (newest first)
        shards_list.sort(key=lambda s: s.creation_time, reverse=True)
        
        # Get filter parameters
        category = request.args.get('category')
        region = request.args.get('region')
        search = request.args.get('search')
        
        # Apply filters
        if category:
            shards_list = [s for s in shards_list if s.category == category]
        
        if region:
            shards_list = [s for s in shards_list if s.region == region]
        
        if search:
            shards_list = [s for s in shards_list if search.lower() in s.data.lower()]
        
        # Get unique categories and regions for filter dropdowns
        categories = sorted(set(s.category for s in all_shards.values()))
        regions = sorted(set(s.region for s in all_shards.values()))
        
        # Render template
        return render_template(
            'shards.html',
            shards=shards_list,
            categories=categories,
            regions=regions,
            selected_category=category,
            selected_region=region,
            search_query=search,
            format_timestamp=format_timestamp
        )
    except Exception as e:
        logger.error(f"Error showing shards: {e}")
        return f"Error showing shards: {e}", 500

@standalone_bp.route('/blockchain')
def blockchain():
    """Show the blockchain."""
    try:
        # Get all blocks
        blocks_dict = db_adapter.load_blocks()
        
        # Convert to list and sort by time
        blocks_list = list(blocks_dict.values())
        blocks_list.sort(key=lambda b: b.timestamp, reverse=True)
        
        # Get filter parameters
        x = request.args.get('x', type=int)
        y = request.args.get('y', type=int)
        z = request.args.get('z', type=int)
        
        # Apply filters
        if x is not None:
            blocks_list = [b for b in blocks_list if b.x_coord == x]
        
        if y is not None:
            blocks_list = [b for b in blocks_list if b.y_coord == y]
        
        if z is not None:
            blocks_list = [b for b in blocks_list if b.z_coord == z]
        
        # Get unique coordinates for filter dropdowns
        x_coords = sorted(set(b.x_coord for b in blocks_dict.values()))
        y_coords = sorted(set(b.y_coord for b in blocks_dict.values()))
        z_coords = sorted(set(b.z_coord for b in blocks_dict.values()))
        
        # Generate blockchain visualization
        if blocks_list:
            time_blocks = sorted(blocks_dict.values(), key=lambda b: b.timestamp)
            
            # Create a base64 encoded PNG for each visualization
            consensus_chart = enhanced_consensus_evolution_chart(blocks_dict, time_blocks)
            fitness_chart = enhanced_fitness_landscape(blocks_dict, time_blocks)
            stats_chart = enhanced_blockchain_stats_chart(blocks_dict, db_adapter.get_blockchain_stats())
        else:
            consensus_chart = None
            fitness_chart = None
            stats_chart = None
        
        # Render template
        return render_template(
            'blockchain.html',
            blocks=blocks_list,
            x_coords=x_coords,
            y_coords=y_coords,
            z_coords=z_coords,
            selected_x=x,
            selected_y=y,
            selected_z=z,
            consensus_chart=consensus_chart,
            fitness_chart=fitness_chart,
            stats_chart=stats_chart,
            format_timestamp=format_timestamp
        )
    except Exception as e:
        logger.error(f"Error showing blockchain: {e}")
        return f"Error showing blockchain: {e}", 500

@standalone_bp.route('/network')
def network():
    """Show network information."""
    try:
        # Get node stats
        stats = get_node_stats()
        
        # In a real implementation, we would get peer information here
        peers = []
        
        # Render template
        return render_template(
            'network.html',
            stats=stats,
            peers=peers,
            format_timestamp=format_timestamp,
            format_duration=format_duration
        )
    except Exception as e:
        logger.error(f"Error showing network info: {e}")
        return f"Error showing network info: {e}", 500

@standalone_bp.route('/leaderboard')
def leaderboard():
    """Show the leaderboard of top contributors."""
    try:
        # Get all users
        users = db_adapter.load_all_user_reputations()
        
        # Get filter parameter
        category = request.args.get('category')
        
        if category:
            # If category filter is applied, we need to calculate category-specific points
            filtered_users = []
            
            for user in users:
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
            users = sorted(filtered_users, key=lambda u: u.total_points, reverse=True)
        else:
            # Sort by total points
            users = sorted(users, key=lambda u: u.total_points, reverse=True)
        
        # Define available categories for filter dropdown
        categories = [
            {"id": "mine_block", "name": "Block Mining"},
            {"id": "create_shard", "name": "Shard Creation"},
            {"id": "preserve_shard", "name": "Shard Preservation"}
        ]
        
        # Render template
        return render_template(
            'leaderboard.html',
            users=users,
            categories=categories,
            selected_category=category,
            format_timestamp=format_timestamp
        )
    except Exception as e:
        logger.error(f"Error showing leaderboard: {e}")
        return f"Error showing leaderboard: {e}", 500

@standalone_bp.route('/shard/<shard_id>')
def view_shard(shard_id):
    """Show details for a specific memory shard."""
    try:
        # Load shards from database
        all_shards = db_adapter.load_shards()
        
        # Find the specific shard
        if shard_id not in all_shards:
            flash(f"Memory shard with ID {shard_id} not found", "error")
            return redirect(url_for('standalone.shards'))
        
        shard = all_shards[shard_id]
        
        # Ensure the metadata is properly handled
        if isinstance(shard.metadata, str):
            try:
                # If metadata is stored as a string (which might be happening), parse it
                import json
                shard.metadata = json.loads(shard.metadata)
                logger.info("Converted metadata from string to dictionary")
            except json.JSONDecodeError:
                logger.error(f"Invalid metadata JSON for shard {shard_id}: {shard.metadata}")
                # Provide a default metadata dictionary to avoid errors
                shard.metadata = {
                    "type": "text",
                    "creator": "unknown",
                    "tags": []
                }
        
        # Render template
        return render_template(
            'shard_detail.html',
            shard=shard,
            format_timestamp=format_timestamp
        )
    except Exception as e:
        logger.error(f"Error showing shard detail: {e}")
        return f"Error showing shard detail: {e}", 500

@standalone_bp.route('/shard/<shard_id>/download')
def download_shard(shard_id):
    """Download a shard as a file."""
    try:
        # Load shards from database
        all_shards = db_adapter.load_shards()
        
        # Find the specific shard
        if shard_id not in all_shards:
            flash(f"Memory shard with ID {shard_id} not found", "error")
            return redirect(url_for('standalone.shards'))
        
        shard = all_shards[shard_id]
        
        # Ensure the metadata is properly handled
        if isinstance(shard.metadata, str):
            try:
                # If metadata is stored as a string, parse it
                import json
                shard.metadata = json.loads(shard.metadata)
                logger.info("Converted metadata from string to dictionary")
            except json.JSONDecodeError:
                logger.error(f"Invalid metadata JSON for shard {shard_id}: {shard.metadata}")
                # Provide a default metadata dictionary to avoid errors
                shard.metadata = {
                    "type": "text",
                    "creator": "unknown",
                    "tags": []
                }
        
        # Determine content type from metadata
        content_type = shard.metadata.get('type', 'text/plain')
        
        # For file types, decode base64 data
        if content_type in ['image/jpeg', 'image/png', 'application/pdf']:
            import base64
            file_data = base64.b64decode(shard.data)
            
            # Create a response with the file data
            response = make_response(file_data)
            response.headers.set('Content-Type', content_type)
            
            # Set filename based on metadata or default
            filename = shard.metadata.get('filename', f'shard_{shard_id}.{content_type.split("/")[1]}')
            response.headers.set('Content-Disposition', f'attachment; filename="{filename}"')
            
            return response
        else:
            # For text data, just download as text file
            response = make_response(shard.data)
            response.headers.set('Content-Type', 'text/plain')
            response.headers.set('Content-Disposition', f'attachment; filename="shard_{shard_id}.txt"')
            
            return response
    except Exception as e:
        logger.error(f"Error downloading shard: {e}")
        return f"Error downloading shard: {e}", 500

@standalone_bp.route('/profile')
@standalone_bp.route('/profile/<user_id>')
def profile(user_id=None):
    """Show a user profile."""
    try:
        # Get user ID (local or specified)
        user_id = user_id or local_node_id
        
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
                flash("Created new user profile for local node")
            else:
                flash(f"User with ID {user_id[:8]}... not found", "error")
                return redirect(url_for('standalone.index'))
        
        # Get recent activity
        recent_history = sorted(user.contribution_history, key=lambda c: c.get('timestamp', 0), reverse=True)
        
        # Function to get badge information
        def get_badge_info(badge_id):
            badge_map = {
                "first_shard": {
                    "title": "First Contribution",
                    "description": "Created your first memory shard",
                    "icon": '<i class="bi bi-star-fill"></i>',
                    "color": "#FFD700"  # Gold
                },
                "dedicated_contributor": {
                    "title": "Dedicated Contributor",
                    "description": "Created at least 5 memory shards",
                    "icon": '<i class="bi bi-trophy-fill"></i>',
                    "color": "#C0C0C0"  # Silver
                },
                "knowledge_keeper": {
                    "title": "Knowledge Keeper",
                    "description": "Created at least 10 memory shards",
                    "icon": '<i class="bi bi-book-fill"></i>',
                    "color": "#CD7F32"  # Bronze
                },
                "diverse_curator": {
                    "title": "Diverse Curator",
                    "description": "Added different types of content to the network",
                    "icon": '<i class="bi bi-grid-3x3-gap-fill"></i>',
                    "color": "#9370DB"  # Purple
                },
                "blockchain_pioneer": {
                    "title": "Blockchain Pioneer",
                    "description": "Mined your first block in the lattice",
                    "icon": '<i class="bi bi-box-fill"></i>',
                    "color": "#20B2AA"  # Light Sea Green
                },
                "lattice_architect": {
                    "title": "Lattice Architect",
                    "description": "Mined at least 5 blocks in the lattice",
                    "icon": '<i class="bi bi-bricks"></i>',
                    "color": "#4682B4"  # Steel Blue
                },
                "network_guardian": {
                    "title": "Network Guardian",
                    "description": "Mined at least 10 blocks in the lattice",
                    "icon": '<i class="bi bi-shield-fill-check"></i>',
                    "color": "#008080"  # Teal
                }
            }
            
            # Return default if badge not found
            if badge_id not in badge_map:
                return {
                    "title": badge_id.replace("_", " ").title(),
                    "description": "Achievement unlocked",
                    "icon": '<i class="bi bi-award-fill"></i>',
                    "color": "#6c757d"  # Gray
                }
                
            return badge_map[badge_id]
        
        # Render template
        return render_template(
            'profile.html',
            user=user,
            recent_history=recent_history,
            format_timestamp=format_timestamp,
            get_badge_info=get_badge_info,
            is_local_user=(user_id == local_node_id)
        )
    except Exception as e:
        logger.error(f"Error showing profile: {e}")
        return f"Error showing profile: {e}", 500

@standalone_bp.route('/api/update_profile', methods=['POST'])
def api_update_profile():
    """API endpoint to update the local user profile."""
    try:
        # Get parameters from form
        username = request.form.get('username')
        display_name = request.form.get('display_name')
        
        # Load user data
        user = db_adapter.load_user_reputation(local_node_id)
        
        if not user:
            # Create a new user record
            user = UserReputation(
                user_id=local_node_id,
                username=username or "local_user",
                display_name=display_name or "Local User"
            )
        else:
            # Update existing user
            if username:
                user.username = username
            if display_name:
                user.display_name = display_name
        
        # Save user reputation
        db_adapter.save_user_reputation(user)
        
        return jsonify({
            "success": True, 
            "message": "Profile successfully updated",
            "user": {
                "username": user.username,
                "display_name": user.display_name
            }
        })
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})