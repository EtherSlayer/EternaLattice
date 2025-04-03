"""
Visualization utilities for the EternaLattice blockchain.
"""
import logging
import json
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import io
import base64

from models import Block, MemoryShard
from blockchain.core import blockchain, get_blocks_by_dimension
import config

# Configure logging
logger = logging.getLogger(__name__)

# Configure matplotlib to use Agg backend
matplotlib.use('Agg')

def generate_3d_lattice_data() -> Dict[str, Any]:
    """
    Generate data for a 3D visualization of the blockchain lattice.
    
    Returns:
        Dict: Data structure for 3D visualization
    """
    # Prepare nodes and links for visualization
    nodes = []
    links = []
    
    # Handle empty blockchain
    if not blockchain:
        return {
            "nodes": [],
            "links": [],
            "dimensions": {
                "x": {"min": 0, "max": 0},
                "y": {"min": 0, "max": 0},
                "z": {"min": 0, "max": 0}
            }
        }
    
    # Track dimension ranges
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    # Add all blocks as nodes
    for coords, block in blockchain.items():
        x, y, z = coords
        
        # Update dimension ranges
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        min_z = min(min_z, z)
        max_z = max(max_z, z)
        
        nodes.append({
            "id": block.hash[:8],
            "x": x,
            "y": y,
            "z": z,
            "value": 1,
            "hash": block.hash,
            "timestamp": block.timestamp
        })
    
    # Add links based on previous_hashes
    for coords, block in blockchain.items():
        for dim, prev_hash in block.previous_hashes.items():
            # Find the node with this hash
            target_node = next((n for n in nodes if n["hash"] == prev_hash), None)
            if target_node:
                links.append({
                    "source": block.hash[:8],
                    "target": target_node["id"],
                    "dimension": dim,
                    "value": 1
                })
    
    return {
        "nodes": nodes,
        "links": links,
        "dimensions": {
            "x": {"name": "Time", "min": min([n["x"] for n in nodes]) if nodes else 0, 
                 "max": max([n["x"] for n in nodes]) if nodes else 0},
            "y": {"name": "Category", "min": min([n["y"] for n in nodes]) if nodes else 0, 
                 "max": max([n["y"] for n in nodes]) if nodes else 0},
            "z": {"name": "Region", "min": min([n["z"] for n in nodes]) if nodes else 0, 
                 "max": max([n["z"] for n in nodes]) if nodes else 0}
        }
    }

def generate_2d_projection(dimension: str = "time") -> Dict[str, Any]:
    """
    Generate a 2D projection of the blockchain along a specific dimension.
    
    Args:
        dimension: The dimension to project onto ("time", "category", or "region")
        
    Returns:
        Dict: Data structure for 2D visualization
    """
    try:
        # Handle empty blockchain
        if not blockchain:
            return {"nodes": [], "links": [], "dimension": dimension}
        
        # Map dimension names to coordinate indices
        dim_map = {"time": 0, "category": 1, "region": 2}
        if dimension not in dim_map:
            logger.error(f"Invalid dimension requested: {dimension}")
            return {"nodes": [], "links": [], "dimension": dimension, "error": "Invalid dimension"}
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all blocks as nodes
        for coords, block in blockchain.items():
            x, y, z = coords
            # Use coordinate value for the selected dimension as node position
            position = coords[dim_map[dimension]]
            
            # Use other two dimensions for the node position
            other_dims = [i for i in range(3) if i != dim_map[dimension]]
            other_pos = [coords[i] for i in other_dims]
            
            # Add the node to the graph
            G.add_node(
                block.hash[:8],
                hash=block.hash,
                position=position,
                other_pos=other_pos,
                timestamp=block.timestamp,
                coords=coords
            )
        
        # Add edges based on previous_hashes 
        for coords, block in blockchain.items():
            for dim, prev_hash in block.previous_hashes.items():
                # Find the target node with this hash
                for node, attrs in G.nodes(data=True):
                    if attrs["hash"] == prev_hash:
                        G.add_edge(block.hash[:8], node, dimension=dim)
                        break
        
        # Prepare the data for serialization
        nodes = []
        for node, data in G.nodes(data=True):
            node_data = {
                "id": node,
                "hash": data["hash"],
                "position": data["position"],
                "other_pos": data["other_pos"],
                "timestamp": data["timestamp"],
                "coords": [data["coords"][0], data["coords"][1], data["coords"][2]]  # Make JSON serializable
            }
            nodes.append(node_data)
        
        links = []
        for u, v, data in G.edges(data=True):
            link_data = {
                "source": u,
                "target": v,
                "dimension": data["dimension"]
            }
            links.append(link_data)
        
        # Return the formatted data
        return {
            "nodes": nodes,
            "links": links,
            "dimension": dimension
        }
    except Exception as e:
        logger.error(f"Error generating 2D projection: {e}")
        return {
            "nodes": [],
            "links": [],
            "dimension": dimension,
            "error": str(e)
        }

def generate_consensus_evolution_chart() -> str:
    """
    Generate a chart showing the evolution of consensus traits over time.
    
    Returns:
        str: Base64 encoded PNG image of the chart
    """
    try:
        # For improved chart rendering, use the enhanced chart functions
        from improved_charts_update import enhanced_consensus_evolution_chart
        
        # Get blocks ordered by time
        time_blocks = sorted(blockchain.values(), key=lambda b: b.coordinates[0])
        
        # Generate the enhanced chart
        return enhanced_consensus_evolution_chart(blockchain, time_blocks)
    except ImportError:
        logger.warning("Enhanced chart functions not available, using fallback")
        # Continue with original implementation
    except Exception as e:
        logger.error(f"Error using enhanced chart: {e}", exc_info=True)
        # Continue with original implementation
        
    # Original implementation as fallback
    """
    Generate a chart showing the evolution of consensus traits over time.
    
    Returns:
        str: Base64 encoded PNG image of the chart
    """
    try:
        # Collect trait data from blocks
        traits_over_time = {}
        
        # Get blocks ordered by time
        time_blocks = sorted(blockchain.values(), key=lambda b: b.coordinates[0])
        
        # Check if we have enough data
        if len(time_blocks) < 2:
            # Create a simple placeholder chart if not enough data
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.style.use('dark_background')
            ax.text(0.5, 0.5, 'Insufficient data - Need more blocks to show trait evolution',
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlabel('Time Dimension', fontsize=12, fontweight='bold')
            ax.set_ylabel('Trait Value', fontsize=12, fontweight='bold')
            ax.set_title('Consensus Trait Evolution', fontsize=16, fontweight='bold')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            # Extract trait data for each trait
            for block in time_blocks:
                for trait, value in block.consensus_traits.items():
                    if trait not in traits_over_time:
                        traits_over_time[trait] = []
                    traits_over_time[trait].append((block.coordinates[0], value))
            
            # Create the plot with enhanced styling
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.style.use('dark_background')
            
            # Define a vibrant color palette for better distinction
            colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F3', '#33FFF3']
            
            # Plot each trait with improved styling
            for i, (trait, values) in enumerate(traits_over_time.items()):
                if not values:
                    continue
                    
                x, y = zip(*values)
                color = colors[i % len(colors)]
                
                ax.plot(x, y, marker='o', linestyle='-', linewidth=2.5, markersize=8,
                       label=trait, color=color, alpha=0.9)
                
                # Add value labels at the last point for better readability
                if len(x) > 0:
                    last_x, last_y = x[-1], y[-1]
                    ax.annotate(f'{last_y:.2f}', (last_x, last_y), 
                               xytext=(7, 0), textcoords='offset points',
                               fontsize=10, color=color, fontweight='bold')
            
            # Better styling for axes and title
            ax.set_title('Consensus Traits Evolution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Time Dimension', fontsize=12, fontweight='bold')
            ax.set_ylabel('Trait Value', fontsize=12, fontweight='bold')
            
            # Improved legend with better positioning and styling
            legend = ax.legend(loc='upper left', fontsize=10, framealpha=0.8, 
                              edgecolor='gray', fancybox=True)
            
            # Set integer ticks for x-axis (time coordinates)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            # Enhanced grid for better readability
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add a subtle background gradient for visual appeal
            ax.set_facecolor('#0A0A0A')
            
            # Enhance tick parameters for better visibility
            ax.tick_params(axis='both', which='major', labelsize=10, colors='white')
            
            # Add a border around the plot area
            for spine in ax.spines.values():
                spine.set_color('gray')
                spine.set_linewidth(0.5)
        
        # Save the plot to a bytes buffer with higher DPI for sharper image
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        
        # Encode as base64
        img_str = base64.b64encode(buf.read()).decode('ascii')
        
        # Close the plot to free memory
        plt.close(fig)
        
        return img_str
        
    except Exception as e:
        logger.error(f"Error generating consensus evolution chart: {e}", exc_info=True)
        
        # Create a simple error chart
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use('dark_background')
        ax.text(0.5, 0.5, f'Error generating chart: {str(e)}',
               ha='center', va='center', fontsize=12, color='red')
        ax.set_title('Consensus Trait Evolution - Error', fontsize=14, fontweight='bold')
        
        # Save to a BytesIO object
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode as base64
        img_str = base64.b64encode(buf.read()).decode('ascii')
        
        # Close the figure to free memory
        plt.close(fig)
        
        return img_str

def generate_fitness_landscape() -> str:
    """
    Generate a visualization of the fitness landscape.
    
    Returns:
        str: Base64 encoded PNG image of the chart
    """
    try:
        # For improved chart rendering, use the enhanced chart functions
        from improved_charts_update import enhanced_fitness_landscape
        
        # Get blocks ordered by time
        time_blocks = sorted(blockchain.values(), key=lambda b: b.coordinates[0])
        
        # Generate the enhanced chart
        return enhanced_fitness_landscape(blockchain, time_blocks)
    except ImportError:
        logger.warning("Enhanced chart functions not available, using fallback")
        # Continue with original implementation
    except Exception as e:
        logger.error(f"Error using enhanced chart: {e}", exc_info=True)
        # Continue with original implementation
        
    # Original implementation as fallback
    """
    Generate a visualization of the fitness landscape.
    
    Returns:
        str: Base64 encoded PNG image of the chart
    """
    try:
        # Collect fitness scores from blocks
        fitness_scores = []
        
        # Get blocks ordered by time
        time_blocks = sorted(blockchain.values(), key=lambda b: b.coordinates[0])
        
        # Extract fitness scores
        for block in time_blocks:
            fitness_scores.append((block.coordinates[0], block.fitness_score))
        
        # Create the plot with improved styling
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use('dark_background')
        
        # Check if we have enough data
        if len(fitness_scores) < 2:
            ax.text(0.5, 0.5, 'Insufficient data - Need more blocks to show fitness landscape',
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 1)
        else:
            x, y = zip(*fitness_scores)
            
            # Create a more vibrant and clear visualization
            # Use a gradient-filled area under the curve for better visual impact
            ax.fill_between(x, 0, y, alpha=0.3, color='#00FFFF')
            
            # Plot the main line with enhanced styling
            ax.plot(x, y, marker='o', linestyle='-', linewidth=3, 
                   color='#00FFFF', markersize=8, markeredgecolor='white', 
                   markeredgewidth=1)
            
            # Add data point annotations for key points
            for i, (x_val, y_val) in enumerate(zip(x, y)):
                # Only annotate first, last and local maxima/minima points
                if i == 0 or i == len(x) - 1 or (i > 0 and i < len(y) - 1 and 
                                                (y[i-1] < y[i] > y[i+1] or y[i-1] > y[i] < y[i+1])):
                    ax.annotate(f'{y_val:.3f}', (x_val, y_val),
                               xytext=(0, 10), textcoords='offset points',
                               fontsize=10, fontweight='bold', color='white',
                               ha='center', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.3', fc='#004040', alpha=0.7, ec='cyan'))
            
            # Set y-axis to start from 0 for better proportion visualization
            ax.set_ylim(bottom=0)
            
            # Add a horizontal line at optimal fitness (1.0) for reference
            ax.axhline(y=1.0, color='#66FF66', linestyle='--', alpha=0.7, linewidth=1)
            ax.text(min(x), 1.02, 'Optimal Fitness', fontsize=9, 
                   color='#66FF66', ha='left', va='bottom')
        
        # Enhanced styling for axes and title
        ax.set_title('Consensus Fitness Landscape', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time Dimension', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fitness Score', fontsize=12, fontweight='bold')
        
        # Use integer ticks for x-axis
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Enhanced grid for better readability
        ax.grid(True, alpha=0.2, linestyle='--')
        
        # Set background color
        ax.set_facecolor('#0A0A0A')
        
        # Enhance tick parameters
        ax.tick_params(axis='both', which='major', labelsize=10, colors='white')
        
        # Add a border around the plot area
        for spine in ax.spines.values():
            spine.set_color('gray')
            spine.set_linewidth(0.5)
            
        # Add descriptive text about fitness score meaning
        if len(fitness_scores) >= 2:
            ax.text(0.02, 0.02, 
                   'Higher fitness scores indicate better adaptation to the network environment',
                   transform=ax.transAxes, fontsize=8, alpha=0.8, color='#BBBBBB')
        
        # Save the plot to a bytes buffer with higher DPI for sharper image
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        
        # Encode as base64
        img_str = base64.b64encode(buf.read()).decode('ascii')
        
        # Close the plot to free memory
        plt.close(fig)
        
        return img_str
        
    except Exception as e:
        logger.error(f"Error generating fitness landscape chart: {e}", exc_info=True)
        
        # Create an error chart
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use('dark_background')
        ax.text(0.5, 0.5, f'Error generating fitness landscape: {str(e)}',
               ha='center', va='center', fontsize=12, color='red')
        ax.set_title('Fitness Landscape - Error', fontsize=14, fontweight='bold')
        
        # Save to a BytesIO object
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode as base64
        img_str = base64.b64encode(buf.read()).decode('ascii')
        
        # Close the figure to free memory
        plt.close(fig)
        
        return img_str

def generate_blockchain_stats_chart() -> str:
    """
    Generate a chart showing statistics about the blockchain.
    
    Returns:
        str: Base64 encoded PNG image of the chart
    """
    try:
        # For improved chart rendering, use the enhanced chart functions
        from improved_charts_update import enhanced_blockchain_stats_chart
        
        # Get blockchain stats
        from blockchain.core import get_blockchain_stats
        stats = get_blockchain_stats()
        
        # Get blocks ordered by time
        time_blocks = sorted(blockchain.values(), key=lambda b: b.coordinates[0])
        
        # Generate the enhanced chart
        return enhanced_blockchain_stats_chart(blockchain, stats)
    except ImportError:
        logger.warning("Enhanced chart functions not available, using fallback")
        # Continue with original implementation
    except Exception as e:
        logger.error(f"Error using enhanced chart: {e}", exc_info=True)
        # Continue with original implementation
        
    # Original implementation as fallback
    """
    Generate a chart showing statistics about the blockchain.
    
    Returns:
        str: Base64 encoded PNG image of the chart
    """
    try:
        # Collect statistics
        block_counts = {}
        shard_refs = {}
        
        # Get blocks ordered by time
        time_blocks = sorted(blockchain.values(), key=lambda b: b.coordinates[0])
        
        # Create the plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.style.use('dark_background')
        
        # Check if we have enough data
        if len(time_blocks) < 2:
            ax1.text(0.5, 0.5, 'Insufficient data - Need more blocks to show growth statistics',
                    ha='center', va='center', fontsize=14, color='white')
            ax1.set_xlim(0, 10)
            ax1.set_ylim(0, 10)
            
            # Enhanced styling even for empty chart
            ax1.set_title('Blockchain Growth Statistics', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Time Dimension', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Total Blocks', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.2, linestyle='--')
            
        else:
            # Calculate cumulative block count and shard references
            total_blocks = 0
            total_shards = 0
            
            for block in time_blocks:
                time_coord = block.coordinates[0]
                total_blocks += 1
                total_shards += len(block.shard_references)
                
                block_counts[time_coord] = total_blocks
                shard_refs[time_coord] = total_shards
            
            # Plot block count on primary y-axis with enhanced styling
            x = list(block_counts.keys())
            y = list(block_counts.values())
            block_color = '#4287f5'  # Bright blue
            
            # Add bar chart as background for blocks (more visually impressive)
            ax1.bar(x, y, alpha=0.3, color=block_color, width=0.7, label='Blocks')
            
            # Plot line on top of bars for trend visibility
            ax1.plot(x, y, marker='o', linestyle='-', linewidth=2.5, 
                   markersize=8, color=block_color, markeredgecolor='white',
                   markeredgewidth=1)
            
            # Add value annotations for blocks
            for i, (x_val, y_val) in enumerate(zip(x, y)):
                if i == len(x) - 1 or i == 0 or y_val % 5 == 0:  # Only annotate some points to avoid clutter
                    ax1.annotate(f'{y_val}', (x_val, y_val),
                               xytext=(0, 7), textcoords='offset points',
                               fontsize=9, fontweight='bold', color=block_color,
                               ha='center')
            
            # Enhanced styling for block axis
            ax1.set_xlabel('Time Dimension', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Total Blocks', fontsize=12, fontweight='bold', color=block_color)
            ax1.tick_params(axis='y', labelcolor=block_color)
            
            # Set y-axis to start from 0 for better proportion visualization
            ax1.set_ylim(bottom=0)
            
            # Plot shard references on secondary y-axis with enhanced styling
            if shard_refs:
                ax2 = ax1.twinx()
                x = list(shard_refs.keys())
                y = list(shard_refs.values())
                shard_color = '#FF9500'  # Bright orange
                
                # Use a different style for shard references - filled area
                ax2.fill_between(x, 0, y, alpha=0.2, color=shard_color)
                ax2.plot(x, y, marker='s', linestyle='-', linewidth=2, 
                       markersize=7, color=shard_color, label='Shard References')
                
                # Add value annotations for shards
                for i, (x_val, y_val) in enumerate(zip(x, y)):
                    if i == len(x) - 1 or i == 0 or y_val % 10 == 0:  # Only annotate some points
                        ax2.annotate(f'{y_val}', (x_val, y_val),
                                   xytext=(0, -15), textcoords='offset points',
                                   fontsize=9, fontweight='bold', color=shard_color,
                                   ha='center')
                
                # Enhanced styling for shard axis
                ax2.set_ylabel('Total Shard References', fontsize=12, 
                             fontweight='bold', color=shard_color)
                ax2.tick_params(axis='y', labelcolor=shard_color)
                ax2.set_ylim(bottom=0)
            
            # Add a combined legend
            handles1, labels1 = ax1.get_legend_handles_labels()
            if shard_refs:
                handles2, labels2 = ax2.get_legend_handles_labels()
                all_handles = handles1 + handles2
                all_labels = labels1 + labels2
                fig.legend(all_handles, all_labels, loc='upper left', 
                         framealpha=0.8, edgecolor='gray', fancybox=True)
            else:
                ax1.legend(loc='upper left', framealpha=0.8, edgecolor='gray', fancybox=True)
            
            # Enhanced title
            fig.suptitle('Blockchain Growth Statistics', fontsize=16, fontweight='bold')
            
            # Add explanatory text
            if len(time_blocks) >= 3:
                fig.text(0.02, 0.02, 
                        'Blocks represent the growth of the blockchain lattice over time.\n'
                        'Shard references show the amount of knowledge preserved in the network.',
                        fontsize=8, alpha=0.8, color='#BBBBBB')
        
        # Use integer ticks for x-axis
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Enhanced grid for better readability
        ax1.grid(True, alpha=0.2, linestyle='--')
        
        # Better background
        ax1.set_facecolor('#0A0A0A')
        
        # Add border
        for spine in ax1.spines.values():
            spine.set_color('gray')
            spine.set_linewidth(0.5)
        
        # Save the plot to a bytes buffer with higher DPI for sharper image
        buf = io.BytesIO()
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for title and note
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        
        # Encode as base64
        img_str = base64.b64encode(buf.read()).decode('ascii')
        
        # Close the plot to free memory
        plt.close(fig)
        
        return img_str
        
    except Exception as e:
        logger.error(f"Error generating blockchain stats chart: {e}", exc_info=True)
        
        # Create an error chart
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use('dark_background')
        ax.text(0.5, 0.5, f'Error generating growth statistics: {str(e)}',
               ha='center', va='center', fontsize=12, color='red')
        ax.set_title('Blockchain Growth Statistics - Error', fontsize=14, fontweight='bold')
        
        # Save to a BytesIO object
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode as base64
        img_str = base64.b64encode(buf.read()).decode('ascii')
        
        # Close the figure to free memory
        plt.close(fig)
        
        return img_str
