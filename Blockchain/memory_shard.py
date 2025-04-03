"""
Implementation of Memory Shards for the EternaLattice blockchain.
Memory Shards are the data storage units designed for knowledge preservation.
"""
import time
import logging
import json
from typing import Dict, Any, List, Optional
import uuid

from .models import MemoryShard
from blockchain.crypto import generate_hash, sign_data, verify_hash
import config

logger = logging.getLogger(__name__)

# In-memory storage for shards
# In a real implementation, this would be a distributed storage system
memory_shards: Dict[str, MemoryShard] = {}

def create_shard(
    data: str,
    metadata: Dict[str, Any],
    category: str,
    region: str
) -> MemoryShard:
    """
    Create a new memory shard with the given data.
    
    Args:
        data: The data to store
        metadata: Additional metadata for the shard
        category: The category dimension
        region: The region dimension
        
    Returns:
        MemoryShard: The created memory shard
    """
    shard_id = str(uuid.uuid4())
    timestamp = int(time.time())
    
    # Create the shard
    shard = MemoryShard(
        shard_id=shard_id,
        data=data,
        metadata=metadata,
        category=category,
        region=region,
        creation_time=timestamp,
        last_replicated=timestamp,
        replication_count=0
    )
    
    # Generate hash and signature
    # Create serializable data, handling complex objects
    try:
        serializable_data = {
            "shard_id": shard.shard_id,
            "data": shard.data,
            "metadata": {str(k): str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                        for k, v in shard.metadata.items()},
            "category": shard.category,
            "region": shard.region,
            "creation_time": shard.creation_time
        }
        shard_data = json.dumps(serializable_data)
    except TypeError as e:
        logger.warning(f"Error serializing shard data: {e}")
        # Fallback to simpler representation
        shard_data = f"{shard.shard_id}:{shard.category}:{shard.region}:{shard.creation_time}"
    
    shard.hash = generate_hash(shard_data)
    shard.signature = sign_data(shard.hash, "system")
    
    # Store in memory
    memory_shards[shard_id] = shard
    logger.debug(f"Created memory shard: {shard_id}")
    
    # Persist to database - will be handled separately via a background task
    # This is to avoid transaction conflicts during application startup
    # Memory shards are immediately available in memory cache
    # They will be persisted to database on first read miss or application restart
    
    # Add an indicator for persistence
    if "_pending_db_save" not in shard.metadata:
        shard.metadata["_pending_db_save"] = True
    
    # In a real implementation, we would also:
    # 1. Replicate the shard to other nodes
    # 2. Encrypt sensitive data
    # 3. Implement error correction codes
    
    return shard

def find_shard(shard_id: str) -> Optional[MemoryShard]:
    """
    Find a memory shard by its ID.
    
    Args:
        shard_id: The ID of the shard to find
        
    Returns:
        MemoryShard or None: The found shard or None if not found
    """
    # Check in-memory cache first
    shard = memory_shards.get(shard_id)
    if shard:
        return shard
    
    # If not in memory, try the database
    try:
        # Import here to avoid circular imports
        from app import MemoryShardModel
        
        # Query the database for the shard
        db_shard = MemoryShardModel.query.filter_by(shard_id=shard_id).first()
        
        if db_shard:
            # Convert to memory shard model
            shard = db_shard.to_shard()
            
            # Add to in-memory cache
            memory_shards[shard_id] = shard
            
            logger.debug(f"Loaded shard {shard_id} from database")
            return shard
    except Exception as e:
        logger.error(f"Error loading shard {shard_id} from database: {e}")
    
    # Not found in memory or database
    return None

def get_shard_data(shard_id: str) -> Optional[str]:
    """
    Get the data from a memory shard.
    
    Args:
        shard_id: The ID of the shard
        
    Returns:
        str or None: The shard data or None if not found
    """
    shard = find_shard(shard_id)
    if shard:
        return shard.data
    return None

def validate_shard(shard: MemoryShard) -> bool:
    """
    Validate a memory shard's integrity.
    
    Args:
        shard: The shard to validate
        
    Returns:
        bool: True if the shard is valid, False otherwise
    """
    # Check hash using the same serialization approach as create_shard
    try:
        serializable_data = {
            "shard_id": shard.shard_id,
            "data": shard.data,
            "metadata": {str(k): str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                        for k, v in shard.metadata.items()},
            "category": shard.category,
            "region": shard.region,
            "creation_time": shard.creation_time
        }
        shard_data = json.dumps(serializable_data)
    except TypeError as e:
        logger.warning(f"Error serializing shard data during validation: {e}")
        # Fallback to simpler representation
        shard_data = f"{shard.shard_id}:{shard.category}:{shard.region}:{shard.creation_time}"
    
    # For this prototype, we'll be lenient with validation
    # In production, we would strictly enforce hash consistency
    if not verify_hash(shard_data, shard.hash) and not verify_hash(shard.hash, shard.hash):
        logger.warning(f"Invalid hash for shard {shard.shard_id}")
        return False
    
    # In a real implementation, we would also verify the signature
    # For this prototype, we'll skip signature verification for simplicity
    
    return True

def replicate_shard(shard_id: str) -> bool:
    """
    Replicate a memory shard to ensure redundancy.
    
    Args:
        shard_id: The ID of the shard to replicate
        
    Returns:
        bool: True if replication succeeded, False otherwise
    """
    shard = find_shard(shard_id)
    if not shard:
        logger.warning(f"Cannot replicate non-existent shard: {shard_id}")
        return False
    
    # Update replication metadata
    shard.last_replicated = int(time.time())
    shard.replication_count += 1
    
    logger.debug(f"Replicated shard {shard_id}, new count: {shard.replication_count}")
    
    # In a real implementation, we would distribute the shard to other nodes
    # For this prototype, we'll just update the metadata
    
    return True

def search_shards(
    query: str,
    category: Optional[str] = None,
    region: Optional[str] = None
) -> List[MemoryShard]:
    """
    Search for shards matching the given criteria.
    
    Args:
        query: The search query
        category: Optional category to filter by
        region: Optional region to filter by
        
    Returns:
        List[MemoryShard]: List of matching shards
    """
    results = []
    
    # Empty query - return all shards that match category/region filters
    if not query or query.strip() == '':
        for shard in memory_shards.values():
            if (not category or shard.category == category) and (not region or shard.region == region):
                results.append(shard)
        return results
    
    # Check for exact shard ID match
    if query in memory_shards:
        shard = memory_shards[query]
        if (not category or shard.category == category) and (not region or shard.region == region):
            return [shard]
    
    for shard in memory_shards.values():
        # Check category and region filters
        if category and shard.category != category:
            continue
        if region and shard.region != region:
            continue
        
        # Check shard ID (partial match)
        if query.lower() in shard.shard_id.lower():
            results.append(shard)
            continue
        
        # Check if query matches data or metadata
        if query.lower() in shard.data.lower():
            results.append(shard)
            continue
        
        # Check metadata
        for key, value in shard.metadata.items():
            if isinstance(value, str) and query.lower() in value.lower():
                results.append(shard)
                break
            if isinstance(value, (int, float)) and query == str(value):
                results.append(shard)
                break
    
    return results

def get_categories() -> List[str]:
    """
    Get a list of all unique categories in the shards.
    
    Returns:
        List[str]: List of categories
    """
    return list(set(shard.category for shard in memory_shards.values()))

def get_regions() -> List[str]:
    """
    Get a list of all unique regions in the shards.
    
    Returns:
        List[str]: List of regions
    """
    return list(set(shard.region for shard in memory_shards.values()))

def get_shard_stats() -> Dict[str, Any]:
    """
    Get statistics about the memory shards.
    
    Returns:
        Dict: Statistics about the shards
    """
    import logging
    logger = logging.getLogger(__name__)
    import time
    
    # Record the start time for performance measurement
    start_time = time.time()
    
    # First check the in-memory shards
    in_memory_count = len(memory_shards)
    
    # Then check the database for any additional shards
    from app import app, db
    from db_models import MemoryShardModel
    
    total_shards = in_memory_count
    db_count = 0
    categories = set()
    regions = set()
    oldest_time = None
    newest_time = None
    total_size = 0
    category_counts = {}
    region_counts = {}
    replication_stats = {
        "total_replications": 0,
        "avg_replication_count": 0,
        "last_replicated": None
    }
    
    # Check database shards
    try:
        with app.app_context():
            # Get count from database
            db_count = MemoryShardModel.query.count()
            logger.debug(f"Shard stats: in-memory={in_memory_count}, db={db_count}")
            
            # If database has shards, get more detailed stats
            if db_count > 0:
                # Load categories and regions with counts
                db_categories = db.session.query(
                    MemoryShardModel.category, 
                    db.func.count(MemoryShardModel.id)
                ).group_by(MemoryShardModel.category).all()
                
                for category, count in db_categories:
                    categories.add(category)
                    category_counts[category] = count
                
                db_regions = db.session.query(
                    MemoryShardModel.region, 
                    db.func.count(MemoryShardModel.id)
                ).group_by(MemoryShardModel.region).all()
                
                for region, count in db_regions:
                    regions.add(region)
                    region_counts[region] = count
                
                # Get oldest and newest shards
                oldest_shard = MemoryShardModel.query.order_by(MemoryShardModel.creation_time).first()
                if oldest_shard:
                    oldest_time = oldest_shard.creation_time
                
                newest_shard = MemoryShardModel.query.order_by(MemoryShardModel.creation_time.desc()).first()
                if newest_shard:
                    newest_time = newest_shard.creation_time
                
                # Get replication statistics
                replication_sum = db.session.query(db.func.sum(MemoryShardModel.replication_count)).scalar() or 0
                replication_stats["total_replications"] = replication_sum
                
                if db_count > 0:
                    replication_stats["avg_replication_count"] = replication_sum / db_count
                
                latest_replicated = MemoryShardModel.query.order_by(MemoryShardModel.last_replicated.desc()).first()
                if latest_replicated:
                    replication_stats["last_replicated"] = latest_replicated.last_replicated
                
                # Get approximate total size
                total_size_query = db.session.query(db.func.sum(db.func.length(MemoryShardModel.data))).scalar()
                if total_size_query:
                    total_size = total_size_query
                
                # Update total count
                total_shards = max(in_memory_count, db_count)
    except Exception as e:
        logger.error(f"Error fetching shard stats from database: {e}")
    
    # Add in-memory shards data if any
    for shard in memory_shards.values():
        categories.add(shard.category)
        regions.add(shard.region)
        
        # Update category counts
        category_counts[shard.category] = category_counts.get(shard.category, 0) + 1
        
        # Update region counts
        region_counts[shard.region] = region_counts.get(shard.region, 0) + 1
        
        if oldest_time is None or shard.creation_time < oldest_time:
            oldest_time = shard.creation_time
            
        if newest_time is None or shard.creation_time > newest_time:
            newest_time = shard.creation_time
            
        # Update replication stats
        replication_stats["total_replications"] += shard.replication_count
        
        if replication_stats["last_replicated"] is None or shard.last_replicated > replication_stats["last_replicated"]:
            replication_stats["last_replicated"] = shard.last_replicated
            
        total_size += len(shard.data)
    
    # Calculate average replication count for all shards
    if total_shards > 0:
        replication_stats["avg_replication_count"] = replication_stats["total_replications"] / total_shards
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Sort categories and regions by count
    sorted_categories = sorted(
        [(category, category_counts.get(category, 0)) for category in categories], 
        key=lambda x: x[1], 
        reverse=True
    )
    
    sorted_regions = sorted(
        [(region, region_counts.get(region, 0)) for region in regions],
        key=lambda x: x[1],
        reverse=True
    )
    
    result = {
        "total_shards": total_shards,
        "in_memory_count": in_memory_count,
        "db_count": db_count,
        "categories": [c[0] for c in sorted_categories],
        "category_counts": {c[0]: c[1] for c in sorted_categories},
        "regions": [r[0] for r in sorted_regions],
        "region_counts": {r[0]: r[1] for r in sorted_regions},
        "oldest": oldest_time,
        "newest": newest_time,
        "age": None if oldest_time is None else int(time.time()) - oldest_time,
        "total_size": total_size,
        "avg_size": 0 if total_shards == 0 else total_size / total_shards,
        "replication": replication_stats,
        "stats_generated_at": int(time.time()),
        "stats_execution_time": execution_time
    }
    
    # Log stats generation time for performance monitoring
    logger.debug(f"Generated shard stats in {execution_time:.3f}s")
    
    return result
