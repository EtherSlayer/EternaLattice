"""
Data models for the EternaLattice blockchain
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import time
import json
from uuid import uuid4
from datetime import datetime

@dataclass
class Block:
    """
    Block in the multi-dimensional blockchain structure.
    Uses coordinates (x, y, z) to represent time, category, and region.
    """
    coordinates: Tuple[int, int, int]  # (time, category, region)
    previous_hashes: Dict[str, str]  # Map of dimension to hash
    timestamp: int = field(default_factory=lambda: int(time.time()))
    nonce: int = 0
    difficulty: int = 1
    hash: str = ""
    signature: str = ""
    shard_references: List[str] = field(default_factory=list)
    fitness_score: float = 0.0
    consensus_traits: Dict[str, float] = field(default_factory=dict)
    miner_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary representation"""
        return {
            "coordinates": self.coordinates,
            "previous_hashes": self.previous_hashes,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "difficulty": self.difficulty,
            "hash": self.hash,
            "signature": self.signature,
            "shard_references": self.shard_references,
            "fitness_score": self.fitness_score,
            "consensus_traits": self.consensus_traits,
            "miner_id": self.miner_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Block':
        """Create block from dictionary representation"""
        return cls(
            coordinates=tuple(data["coordinates"]),
            previous_hashes=data["previous_hashes"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            difficulty=data["difficulty"],
            hash=data["hash"],
            signature=data["signature"],
            shard_references=data["shard_references"],
            fitness_score=data["fitness_score"],
            consensus_traits=data["consensus_traits"],
            miner_id=data["miner_id"]
        )

    def to_json(self) -> str:
        """Convert block to JSON string"""
        try:
            return json.dumps(self.to_dict())
        except TypeError:
            # Handle non-serializable objects by converting them to strings
            block_dict = self.to_dict()
            serializable_dict = {}
            
            for key, value in block_dict.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries
                    serializable_dict[key] = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                                              for k, v in value.items()}
                elif isinstance(value, list):
                    # Handle lists
                    serializable_dict[key] = [str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item 
                                             for item in value]
                else:
                    # Convert non-serializable types to strings
                    serializable_dict[key] = str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value
            
            return json.dumps(serializable_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Block':
        """Create block from JSON string"""
        return cls.from_dict(json.loads(json_str))

@dataclass
class MemoryShard:
    """
    Memory Shard containing knowledge data, designed for self-replication
    and resilience across the network.
    """
    shard_id: str = field(default_factory=lambda: str(uuid4()))
    data: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    category: str = ""
    region: str = ""
    creation_time: int = field(default_factory=lambda: int(time.time()))
    last_replicated: int = field(default_factory=lambda: int(time.time()))
    replication_count: int = 0
    signature: str = ""
    hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory shard to dictionary representation"""
        return {
            "shard_id": self.shard_id,
            "data": self.data,
            "metadata": self.metadata,
            "category": self.category,
            "region": self.region,
            "creation_time": self.creation_time,
            "last_replicated": self.last_replicated,
            "replication_count": self.replication_count,
            "signature": self.signature,
            "hash": self.hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryShard':
        """Create memory shard from dictionary representation"""
        return cls(
            shard_id=data["shard_id"],
            data=data["data"],
            metadata=data["metadata"],
            category=data["category"],
            region=data["region"],
            creation_time=data["creation_time"],
            last_replicated=data["last_replicated"],
            replication_count=data["replication_count"],
            signature=data["signature"],
            hash=data["hash"]
        )

    def to_json(self) -> str:
        """Convert memory shard to JSON string"""
        try:
            return json.dumps(self.to_dict())
        except TypeError:
            # Handle non-serializable objects by converting them to strings
            shard_dict = self.to_dict()
            serializable_dict = {}
            
            for key, value in shard_dict.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries
                    serializable_dict[key] = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                                             for k, v in value.items()}
                elif isinstance(value, list):
                    # Handle lists
                    serializable_dict[key] = [str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item 
                                            for item in value]
                else:
                    # Convert non-serializable types to strings
                    serializable_dict[key] = str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value
            
            return json.dumps(serializable_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MemoryShard':
        """Create memory shard from JSON string"""
        return cls.from_dict(json.loads(json_str))

@dataclass
class UserReputation:
    """
    User reputation system for tracking contributions and achievements in the EternaLattice network.
    """
    user_id: str = field(default_factory=lambda: str(uuid4()))
    username: str = ""
    display_name: str = ""
    total_points: int = 0
    level: int = 1
    mined_blocks: int = 0
    created_shards: int = 0
    preserved_shards: int = 0  # Count of shards the user is helping to preserve
    badges: Set[str] = field(default_factory=set)
    last_contribution: int = field(default_factory=lambda: int(time.time()))
    created_at: int = field(default_factory=lambda: int(time.time()))
    contribution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user reputation to dictionary representation"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "total_points": self.total_points,
            "level": self.level,
            "mined_blocks": self.mined_blocks,
            "created_shards": self.created_shards,
            "preserved_shards": self.preserved_shards,
            "badges": list(self.badges),  # Convert set to list for JSON serialization
            "last_contribution": self.last_contribution,
            "created_at": self.created_at,
            "contribution_history": self.contribution_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserReputation':
        """Create user reputation from dictionary representation"""
        user = cls(
            user_id=data["user_id"],
            username=data["username"],
            display_name=data["display_name"],
            total_points=data["total_points"],
            level=data["level"],
            mined_blocks=data["mined_blocks"],
            created_shards=data["created_shards"],
            preserved_shards=data["preserved_shards"],
            last_contribution=data["last_contribution"],
            created_at=data["created_at"],
            contribution_history=data["contribution_history"]
        )
        # Convert list back to set for badges
        user.badges = set(data["badges"])
        return user
    
    def to_json(self) -> str:
        """Convert user reputation to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UserReputation':
        """Create user reputation from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    def add_points(self, points: int, activity: str, description: str = "") -> None:
        """
        Add reputation points for a user activity
        
        Args:
            points: Number of points to add
            activity: Type of activity (e.g., 'mine_block', 'create_shard')
            description: Optional description of the activity
        """
        self.total_points += points
        self.last_contribution = int(time.time())
        
        # Record contribution
        contribution = {
            "timestamp": self.last_contribution,
            "points": points,
            "activity": activity,
            "description": description
        }
        self.contribution_history.append(contribution)
        
        # Update specific counters
        if activity == 'mine_block':
            self.mined_blocks += 1
        elif activity == 'create_shard':
            self.created_shards += 1
        elif activity == 'preserve_shard':
            self.preserved_shards += 1
        
        # Calculate level (square root formula for slower level progression)
        self.level = max(1, int((self.total_points / 100) ** 0.5))
        
        # Check for new badges
        self._update_badges()
    
    def _update_badges(self) -> None:
        """Update user badges based on their achievements"""
        # Shard creator badges
        if self.created_shards >= 1000:
            self.badges.add("Archivist Master")
        elif self.created_shards >= 500:
            self.badges.add("Senior Archivist")
        elif self.created_shards >= 100:
            self.badges.add("Archivist")
        elif self.created_shards >= 10:
            self.badges.add("Knowledge Collector")
        
        # Block miner badges
        if self.mined_blocks >= 1000:
            self.badges.add("Master Builder")
        elif self.mined_blocks >= 500:
            self.badges.add("Lattice Architect")
        elif self.mined_blocks >= 100:
            self.badges.add("Lattice Constructor")
        elif self.mined_blocks >= 10:
            self.badges.add("Block Miner")
        
        # Knowledge preservation badges
        if self.preserved_shards >= 1000:
            self.badges.add("Knowledge Guardian")
        elif self.preserved_shards >= 500:
            self.badges.add("Knowledge Keeper")
        elif self.preserved_shards >= 100:
            self.badges.add("Knowledge Preserver")
        
        # Level badges
        if self.level >= 50:
            self.badges.add("EternaLattice Master")
        elif self.level >= 25:
            self.badges.add("EternaLattice Expert")
        elif self.level >= 10:
            self.badges.add("EternaLattice Adept")
        elif self.level >= 5:
            self.badges.add("EternaLattice Enthusiast")

@dataclass
class Node:
    """
    Network node in the EternaLattice network.
    """
    node_id: str = field(default_factory=lambda: str(uuid4()))
    public_key: str = ""
    address: str = ""
    port: int = 0
    last_seen: int = field(default_factory=lambda: int(time.time()))
    traits: Dict[str, float] = field(default_factory=dict)
    fitness_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation"""
        return {
            "node_id": self.node_id,
            "public_key": self.public_key,
            "address": self.address,
            "port": self.port,
            "last_seen": self.last_seen,
            "traits": self.traits,
            "fitness_history": self.fitness_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create node from dictionary representation"""
        return cls(
            node_id=data["node_id"],
            public_key=data["public_key"],
            address=data["address"],
            port=data["port"],
            last_seen=data["last_seen"],
            traits=data["traits"],
            fitness_history=data["fitness_history"]
        )
