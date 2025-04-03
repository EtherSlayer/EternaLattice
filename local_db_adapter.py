"""
Database adapter for EternaLattice in standalone node mode.
This module provides a SQLite-based database backend for the node.
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from models import Block, MemoryShard, UserReputation
from local_config import DATA_DIR, SQLITE_DB_PATH

logger = logging.getLogger('eternalattice.db')

class LocalDatabaseAdapter:
    """Adapter class for handling SQLite database operations in standalone mode."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the database adapter."""
        self.db_path = db_path or SQLITE_DB_PATH
        self._ensure_tables()
        logger.info(f"Local database initialized at {self.db_path}")
    
    def _get_connection(self):
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
    
    def _ensure_tables(self):
        """Ensure that all required tables exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create blocks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            x_coord INTEGER NOT NULL,
            y_coord INTEGER NOT NULL,
            z_coord INTEGER NOT NULL,
            hash TEXT NOT NULL UNIQUE,
            previous_hashes TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            nonce INTEGER NOT NULL,
            difficulty INTEGER NOT NULL,
            signature TEXT NOT NULL,
            shard_references TEXT NOT NULL,
            fitness_score REAL NOT NULL,
            consensus_traits TEXT NOT NULL,
            miner_id TEXT NOT NULL,
            UNIQUE(x_coord, y_coord, z_coord)
        )
        ''')
        
        # Create memory_shards table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory_shards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shard_id TEXT NOT NULL UNIQUE,
            data TEXT NOT NULL,
            shard_metadata TEXT NOT NULL,
            category TEXT NOT NULL,
            region TEXT NOT NULL,
            creation_time INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_replicated INTEGER NOT NULL,
            replication_count INTEGER NOT NULL,
            signature TEXT NOT NULL,
            hash TEXT NOT NULL
        )
        ''')
        
        # Create user_reputations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_reputations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL UNIQUE,
            username TEXT NOT NULL,
            display_name TEXT NOT NULL,
            total_points INTEGER NOT NULL DEFAULT 0,
            level INTEGER NOT NULL DEFAULT 1,
            mined_blocks INTEGER NOT NULL DEFAULT 0,
            created_shards INTEGER NOT NULL DEFAULT 0,
            preserved_shards INTEGER NOT NULL DEFAULT 0,
            badges TEXT NOT NULL,
            last_contribution INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            contribution_history TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create peer_nodes table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS peer_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL UNIQUE,
            public_key TEXT NOT NULL,
            address TEXT NOT NULL,
            port INTEGER NOT NULL,
            last_seen INTEGER NOT NULL,
            traits TEXT NOT NULL,
            fitness_history TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    # Block operations
    def save_block(self, block: Block) -> bool:
        """Save a block to the database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Convert block data to database format
            x, y, z = block.coordinates
            previous_hashes = json.dumps(block.previous_hashes)
            shard_references = json.dumps(block.shard_references)
            consensus_traits = json.dumps(block.consensus_traits)
            
            cursor.execute('''
            INSERT OR REPLACE INTO blocks 
            (x_coord, y_coord, z_coord, hash, previous_hashes, timestamp, nonce, 
            difficulty, signature, shard_references, fitness_score, consensus_traits, miner_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                x, y, z, block.hash, previous_hashes, block.timestamp, block.nonce,
                block.difficulty, block.signature, shard_references, block.fitness_score,
                consensus_traits, block.miner_id
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving block to database: {e}")
            return False
    
    def load_blocks(self) -> Dict[Tuple[int, int, int], Block]:
        """Load all blocks from the database."""
        blocks = {}
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM blocks ORDER BY timestamp ASC
            ''')
            
            rows = cursor.fetchall()
            for row in rows:
                # Convert database row to Block object
                coordinates = (row['x_coord'], row['y_coord'], row['z_coord'])
                previous_hashes = json.loads(row['previous_hashes'])
                shard_references = json.loads(row['shard_references'])
                consensus_traits = json.loads(row['consensus_traits'])
                
                block = Block(
                    coordinates=coordinates,
                    previous_hashes=previous_hashes,
                    timestamp=row['timestamp'],
                    nonce=row['nonce'],
                    difficulty=row['difficulty'],
                    hash=row['hash'],
                    signature=row['signature'],
                    shard_references=shard_references,
                    fitness_score=row['fitness_score'],
                    consensus_traits=consensus_traits,
                    miner_id=row['miner_id']
                )
                
                blocks[coordinates] = block
            
            conn.close()
            logger.info(f"Loaded {len(blocks)} blocks from database")
            return blocks
        except Exception as e:
            logger.error(f"Error loading blocks from database: {e}")
            return {}
    
    def delete_block(self, coordinates: Tuple[int, int, int]) -> bool:
        """Delete a block from the database by its coordinates."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            x, y, z = coordinates
            cursor.execute('''
            DELETE FROM blocks WHERE x_coord = ? AND y_coord = ? AND z_coord = ?
            ''', (x, y, z))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error deleting block from database: {e}")
            return False
    
    # MemoryShard operations
    def save_shard(self, shard: MemoryShard) -> bool:
        """Save a memory shard to the database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Convert shard data to database format
            metadata = json.dumps(shard.metadata)
            
            cursor.execute('''
            INSERT OR REPLACE INTO memory_shards 
            (shard_id, data, shard_metadata, category, region, creation_time,
            last_replicated, replication_count, signature, hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                shard.shard_id, shard.data, metadata, shard.category, shard.region,
                shard.creation_time, shard.last_replicated, shard.replication_count,
                shard.signature, shard.hash
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving shard to database: {e}")
            return False
    
    def load_shards(self) -> Dict[str, MemoryShard]:
        """Load all memory shards from the database."""
        shards = {}
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM memory_shards
            ''')
            
            rows = cursor.fetchall()
            for row in rows:
                # Convert database row to MemoryShard object
                metadata = json.loads(row['shard_metadata'])
                
                shard = MemoryShard(
                    shard_id=row['shard_id'],
                    data=row['data'],
                    metadata=metadata,
                    category=row['category'],
                    region=row['region'],
                    creation_time=row['creation_time'],
                    last_replicated=row['last_replicated'],
                    replication_count=row['replication_count'],
                    signature=row['signature'],
                    hash=row['hash']
                )
                
                shards[shard.shard_id] = shard
            
            conn.close()
            logger.info(f"Loaded {len(shards)} shards from database")
            return shards
        except Exception as e:
            logger.error(f"Error loading shards from database: {e}")
            return {}
    
    def delete_shard(self, shard_id: str) -> bool:
        """Delete a memory shard from the database by its ID."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            DELETE FROM memory_shards WHERE shard_id = ?
            ''', (shard_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error deleting shard from database: {e}")
            return False
    
    # UserReputation operations
    def save_user_reputation(self, user: UserReputation) -> bool:
        """Save a user's reputation data to the database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Convert user data to database format
            badges = json.dumps(list(user.badges))
            contribution_history = json.dumps(user.contribution_history)
            
            cursor.execute('''
            INSERT OR REPLACE INTO user_reputations 
            (user_id, username, display_name, total_points, level, mined_blocks,
            created_shards, preserved_shards, badges, last_contribution, created_at,
            contribution_history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user.user_id, user.username, user.display_name, user.total_points,
                user.level, user.mined_blocks, user.created_shards, user.preserved_shards,
                badges, user.last_contribution, user.created_at, contribution_history
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving user reputation to database: {e}")
            return False
    
    def load_user_reputation(self, user_id: str) -> Optional[UserReputation]:
        """Load a user's reputation data from the database by user ID."""
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM user_reputations WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            if row is None:
                conn.close()
                return None
            
            # Convert database row to UserReputation object
            badges = set(json.loads(row['badges']))
            contribution_history = json.loads(row['contribution_history'])
            
            user = UserReputation(
                user_id=row['user_id'],
                username=row['username'],
                display_name=row['display_name'],
                total_points=row['total_points'],
                level=row['level'],
                mined_blocks=row['mined_blocks'],
                created_shards=row['created_shards'],
                preserved_shards=row['preserved_shards'],
                badges=badges,
                last_contribution=row['last_contribution'],
                created_at=row['created_at'],
                contribution_history=contribution_history
            )
            
            conn.close()
            return user
        except Exception as e:
            logger.error(f"Error loading user reputation from database: {e}")
            return None
    
    def load_all_user_reputations(self) -> List[UserReputation]:
        """Load all user reputation data from the database."""
        users = []
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM user_reputations ORDER BY total_points DESC
            ''')
            
            rows = cursor.fetchall()
            for row in rows:
                # Convert database row to UserReputation object
                badges = set(json.loads(row['badges']))
                contribution_history = json.loads(row['contribution_history'])
                
                user = UserReputation(
                    user_id=row['user_id'],
                    username=row['username'],
                    display_name=row['display_name'],
                    total_points=row['total_points'],
                    level=row['level'],
                    mined_blocks=row['mined_blocks'],
                    created_shards=row['created_shards'],
                    preserved_shards=row['preserved_shards'],
                    badges=badges,
                    last_contribution=row['last_contribution'],
                    created_at=row['created_at'],
                    contribution_history=contribution_history
                )
                
                users.append(user)
            
            conn.close()
            logger.info(f"Loaded {len(users)} user reputations from database")
            return users
        except Exception as e:
            logger.error(f"Error loading user reputations from database: {e}")
            return []
    
    # Statistics and queries
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get statistics about the blockchain."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get block count
            cursor.execute('SELECT COUNT(*) FROM blocks')
            block_count = cursor.fetchone()[0]
            
            # Get dimension ranges
            cursor.execute('SELECT MIN(x_coord), MAX(x_coord) FROM blocks')
            time_min, time_max = cursor.fetchone()
            
            cursor.execute('SELECT MIN(y_coord), MAX(y_coord) FROM blocks')
            category_min, category_max = cursor.fetchone()
            
            cursor.execute('SELECT MIN(z_coord), MAX(z_coord) FROM blocks')
            region_min, region_max = cursor.fetchone()
            
            # Get shard count
            cursor.execute('SELECT COUNT(*) FROM memory_shards')
            shard_count = cursor.fetchone()[0]
            
            # Get average difficulty and fitness
            cursor.execute('SELECT AVG(difficulty), AVG(fitness_score) FROM blocks')
            avg_difficulty, avg_fitness = cursor.fetchone()
            
            # Get shard counts in blocks
            cursor.execute('''
            SELECT blocks.hash, blocks.shard_references
            FROM blocks
            ''')
            
            shards_in_blocks = set()
            for _, shard_refs in cursor.fetchall():
                shard_list = json.loads(shard_refs)
                shards_in_blocks.update(shard_list)
            
            conn.close()
            
            # Compile stats
            stats = {
                'blocks': block_count,
                'dimensions': {
                    'time': {'min': time_min or 0, 'max': time_max or 0},
                    'category': {'min': category_min or 0, 'max': category_max or 0},
                    'region': {'min': region_min or 0, 'max': region_max or 0}
                },
                'shards': {
                    'total': shard_count,
                    'in_blocks': len(shards_in_blocks),
                    'unconfirmed': shard_count - len(shards_in_blocks)
                },
                'avg_difficulty': avg_difficulty or 1.0,
                'avg_fitness': avg_fitness or 0.0
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting blockchain stats from database: {e}")
            return {
                'blocks': 0,
                'dimensions': {
                    'time': {'min': 0, 'max': 0},
                    'category': {'min': 0, 'max': 0},
                    'region': {'min': 0, 'max': 0}
                },
                'shards': {
                    'total': 0,
                    'in_blocks': 0,
                    'unconfirmed': 0
                },
                'avg_difficulty': 1.0,
                'avg_fitness': 0.0
            }
    
    def search_shards(self, query: str, category: Optional[str] = None, region: Optional[str] = None) -> List[MemoryShard]:
        """Search for memory shards by content, category, or region."""
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Base query
            sql = '''
            SELECT * FROM memory_shards WHERE data LIKE ?
            '''
            params = [f'%{query}%']
            
            # Add category filter if provided
            if category:
                sql += ' AND category = ?'
                params.append(category)
            
            # Add region filter if provided
            if region:
                sql += ' AND region = ?'
                params.append(region)
            
            cursor.execute(sql, params)
            
            shards = []
            for row in cursor.fetchall():
                # Convert database row to MemoryShard object
                metadata = json.loads(row['shard_metadata'])
                
                shard = MemoryShard(
                    shard_id=row['shard_id'],
                    data=row['data'],
                    metadata=metadata,
                    category=row['category'],
                    region=row['region'],
                    creation_time=row['creation_time'],
                    last_replicated=row['last_replicated'],
                    replication_count=row['replication_count'],
                    signature=row['signature'],
                    hash=row['hash']
                )
                
                shards.append(shard)
            
            conn.close()
            return shards
        except Exception as e:
            logger.error(f"Error searching shards in database: {e}")
            return []