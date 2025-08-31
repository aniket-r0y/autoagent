"""
Memory Manager - Advanced memory system for context and conversation management
"""

import asyncio
import logging
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MemoryManager:
    """Advanced memory management system for AI agent"""
    
    def __init__(self, settings):
        self.settings = settings
        self.db_path = Path("data/memory.db")
        self.connection = None
        self.short_term_memory = deque(maxlen=100)
        self.context_cache = {}
        self.memory_embeddings = {}
        
        # Memory parameters
        self.max_context_length = 4000
        self.relevance_threshold = 0.7
        self.memory_decay_factor = 0.95
        self.cleanup_interval = 3600  # 1 hour
        
    async def initialize(self):
        """Initialize memory management system"""
        try:
            logger.info("Initializing Memory Manager...")
            
            # Create data directory
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize database
            await self.init_database()
            
            # Load recent memories
            await self.load_recent_memories()
            
            # Start cleanup task
            asyncio.create_task(self.periodic_cleanup())
            
            logger.info("Memory Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memory Manager: {e}")
            raise
    
    async def init_database(self):
        """Initialize SQLite database for persistent memory"""
        try:
            self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = self.connection.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TEXT,
                    user_input TEXT,
                    ai_response TEXT,
                    context TEXT,
                    relevance_score REAL,
                    tags TEXT,
                    embedding BLOB
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_type TEXT,
                    context_key TEXT,
                    context_value TEXT,
                    timestamp TEXT,
                    expiry_date TEXT,
                    importance REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT,
                    content TEXT,
                    source TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    last_accessed TEXT,
                    access_count INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TEXT
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_context_type ON user_context(context_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge_base(topic)')
            
            self.connection.commit()
            logger.info("Memory database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    async def load_recent_memories(self):
        """Load recent memories into short-term memory"""
        try:
            cursor = self.connection.cursor()
            
            # Load recent conversations
            cursor.execute('''
                SELECT user_input, ai_response, context, timestamp
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT 50
            ''')
            
            recent_conversations = cursor.fetchall()
            
            for conv in reversed(recent_conversations):  # Reverse to maintain chronological order
                memory_item = {
                    'type': 'conversation',
                    'user_input': conv[0],
                    'ai_response': conv[1],
                    'context': json.loads(conv[2]) if conv[2] else {},
                    'timestamp': conv[3]
                }
                self.short_term_memory.append(memory_item)
            
            logger.info(f"Loaded {len(recent_conversations)} recent memories")
            
        except Exception as e:
            logger.error(f"Error loading recent memories: {e}")
    
    async def store_interaction(self, user_input: str, ai_response: Dict[str, Any], context: Dict[str, Any] = None):
        """Store a user-AI interaction in memory"""
        try:
            if context is None:
                context = {}
            
            # Add to short-term memory
            memory_item = {
                'type': 'interaction',
                'user_input': user_input,
                'ai_response': ai_response,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'relevance_score': 1.0  # Initially high relevance
            }
            
            self.short_term_memory.append(memory_item)
            
            # Store in database
            await self.store_conversation_db(
                user_input=user_input,
                ai_response=json.dumps(ai_response),
                context=json.dumps(context),
                relevance_score=1.0
            )
            
            # Extract and store context information
            await self.extract_and_store_context(user_input, ai_response, context)
            
            logger.debug(f"Stored interaction: {user_input[:50]}...")
            
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
    
    async def store_conversation_db(self, user_input: str, ai_response: str, context: str, relevance_score: float):
        """Store conversation in database"""
        try:
            cursor = self.connection.cursor()
            
            # Generate session ID based on time window
            session_id = self.generate_session_id()
            
            # Generate tags from input
            tags = await self.generate_tags(user_input)
            
            cursor.execute('''
                INSERT INTO conversations (session_id, timestamp, user_input, ai_response, context, relevance_score, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                datetime.now().isoformat(),
                user_input,
                ai_response,
                context,
                relevance_score,
                json.dumps(tags)
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing conversation in database: {e}")
    
    async def extract_and_store_context(self, user_input: str, ai_response: Dict, context: Dict):
        """Extract and store contextual information"""
        try:
            cursor = self.connection.cursor()
            
            # Extract user preferences
            if 'preference' in user_input.lower():
                preference_value = user_input
                cursor.execute('''
                    INSERT OR REPLACE INTO user_context (context_type, context_key, context_value, timestamp, importance)
                    VALUES (?, ?, ?, ?, ?)
                ''', ('preference', 'user_preference', preference_value, datetime.now().isoformat(), 0.8))
            
            # Extract application usage patterns
            if 'application' in context:
                app_name = context['application']
                cursor.execute('''
                    INSERT INTO user_context (context_type, context_key, context_value, timestamp, importance)
                    VALUES (?, ?, ?, ?, ?)
                ''', ('app_usage', app_name, user_input, datetime.now().isoformat(), 0.6))
            
            # Extract time-based patterns
            current_hour = datetime.now().hour
            cursor.execute('''
                INSERT INTO user_context (context_type, context_key, context_value, timestamp, importance)
                VALUES (?, ?, ?, ?, ?)
            ''', ('time_pattern', f'hour_{current_hour}', user_input, datetime.now().isoformat(), 0.5))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
    
    async def get_relevant_context(self, query: str, max_items: int = 10) -> Dict[str, Any]:
        """Get relevant context for a given query"""
        try:
            relevant_context = {
                'recent_conversations': [],
                'user_preferences': [],
                'app_usage': [],
                'knowledge': [],
                'patterns': []
            }
            
            # Get recent conversations from short-term memory
            query_lower = query.lower()
            for memory in list(self.short_term_memory):
                if memory['type'] == 'interaction':
                    # Simple keyword matching for relevance
                    input_text = memory['user_input'].lower()
                    if any(word in input_text for word in query_lower.split() if len(word) > 2):
                        relevant_context['recent_conversations'].append({
                            'input': memory['user_input'],
                            'response': memory['ai_response'],
                            'timestamp': memory['timestamp']
                        })
            
            # Get relevant context from database
            cursor = self.connection.cursor()
            
            # Get user preferences
            cursor.execute('''
                SELECT context_value, importance, timestamp
                FROM user_context
                WHERE context_type = 'preference'
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            ''', (max_items,))
            
            preferences = cursor.fetchall()
            relevant_context['user_preferences'] = [
                {'value': pref[0], 'importance': pref[1], 'timestamp': pref[2]}
                for pref in preferences
            ]
            
            # Get app usage patterns
            cursor.execute('''
                SELECT context_key, context_value, timestamp
                FROM user_context
                WHERE context_type = 'app_usage'
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (max_items,))
            
            app_usage = cursor.fetchall()
            relevant_context['app_usage'] = [
                {'app': usage[0], 'context': usage[1], 'timestamp': usage[2]}
                for usage in app_usage
            ]
            
            # Get relevant knowledge
            cursor.execute('''
                SELECT topic, content, confidence, last_accessed
                FROM knowledge_base
                WHERE content LIKE ?
                ORDER BY confidence DESC, last_accessed DESC
                LIMIT ?
            ''', (f'%{query}%', max_items))
            
            knowledge = cursor.fetchall()
            relevant_context['knowledge'] = [
                {'topic': k[0], 'content': k[1], 'confidence': k[2], 'last_accessed': k[3]}
                for k in knowledge
            ]
            
            return relevant_context
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return {'error': str(e)}
    
    async def store_knowledge(self, topic: str, content: str, source: str = 'user', confidence: float = 0.8):
        """Store knowledge in the knowledge base"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO knowledge_base (topic, content, source, confidence, timestamp, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                topic,
                content,
                source,
                confidence,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                0
            ))
            
            self.connection.commit()
            logger.debug(f"Stored knowledge: {topic}")
            
        except Exception as e:
            logger.error(f"Error storing knowledge: {e}")
    
    async def update_knowledge_access(self, knowledge_id: int):
        """Update knowledge access statistics"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                UPDATE knowledge_base
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
            ''', (datetime.now().isoformat(), knowledge_id))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error updating knowledge access: {e}")
    
    async def decay_memory_relevance(self):
        """Apply decay to memory relevance scores"""
        try:
            # Decay short-term memory
            for memory in self.short_term_memory:
                if 'relevance_score' in memory:
                    memory['relevance_score'] *= self.memory_decay_factor
            
            # Decay database memories
            cursor = self.connection.cursor()
            cursor.execute('''
                UPDATE conversations
                SET relevance_score = relevance_score * ?
                WHERE timestamp < ?
            ''', (self.memory_decay_factor, (datetime.now() - timedelta(hours=1)).isoformat()))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error applying memory decay: {e}")
    
    async def cleanup_old_data(self):
        """Clean up old and irrelevant data"""
        try:
            cursor = self.connection.cursor()
            
            # Remove very old conversations (older than 30 days)
            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
            cursor.execute('DELETE FROM conversations WHERE timestamp < ?', (cutoff_date,))
            
            # Remove low-relevance conversations (relevance < 0.1)
            cursor.execute('DELETE FROM conversations WHERE relevance_score < 0.1')
            
            # Remove expired context
            cursor.execute('DELETE FROM user_context WHERE expiry_date IS NOT NULL AND expiry_date < ?', 
                         (datetime.now().isoformat(),))
            
            # Remove rarely accessed knowledge (not accessed in 60 days)
            old_knowledge_cutoff = (datetime.now() - timedelta(days=60)).isoformat()
            cursor.execute('DELETE FROM knowledge_base WHERE last_accessed < ? AND access_count < 5', 
                         (old_knowledge_cutoff,))
            
            self.connection.commit()
            
            # Vacuum database to reclaim space
            cursor.execute('VACUUM')
            
            logger.info("Memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    async def periodic_cleanup(self):
        """Periodic cleanup task"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.decay_memory_relevance()
                await self.cleanup_old_data()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    def generate_session_id(self) -> str:
        """Generate session ID based on time window"""
        # Create 1-hour session windows
        hour_window = datetime.now().replace(minute=0, second=0, microsecond=0)
        return hashlib.md5(hour_window.isoformat().encode()).hexdigest()[:8]
    
    async def generate_tags(self, text: str) -> List[str]:
        """Generate tags from text content"""
        try:
            # Simple keyword extraction
            words = text.lower().split()
            
            # Filter out common words and extract meaningful tags
            stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
            tags = [word for word in words if len(word) > 3 and word not in stop_words]
            
            # Limit to top 5 tags
            return tags[:5]
            
        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            cursor = self.connection.cursor()
            
            # Count conversations
            cursor.execute('SELECT COUNT(*) FROM conversations')
            conversation_count = cursor.fetchone()[0]
            
            # Count context items
            cursor.execute('SELECT COUNT(*) FROM user_context')
            context_count = cursor.fetchone()[0]
            
            # Count knowledge items
            cursor.execute('SELECT COUNT(*) FROM knowledge_base')
            knowledge_count = cursor.fetchone()[0]
            
            # Get database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                'short_term_memories': len(self.short_term_memory),
                'total_conversations': conversation_count,
                'context_items': context_count,
                'knowledge_items': knowledge_count,
                'database_size_mb': db_size / (1024 * 1024),
                'cache_size': len(self.context_cache)
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {'error': str(e)}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory system summary"""
        try:
            stats = self.get_memory_stats()
            
            # Get recent activity
            recent_memories = list(self.short_term_memory)[-5:]
            recent_activity = [
                {
                    'type': mem.get('type', 'unknown'),
                    'timestamp': mem.get('timestamp', ''),
                    'preview': mem.get('user_input', '')[:50] + '...' if mem.get('user_input') else ''
                }
                for mem in recent_memories
            ]
            
            return {
                'stats': stats,
                'recent_activity': recent_activity,
                'health': 'good' if stats.get('total_conversations', 0) > 0 else 'initializing'
            }
            
        except Exception as e:
            logger.error(f"Error getting memory summary: {e}")
            return {'error': str(e)}
    
    async def search_memory(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search through memory for specific content"""
        try:
            cursor = self.connection.cursor()
            
            # Search conversations
            cursor.execute('''
                SELECT user_input, ai_response, timestamp, relevance_score
                FROM conversations
                WHERE user_input LIKE ? OR ai_response LIKE ?
                ORDER BY relevance_score DESC, timestamp DESC
                LIMIT ?
            ''', (f'%{query}%', f'%{query}%', limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'type': 'conversation',
                    'user_input': row[0],
                    'ai_response': row[1],
                    'timestamp': row[2],
                    'relevance_score': row[3]
                })
            
            # Search knowledge base
            cursor.execute('''
                SELECT topic, content, confidence, timestamp
                FROM knowledge_base
                WHERE topic LIKE ? OR content LIKE ?
                ORDER BY confidence DESC, timestamp DESC
                LIMIT ?
            ''', (f'%{query}%', f'%{query}%', limit // 2))
            
            for row in cursor.fetchall():
                results.append({
                    'type': 'knowledge',
                    'topic': row[0],
                    'content': row[1],
                    'confidence': row[2],
                    'timestamp': row[3]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return []
    
    async def export_memory(self, format_type: str = 'json') -> str:
        """Export memory data"""
        try:
            cursor = self.connection.cursor()
            
            # Export conversations
            cursor.execute('SELECT * FROM conversations ORDER BY timestamp DESC')
            conversations = cursor.fetchall()
            
            # Export context
            cursor.execute('SELECT * FROM user_context ORDER BY timestamp DESC')
            context = cursor.fetchall()
            
            # Export knowledge
            cursor.execute('SELECT * FROM knowledge_base ORDER BY timestamp DESC')
            knowledge = cursor.fetchall()
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'conversations': conversations,
                'context': context,
                'knowledge': knowledge,
                'short_term_memory': list(self.short_term_memory)
            }
            
            if format_type == 'json':
                return json.dumps(export_data, indent=2)
            else:
                return str(export_data)
                
        except Exception as e:
            logger.error(f"Error exporting memory: {e}")
            return f"Error: {e}"
    
    async def shutdown(self):
        """Shutdown memory manager"""
        logger.info("Shutting down Memory Manager...")
        
        try:
            # Final cleanup
            await self.cleanup_old_data()
            
            # Close database connection
            if self.connection:
                self.connection.close()
                
        except Exception as e:
            logger.error(f"Error during memory manager shutdown: {e}")
