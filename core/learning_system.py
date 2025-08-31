"""
Learning System - Advanced machine learning for user behavior and pattern recognition
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
from collections import defaultdict, deque

# Setup logger first
logger = logging.getLogger(__name__)

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("Scikit-learn not available, machine learning features disabled")


class LearningSystem:
    """Advanced learning system for user behavior analysis and prediction"""
    
    def __init__(self, settings):
        self.settings = settings
        self.interaction_data = []
        self.user_patterns = {}
        self.predictive_models = {}
        self.anomaly_detector = None
        self.behavior_clusters = None
        self.learning_enabled = ML_AVAILABLE
        
        # Learning parameters
        self.max_interactions = 10000
        self.pattern_window = 100
        self.min_pattern_frequency = 3
        self.learning_rate = 0.01
        
        # Data storage
        self.data_dir = Path("data/learning")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self):
        """Initialize learning system"""
        try:
            logger.info("Initializing Learning System...")
            
            if not self.learning_enabled:
                logger.warning("Learning system initialized with limited functionality")
                return
            
            # Load existing data
            await self.load_interaction_data()
            await self.load_models()
            
            # Initialize models if not loaded
            if not self.predictive_models:
                await self.initialize_models()
            
            logger.info("Learning System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Learning System: {e}")
            raise
    
    async def load_interaction_data(self):
        """Load historical interaction data"""
        try:
            data_file = self.data_dir / "interactions.json"
            if data_file.exists():
                with open(data_file, 'r') as f:
                    self.interaction_data = json.load(f)
                logger.info(f"Loaded {len(self.interaction_data)} historical interactions")
            
            # Load user patterns
            patterns_file = self.data_dir / "user_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    self.user_patterns = json.load(f)
                logger.info(f"Loaded {len(self.user_patterns)} user patterns")
                
        except Exception as e:
            logger.error(f"Error loading interaction data: {e}")
    
    async def load_models(self):
        """Load trained machine learning models"""
        try:
            if not self.learning_enabled:
                return
            
            models_dir = self.data_dir / "models"
            if models_dir.exists():
                # Load predictive models
                for model_file in models_dir.glob("*.pkl"):
                    model_name = model_file.stem
                    try:
                        self.predictive_models[model_name] = joblib.load(model_file)
                        logger.info(f"Loaded model: {model_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load model {model_name}: {e}")
                
                # Load anomaly detector
                anomaly_file = models_dir / "anomaly_detector.pkl"
                if anomaly_file.exists():
                    self.anomaly_detector = joblib.load(anomaly_file)
                    logger.info("Loaded anomaly detector")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def initialize_models(self):
        """Initialize new machine learning models"""
        try:
            if not self.learning_enabled:
                return
            
            logger.info("Initializing new machine learning models...")
            
            # Command prediction model
            self.predictive_models['command_prediction'] = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            
            # Time-based pattern model
            self.predictive_models['time_pattern'] = RandomForestClassifier(
                n_estimators=50, random_state=42
            )
            
            # User preference model
            self.predictive_models['user_preference'] = RandomForestClassifier(
                n_estimators=75, random_state=42
            )
            
            # Anomaly detection model
            self.anomaly_detector = IsolationForest(
                contamination=0.1, random_state=42
            )
            
            logger.info("Machine learning models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def record_interaction(self, command: str, analysis: Dict, result: Dict, llm_response: Dict):
        """Record a user interaction for learning"""
        try:
            interaction = {
                'timestamp': datetime.now().isoformat(),
                'command': command,
                'analysis': analysis,
                'result': result,
                'llm_response': llm_response,
                'success': result.get('success', False),
                'execution_time': result.get('execution_time', 0),
                'user_satisfaction': None,  # To be filled by feedback
                'context': {
                    'hour': datetime.now().hour,
                    'day_of_week': datetime.now().weekday(),
                    'command_length': len(command),
                    'command_complexity': analysis.get('complexity', 0),
                    'previous_command': self.get_last_command()
                }
            }
            
            self.interaction_data.append(interaction)
            
            # Limit data size
            if len(self.interaction_data) > self.max_interactions:
                self.interaction_data = self.interaction_data[-self.max_interactions:]
            
            # Trigger pattern analysis if enough new data
            if len(self.interaction_data) % 50 == 0:
                asyncio.create_task(self.analyze_patterns())
            
            logger.debug(f"Recorded interaction: {command[:50]}...")
            
        except Exception as e:
            logger.error(f"Error recording interaction: {e}")
    
    async def analyze_patterns(self):
        """Analyze user interaction patterns"""
        try:
            if not self.learning_enabled or len(self.interaction_data) < 10:
                return
            
            logger.info("Analyzing user patterns...")
            
            # Analyze command patterns
            await self.analyze_command_patterns()
            
            # Analyze temporal patterns
            await self.analyze_temporal_patterns()
            
            # Analyze success patterns
            await self.analyze_success_patterns()
            
            # Detect usage anomalies
            await self.detect_anomalies()
            
            # Save updated patterns
            await self.save_patterns()
            
            logger.info("Pattern analysis completed")
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
    
    async def analyze_command_patterns(self):
        """Analyze patterns in user commands"""
        try:
            # Extract command features
            commands = [interaction['command'] for interaction in self.interaction_data]
            
            # Common command sequences
            command_sequences = []
            for i in range(len(commands) - 1):
                command_sequences.append((commands[i], commands[i + 1]))
            
            # Find frequent sequences
            sequence_counts = defaultdict(int)
            for seq in command_sequences:
                sequence_counts[seq] += 1
            
            # Store frequent sequences
            frequent_sequences = {
                seq: count for seq, count in sequence_counts.items() 
                if count >= self.min_pattern_frequency
            }
            
            self.user_patterns['command_sequences'] = frequent_sequences
            
            # Command categories
            command_categories = defaultdict(list)
            for interaction in self.interaction_data:
                category = interaction['analysis'].get('category', 'general')
                command_categories[category].append(interaction['command'])
            
            self.user_patterns['command_categories'] = dict(command_categories)
            
            logger.debug(f"Found {len(frequent_sequences)} frequent command sequences")
            
        except Exception as e:
            logger.error(f"Error analyzing command patterns: {e}")
    
    async def analyze_temporal_patterns(self):
        """Analyze temporal usage patterns"""
        try:
            # Extract temporal features
            temporal_data = []
            for interaction in self.interaction_data:
                timestamp = datetime.fromisoformat(interaction['timestamp'])
                temporal_data.append({
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.weekday(),
                    'command': interaction['command'],
                    'success': interaction['success']
                })
            
            # Analyze usage by hour
            hourly_usage = defaultdict(int)
            for data in temporal_data:
                hourly_usage[data['hour']] += 1
            
            # Analyze usage by day of week
            daily_usage = defaultdict(int)
            for data in temporal_data:
                daily_usage[data['day_of_week']] += 1
            
            # Find peak usage times
            peak_hour = max(hourly_usage.items(), key=lambda x: x[1])[0]
            peak_day = max(daily_usage.items(), key=lambda x: x[1])[0]
            
            self.user_patterns['temporal'] = {
                'hourly_usage': dict(hourly_usage),
                'daily_usage': dict(daily_usage),
                'peak_hour': peak_hour,
                'peak_day': peak_day,
                'total_interactions': len(temporal_data)
            }
            
            logger.debug(f"Peak usage: Hour {peak_hour}, Day {peak_day}")
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
    
    async def analyze_success_patterns(self):
        """Analyze patterns in successful vs failed interactions"""
        try:
            successful = [i for i in self.interaction_data if i['success']]
            failed = [i for i in self.interaction_data if not i['success']]
            
            success_rate = len(successful) / len(self.interaction_data) if self.interaction_data else 0
            
            # Analyze success by command type
            success_by_category = defaultdict(lambda: {'total': 0, 'successful': 0})
            for interaction in self.interaction_data:
                category = interaction['analysis'].get('category', 'general')
                success_by_category[category]['total'] += 1
                if interaction['success']:
                    success_by_category[category]['successful'] += 1
            
            # Calculate success rates by category
            category_success_rates = {}
            for category, stats in success_by_category.items():
                if stats['total'] > 0:
                    category_success_rates[category] = stats['successful'] / stats['total']
            
            self.user_patterns['success'] = {
                'overall_success_rate': success_rate,
                'category_success_rates': category_success_rates,
                'total_successful': len(successful),
                'total_failed': len(failed)
            }
            
            logger.debug(f"Overall success rate: {success_rate:.2%}")
            
        except Exception as e:
            logger.error(f"Error analyzing success patterns: {e}")
    
    async def detect_anomalies(self):
        """Detect anomalous usage patterns"""
        try:
            if not self.learning_enabled or len(self.interaction_data) < 50:
                return
            
            # Prepare features for anomaly detection
            features = []
            for interaction in self.interaction_data:
                context = interaction['context']
                feature_vector = [
                    context['hour'],
                    context['day_of_week'],
                    context['command_length'],
                    context['command_complexity'],
                    1 if interaction['success'] else 0
                ]
                features.append(feature_vector)
            
            features_array = np.array(features)
            
            # Fit anomaly detector if not trained
            if not hasattr(self.anomaly_detector, 'decision_function'):
                self.anomaly_detector.fit(features_array)
            
            # Detect anomalies in recent interactions
            recent_features = features_array[-50:]  # Last 50 interactions
            anomaly_scores = self.anomaly_detector.decision_function(recent_features)
            anomalies = self.anomaly_detector.predict(recent_features)
            
            # Record anomalies
            anomaly_count = np.sum(anomalies == -1)
            self.user_patterns['anomalies'] = {
                'recent_anomalies': int(anomaly_count),
                'anomaly_rate': float(anomaly_count / len(recent_features)),
                'last_check': datetime.now().isoformat()
            }
            
            if anomaly_count > 0:
                logger.info(f"Detected {anomaly_count} anomalous interactions")
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
    
    async def predict_next_command(self, current_context: Dict) -> Dict[str, Any]:
        """Predict what the user might want to do next"""
        try:
            if not self.learning_enabled or not self.interaction_data:
                return {'prediction': None, 'confidence': 0.0}
            
            # Analyze recent command patterns
            recent_commands = [i['command'] for i in self.interaction_data[-10:]]
            
            # Simple pattern matching for now
            if len(recent_commands) >= 2:
                last_command = recent_commands[-1]
                
                # Find what typically follows this command
                following_commands = defaultdict(int)
                for i in range(len(self.interaction_data) - 1):
                    if self.interaction_data[i]['command'] == last_command:
                        next_command = self.interaction_data[i + 1]['command']
                        following_commands[next_command] += 1
                
                if following_commands:
                    # Get most frequent following command
                    predicted_command = max(following_commands.items(), key=lambda x: x[1])
                    total_occurrences = sum(following_commands.values())
                    confidence = predicted_command[1] / total_occurrences
                    
                    return {
                        'prediction': predicted_command[0],
                        'confidence': confidence,
                        'method': 'sequence_pattern'
                    }
            
            # Fallback to most common command in current context
            current_hour = datetime.now().hour
            commands_at_hour = [
                i['command'] for i in self.interaction_data 
                if datetime.fromisoformat(i['timestamp']).hour == current_hour
            ]
            
            if commands_at_hour:
                command_counts = defaultdict(int)
                for cmd in commands_at_hour:
                    command_counts[cmd] += 1
                
                most_common = max(command_counts.items(), key=lambda x: x[1])
                confidence = most_common[1] / len(commands_at_hour)
                
                return {
                    'prediction': most_common[0],
                    'confidence': confidence,
                    'method': 'temporal_pattern'
                }
            
            return {'prediction': None, 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Error predicting next command: {e}")
            return {'prediction': None, 'confidence': 0.0, 'error': str(e)}
    
    async def get_user_preferences(self) -> Dict[str, Any]:
        """Get learned user preferences"""
        try:
            preferences = {}
            
            if 'command_categories' in self.user_patterns:
                # Most used categories
                categories = self.user_patterns['command_categories']
                category_counts = {cat: len(commands) for cat, commands in categories.items()}
                preferences['preferred_categories'] = sorted(
                    category_counts.items(), key=lambda x: x[1], reverse=True
                )
            
            if 'temporal' in self.user_patterns:
                temporal = self.user_patterns['temporal']
                preferences['peak_usage_hour'] = temporal.get('peak_hour')
                preferences['peak_usage_day'] = temporal.get('peak_day')
            
            if 'success' in self.user_patterns:
                success = self.user_patterns['success']
                preferences['success_rate'] = success.get('overall_success_rate')
                preferences['best_categories'] = sorted(
                    success.get('category_success_rates', {}).items(),
                    key=lambda x: x[1], reverse=True
                )
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}
    
    async def update_models(self):
        """Update machine learning models with new data"""
        try:
            if not self.learning_enabled or len(self.interaction_data) < 100:
                return
            
            logger.info("Updating machine learning models...")
            
            # Prepare training data
            X, y = self.prepare_training_data()
            
            if len(X) > 0:
                # Update command prediction model
                await self.update_command_model(X, y)
                
                # Save updated models
                await self.save_models()
                
                logger.info("Machine learning models updated")
            
        except Exception as e:
            logger.error(f"Error updating models: {e}")
    
    def prepare_training_data(self) -> Tuple[List, List]:
        """Prepare training data from interactions"""
        try:
            X = []  # Features
            y = []  # Labels (success/failure)
            
            for interaction in self.interaction_data:
                if interaction.get('context'):
                    context = interaction['context']
                    features = [
                        context.get('hour', 0),
                        context.get('day_of_week', 0),
                        context.get('command_length', 0),
                        context.get('command_complexity', 0)
                    ]
                    X.append(features)
                    y.append(1 if interaction.get('success', False) else 0)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return [], []
    
    async def update_command_model(self, X: List, y: List):
        """Update the command prediction model"""
        try:
            if len(X) < 50:  # Need minimum data for training
                return
            
            X_array = np.array(X)
            y_array = np.array(y)
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_array, y_array, test_size=0.2, random_state=42
            )
            
            # Train model
            model = self.predictive_models.get('command_prediction')
            if model:
                model.fit(X_train, y_train)
                
                # Evaluate model
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                
                logger.info(f"Command model accuracy: {accuracy:.2%}")
            
        except Exception as e:
            logger.error(f"Error updating command model: {e}")
    
    async def save_patterns(self):
        """Save learned patterns to disk"""
        try:
            patterns_file = self.data_dir / "user_patterns.json"
            with open(patterns_file, 'w') as f:
                json.dump(self.user_patterns, f, indent=2)
            
            # Save interaction data
            interactions_file = self.data_dir / "interactions.json"
            with open(interactions_file, 'w') as f:
                json.dump(self.interaction_data[-1000:], f, indent=2)  # Keep last 1000
                
            logger.debug("Patterns and interactions saved")
            
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    async def save_models(self):
        """Save machine learning models to disk"""
        try:
            if not self.learning_enabled:
                return
            
            models_dir = self.data_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Save predictive models
            for name, model in self.predictive_models.items():
                model_file = models_dir / f"{name}.pkl"
                joblib.dump(model, model_file)
            
            # Save anomaly detector
            if self.anomaly_detector:
                anomaly_file = models_dir / "anomaly_detector.pkl"
                joblib.dump(self.anomaly_detector, anomaly_file)
            
            logger.debug("Machine learning models saved")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def get_last_command(self) -> Optional[str]:
        """Get the last command executed"""
        try:
            if self.interaction_data:
                return self.interaction_data[-1]['command']
            return None
        except:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        try:
            stats = {
                'total_interactions': len(self.interaction_data),
                'patterns_discovered': len(self.user_patterns),
                'learning_enabled': self.learning_enabled,
                'models_loaded': len(self.predictive_models),
                'data_size_mb': len(str(self.interaction_data)) / (1024 * 1024)
            }
            
            if 'success' in self.user_patterns:
                stats['success_rate'] = self.user_patterns['success'].get('overall_success_rate', 0)
            
            if 'temporal' in self.user_patterns:
                stats['peak_hour'] = self.user_patterns['temporal'].get('peak_hour')
                stats['peak_day'] = self.user_patterns['temporal'].get('peak_day')
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of learning system"""
        try:
            return {
                "healthy": True,
                "learning_enabled": self.learning_enabled,
                "interactions_recorded": len(self.interaction_data),
                "patterns_discovered": len(self.user_patterns),
                "models_loaded": len(self.predictive_models)
            }
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def optimize_models(self):
        """Optimize model performance and memory usage"""
        try:
            logger.info("Optimizing learning models...")
            
            # Clean old interaction data
            cutoff_date = datetime.now() - timedelta(days=30)
            self.interaction_data = [
                i for i in self.interaction_data 
                if datetime.fromisoformat(i['timestamp']) > cutoff_date
            ]
            
            # Re-analyze patterns with cleaned data
            if len(self.interaction_data) > 50:
                await self.analyze_patterns()
            
            logger.info("Model optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing models: {e}")
    
    async def shutdown(self):
        """Shutdown learning system"""
        logger.info("Shutting down Learning System...")
        
        try:
            # Save all data before shutdown
            await self.save_patterns()
            await self.save_models()
        except Exception as e:
            logger.error(f"Error saving data during shutdown: {e}")
