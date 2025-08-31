"""
AI Orchestrator - Central coordination system for all AI agent components
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from core.llm_manager import LLMManager
from core.computer_vision import ComputerVision
from core.ui_automation import UIAutomation
from core.learning_system import LearningSystem
from core.memory_manager import MemoryManager
from modules.social_media_handler import SocialMediaHandler
from modules.browser_controller import BrowserController
from modules.notification_manager import NotificationManager
from modules.content_generator import ContentGenerator
from modules.task_scheduler import TaskScheduler
from modules.file_processor import FileProcessor
from modules.security_manager import SecurityManager
from utils.nlp_processor import NLPProcessor
from utils.voice_handler import VoiceHandler
from utils.translation_service import TranslationService

logger = logging.getLogger(__name__)

class AIOrchestrator:
    """Central orchestrator for all AI agent components"""
    
    def __init__(self, settings):
        self.settings = settings
        self.is_running = False
        self.start_time = datetime.now()
        self.last_activity = datetime.now()
        self.active_tasks = {}
        self.command_queue = asyncio.Queue()
        
        # Initialize components
        self.llm_manager = None
        self.computer_vision = None
        self.ui_automation = None
        self.learning_system = None
        self.memory_manager = None
        self.social_media = None
        self.browser_controller = None
        self.notification_manager = None
        self.content_generator = None
        self.task_scheduler = None
        self.file_processor = None
        self.security_manager = None
        self.nlp_processor = None
        self.voice_handler = None
        self.translation_service = None
        
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("Initializing AI Orchestrator components...")
            
            # Core AI components
            self.llm_manager = LLMManager(self.settings)
            await self.llm_manager.initialize()
            
            self.computer_vision = ComputerVision(self.settings)
            await self.computer_vision.initialize()
            
            self.ui_automation = UIAutomation(self.computer_vision)
            await self.ui_automation.initialize()
            
            self.learning_system = LearningSystem(self.settings)
            await self.learning_system.initialize()
            
            self.memory_manager = MemoryManager(self.settings)
            await self.memory_manager.initialize()
            
            # Module components
            self.social_media = SocialMediaHandler(self.llm_manager, self.settings)
            await self.social_media.initialize()
            
            self.browser_controller = BrowserController(self.ui_automation, self.settings)
            await self.browser_controller.initialize()
            
            self.notification_manager = NotificationManager(self.settings)
            await self.notification_manager.initialize()
            
            self.content_generator = ContentGenerator(self.llm_manager, self.settings)
            await self.content_generator.initialize()
            
            self.task_scheduler = TaskScheduler(self.settings)
            await self.task_scheduler.initialize()
            
            self.file_processor = FileProcessor(self.llm_manager, self.settings)
            await self.file_processor.initialize()
            
            self.security_manager = SecurityManager(self.settings)
            await self.security_manager.initialize()
            
            # Utility components
            self.nlp_processor = NLPProcessor(self.settings)
            await self.nlp_processor.initialize()
            
            self.voice_handler = VoiceHandler(self.settings)
            await self.voice_handler.initialize()
            
            self.translation_service = TranslationService(self.settings)
            await self.translation_service.initialize()
            
            logger.info("All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def run(self):
        """Main orchestrator loop"""
        self.is_running = True
        logger.info("AI Orchestrator started")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self.command_processor()),
            asyncio.create_task(self.learning_loop()),
            asyncio.create_task(self.monitoring_loop()),
            asyncio.create_task(self.auto_optimization_loop()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in orchestrator loop: {e}")
        finally:
            self.is_running = False
    
    async def command_processor(self):
        """Process commands from the queue"""
        while self.is_running:
            try:
                command = await asyncio.wait_for(self.command_queue.get(), timeout=1.0)
                await self.process_command(command)
                self.last_activity = datetime.now()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing command: {e}")
    
    async def process_command(self, command: str) -> Dict[str, Any]:
        """Process a single command using AI reasoning"""
        try:
            logger.info(f"Processing command: {command}")
            
            # Analyze command with NLP
            analysis = await self.nlp_processor.analyze_command(command)
            
            # Get context from memory
            context = await self.memory_manager.get_relevant_context(command)
            
            # Generate response using LLM
            llm_response = await self.llm_manager.process_command(
                command, analysis, context
            )
            
            # Execute the planned actions
            result = await self.execute_actions(llm_response['actions'])
            
            # Learn from the interaction
            await self.learning_system.record_interaction(
                command, analysis, result, llm_response
            )
            
            # Store in memory
            await self.memory_manager.store_interaction(command, result)
            
            return {
                'command': command,
                'analysis': analysis,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing command '{command}': {e}")
            return {'error': str(e), 'command': command}
    
    async def execute_actions(self, actions: List[Dict]) -> Dict[str, Any]:
        """Execute a list of actions"""
        results = []
        
        for action in actions:
            try:
                action_type = action.get('type')
                params = action.get('parameters', {})
                
                if action_type == 'ui_interaction':
                    result = await self.ui_automation.perform_action(params)
                elif action_type == 'browser_control':
                    result = await self.browser_controller.execute_action(params)
                elif action_type == 'social_media':
                    result = await self.social_media.handle_action(params)
                elif action_type == 'file_operation':
                    result = await self.file_processor.process_action(params)
                elif action_type == 'content_generation':
                    result = await self.content_generator.generate_content(params)
                elif action_type == 'schedule_task':
                    result = await self.task_scheduler.schedule_task(params)
                elif action_type == 'system_command':
                    result = await self.execute_system_command(params)
                else:
                    result = {'error': f'Unknown action type: {action_type}'}
                
                results.append({
                    'action': action,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error executing action {action}: {e}")
                results.append({
                    'action': action,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return {'actions_executed': len(results), 'results': results}
    
    async def learning_loop(self):
        """Continuous learning loop"""
        while self.is_running:
            try:
                await self.learning_system.analyze_patterns()
                await self.learning_system.update_models()
                await asyncio.sleep(300)  # Learn every 5 minutes
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(60)
    
    async def monitoring_loop(self):
        """System monitoring loop"""
        while self.is_running:
            try:
                await self.monitor_system_health()
                await self.notification_manager.process_notifications()
                await asyncio.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def auto_optimization_loop(self):
        """Auto-optimization loop"""
        while self.is_running:
            try:
                await self.optimize_performance()
                await asyncio.sleep(1800)  # Optimize every 30 minutes
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(300)
    
    def queue_command(self, command: str) -> str:
        """Queue a command for processing"""
        try:
            self.command_queue.put_nowait(command)
            return f"Command queued: {command}"
        except Exception as e:
            return f"Error queuing command: {e}"
    
    def get_active_tasks(self) -> List[Dict]:
        """Get currently active tasks"""
        return list(self.active_tasks.values())
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            return {
                'rss': process.memory_info().rss / 1024 / 1024,  # MB
                'vms': process.memory_info().vms / 1024 / 1024,  # MB
                'percent': process.memory_percent()
            }
        except:
            return {'error': 'Unable to get memory usage'}
    
    def get_uptime(self) -> str:
        """Get system uptime"""
        uptime = datetime.now() - self.start_time
        return str(uptime).split('.')[0]  # Remove microseconds
    
    def get_last_activity(self) -> str:
        """Get last activity timestamp"""
        return self.last_activity.isoformat()
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings"""
        return self.settings.to_dict()
    
    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        """Update settings"""
        try:
            self.settings.update(new_settings)
            return True
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return False
    
    async def monitor_system_health(self):
        """Monitor system health"""
        try:
            # Check component health
            health_status = {
                'llm_manager': await self.llm_manager.health_check(),
                'computer_vision': await self.computer_vision.health_check(),
                'ui_automation': await self.ui_automation.health_check(),
                'learning_system': await self.learning_system.health_check(),
            }
            
            # Log any unhealthy components
            for component, status in health_status.items():
                if not status.get('healthy', False):
                    logger.warning(f"Component {component} is unhealthy: {status}")
                    
        except Exception as e:
            logger.error(f"Error monitoring system health: {e}")
    
    async def optimize_performance(self):
        """Optimize system performance"""
        try:
            # Optimize memory usage
            await self.memory_manager.cleanup_old_data()
            
            # Optimize learning models
            await self.learning_system.optimize_models()
            
            # Clean up temporary files
            await self.file_processor.cleanup_temp_files()
            
            logger.info("Performance optimization completed")
            
        except Exception as e:
            logger.error(f"Error during performance optimization: {e}")
    
    async def execute_system_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute system-level commands"""
        try:
            command_type = params.get('command_type')
            
            if command_type == 'shutdown':
                logger.info("Shutdown command received")
                return {'status': 'shutting_down'}
            elif command_type == 'restart':
                logger.info("Restart command received")
                return {'status': 'restarting'}
            elif command_type == 'update':
                return await self.perform_self_update()
            else:
                return {'error': f'Unknown system command: {command_type}'}
                
        except Exception as e:
            logger.error(f"Error executing system command: {e}")
            return {'error': str(e)}
    
    async def perform_self_update(self) -> Dict[str, Any]:
        """Perform self-update of the AI agent"""
        try:
            logger.info("Performing self-update...")
            
            # Check for updates
            updates_available = await self.check_for_updates()
            
            if updates_available:
                # Download and apply updates
                success = await self.apply_updates()
                return {'status': 'updated' if success else 'update_failed'}
            else:
                return {'status': 'no_updates_available'}
                
        except Exception as e:
            logger.error(f"Error during self-update: {e}")
            return {'error': str(e)}
    
    async def check_for_updates(self) -> bool:
        """Check if updates are available"""
        # This would implement actual update checking logic
        return False
    
    async def apply_updates(self) -> bool:
        """Apply available updates"""
        # This would implement actual update application logic
        return True
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("Shutting down AI Orchestrator...")
        self.is_running = False
        
        # Shutdown all components
        components = [
            self.llm_manager, self.computer_vision, self.ui_automation,
            self.learning_system, self.memory_manager, self.social_media,
            self.browser_controller, self.notification_manager,
            self.content_generator, self.task_scheduler, self.file_processor,
            self.security_manager, self.nlp_processor, self.voice_handler,
            self.translation_service
        ]
        
        for component in components:
            if component and hasattr(component, 'shutdown'):
                try:
                    await component.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down component {component.__class__.__name__}: {e}")
        
        logger.info("AI Orchestrator shutdown complete")
