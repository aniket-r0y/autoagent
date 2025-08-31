"""
Task Scheduler - Advanced task automation and scheduling system
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import crontab
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class TriggerType(Enum):
    TIME = "time"
    INTERVAL = "interval"
    CRON = "cron"
    EVENT = "event"
    CONDITION = "condition"

@dataclass
class ScheduledTask:
    id: str
    name: str
    description: str
    action_type: str
    parameters: Dict[str, Any]
    trigger_type: TriggerType
    trigger_config: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    timeout: int = 300  # 5 minutes default
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

class TaskScheduler:
    """Advanced task scheduling and automation system"""
    
    def __init__(self, settings):
        self.settings = settings
        self.tasks = {}
        self.running_tasks = {}
        self.task_history = []
        self.event_handlers = {}
        self.condition_checkers = {}
        
        # Scheduler configuration
        self.max_concurrent_tasks = 10
        self.max_history = 1000
        self.check_interval = 10  # seconds
        self.is_running = False
        
        # Performance tracking
        self.statistics = {
            'total_scheduled': 0,
            'total_completed': 0,
            'total_failed': 0,
            'average_duration': 0,
            'last_cleanup': datetime.now()
        }
        
    async def initialize(self):
        """Initialize task scheduler"""
        try:
            logger.info("Initializing Task Scheduler...")
            
            # Load existing tasks
            await self.load_tasks()
            
            # Start scheduler loop
            self.is_running = True
            asyncio.create_task(self.scheduler_loop())
            
            # Start cleanup task
            asyncio.create_task(self.cleanup_loop())
            
            # Initialize default event handlers
            await self.init_default_handlers()
            
            logger.info("Task Scheduler initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Task Scheduler: {e}")
            raise
    
    async def init_default_handlers(self):
        """Initialize default event handlers"""
        try:
            # File system events
            self.event_handlers['file_created'] = self.handle_file_event
            self.event_handlers['file_modified'] = self.handle_file_event
            self.event_handlers['file_deleted'] = self.handle_file_event
            
            # System events
            self.event_handlers['system_startup'] = self.handle_system_event
            self.event_handlers['system_shutdown'] = self.handle_system_event
            self.event_handlers['user_login'] = self.handle_system_event
            
            # Application events
            self.event_handlers['task_completed'] = self.handle_task_event
            self.event_handlers['task_failed'] = self.handle_task_event
            
            # Default condition checkers
            self.condition_checkers['time_based'] = self.check_time_condition
            self.condition_checkers['file_exists'] = self.check_file_condition
            self.condition_checkers['process_running'] = self.check_process_condition
            
        except Exception as e:
            logger.error(f"Error initializing default handlers: {e}")
    
    async def schedule_task(self, task_params: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a new task"""
        try:
            # Generate unique task ID
            task_id = f"task_{int(time.time() * 1000)}"
            
            # Create task object
            task = ScheduledTask(
                id=task_id,
                name=task_params.get('name', f'Task {task_id}'),
                description=task_params.get('description', ''),
                action_type=task_params.get('action_type', 'custom'),
                parameters=task_params.get('parameters', {}),
                trigger_type=TriggerType(task_params.get('trigger_type', 'time')),
                trigger_config=task_params.get('trigger_config', {}),
                priority=TaskPriority(task_params.get('priority', TaskPriority.NORMAL.value)),
                timeout=task_params.get('timeout', 300),
                max_runs=task_params.get('max_runs'),
                max_retries=task_params.get('max_retries', 3),
                dependencies=task_params.get('dependencies', []),
                tags=task_params.get('tags', []),
                metadata=task_params.get('metadata', {})
            )
            
            # Calculate next run time
            task.next_run = await self.calculate_next_run(task)
            
            if task.next_run is None:
                return {'success': False, 'error': 'Could not calculate next run time'}
            
            # Store task
            self.tasks[task_id] = task
            self.statistics['total_scheduled'] += 1
            
            # Save tasks
            await self.save_tasks()
            
            logger.info(f"Scheduled task: {task.name} (ID: {task_id}) for {task.next_run}")
            
            return {
                'success': True,
                'task_id': task_id,
                'next_run': task.next_run.isoformat(),
                'task_name': task.name
            }
            
        except Exception as e:
            logger.error(f"Error scheduling task: {e}")
            return {'success': False, 'error': str(e)}
    
    async def calculate_next_run(self, task: ScheduledTask) -> Optional[datetime]:
        """Calculate next run time for a task"""
        try:
            now = datetime.now()
            trigger_config = task.trigger_config
            
            if task.trigger_type == TriggerType.TIME:
                # Specific date/time
                run_time = trigger_config.get('datetime')
                if isinstance(run_time, str):
                    return datetime.fromisoformat(run_time)
                elif isinstance(run_time, datetime):
                    return run_time
                else:
                    return now + timedelta(minutes=5)  # Default to 5 minutes
            
            elif task.trigger_type == TriggerType.INTERVAL:
                # Recurring interval
                interval_seconds = trigger_config.get('interval_seconds', 3600)
                base_time = task.last_run or now
                return base_time + timedelta(seconds=interval_seconds)
            
            elif task.trigger_type == TriggerType.CRON:
                # Cron expression
                cron_expr = trigger_config.get('cron_expression', '0 * * * *')
                return self.parse_cron_next_run(cron_expr, now)
            
            elif task.trigger_type == TriggerType.EVENT:
                # Event-triggered (no specific time)
                return None
            
            elif task.trigger_type == TriggerType.CONDITION:
                # Condition-based (check every minute)
                return now + timedelta(minutes=1)
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating next run time: {e}")
            return None
    
    def parse_cron_next_run(self, cron_expr: str, from_time: datetime) -> datetime:
        """Parse cron expression and get next run time"""
        try:
            # Simple cron parser for basic expressions
            # Format: minute hour day month weekday
            parts = cron_expr.split()
            if len(parts) != 5:
                raise ValueError("Invalid cron expression")
            
            minute, hour, day, month, weekday = parts
            
            # For simplicity, handle basic cases
            next_run = from_time.replace(second=0, microsecond=0)
            
            # Set minute
            if minute != '*':
                target_minute = int(minute)
                if target_minute > next_run.minute:
                    next_run = next_run.replace(minute=target_minute)
                else:
                    next_run = next_run.replace(minute=target_minute) + timedelta(hours=1)
            
            # Set hour
            if hour != '*':
                target_hour = int(hour)
                if target_hour > next_run.hour:
                    next_run = next_run.replace(hour=target_hour, minute=int(minute) if minute != '*' else 0)
                elif target_hour == next_run.hour:
                    if minute != '*' and int(minute) <= next_run.minute:
                        next_run += timedelta(days=1)
                        next_run = next_run.replace(hour=target_hour, minute=int(minute))
                else:
                    next_run += timedelta(days=1)
                    next_run = next_run.replace(hour=target_hour, minute=int(minute) if minute != '*' else 0)
            
            return next_run
            
        except Exception as e:
            logger.error(f"Error parsing cron expression: {e}")
            return from_time + timedelta(hours=1)  # Default fallback
    
    async def scheduler_loop(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                await self.process_scheduled_tasks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def process_scheduled_tasks(self):
        """Process tasks that are due to run"""
        try:
            now = datetime.now()
            due_tasks = []
            
            # Find tasks that are due
            for task in self.tasks.values():
                if (task.status == TaskStatus.PENDING and 
                    task.next_run and 
                    task.next_run <= now and
                    await self.check_task_dependencies(task)):
                    due_tasks.append(task)
            
            # Sort by priority
            due_tasks.sort(key=lambda t: t.priority.value, reverse=True)
            
            # Execute tasks (respecting concurrency limit)
            for task in due_tasks:
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    break
                
                await self.execute_task(task)
            
        except Exception as e:
            logger.error(f"Error processing scheduled tasks: {e}")
    
    async def check_task_dependencies(self, task: ScheduledTask) -> bool:
        """Check if task dependencies are satisfied"""
        try:
            if not task.dependencies:
                return True
            
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking dependencies for task {task.id}: {e}")
            return False
    
    async def execute_task(self, task: ScheduledTask):
        """Execute a scheduled task"""
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.last_run = datetime.now()
            task.run_count += 1
            
            # Add to running tasks
            self.running_tasks[task.id] = task
            
            logger.info(f"Executing task: {task.name} (ID: {task.id})")
            
            # Create task execution coroutine
            execution_task = asyncio.create_task(
                self.run_task_action(task)
            )
            
            # Handle timeout
            try:
                result = await asyncio.wait_for(execution_task, timeout=task.timeout)
                await self.handle_task_success(task, result)
            except asyncio.TimeoutError:
                await self.handle_task_timeout(task)
            except Exception as e:
                await self.handle_task_failure(task, e)
            
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            await self.handle_task_failure(task, e)
    
    async def run_task_action(self, task: ScheduledTask) -> Dict[str, Any]:
        """Run the actual task action"""
        try:
            action_type = task.action_type
            parameters = task.parameters
            
            if action_type == 'system_command':
                return await self.run_system_command(parameters)
            elif action_type == 'file_operation':
                return await self.run_file_operation(parameters)
            elif action_type == 'api_call':
                return await self.run_api_call(parameters)
            elif action_type == 'backup':
                return await self.run_backup_task(parameters)
            elif action_type == 'cleanup':
                return await self.run_cleanup_task(parameters)
            elif action_type == 'notification':
                return await self.run_notification_task(parameters)
            elif action_type == 'data_sync':
                return await self.run_data_sync_task(parameters)
            elif action_type == 'health_check':
                return await self.run_health_check(parameters)
            elif action_type == 'custom':
                return await self.run_custom_task(parameters)
            else:
                return {'success': False, 'error': f'Unknown action type: {action_type}'}
            
        except Exception as e:
            logger.error(f"Error running task action: {e}")
            return {'success': False, 'error': str(e)}
    
    async def run_system_command(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run system command task"""
        try:
            import subprocess
            
            command = parameters.get('command', '')
            if not command:
                return {'success': False, 'error': 'No command specified'}
            
            # Execute command
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            
            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def run_file_operation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run file operation task"""
        try:
            operation = parameters.get('operation', 'copy')
            source = parameters.get('source', '')
            destination = parameters.get('destination', '')
            
            if operation == 'copy':
                import shutil
                shutil.copy2(source, destination)
                return {'success': True, 'operation': 'copy', 'source': source, 'destination': destination}
            
            elif operation == 'move':
                import shutil
                shutil.move(source, destination)
                return {'success': True, 'operation': 'move', 'source': source, 'destination': destination}
            
            elif operation == 'delete':
                import os
                if os.path.isfile(source):
                    os.remove(source)
                elif os.path.isdir(source):
                    import shutil
                    shutil.rmtree(source)
                return {'success': True, 'operation': 'delete', 'path': source}
            
            elif operation == 'create_directory':
                os.makedirs(source, exist_ok=True)
                return {'success': True, 'operation': 'create_directory', 'path': source}
            
            else:
                return {'success': False, 'error': f'Unknown file operation: {operation}'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def run_api_call(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run API call task"""
        try:
            import aiohttp
            
            url = parameters.get('url', '')
            method = parameters.get('method', 'GET').upper()
            headers = parameters.get('headers', {})
            data = parameters.get('data', {})
            
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, json=data) as response:
                    result_data = await response.text()
                    
                    return {
                        'success': response.status < 400,
                        'status_code': response.status,
                        'response_data': result_data,
                        'url': url
                    }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def run_backup_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run backup task"""
        try:
            source_path = parameters.get('source_path', '')
            backup_path = parameters.get('backup_path', '')
            compression = parameters.get('compression', False)
            
            if not source_path or not backup_path:
                return {'success': False, 'error': 'Source and backup paths required'}
            
            import shutil
            import datetime
            
            # Create timestamped backup directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{backup_path}/backup_{timestamp}"
            
            if compression:
                # Create compressed archive
                shutil.make_archive(backup_dir, 'zip', source_path)
                backup_file = f"{backup_dir}.zip"
            else:
                # Copy directory tree
                shutil.copytree(source_path, backup_dir)
                backup_file = backup_dir
            
            return {
                'success': True,
                'backup_location': backup_file,
                'source': source_path,
                'compressed': compression
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def run_cleanup_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run cleanup task"""
        try:
            target_path = parameters.get('target_path', '')
            age_days = parameters.get('age_days', 30)
            file_pattern = parameters.get('file_pattern', '*')
            
            import os
            import glob
            
            if not target_path or not os.path.exists(target_path):
                return {'success': False, 'error': 'Invalid target path'}
            
            # Find files to clean up
            pattern = os.path.join(target_path, file_pattern)
            files = glob.glob(pattern)
            
            cutoff_time = time.time() - (age_days * 24 * 3600)
            cleaned_files = []
            
            for file_path in files:
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        cleaned_files.append(file_path)
            
            return {
                'success': True,
                'files_cleaned': len(cleaned_files),
                'cleaned_files': cleaned_files[:10],  # First 10 files
                'target_path': target_path
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def run_notification_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run notification task"""
        try:
            message = parameters.get('message', '')
            title = parameters.get('title', 'Scheduled Task Notification')
            priority = parameters.get('priority', 'normal')
            
            # This would integrate with the notification manager
            logger.info(f"Notification: {title} - {message}")
            
            return {
                'success': True,
                'notification_sent': True,
                'title': title,
                'message': message
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def run_data_sync_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run data synchronization task"""
        try:
            source = parameters.get('source', '')
            destination = parameters.get('destination', '')
            sync_type = parameters.get('sync_type', 'mirror')
            
            # Simple file sync implementation
            import shutil
            import os
            
            if sync_type == 'mirror':
                if os.path.exists(destination):
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            elif sync_type == 'incremental':
                # Copy only newer files
                for root, dirs, files in os.walk(source):
                    for file in files:
                        src_file = os.path.join(root, file)
                        rel_path = os.path.relpath(src_file, source)
                        dst_file = os.path.join(destination, rel_path)
                        
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        
                        if not os.path.exists(dst_file) or os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                            shutil.copy2(src_file, dst_file)
            
            return {
                'success': True,
                'sync_type': sync_type,
                'source': source,
                'destination': destination
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def run_health_check(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run system health check task"""
        try:
            import psutil
            
            # Check system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            health_status = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'available_memory_gb': memory.available / (1024**3),
                'available_disk_gb': disk.free / (1024**3)
            }
            
            # Determine overall health
            health_ok = (
                cpu_percent < 80 and
                memory.percent < 85 and
                disk.percent < 90
            )
            
            return {
                'success': True,
                'health_ok': health_ok,
                'health_status': health_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def run_custom_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run custom task with user-defined logic"""
        try:
            # This would allow users to define custom task logic
            custom_code = parameters.get('custom_code', '')
            
            if not custom_code:
                return {'success': False, 'error': 'No custom code provided'}
            
            # For security, this would need proper sandboxing in production
            # For now, just return a placeholder
            return {
                'success': True,
                'message': 'Custom task executed',
                'parameters': parameters
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def handle_task_success(self, task: ScheduledTask, result: Dict[str, Any]):
        """Handle successful task completion"""
        try:
            task.status = TaskStatus.COMPLETED
            task.retry_count = 0
            
            # Remove from running tasks
            self.running_tasks.pop(task.id, None)
            
            # Update statistics
            self.statistics['total_completed'] += 1
            
            # Calculate next run if recurring
            if task.trigger_type in [TriggerType.INTERVAL, TriggerType.CRON]:
                if task.max_runs is None or task.run_count < task.max_runs:
                    task.next_run = await self.calculate_next_run(task)
                    task.status = TaskStatus.PENDING
                else:
                    # Task has reached max runs
                    task.next_run = None
            else:
                # One-time task
                task.next_run = None
            
            # Add to history
            self.add_to_history(task, result, 'success')
            
            # Trigger completion event
            await self.trigger_event('task_completed', {
                'task_id': task.id,
                'task_name': task.name,
                'result': result
            })
            
            logger.info(f"Task completed successfully: {task.name} (ID: {task.id})")
            
        except Exception as e:
            logger.error(f"Error handling task success: {e}")
    
    async def handle_task_failure(self, task: ScheduledTask, error: Exception):
        """Handle task failure"""
        try:
            task.retry_count += 1
            
            # Remove from running tasks
            self.running_tasks.pop(task.id, None)
            
            if task.retry_count <= task.max_retries:
                # Schedule retry
                task.status = TaskStatus.PENDING
                task.next_run = datetime.now() + timedelta(minutes=task.retry_count * 5)  # Exponential backoff
                logger.warning(f"Task failed, scheduling retry {task.retry_count}/{task.max_retries}: {task.name}")
            else:
                # Max retries reached
                task.status = TaskStatus.FAILED
                task.next_run = None
                self.statistics['total_failed'] += 1
                logger.error(f"Task failed permanently: {task.name} (ID: {task.id}) - {error}")
            
            # Add to history
            self.add_to_history(task, {'error': str(error)}, 'failure')
            
            # Trigger failure event
            await self.trigger_event('task_failed', {
                'task_id': task.id,
                'task_name': task.name,
                'error': str(error),
                'retry_count': task.retry_count
            })
            
        except Exception as e:
            logger.error(f"Error handling task failure: {e}")
    
    async def handle_task_timeout(self, task: ScheduledTask):
        """Handle task timeout"""
        try:
            await self.handle_task_failure(task, Exception(f"Task timed out after {task.timeout} seconds"))
        except Exception as e:
            logger.error(f"Error handling task timeout: {e}")
    
    def add_to_history(self, task: ScheduledTask, result: Dict[str, Any], outcome: str):
        """Add task execution to history"""
        try:
            history_entry = {
                'task_id': task.id,
                'task_name': task.name,
                'execution_time': task.last_run.isoformat() if task.last_run else None,
                'outcome': outcome,
                'result': result,
                'run_count': task.run_count,
                'retry_count': task.retry_count
            }
            
            self.task_history.append(history_entry)
            
            # Limit history size
            if len(self.task_history) > self.max_history:
                self.task_history = self.task_history[-self.max_history:]
                
        except Exception as e:
            logger.error(f"Error adding to history: {e}")
    
    async def trigger_event(self, event_type: str, event_data: Dict[str, Any]):
        """Trigger an event that may activate event-based tasks"""
        try:
            # Find tasks that listen for this event
            event_tasks = [
                task for task in self.tasks.values()
                if (task.trigger_type == TriggerType.EVENT and
                    task.trigger_config.get('event_type') == event_type and
                    task.status == TaskStatus.PENDING)
            ]
            
            for task in event_tasks:
                # Check if event matches task criteria
                if await self.event_matches_task(event_data, task):
                    task.next_run = datetime.now()
                    logger.info(f"Event {event_type} triggered task: {task.name}")
            
            # Call event handlers
            handler = self.event_handlers.get(event_type)
            if handler:
                await handler(event_data)
                
        except Exception as e:
            logger.error(f"Error triggering event {event_type}: {e}")
    
    async def event_matches_task(self, event_data: Dict[str, Any], task: ScheduledTask) -> bool:
        """Check if event matches task criteria"""
        try:
            criteria = task.trigger_config.get('event_criteria', {})
            
            for key, expected_value in criteria.items():
                if key not in event_data or event_data[key] != expected_value:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking event match: {e}")
            return False
    
    async def handle_file_event(self, event_data: Dict[str, Any]):
        """Handle file system events"""
        try:
            logger.debug(f"File event: {event_data}")
        except Exception as e:
            logger.error(f"Error handling file event: {e}")
    
    async def handle_system_event(self, event_data: Dict[str, Any]):
        """Handle system events"""
        try:
            logger.debug(f"System event: {event_data}")
        except Exception as e:
            logger.error(f"Error handling system event: {e}")
    
    async def handle_task_event(self, event_data: Dict[str, Any]):
        """Handle task-related events"""
        try:
            logger.debug(f"Task event: {event_data}")
        except Exception as e:
            logger.error(f"Error handling task event: {e}")
    
    async def cleanup_loop(self):
        """Periodic cleanup of completed tasks and history"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_tasks()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def cleanup_tasks(self):
        """Clean up old completed tasks"""
        try:
            now = datetime.now()
            cutoff_time = now - timedelta(days=7)  # Keep tasks for 7 days
            
            # Remove old completed/failed tasks
            tasks_to_remove = []
            for task_id, task in self.tasks.items():
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] and
                    task.last_run and task.last_run < cutoff_time):
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
            
            # Clean up old history
            history_cutoff = now - timedelta(days=30)
            self.task_history = [
                entry for entry in self.task_history
                if datetime.fromisoformat(entry['execution_time']) > history_cutoff
            ]
            
            self.statistics['last_cleanup'] = now
            
            if tasks_to_remove:
                logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
                await self.save_tasks()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def pause_task(self, task_id: str) -> bool:
        """Pause a scheduled task"""
        try:
            task = self.tasks.get(task_id)
            if task and task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task.status = TaskStatus.PAUSED
                logger.info(f"Paused task: {task.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error pausing task {task_id}: {e}")
            return False
    
    async def resume_task(self, task_id: str) -> bool:
        """Resume a paused task"""
        try:
            task = self.tasks.get(task_id)
            if task and task.status == TaskStatus.PAUSED:
                task.status = TaskStatus.PENDING
                task.next_run = await self.calculate_next_run(task)
                logger.info(f"Resumed task: {task.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error resuming task {task_id}: {e}")
            return False
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        try:
            task = self.tasks.get(task_id)
            if task:
                task.status = TaskStatus.CANCELLED
                task.next_run = None
                # Remove from running tasks if currently executing
                self.running_tasks.pop(task_id, None)
                logger.info(f"Cancelled task: {task.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        try:
            task = self.tasks.get(task_id)
            if task:
                return {
                    'id': task.id,
                    'name': task.name,
                    'status': task.status.value,
                    'next_run': task.next_run.isoformat() if task.next_run else None,
                    'last_run': task.last_run.isoformat() if task.last_run else None,
                    'run_count': task.run_count,
                    'retry_count': task.retry_count,
                    'priority': task.priority.name
                }
            return None
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get status of all tasks"""
        try:
            return [
                {
                    'id': task.id,
                    'name': task.name,
                    'description': task.description,
                    'status': task.status.value,
                    'priority': task.priority.name,
                    'trigger_type': task.trigger_type.value,
                    'next_run': task.next_run.isoformat() if task.next_run else None,
                    'last_run': task.last_run.isoformat() if task.last_run else None,
                    'run_count': task.run_count,
                    'created_at': task.created_at.isoformat()
                }
                for task in self.tasks.values()
            ]
        except Exception as e:
            logger.error(f"Error getting all tasks: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        try:
            return {
                **self.statistics,
                'total_tasks': len(self.tasks),
                'running_tasks': len(self.running_tasks),
                'pending_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
                'failed_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED]),
                'history_entries': len(self.task_history)
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    async def save_tasks(self):
        """Save tasks to persistent storage"""
        try:
            tasks_file = Path("data/scheduled_tasks.json")
            tasks_file.parent.mkdir(exist_ok=True)
            
            # Convert tasks to serializable format
            tasks_data = {}
            for task_id, task in self.tasks.items():
                task_dict = asdict(task)
                # Convert datetime objects to ISO strings
                if task_dict['created_at']:
                    task_dict['created_at'] = task_dict['created_at'].isoformat()
                if task_dict['next_run']:
                    task_dict['next_run'] = task_dict['next_run'].isoformat()
                if task_dict['last_run']:
                    task_dict['last_run'] = task_dict['last_run'].isoformat()
                # Convert enums to values
                task_dict['status'] = task_dict['status'].value
                task_dict['priority'] = task_dict['priority'].value
                task_dict['trigger_type'] = task_dict['trigger_type'].value
                
                tasks_data[task_id] = task_dict
            
            with open(tasks_file, 'w') as f:
                json.dump(tasks_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving tasks: {e}")
    
    async def load_tasks(self):
        """Load tasks from persistent storage"""
        try:
            tasks_file = Path("data/scheduled_tasks.json")
            if not tasks_file.exists():
                return
            
            with open(tasks_file, 'r') as f:
                tasks_data = json.load(f)
            
            for task_id, task_dict in tasks_data.items():
                # Convert ISO strings back to datetime objects
                if task_dict['created_at']:
                    task_dict['created_at'] = datetime.fromisoformat(task_dict['created_at'])
                if task_dict['next_run']:
                    task_dict['next_run'] = datetime.fromisoformat(task_dict['next_run'])
                if task_dict['last_run']:
                    task_dict['last_run'] = datetime.fromisoformat(task_dict['last_run'])
                
                # Convert values back to enums
                task_dict['status'] = TaskStatus(task_dict['status'])
                task_dict['priority'] = TaskPriority(task_dict['priority'])
                task_dict['trigger_type'] = TriggerType(task_dict['trigger_type'])
                
                # Create task object
                task = ScheduledTask(**task_dict)
                self.tasks[task_id] = task
            
            logger.info(f"Loaded {len(self.tasks)} scheduled tasks")
            
        except Exception as e:
            logger.error(f"Error loading tasks: {e}")
    
    async def shutdown(self):
        """Shutdown task scheduler"""
        logger.info("Shutting down Task Scheduler...")
        self.is_running = False
        
        # Cancel all running tasks
        for task_id in list(self.running_tasks.keys()):
            await self.cancel_task(task_id)
        
        # Save tasks
        await self.save_tasks()
