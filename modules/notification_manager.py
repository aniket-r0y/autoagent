"""
Notification Manager - Smart notification handling and filtering system
"""

import asyncio
import logging
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class NotificationPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class NotificationStatus(Enum):
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    DISMISSED = "dismissed"
    EXPIRED = "expired"

@dataclass
class Notification:
    id: str
    title: str
    message: str
    source: str
    priority: NotificationPriority
    timestamp: datetime
    status: NotificationStatus = NotificationStatus.PENDING
    category: str = "general"
    actions: List[Dict] = None
    metadata: Dict = None
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.actions is None:
            self.actions = []
        if self.metadata is None:
            self.metadata = {}

class NotificationManager:
    """Advanced notification management and filtering system"""
    
    def __init__(self, settings):
        self.settings = settings
        self.notifications = []
        self.notification_handlers = {}
        self.filters = []
        self.auto_responses = {}
        self.priority_rules = {}
        self.statistics = {
            'total_received': 0,
            'filtered_out': 0,
            'auto_responded': 0,
            'user_actions': 0
        }
        
        # Configuration
        self.max_notifications = 1000
        self.auto_cleanup_interval = 3600  # 1 hour
        self.notification_timeout = timedelta(hours=24)
        
    async def initialize(self):
        """Initialize notification manager"""
        try:
            logger.info("Initializing Notification Manager...")
            
            # Load existing notifications
            await self.load_notifications()
            
            # Load configuration
            await self.load_configuration()
            
            # Start background tasks
            asyncio.create_task(self.cleanup_expired_notifications())
            asyncio.create_task(self.process_notifications())
            
            # Initialize platform-specific handlers
            await self.init_platform_handlers()
            
            logger.info("Notification Manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Notification Manager: {e}")
            raise
    
    async def init_platform_handlers(self):
        """Initialize platform-specific notification handlers"""
        try:
            # Windows notifications
            if os.name == 'nt':
                self.notification_handlers['windows'] = await self.init_windows_notifications()
            
            # macOS notifications
            elif os.name == 'posix' and os.uname().sysname == 'Darwin':
                self.notification_handlers['macos'] = await self.init_macos_notifications()
            
            # Linux notifications
            elif os.name == 'posix':
                self.notification_handlers['linux'] = await self.init_linux_notifications()
            
            # Email notifications
            self.notification_handlers['email'] = await self.init_email_notifications()
            
            # Slack notifications
            self.notification_handlers['slack'] = await self.init_slack_notifications()
            
            # Discord notifications
            self.notification_handlers['discord'] = await self.init_discord_notifications()
            
        except Exception as e:
            logger.error(f"Error initializing platform handlers: {e}")
    
    async def init_windows_notifications(self):
        """Initialize Windows notification handling"""
        try:
            # Windows 10+ notification monitoring would go here
            return {'enabled': True, 'type': 'windows_toast'}
        except Exception as e:
            logger.error(f"Error initializing Windows notifications: {e}")
            return {'enabled': False, 'error': str(e)}
    
    async def init_macos_notifications(self):
        """Initialize macOS notification handling"""
        try:
            # macOS notification center integration would go here
            return {'enabled': True, 'type': 'macos_notification_center'}
        except Exception as e:
            logger.error(f"Error initializing macOS notifications: {e}")
            return {'enabled': False, 'error': str(e)}
    
    async def init_linux_notifications(self):
        """Initialize Linux notification handling"""
        try:
            # D-Bus notification monitoring would go here
            return {'enabled': True, 'type': 'dbus_notifications'}
        except Exception as e:
            logger.error(f"Error initializing Linux notifications: {e}")
            return {'enabled': False, 'error': str(e)}
    
    async def init_email_notifications(self):
        """Initialize email notification monitoring"""
        try:
            email_config = {
                'enabled': False,
                'imap_server': os.getenv('EMAIL_IMAP_SERVER'),
                'username': os.getenv('EMAIL_USERNAME'),
                'password': os.getenv('EMAIL_PASSWORD'),
                'port': int(os.getenv('EMAIL_IMAP_PORT', '993'))
            }
            
            if all([email_config['imap_server'], email_config['username'], email_config['password']]):
                email_config['enabled'] = True
                # Start email monitoring task
                asyncio.create_task(self.monitor_email_notifications())
            
            return email_config
        except Exception as e:
            logger.error(f"Error initializing email notifications: {e}")
            return {'enabled': False, 'error': str(e)}
    
    async def init_slack_notifications(self):
        """Initialize Slack notification integration"""
        try:
            slack_token = os.getenv('SLACK_BOT_TOKEN')
            if slack_token:
                return {'enabled': True, 'token': slack_token}
            return {'enabled': False}
        except Exception as e:
            logger.error(f"Error initializing Slack notifications: {e}")
            return {'enabled': False, 'error': str(e)}
    
    async def init_discord_notifications(self):
        """Initialize Discord notification integration"""
        try:
            discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
            if discord_webhook:
                return {'enabled': True, 'webhook_url': discord_webhook}
            return {'enabled': False}
        except Exception as e:
            logger.error(f"Error initializing Discord notifications: {e}")
            return {'enabled': False, 'error': str(e)}
    
    async def create_notification(self, 
                                title: str, 
                                message: str, 
                                source: str, 
                                priority: NotificationPriority = NotificationPriority.MEDIUM,
                                category: str = "general",
                                actions: List[Dict] = None,
                                metadata: Dict = None) -> str:
        """Create a new notification"""
        try:
            notification_id = f"notif_{int(time.time() * 1000)}"
            
            notification = Notification(
                id=notification_id,
                title=title,
                message=message,
                source=source,
                priority=priority,
                timestamp=datetime.now(),
                category=category,
                actions=actions or [],
                metadata=metadata or {},
                expires_at=datetime.now() + self.notification_timeout
            )
            
            # Apply filters
            if await self.should_filter_notification(notification):
                self.statistics['filtered_out'] += 1
                logger.debug(f"Notification filtered: {title}")
                return notification_id
            
            # Add to queue
            self.notifications.append(notification)
            self.statistics['total_received'] += 1
            
            # Apply priority rules
            await self.apply_priority_rules(notification)
            
            # Check for auto-response
            await self.check_auto_response(notification)
            
            # Trigger handlers
            await self.trigger_notification_handlers(notification)
            
            # Cleanup if too many notifications
            if len(self.notifications) > self.max_notifications:
                await self.cleanup_old_notifications()
            
            logger.info(f"Created notification: {title} (Priority: {priority.name})")
            return notification_id
            
        except Exception as e:
            logger.error(f"Error creating notification: {e}")
            return ""
    
    async def should_filter_notification(self, notification: Notification) -> bool:
        """Check if notification should be filtered out"""
        try:
            for filter_rule in self.filters:
                if await self.apply_filter_rule(notification, filter_rule):
                    return True
            return False
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return False
    
    async def apply_filter_rule(self, notification: Notification, filter_rule: Dict) -> bool:
        """Apply a single filter rule"""
        try:
            rule_type = filter_rule.get('type')
            
            if rule_type == 'source_blacklist':
                blacklisted_sources = filter_rule.get('sources', [])
                return notification.source in blacklisted_sources
            
            elif rule_type == 'keyword_filter':
                blocked_keywords = filter_rule.get('keywords', [])
                text_to_check = f"{notification.title} {notification.message}".lower()
                return any(keyword.lower() in text_to_check for keyword in blocked_keywords)
            
            elif rule_type == 'priority_filter':
                min_priority = filter_rule.get('min_priority', NotificationPriority.LOW)
                return notification.priority.value < min_priority.value
            
            elif rule_type == 'time_filter':
                start_time = filter_rule.get('start_time', '00:00')
                end_time = filter_rule.get('end_time', '23:59')
                current_time = datetime.now().strftime('%H:%M')
                
                # Simple time range check
                return not (start_time <= current_time <= end_time)
            
            elif rule_type == 'frequency_filter':
                # Limit notifications from same source
                max_per_hour = filter_rule.get('max_per_hour', 10)
                recent_count = sum(1 for n in self.notifications 
                                 if n.source == notification.source 
                                 and n.timestamp > datetime.now() - timedelta(hours=1))
                return recent_count >= max_per_hour
            
            return False
            
        except Exception as e:
            logger.error(f"Error applying filter rule: {e}")
            return False
    
    async def apply_priority_rules(self, notification: Notification):
        """Apply priority adjustment rules"""
        try:
            for rule in self.priority_rules.get(notification.category, []):
                if await self.matches_priority_rule(notification, rule):
                    new_priority = NotificationPriority(rule.get('new_priority', notification.priority.value))
                    notification.priority = new_priority
                    logger.debug(f"Priority adjusted to {new_priority.name} for notification: {notification.title}")
                    break
        except Exception as e:
            logger.error(f"Error applying priority rules: {e}")
    
    async def matches_priority_rule(self, notification: Notification, rule: Dict) -> bool:
        """Check if notification matches a priority rule"""
        try:
            conditions = rule.get('conditions', [])
            
            for condition in conditions:
                condition_type = condition.get('type')
                
                if condition_type == 'keyword':
                    keywords = condition.get('keywords', [])
                    text = f"{notification.title} {notification.message}".lower()
                    if not any(keyword.lower() in text for keyword in keywords):
                        return False
                
                elif condition_type == 'source':
                    required_sources = condition.get('sources', [])
                    if notification.source not in required_sources:
                        return False
                
                elif condition_type == 'time_sensitive':
                    age_limit = condition.get('age_limit_minutes', 30)
                    if (datetime.now() - notification.timestamp).total_seconds() > age_limit * 60:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking priority rule: {e}")
            return False
    
    async def check_auto_response(self, notification: Notification):
        """Check if notification should trigger an auto-response"""
        try:
            for response_rule in self.auto_responses.get(notification.category, []):
                if await self.matches_auto_response_rule(notification, response_rule):
                    await self.execute_auto_response(notification, response_rule)
                    self.statistics['auto_responded'] += 1
                    break
        except Exception as e:
            logger.error(f"Error checking auto-response: {e}")
    
    async def matches_auto_response_rule(self, notification: Notification, rule: Dict) -> bool:
        """Check if notification matches auto-response rule"""
        try:
            triggers = rule.get('triggers', [])
            
            for trigger in triggers:
                trigger_type = trigger.get('type')
                
                if trigger_type == 'keyword':
                    keywords = trigger.get('keywords', [])
                    text = f"{notification.title} {notification.message}".lower()
                    if any(keyword.lower() in text for keyword in keywords):
                        return True
                
                elif trigger_type == 'priority':
                    required_priority = NotificationPriority(trigger.get('priority', 3))
                    if notification.priority.value >= required_priority.value:
                        return True
                
                elif trigger_type == 'source':
                    required_sources = trigger.get('sources', [])
                    if notification.source in required_sources:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking auto-response rule: {e}")
            return False
    
    async def execute_auto_response(self, notification: Notification, rule: Dict):
        """Execute an auto-response action"""
        try:
            actions = rule.get('actions', [])
            
            for action in actions:
                action_type = action.get('type')
                
                if action_type == 'acknowledge':
                    notification.status = NotificationStatus.ACKNOWLEDGED
                
                elif action_type == 'dismiss':
                    notification.status = NotificationStatus.DISMISSED
                
                elif action_type == 'forward':
                    # Forward to another system
                    destination = action.get('destination')
                    await self.forward_notification(notification, destination)
                
                elif action_type == 'log':
                    # Log to file
                    log_file = action.get('log_file', 'notifications.log')
                    await self.log_notification(notification, log_file)
                
                elif action_type == 'webhook':
                    # Send to webhook
                    webhook_url = action.get('webhook_url')
                    await self.send_webhook_notification(notification, webhook_url)
            
        except Exception as e:
            logger.error(f"Error executing auto-response: {e}")
    
    async def trigger_notification_handlers(self, notification: Notification):
        """Trigger platform-specific notification handlers"""
        try:
            # Only trigger for high priority notifications
            if notification.priority.value >= NotificationPriority.HIGH.value:
                
                # System notifications
                if notification.priority == NotificationPriority.URGENT:
                    await self.send_system_notification(notification)
                
                # Email for critical notifications
                if notification.priority == NotificationPriority.CRITICAL:
                    await self.send_email_notification(notification)
                
                # Slack for important work notifications
                if notification.category in ['work', 'security', 'system']:
                    await self.send_slack_notification(notification)
        
        except Exception as e:
            logger.error(f"Error triggering notification handlers: {e}")
    
    async def send_system_notification(self, notification: Notification):
        """Send system-level notification"""
        try:
            # This would use platform-specific APIs
            logger.info(f"System notification: {notification.title}")
        except Exception as e:
            logger.error(f"Error sending system notification: {e}")
    
    async def send_email_notification(self, notification: Notification):
        """Send email notification"""
        try:
            email_handler = self.notification_handlers.get('email', {})
            if email_handler.get('enabled', False):
                # Email sending logic would go here
                logger.info(f"Email notification sent: {notification.title}")
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    async def send_slack_notification(self, notification: Notification):
        """Send Slack notification"""
        try:
            slack_handler = self.notification_handlers.get('slack', {})
            if slack_handler.get('enabled', False):
                # Slack API call would go here
                logger.info(f"Slack notification sent: {notification.title}")
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    async def process_notifications(self):
        """Main notification processing loop"""
        while True:
            try:
                # Process pending notifications
                pending_notifications = [n for n in self.notifications if n.status == NotificationStatus.PENDING]
                
                for notification in pending_notifications:
                    # Check if expired
                    if notification.expires_at and datetime.now() > notification.expires_at:
                        notification.status = NotificationStatus.EXPIRED
                        continue
                    
                    # Process based on priority
                    await self.process_notification(notification)
                
                await asyncio.sleep(10)  # Process every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in notification processing loop: {e}")
                await asyncio.sleep(60)
    
    async def process_notification(self, notification: Notification):
        """Process a single notification"""
        try:
            # Update metadata
            notification.metadata['processed_at'] = datetime.now().isoformat()
            
            # Log for analytics
            logger.debug(f"Processing notification: {notification.title} from {notification.source}")
            
        except Exception as e:
            logger.error(f"Error processing notification {notification.id}: {e}")
    
    async def get_notifications(self, 
                              status: Optional[NotificationStatus] = None,
                              category: Optional[str] = None,
                              priority: Optional[NotificationPriority] = None,
                              limit: int = 50) -> List[Dict]:
        """Get notifications with filtering"""
        try:
            filtered_notifications = self.notifications
            
            if status:
                filtered_notifications = [n for n in filtered_notifications if n.status == status]
            
            if category:
                filtered_notifications = [n for n in filtered_notifications if n.category == category]
            
            if priority:
                filtered_notifications = [n for n in filtered_notifications if n.priority == priority]
            
            # Sort by timestamp (newest first)
            filtered_notifications.sort(key=lambda n: n.timestamp, reverse=True)
            
            # Limit results
            filtered_notifications = filtered_notifications[:limit]
            
            # Convert to dictionaries
            return [self.notification_to_dict(n) for n in filtered_notifications]
            
        except Exception as e:
            logger.error(f"Error getting notifications: {e}")
            return []
    
    def notification_to_dict(self, notification: Notification) -> Dict:
        """Convert notification to dictionary"""
        return {
            'id': notification.id,
            'title': notification.title,
            'message': notification.message,
            'source': notification.source,
            'priority': notification.priority.name,
            'status': notification.status.value,
            'category': notification.category,
            'timestamp': notification.timestamp.isoformat(),
            'expires_at': notification.expires_at.isoformat() if notification.expires_at else None,
            'actions': notification.actions,
            'metadata': notification.metadata
        }
    
    async def acknowledge_notification(self, notification_id: str) -> bool:
        """Acknowledge a notification"""
        try:
            notification = self.get_notification_by_id(notification_id)
            if notification:
                notification.status = NotificationStatus.ACKNOWLEDGED
                self.statistics['user_actions'] += 1
                logger.info(f"Acknowledged notification: {notification_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error acknowledging notification: {e}")
            return False
    
    async def dismiss_notification(self, notification_id: str) -> bool:
        """Dismiss a notification"""
        try:
            notification = self.get_notification_by_id(notification_id)
            if notification:
                notification.status = NotificationStatus.DISMISSED
                self.statistics['user_actions'] += 1
                logger.info(f"Dismissed notification: {notification_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error dismissing notification: {e}")
            return False
    
    def get_notification_by_id(self, notification_id: str) -> Optional[Notification]:
        """Get notification by ID"""
        return next((n for n in self.notifications if n.id == notification_id), None)
    
    async def cleanup_expired_notifications(self):
        """Clean up expired notifications"""
        while True:
            try:
                current_time = datetime.now()
                
                # Mark expired notifications
                for notification in self.notifications:
                    if (notification.expires_at and current_time > notification.expires_at 
                        and notification.status == NotificationStatus.PENDING):
                        notification.status = NotificationStatus.EXPIRED
                
                # Remove old dismissed/acknowledged notifications
                cutoff_time = current_time - timedelta(days=7)
                self.notifications = [
                    n for n in self.notifications
                    if not (n.status in [NotificationStatus.DISMISSED, NotificationStatus.ACKNOWLEDGED] 
                           and n.timestamp < cutoff_time)
                ]
                
                await asyncio.sleep(self.auto_cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(300)
    
    async def cleanup_old_notifications(self):
        """Remove oldest notifications when limit is exceeded"""
        try:
            if len(self.notifications) > self.max_notifications:
                # Sort by timestamp and keep newest
                self.notifications.sort(key=lambda n: n.timestamp, reverse=True)
                self.notifications = self.notifications[:self.max_notifications]
                logger.info(f"Cleaned up old notifications, keeping {self.max_notifications} newest")
        except Exception as e:
            logger.error(f"Error cleaning up old notifications: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics"""
        active_count = len([n for n in self.notifications if n.status == NotificationStatus.PENDING])
        
        return {
            **self.statistics,
            'active_notifications': active_count,
            'total_notifications': len(self.notifications),
            'by_priority': {
                priority.name: len([n for n in self.notifications if n.priority == priority])
                for priority in NotificationPriority
            },
            'by_status': {
                status.value: len([n for n in self.notifications if n.status == status])
                for status in NotificationStatus
            }
        }
    
    async def load_notifications(self):
        """Load notifications from persistent storage"""
        try:
            notifications_file = Path("data/notifications.json")
            if notifications_file.exists():
                with open(notifications_file, 'r') as f:
                    data = json.load(f)
                    # Convert back to Notification objects
                    for item in data:
                        notification = Notification(
                            id=item['id'],
                            title=item['title'],
                            message=item['message'],
                            source=item['source'],
                            priority=NotificationPriority[item['priority']],
                            timestamp=datetime.fromisoformat(item['timestamp']),
                            status=NotificationStatus(item['status']),
                            category=item.get('category', 'general'),
                            actions=item.get('actions', []),
                            metadata=item.get('metadata', {}),
                            expires_at=datetime.fromisoformat(item['expires_at']) if item.get('expires_at') else None
                        )
                        self.notifications.append(notification)
                
                logger.info(f"Loaded {len(self.notifications)} notifications")
        except Exception as e:
            logger.error(f"Error loading notifications: {e}")
    
    async def save_notifications(self):
        """Save notifications to persistent storage"""
        try:
            notifications_file = Path("data/notifications.json")
            notifications_file.parent.mkdir(exist_ok=True)
            
            data = [self.notification_to_dict(n) for n in self.notifications]
            
            with open(notifications_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving notifications: {e}")
    
    async def load_configuration(self):
        """Load notification configuration"""
        try:
            config_file = Path("data/notification_config.json")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    
                    self.filters = config.get('filters', [])
                    self.auto_responses = config.get('auto_responses', {})
                    self.priority_rules = config.get('priority_rules', {})
                    
                logger.info("Loaded notification configuration")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    async def shutdown(self):
        """Shutdown notification manager"""
        logger.info("Shutting down Notification Manager...")
        await self.save_notifications()
