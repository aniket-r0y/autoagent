"""
Settings - Configuration management for the AI Agent system
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class LLMSettings:
    """LLM configuration settings"""
    default_model: str = "gemini"
    timeout: int = 30
    max_tokens: int = 2048
    temperature: float = 0.7
    enable_streaming: bool = True
    enable_function_calling: bool = True
    fallback_models: list = None
    
    def __post_init__(self):
        if self.fallback_models is None:
            self.fallback_models = ["ollama", "lm_studio"]

@dataclass
class ComputerVisionSettings:
    """Computer vision configuration settings"""
    ocr_enabled: bool = True
    template_matching_threshold: float = 0.8
    screenshot_quality: int = 90
    element_detection_confidence: float = 0.7
    cache_templates: bool = True
    max_cache_size: int = 100

@dataclass
class UIAutomationSettings:
    """UI automation configuration settings"""
    action_delay: float = 0.1
    wait_timeout: int = 10
    screenshot_on_action: bool = False
    element_highlight: bool = True
    safe_mode: bool = True
    retry_attempts: int = 3

@dataclass
class LearningSettings:
    """Learning system configuration settings"""
    enabled: bool = True
    max_interactions: int = 10000
    pattern_window: int = 100
    min_pattern_frequency: int = 3
    learning_rate: float = 0.01
    auto_optimization: bool = True
    retention_days: int = 365

@dataclass
class MemorySettings:
    """Memory management configuration settings"""
    max_context_length: int = 4000
    relevance_threshold: float = 0.7
    memory_decay_factor: float = 0.95
    cleanup_interval: int = 3600
    database_path: str = "data/memory.db"
    enable_encryption: bool = True

@dataclass
class SecuritySettings:
    """Security configuration settings"""
    encryption_enabled: bool = True
    max_failed_attempts: int = 5
    lockout_duration: int = 1800
    session_timeout: int = 3600
    audit_enabled: bool = True
    audit_retention_days: int = 90
    require_2fa: bool = False

@dataclass
class NotificationSettings:
    """Notification system configuration settings"""
    enabled: bool = True
    max_notifications: int = 1000
    auto_cleanup: bool = True
    cleanup_interval: int = 3600
    priority_filtering: bool = True
    real_time_alerts: bool = True

@dataclass
class VoiceSettings:
    """Voice interaction configuration settings"""
    enabled: bool = False
    wake_words: list = None
    language: str = "en-US"
    voice_speed: int = 150
    voice_volume: float = 0.8
    continuous_listening: bool = False
    noise_threshold: int = 300
    
    def __post_init__(self):
        if self.wake_words is None:
            self.wake_words = ["hey agent", "computer", "assistant"]

@dataclass
class TranslationSettings:
    """Translation service configuration settings"""
    enabled: bool = True
    default_language: str = "en"
    cache_enabled: bool = True
    cache_max_size: int = 1000
    auto_detect: bool = True
    confidence_threshold: float = 0.8

@dataclass
class ContentGenerationSettings:
    """Content generation configuration settings"""
    enabled: bool = True
    max_history: int = 1000
    image_generation: bool = True
    video_generation: bool = True
    audio_generation: bool = True
    quality_preset: str = "balanced"  # low, balanced, high

@dataclass
class TaskSchedulerSettings:
    """Task scheduler configuration settings"""
    enabled: bool = True
    max_concurrent_tasks: int = 10
    max_history: int = 1000
    check_interval: int = 10
    enable_persistence: bool = True
    auto_cleanup: bool = True

@dataclass
class FileProcessorSettings:
    """File processor configuration settings"""
    enabled: bool = True
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    temp_cleanup: bool = True
    auto_virus_scan: bool = False
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["txt", "pdf", "docx", "xlsx", "png", "jpg", "mp4", "mp3"]

@dataclass
class SystemSettings:
    """System-wide configuration settings"""
    debug_mode: bool = False
    log_level: str = "INFO"
    max_log_size: int = 50 * 1024 * 1024  # 50MB
    log_retention_days: int = 30
    performance_monitoring: bool = True
    auto_update: bool = False
    data_directory: str = "data"
    temp_directory: str = "temp"
    backup_enabled: bool = True
    backup_interval: int = 86400  # 24 hours

class Settings:
    """Main settings manager for the AI Agent system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/config.json"
        self.config_path = Path(self.config_file)
        
        # Initialize all setting categories
        self.llm = LLMSettings()
        self.computer_vision = ComputerVisionSettings()
        self.ui_automation = UIAutomationSettings()
        self.learning = LearningSettings()
        self.memory = MemorySettings()
        self.security = SecuritySettings()
        self.notifications = NotificationSettings()
        self.voice = VoiceSettings()
        self.translation = TranslationSettings()
        self.content_generation = ContentGenerationSettings()
        self.task_scheduler = TaskSchedulerSettings()
        self.file_processor = FileProcessorSettings()
        self.system = SystemSettings()
        
        # Load configuration
        self.load_config()
        
        # Apply environment variable overrides
        self.apply_env_overrides()
        
        # Validate settings
        self.validate_settings()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update settings from config file
                self._update_settings_from_dict(config_data)
                
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.info("No config file found, using default settings")
                # Create default config file
                self.save_config()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default settings")
    
    def _update_settings_from_dict(self, config_data: Dict[str, Any]):
        """Update settings objects from dictionary"""
        try:
            setting_objects = {
                'llm': self.llm,
                'computer_vision': self.computer_vision,
                'ui_automation': self.ui_automation,
                'learning': self.learning,
                'memory': self.memory,
                'security': self.security,
                'notifications': self.notifications,
                'voice': self.voice,
                'translation': self.translation,
                'content_generation': self.content_generation,
                'task_scheduler': self.task_scheduler,
                'file_processor': self.file_processor,
                'system': self.system
            }
            
            for category, settings_obj in setting_objects.items():
                if category in config_data:
                    category_data = config_data[category]
                    for key, value in category_data.items():
                        if hasattr(settings_obj, key):
                            setattr(settings_obj, key, value)
                        else:
                            logger.warning(f"Unknown setting: {category}.{key}")
                            
        except Exception as e:
            logger.error(f"Error updating settings from dict: {e}")
    
    def apply_env_overrides(self):
        """Apply environment variable overrides"""
        try:
            # LLM settings
            if os.getenv("DEFAULT_MODEL"):
                self.llm.default_model = os.getenv("DEFAULT_MODEL")
            if os.getenv("LLM_TIMEOUT"):
                self.llm.timeout = int(os.getenv("LLM_TIMEOUT"))
            
            # Security settings
            if os.getenv("ENCRYPTION_ENABLED"):
                self.security.encryption_enabled = os.getenv("ENCRYPTION_ENABLED").lower() == "true"
            if os.getenv("SESSION_TIMEOUT"):
                self.security.session_timeout = int(os.getenv("SESSION_TIMEOUT"))
            
            # System settings
            if os.getenv("DEBUG_MODE"):
                self.system.debug_mode = os.getenv("DEBUG_MODE").lower() == "true"
            if os.getenv("LOG_LEVEL"):
                self.system.log_level = os.getenv("LOG_LEVEL").upper()
            if os.getenv("DATA_DIRECTORY"):
                self.system.data_directory = os.getenv("DATA_DIRECTORY")
            
            # Voice settings
            if os.getenv("VOICE_ENABLED"):
                self.voice.enabled = os.getenv("VOICE_ENABLED").lower() == "true"
            if os.getenv("VOICE_LANGUAGE"):
                self.voice.language = os.getenv("VOICE_LANGUAGE")
            
            logger.debug("Environment variable overrides applied")
            
        except Exception as e:
            logger.error(f"Error applying environment overrides: {e}")
    
    def validate_settings(self):
        """Validate configuration settings"""
        try:
            # Validate LLM settings
            if self.llm.temperature < 0 or self.llm.temperature > 2:
                logger.warning("Invalid LLM temperature, setting to default (0.7)")
                self.llm.temperature = 0.7
            
            if self.llm.max_tokens < 1:
                logger.warning("Invalid max_tokens, setting to default (2048)")
                self.llm.max_tokens = 2048
            
            # Validate file processor settings
            if self.file_processor.max_file_size < 1024:
                logger.warning("Max file size too small, setting to 1MB minimum")
                self.file_processor.max_file_size = 1024 * 1024
            
            # Validate security settings
            if self.security.max_failed_attempts < 1:
                logger.warning("Invalid max_failed_attempts, setting to 3")
                self.security.max_failed_attempts = 3
            
            if self.security.session_timeout < 300:
                logger.warning("Session timeout too short, setting to 5 minutes minimum")
                self.security.session_timeout = 300
            
            # Validate system settings
            if self.system.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                logger.warning("Invalid log level, setting to INFO")
                self.system.log_level = "INFO"
            
            # Create required directories
            self._create_directories()
            
            logger.debug("Settings validation completed")
            
        except Exception as e:
            logger.error(f"Error validating settings: {e}")
    
    def _create_directories(self):
        """Create required directories"""
        try:
            directories = [
                self.system.data_directory,
                self.system.temp_directory,
                "logs",
                "config",
                f"{self.system.data_directory}/learning",
                f"{self.system.data_directory}/templates",
                f"{self.system.data_directory}/secure"
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = self.to_dict()
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'llm': asdict(self.llm),
            'computer_vision': asdict(self.computer_vision),
            'ui_automation': asdict(self.ui_automation),
            'learning': asdict(self.learning),
            'memory': asdict(self.memory),
            'security': asdict(self.security),
            'notifications': asdict(self.notifications),
            'voice': asdict(self.voice),
            'translation': asdict(self.translation),
            'content_generation': asdict(self.content_generation),
            'task_scheduler': asdict(self.task_scheduler),
            'file_processor': asdict(self.file_processor),
            'system': asdict(self.system),
            'metadata': {
                'version': '1.0.0',
                'last_updated': datetime.now().isoformat(),
                'config_file': str(self.config_path)
            }
        }
    
    def update(self, updates: Dict[str, Any]) -> bool:
        """Update settings from dictionary"""
        try:
            self._update_settings_from_dict(updates)
            self.validate_settings()
            self.save_config()
            logger.info("Settings updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return False
    
    def get_setting(self, category: str, key: str, default: Any = None) -> Any:
        """Get a specific setting value"""
        try:
            if hasattr(self, category):
                category_obj = getattr(self, category)
                if hasattr(category_obj, key):
                    return getattr(category_obj, key)
            return default
        except Exception as e:
            logger.error(f"Error getting setting {category}.{key}: {e}")
            return default
    
    def set_setting(self, category: str, key: str, value: Any) -> bool:
        """Set a specific setting value"""
        try:
            if hasattr(self, category):
                category_obj = getattr(self, category)
                if hasattr(category_obj, key):
                    setattr(category_obj, key, value)
                    self.save_config()
                    return True
            return False
        except Exception as e:
            logger.error(f"Error setting {category}.{key}: {e}")
            return False
    
    def reset_to_defaults(self, category: Optional[str] = None):
        """Reset settings to defaults"""
        try:
            if category:
                # Reset specific category
                if category == 'llm':
                    self.llm = LLMSettings()
                elif category == 'computer_vision':
                    self.computer_vision = ComputerVisionSettings()
                elif category == 'ui_automation':
                    self.ui_automation = UIAutomationSettings()
                elif category == 'learning':
                    self.learning = LearningSettings()
                elif category == 'memory':
                    self.memory = MemorySettings()
                elif category == 'security':
                    self.security = SecuritySettings()
                elif category == 'notifications':
                    self.notifications = NotificationSettings()
                elif category == 'voice':
                    self.voice = VoiceSettings()
                elif category == 'translation':
                    self.translation = TranslationSettings()
                elif category == 'content_generation':
                    self.content_generation = ContentGenerationSettings()
                elif category == 'task_scheduler':
                    self.task_scheduler = TaskSchedulerSettings()
                elif category == 'file_processor':
                    self.file_processor = FileProcessorSettings()
                elif category == 'system':
                    self.system = SystemSettings()
                
                logger.info(f"Reset {category} settings to defaults")
            else:
                # Reset all settings
                self.__init__(self.config_file)
                logger.info("All settings reset to defaults")
            
            self.save_config()
            
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
    
    def export_config(self, export_path: str) -> bool:
        """Export configuration to file"""
        try:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = self.to_dict()
            
            with open(export_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, import_path: str) -> bool:
        """Import configuration from file"""
        try:
            import_file = Path(import_path)
            
            if not import_file.exists():
                logger.error(f"Import file not found: {import_path}")
                return False
            
            with open(import_file, 'r') as f:
                config_data = json.load(f)
            
            # Remove metadata before importing
            if 'metadata' in config_data:
                del config_data['metadata']
            
            self._update_settings_from_dict(config_data)
            self.validate_settings()
            self.save_config()
            
            logger.info(f"Configuration imported from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        try:
            return {
                'config_file': str(self.config_path),
                'categories': {
                    'llm': f"Model: {self.llm.default_model}, Timeout: {self.llm.timeout}s",
                    'security': f"Encryption: {self.security.encryption_enabled}, Session: {self.security.session_timeout}s",
                    'voice': f"Enabled: {self.voice.enabled}, Language: {self.voice.language}",
                    'learning': f"Enabled: {self.learning.enabled}, Max interactions: {self.learning.max_interactions}",
                    'system': f"Debug: {self.system.debug_mode}, Log level: {self.system.log_level}"
                },
                'directories': {
                    'data': self.system.data_directory,
                    'temp': self.system.temp_directory,
                    'memory_db': self.memory.database_path
                }
            }
        except Exception as e:
            logger.error(f"Error getting config summary: {e}")
            return {'error': str(e)}
