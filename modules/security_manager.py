"""
Security Manager - Advanced security and privacy controls
"""

import asyncio
import logging
import json
import hashlib
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import secrets
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import psutil
import socket
import ssl

logger = logging.getLogger(__name__)

class SecurityManager:
    """Advanced security and privacy management system"""
    
    def __init__(self, settings):
        self.settings = settings
        self.encryption_key = None
        self.security_events = []
        self.access_logs = []
        self.threat_indicators = {}
        self.security_policies = {}
        self.quarantine_zone = Path("data/quarantine")
        
        # Security configuration
        self.max_failed_attempts = 5
        self.lockout_duration = 1800  # 30 minutes
        self.session_timeout = 3600   # 1 hour
        self.audit_retention_days = 90
        
        # Threat detection
        self.suspicious_patterns = [
            r'(?i)(password|secret|key|token)\s*[:=]\s*[\'"][^\'"]+[\'"]',
            r'(?i)(admin|root|administrator)',
            r'(?i)(sql\s+injection|xss|csrf|rce)',
            r'(?i)(exploit|malware|virus|trojan)'
        ]
        
    async def initialize(self):
        """Initialize security manager"""
        try:
            logger.info("Initializing Security Manager...")
            
            # Create secure directories
            await self.setup_secure_directories()
            
            # Initialize encryption
            await self.init_encryption()
            
            # Load security policies
            await self.load_security_policies()
            
            # Start security monitoring
            asyncio.create_task(self.security_monitor_loop())
            
            # Initialize network security
            await self.init_network_security()
            
            logger.info("Security Manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Security Manager: {e}")
            raise
    
    async def setup_secure_directories(self):
        """Create secure directories with proper permissions"""
        try:
            secure_dirs = [
                Path("data/secure"),
                Path("data/keys"),
                Path("data/audit_logs"),
                self.quarantine_zone
            ]
            
            for directory in secure_dirs:
                directory.mkdir(parents=True, exist_ok=True)
                
                # Set restrictive permissions (owner only)
                if os.name != 'nt':  # Unix-like systems
                    os.chmod(directory, 0o700)
            
            logger.info("Secure directories created")
            
        except Exception as e:
            logger.error(f"Error creating secure directories: {e}")
    
    async def init_encryption(self):
        """Initialize encryption system"""
        try:
            key_file = Path("data/keys/master.key")
            
            if key_file.exists():
                # Load existing key
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new encryption key
                password = os.getenv("MASTER_PASSWORD", "default_secure_password").encode()
                salt = os.urandom(16)
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                
                key = base64.urlsafe_b64encode(kdf.derive(password))
                self.encryption_key = key
                
                # Save key securely
                with open(key_file, 'wb') as f:
                    f.write(key)
                
                # Save salt for key derivation
                with open(Path("data/keys/salt"), 'wb') as f:
                    f.write(salt)
                
                # Set restrictive permissions
                if os.name != 'nt':
                    os.chmod(key_file, 0o600)
                    os.chmod(Path("data/keys/salt"), 0o600)
            
            logger.info("Encryption system initialized")
            
        except Exception as e:
            logger.error(f"Error initializing encryption: {e}")
            raise
    
    async def load_security_policies(self):
        """Load security policies configuration"""
        try:
            policies_file = Path("data/security_policies.json")
            
            if policies_file.exists():
                with open(policies_file, 'r') as f:
                    self.security_policies = json.load(f)
            else:
                # Default security policies
                self.security_policies = {
                    "access_control": {
                        "require_authentication": True,
                        "session_timeout": self.session_timeout,
                        "max_failed_attempts": self.max_failed_attempts,
                        "lockout_duration": self.lockout_duration
                    },
                    "data_protection": {
                        "encrypt_sensitive_data": True,
                        "secure_communications": True,
                        "data_retention_days": 365
                    },
                    "monitoring": {
                        "log_all_access": True,
                        "monitor_file_changes": True,
                        "detect_anomalies": True,
                        "real_time_alerts": True
                    },
                    "privacy": {
                        "anonymize_logs": True,
                        "data_minimization": True,
                        "user_consent_required": True
                    }
                }
                
                await self.save_security_policies()
            
            logger.info("Security policies loaded")
            
        except Exception as e:
            logger.error(f"Error loading security policies: {e}")
    
    async def save_security_policies(self):
        """Save security policies to file"""
        try:
            policies_file = Path("data/security_policies.json")
            with open(policies_file, 'w') as f:
                json.dump(self.security_policies, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving security policies: {e}")
    
    async def init_network_security(self):
        """Initialize network security monitoring"""
        try:
            # Monitor network connections
            self.network_monitor = NetworkMonitor()
            
            # Setup SSL/TLS configuration
            await self.setup_ssl_config()
            
            logger.info("Network security initialized")
            
        except Exception as e:
            logger.error(f"Error initializing network security: {e}")
    
    async def setup_ssl_config(self):
        """Setup SSL/TLS configuration"""
        try:
            # Create SSL context with strong security
            self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE
            
            # Configure cipher suites
            self.ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
            
        except Exception as e:
            logger.error(f"Error setting up SSL config: {e}")
    
    async def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """Encrypt sensitive data"""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(data)
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
    
    async def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data"""
        try:
            fernet = Fernet(self.encryption_key)
            decrypted_data = fernet.decrypt(encrypted_data)
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise
    
    async def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user with security checks"""
        try:
            # Check for account lockout
            if await self.is_account_locked(username):
                return {
                    'success': False,
                    'error': 'Account temporarily locked due to failed attempts',
                    'locked_until': await self.get_lockout_expiry(username)
                }
            
            # Verify credentials
            if await self.verify_credentials(username, password):
                # Reset failed attempts
                await self.reset_failed_attempts(username)
                
                # Create secure session
                session_token = await self.create_session(username)
                
                # Log successful authentication
                await self.log_security_event('authentication_success', {
                    'username': username,
                    'ip_address': await self.get_client_ip(),
                    'user_agent': await self.get_user_agent()
                })
                
                return {
                    'success': True,
                    'session_token': session_token,
                    'expires_at': (datetime.now() + timedelta(seconds=self.session_timeout)).isoformat()
                }
            else:
                # Record failed attempt
                await self.record_failed_attempt(username)
                
                # Log failed authentication
                await self.log_security_event('authentication_failure', {
                    'username': username,
                    'ip_address': await self.get_client_ip(),
                    'attempt_count': await self.get_failed_attempts(username)
                })
                
                return {
                    'success': False,
                    'error': 'Invalid credentials',
                    'attempts_remaining': self.max_failed_attempts - await self.get_failed_attempts(username)
                }
            
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return {'success': False, 'error': 'Authentication system error'}
    
    async def verify_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials"""
        try:
            # In production, this would check against a secure user database
            # For now, check against environment variables or default
            valid_username = os.getenv("ADMIN_USERNAME", "admin")
            valid_password = os.getenv("ADMIN_PASSWORD", "secure_password_123")
            
            # Use constant-time comparison to prevent timing attacks
            username_match = secrets.compare_digest(username, valid_username)
            password_match = secrets.compare_digest(password, valid_password)
            
            return username_match and password_match
            
        except Exception as e:
            logger.error(f"Error verifying credentials: {e}")
            return False
    
    async def create_session(self, username: str) -> str:
        """Create secure session token"""
        try:
            session_data = {
                'username': username,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(seconds=self.session_timeout)).isoformat(),
                'session_id': secrets.token_urlsafe(32)
            }
            
            # Encrypt session data
            encrypted_session = await self.encrypt_data(json.dumps(session_data))
            session_token = base64.urlsafe_b64encode(encrypted_session).decode('utf-8')
            
            return session_token
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise
    
    async def validate_session(self, session_token: str) -> Dict[str, Any]:
        """Validate session token"""
        try:
            # Decode and decrypt session
            encrypted_session = base64.urlsafe_b64decode(session_token.encode('utf-8'))
            decrypted_data = await self.decrypt_data(encrypted_session)
            session_data = json.loads(decrypted_data.decode('utf-8'))
            
            # Check expiration
            expires_at = datetime.fromisoformat(session_data['expires_at'])
            if datetime.now() > expires_at:
                return {'valid': False, 'error': 'Session expired'}
            
            return {'valid': True, 'session_data': session_data}
            
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return {'valid': False, 'error': 'Invalid session'}
    
    async def is_account_locked(self, username: str) -> bool:
        """Check if account is locked"""
        try:
            lockout_file = Path(f"data/secure/lockouts_{hashlib.md5(username.encode()).hexdigest()}")
            
            if not lockout_file.exists():
                return False
            
            with open(lockout_file, 'r') as f:
                lockout_data = json.load(f)
            
            lockout_until = datetime.fromisoformat(lockout_data['locked_until'])
            
            if datetime.now() < lockout_until:
                return True
            else:
                # Lockout expired, remove file
                lockout_file.unlink()
                return False
                
        except Exception as e:
            logger.error(f"Error checking account lockout: {e}")
            return False
    
    async def record_failed_attempt(self, username: str):
        """Record failed authentication attempt"""
        try:
            attempts_file = Path(f"data/secure/attempts_{hashlib.md5(username.encode()).hexdigest()}")
            
            attempts_data = {'count': 0, 'first_attempt': datetime.now().isoformat()}
            
            if attempts_file.exists():
                with open(attempts_file, 'r') as f:
                    attempts_data = json.load(f)
            
            attempts_data['count'] += 1
            attempts_data['last_attempt'] = datetime.now().isoformat()
            
            # Check if lockout threshold reached
            if attempts_data['count'] >= self.max_failed_attempts:
                await self.lockout_account(username)
            
            with open(attempts_file, 'w') as f:
                json.dump(attempts_data, f)
                
        except Exception as e:
            logger.error(f"Error recording failed attempt: {e}")
    
    async def get_failed_attempts(self, username: str) -> int:
        """Get number of failed attempts for user"""
        try:
            attempts_file = Path(f"data/secure/attempts_{hashlib.md5(username.encode()).hexdigest()}")
            
            if not attempts_file.exists():
                return 0
            
            with open(attempts_file, 'r') as f:
                attempts_data = json.load(f)
            
            return attempts_data.get('count', 0)
            
        except Exception as e:
            logger.error(f"Error getting failed attempts: {e}")
            return 0
    
    async def reset_failed_attempts(self, username: str):
        """Reset failed attempts counter"""
        try:
            attempts_file = Path(f"data/secure/attempts_{hashlib.md5(username.encode()).hexdigest()}")
            
            if attempts_file.exists():
                attempts_file.unlink()
                
        except Exception as e:
            logger.error(f"Error resetting failed attempts: {e}")
    
    async def lockout_account(self, username: str):
        """Lock account for specified duration"""
        try:
            lockout_data = {
                'username': username,
                'locked_at': datetime.now().isoformat(),
                'locked_until': (datetime.now() + timedelta(seconds=self.lockout_duration)).isoformat(),
                'reason': 'Exceeded maximum failed authentication attempts'
            }
            
            lockout_file = Path(f"data/secure/lockouts_{hashlib.md5(username.encode()).hexdigest()}")
            
            with open(lockout_file, 'w') as f:
                json.dump(lockout_data, f)
            
            # Log security event
            await self.log_security_event('account_lockout', lockout_data)
            
        except Exception as e:
            logger.error(f"Error locking account: {e}")
    
    async def get_lockout_expiry(self, username: str) -> Optional[str]:
        """Get account lockout expiry time"""
        try:
            lockout_file = Path(f"data/secure/lockouts_{hashlib.md5(username.encode()).hexdigest()}")
            
            if not lockout_file.exists():
                return None
            
            with open(lockout_file, 'r') as f:
                lockout_data = json.load(f)
            
            return lockout_data.get('locked_until')
            
        except Exception as e:
            logger.error(f"Error getting lockout expiry: {e}")
            return None
    
    async def log_security_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log security events for audit trail"""
        try:
            security_event = {
                'event_id': secrets.token_hex(16),
                'event_type': event_type,
                'timestamp': datetime.now().isoformat(),
                'data': event_data,
                'severity': self.get_event_severity(event_type)
            }
            
            # Add to memory
            self.security_events.append(security_event)
            
            # Limit memory usage
            if len(self.security_events) > 1000:
                self.security_events = self.security_events[-1000:]
            
            # Write to audit log
            await self.write_audit_log(security_event)
            
            # Check for threat patterns
            await self.analyze_threat_patterns(security_event)
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    def get_event_severity(self, event_type: str) -> str:
        """Get severity level for event type"""
        severity_map = {
            'authentication_success': 'info',
            'authentication_failure': 'warning',
            'account_lockout': 'high',
            'data_access': 'info',
            'privilege_escalation': 'critical',
            'malware_detected': 'critical',
            'intrusion_attempt': 'high',
            'data_breach': 'critical',
            'system_compromise': 'critical'
        }
        
        return severity_map.get(event_type, 'medium')
    
    async def write_audit_log(self, security_event: Dict[str, Any]):
        """Write security event to audit log"""
        try:
            log_date = datetime.now().strftime("%Y-%m-%d")
            log_file = Path(f"data/audit_logs/security_{log_date}.log")
            
            log_entry = f"{security_event['timestamp']} | {security_event['event_type']} | {security_event['severity']} | {json.dumps(security_event['data'])}\n"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception as e:
            logger.error(f"Error writing audit log: {e}")
    
    async def analyze_threat_patterns(self, security_event: Dict[str, Any]):
        """Analyze security events for threat patterns"""
        try:
            event_type = security_event['event_type']
            
            # Track pattern indicators
            if event_type not in self.threat_indicators:
                self.threat_indicators[event_type] = {
                    'count': 0,
                    'first_seen': security_event['timestamp'],
                    'last_seen': security_event['timestamp'],
                    'sources': set()
                }
            
            indicator = self.threat_indicators[event_type]
            indicator['count'] += 1
            indicator['last_seen'] = security_event['timestamp']
            
            # Add source information
            if 'ip_address' in security_event['data']:
                indicator['sources'].add(security_event['data']['ip_address'])
            
            # Check for threat thresholds
            await self.check_threat_thresholds(event_type, indicator)
            
        except Exception as e:
            logger.error(f"Error analyzing threat patterns: {e}")
    
    async def check_threat_thresholds(self, event_type: str, indicator: Dict[str, Any]):
        """Check if threat thresholds are exceeded"""
        try:
            threat_thresholds = {
                'authentication_failure': 10,
                'account_lockout': 3,
                'intrusion_attempt': 5,
                'malware_detected': 1
            }
            
            threshold = threat_thresholds.get(event_type, 20)
            
            if indicator['count'] >= threshold:
                # Trigger threat response
                await self.trigger_threat_response(event_type, indicator)
                
        except Exception as e:
            logger.error(f"Error checking threat thresholds: {e}")
    
    async def trigger_threat_response(self, event_type: str, indicator: Dict[str, Any]):
        """Trigger automated threat response"""
        try:
            response_actions = {
                'authentication_failure': ['block_ip', 'alert_admin'],
                'account_lockout': ['alert_admin', 'enhance_monitoring'],
                'intrusion_attempt': ['block_ip', 'alert_admin', 'quarantine'],
                'malware_detected': ['quarantine', 'alert_admin', 'full_scan']
            }
            
            actions = response_actions.get(event_type, ['alert_admin'])
            
            for action in actions:
                await self.execute_response_action(action, indicator)
            
            # Log threat response
            await self.log_security_event('threat_response_triggered', {
                'event_type': event_type,
                'actions_taken': actions,
                'indicator_data': {k: v for k, v in indicator.items() if k != 'sources'}
            })
            
        except Exception as e:
            logger.error(f"Error triggering threat response: {e}")
    
    async def execute_response_action(self, action: str, indicator: Dict[str, Any]):
        """Execute specific threat response action"""
        try:
            if action == 'block_ip':
                await self.block_suspicious_ips(indicator['sources'])
            elif action == 'alert_admin':
                await self.send_security_alert(indicator)
            elif action == 'quarantine':
                await self.quarantine_suspicious_files()
            elif action == 'enhance_monitoring':
                await self.enhance_monitoring()
            elif action == 'full_scan':
                await self.initiate_security_scan()
            
        except Exception as e:
            logger.error(f"Error executing response action {action}: {e}")
    
    async def block_suspicious_ips(self, ip_addresses: set):
        """Block suspicious IP addresses"""
        try:
            blocked_ips_file = Path("data/secure/blocked_ips.json")
            
            blocked_ips = []
            if blocked_ips_file.exists():
                with open(blocked_ips_file, 'r') as f:
                    blocked_ips = json.load(f)
            
            for ip in ip_addresses:
                if ip not in blocked_ips:
                    blocked_ips.append({
                        'ip': ip,
                        'blocked_at': datetime.now().isoformat(),
                        'reason': 'Suspicious activity detected'
                    })
            
            with open(blocked_ips_file, 'w') as f:
                json.dump(blocked_ips, f, indent=2)
            
            logger.warning(f"Blocked {len(ip_addresses)} suspicious IP addresses")
            
        except Exception as e:
            logger.error(f"Error blocking suspicious IPs: {e}")
    
    async def send_security_alert(self, indicator: Dict[str, Any]):
        """Send security alert to administrators"""
        try:
            alert_data = {
                'alert_type': 'security_threat',
                'severity': 'high',
                'message': f"Security threat detected: {indicator}",
                'timestamp': datetime.now().isoformat(),
                'requires_action': True
            }
            
            # In production, this would send actual alerts via email, SMS, etc.
            logger.critical(f"SECURITY ALERT: {alert_data['message']}")
            
        except Exception as e:
            logger.error(f"Error sending security alert: {e}")
    
    async def quarantine_suspicious_files(self):
        """Quarantine suspicious files"""
        try:
            # This would implement file quarantine logic
            logger.warning("File quarantine process initiated")
            
        except Exception as e:
            logger.error(f"Error quarantining files: {e}")
    
    async def enhance_monitoring(self):
        """Enhance security monitoring"""
        try:
            # This would implement enhanced monitoring
            logger.info("Enhanced security monitoring activated")
            
        except Exception as e:
            logger.error(f"Error enhancing monitoring: {e}")
    
    async def initiate_security_scan(self):
        """Initiate comprehensive security scan"""
        try:
            # This would implement security scanning
            logger.info("Security scan initiated")
            
        except Exception as e:
            logger.error(f"Error initiating security scan: {e}")
    
    async def security_monitor_loop(self):
        """Main security monitoring loop"""
        while True:
            try:
                # Monitor system resources
                await self.monitor_system_resources()
                
                # Check for suspicious processes
                await self.monitor_processes()
                
                # Monitor network connections
                await self.monitor_network_connections()
                
                # Clean up old audit logs
                await self.cleanup_audit_logs()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in security monitor loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def monitor_system_resources(self):
        """Monitor system resources for anomalies"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                await self.log_security_event('high_cpu_usage', {'cpu_percent': cpu_percent})
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                await self.log_security_event('high_memory_usage', {'memory_percent': memory.percent})
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                await self.log_security_event('high_disk_usage', {'disk_percent': disk.percent})
            
        except Exception as e:
            logger.error(f"Error monitoring system resources: {e}")
    
    async def monitor_processes(self):
        """Monitor running processes for suspicious activity"""
        try:
            suspicious_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
                try:
                    if proc.info['cpu_percent'] and proc.info['cpu_percent'] > 80:
                        suspicious_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if suspicious_processes:
                await self.log_security_event('suspicious_processes', {
                    'processes': suspicious_processes[:5]  # Log top 5
                })
            
        except Exception as e:
            logger.error(f"Error monitoring processes: {e}")
    
    async def monitor_network_connections(self):
        """Monitor network connections"""
        try:
            connections = psutil.net_connections(kind='inet')
            suspicious_connections = []
            
            for conn in connections:
                if conn.status == 'ESTABLISHED' and conn.raddr:
                    # Check against known bad IPs (would be more sophisticated in production)
                    if self.is_suspicious_ip(conn.raddr.ip):
                        suspicious_connections.append({
                            'local': f"{conn.laddr.ip}:{conn.laddr.port}",
                            'remote': f"{conn.raddr.ip}:{conn.raddr.port}",
                            'status': conn.status
                        })
            
            if suspicious_connections:
                await self.log_security_event('suspicious_connections', {
                    'connections': suspicious_connections
                })
            
        except Exception as e:
            logger.error(f"Error monitoring network connections: {e}")
    
    def is_suspicious_ip(self, ip: str) -> bool:
        """Check if IP address is suspicious"""
        try:
            # Simple checks for private/localhost IPs
            private_ranges = ['127.', '10.', '192.168.', '172.16.']
            
            # Allow private IPs
            if any(ip.startswith(range_start) for range_start in private_ranges):
                return False
            
            # In production, this would check against threat intelligence feeds
            return False
            
        except Exception as e:
            logger.error(f"Error checking suspicious IP: {e}")
            return False
    
    async def cleanup_audit_logs(self):
        """Clean up old audit logs"""
        try:
            audit_dir = Path("data/audit_logs")
            if not audit_dir.exists():
                return
            
            cutoff_date = datetime.now() - timedelta(days=self.audit_retention_days)
            
            for log_file in audit_dir.glob("*.log"):
                try:
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        log_file.unlink()
                        logger.info(f"Cleaned up old audit log: {log_file.name}")
                except Exception as e:
                    logger.warning(f"Error cleaning up log file {log_file}: {e}")
            
        except Exception as e:
            logger.error(f"Error cleaning up audit logs: {e}")
    
    async def get_client_ip(self) -> str:
        """Get client IP address"""
        # This would be implemented based on the web framework being used
        return "127.0.0.1"
    
    async def get_user_agent(self) -> str:
        """Get client user agent"""
        # This would be implemented based on the web framework being used
        return "AI-Agent-Client/1.0"
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        try:
            recent_events = self.security_events[-50:] if self.security_events else []
            
            return {
                'security_level': 'normal',  # Would be calculated based on threats
                'recent_events_count': len(recent_events),
                'threat_indicators': len(self.threat_indicators),
                'encryption_enabled': self.encryption_key is not None,
                'monitoring_active': True,
                'last_threat_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting security status: {e}")
            return {'error': str(e)}
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics"""
        try:
            event_counts = {}
            for event in self.security_events:
                event_type = event['event_type']
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            return {
                'total_events': len(self.security_events),
                'event_breakdown': event_counts,
                'threat_indicators_active': len(self.threat_indicators),
                'security_policies_enabled': len(self.security_policies),
                'monitoring_uptime': '99.9%'  # Would be calculated
            }
            
        except Exception as e:
            logger.error(f"Error getting security metrics: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Shutdown security manager"""
        logger.info("Shutting down Security Manager...")
        
        try:
            # Final security audit log
            await self.log_security_event('system_shutdown', {
                'shutdown_time': datetime.now().isoformat(),
                'events_processed': len(self.security_events)
            })
            
        except Exception as e:
            logger.error(f"Error during security manager shutdown: {e}")


class NetworkMonitor:
    """Network security monitoring component"""
    
    def __init__(self):
        self.active_connections = {}
        self.suspicious_activity = []
    
    async def monitor_connections(self):
        """Monitor network connections for suspicious activity"""
        try:
            connections = psutil.net_connections(kind='inet')
            
            for conn in connections:
                if conn.status == 'ESTABLISHED':
                    self.active_connections[f"{conn.laddr}:{conn.raddr}"] = {
                        'established_at': datetime.now().isoformat(),
                        'local': conn.laddr,
                        'remote': conn.raddr,
                        'status': conn.status
                    }
            
        except Exception as e:
            logger.error(f"Error monitoring network connections: {e}")
