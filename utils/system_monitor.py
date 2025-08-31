"""
System Monitor - Advanced system monitoring and performance tracking
"""

import asyncio
import logging
import psutil
import platform
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import threading
import queue

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Advanced system monitoring and performance tracking"""
    
    def __init__(self):
        self.monitoring_active = False
        self.metrics_history = []
        self.alerts = []
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 90.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'temperature_warning': 70.0,
            'temperature_critical': 80.0
        }
        
        # Performance tracking
        self.performance_metrics = {
            'system_startup_time': None,
            'average_cpu_usage': 0.0,
            'peak_memory_usage': 0.0,
            'disk_io_stats': {},
            'network_stats': {},
            'process_count': 0
        }
        
        # System information
        self.system_info = None
        self.last_update = None
        self.update_interval = 5  # seconds
        
    async def initialize(self):
        """Initialize system monitor"""
        try:
            logger.info("Initializing System Monitor...")
            
            # Gather initial system information
            await self.gather_system_info()
            
            # Record startup time
            self.performance_metrics['system_startup_time'] = datetime.now().isoformat()
            
            # Load historical data
            await self.load_historical_data()
            
            logger.info("System Monitor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize System Monitor: {e}")
            raise
    
    async def gather_system_info(self):
        """Gather comprehensive system information"""
        try:
            self.system_info = {
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                    'architecture': platform.architecture(),
                    'hostname': platform.node()
                },
                'cpu': {
                    'physical_cores': psutil.cpu_count(logical=False),
                    'logical_cores': psutil.cpu_count(logical=True),
                    'max_frequency': psutil.cpu_freq().max if psutil.cpu_freq() else None,
                    'min_frequency': psutil.cpu_freq().min if psutil.cpu_freq() else None,
                    'current_frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
                },
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'total_swap': psutil.swap_memory().total,
                    'available_swap': psutil.swap_memory().free
                },
                'disk': await self.get_disk_info(),
                'network': await self.get_network_info(),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error gathering system info: {e}")
            self.system_info = {'error': str(e)}
    
    async def get_disk_info(self) -> Dict[str, Any]:
        """Get disk information for all mounted drives"""
        try:
            disk_info = {}
            
            # Get disk partitions
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                try:
                    partition_usage = psutil.disk_usage(partition.mountpoint)
                    disk_info[partition.device] = {
                        'mountpoint': partition.mountpoint,
                        'filesystem': partition.fstype,
                        'total': partition_usage.total,
                        'used': partition_usage.used,
                        'free': partition_usage.free,
                        'percentage': (partition_usage.used / partition_usage.total) * 100
                    }
                except PermissionError:
                    # This can happen on Windows
                    continue
            
            return disk_info
            
        except Exception as e:
            logger.error(f"Error getting disk info: {e}")
            return {}
    
    async def get_network_info(self) -> Dict[str, Any]:
        """Get network interface information"""
        try:
            network_info = {}
            
            # Get network interfaces
            interfaces = psutil.net_if_addrs()
            
            for interface_name, addresses in interfaces.items():
                interface_info = {
                    'addresses': [],
                    'is_up': interface_name in psutil.net_if_stats() and psutil.net_if_stats()[interface_name].isup
                }
                
                for addr in addresses:
                    interface_info['addresses'].append({
                        'family': str(addr.family),
                        'address': addr.address,
                        'netmask': addr.netmask,
                        'broadcast': addr.broadcast
                    })
                
                network_info[interface_name] = interface_info
            
            return network_info
            
        except Exception as e:
            logger.error(f"Error getting network info: {e}")
            return {}
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        self.monitoring_active = True
        logger.info("System monitoring started")
        
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = await self.collect_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Limit history size
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Check thresholds and generate alerts
                await self.check_thresholds(metrics)
                
                # Update performance metrics
                await self.update_performance_metrics(metrics)
                
                self.last_update = datetime.now()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'usage_percent': psutil.cpu_percent(interval=1),
                    'per_cpu': psutil.cpu_percent(interval=1, percpu=True),
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                    'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
                },
                'memory': {
                    'virtual': {
                        'total': psutil.virtual_memory().total,
                        'available': psutil.virtual_memory().available,
                        'used': psutil.virtual_memory().used,
                        'percentage': psutil.virtual_memory().percent
                    },
                    'swap': {
                        'total': psutil.swap_memory().total,
                        'used': psutil.swap_memory().used,
                        'free': psutil.swap_memory().free,
                        'percentage': psutil.swap_memory().percent
                    }
                },
                'disk': await self.get_disk_usage(),
                'network': await self.get_network_stats(),
                'processes': await self.get_process_stats(),
                'system': {
                    'uptime': time.time() - psutil.boot_time(),
                    'users': len(psutil.users()),
                    'temperature': await self.get_temperature_info()
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def get_disk_usage(self) -> Dict[str, Any]:
        """Get current disk usage"""
        try:
            disk_usage = {}
            
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.device] = {
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percentage': (usage.used / usage.total) * 100
                    }
                except PermissionError:
                    continue
            
            # Get disk I/O statistics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_usage['io_stats'] = {
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count,
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_time': disk_io.read_time,
                    'write_time': disk_io.write_time
                }
            
            return disk_usage
            
        except Exception as e:
            logger.error(f"Error getting disk usage: {e}")
            return {}
    
    async def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        try:
            network_stats = {}
            
            # Get overall network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                network_stats['total'] = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_received': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_received': net_io.packets_recv,
                    'errors_in': net_io.errin,
                    'errors_out': net_io.errout,
                    'dropped_in': net_io.dropin,
                    'dropped_out': net_io.dropout
                }
            
            # Get per-interface statistics
            net_io_per_interface = psutil.net_io_counters(pernic=True)
            network_stats['interfaces'] = {}
            
            for interface, stats in net_io_per_interface.items():
                network_stats['interfaces'][interface] = {
                    'bytes_sent': stats.bytes_sent,
                    'bytes_received': stats.bytes_recv,
                    'packets_sent': stats.packets_sent,
                    'packets_received': stats.packets_recv
                }
            
            return network_stats
            
        except Exception as e:
            logger.error(f"Error getting network stats: {e}")
            return {}
    
    async def get_process_stats(self) -> Dict[str, Any]:
        """Get process statistics"""
        try:
            processes = list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']))
            
            # Sort by CPU usage
            processes.sort(key=lambda p: p.info['cpu_percent'] or 0, reverse=True)
            
            process_stats = {
                'total_count': len(processes),
                'top_cpu': [],
                'top_memory': []
            }
            
            # Top 5 CPU consuming processes
            for proc in processes[:5]:
                if proc.info['cpu_percent'] and proc.info['cpu_percent'] > 0:
                    process_stats['top_cpu'].append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent']
                    })
            
            # Sort by memory usage and get top 5
            processes.sort(key=lambda p: p.info['memory_percent'] or 0, reverse=True)
            for proc in processes[:5]:
                if proc.info['memory_percent'] and proc.info['memory_percent'] > 0:
                    process_stats['top_memory'].append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'memory_percent': proc.info['memory_percent']
                    })
            
            return process_stats
            
        except Exception as e:
            logger.error(f"Error getting process stats: {e}")
            return {'total_count': 0, 'top_cpu': [], 'top_memory': []}
    
    async def get_temperature_info(self) -> Dict[str, Any]:
        """Get system temperature information"""
        try:
            temperatures = {}
            
            if hasattr(psutil, 'sensors_temperatures'):
                temp_sensors = psutil.sensors_temperatures()
                
                for sensor_name, sensor_list in temp_sensors.items():
                    temperatures[sensor_name] = []
                    for sensor in sensor_list:
                        temp_info = {
                            'label': sensor.label or 'Unknown',
                            'current': sensor.current,
                            'high': sensor.high,
                            'critical': sensor.critical
                        }
                        temperatures[sensor_name].append(temp_info)
            
            return temperatures
            
        except Exception as e:
            logger.debug(f"Temperature sensors not available: {e}")
            return {}
    
    async def check_thresholds(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts"""
        try:
            alerts_generated = []
            
            # Check CPU usage
            cpu_usage = metrics.get('cpu', {}).get('usage_percent', 0)
            if cpu_usage > self.thresholds['cpu_critical']:
                alert = await self.create_alert('critical', 'cpu', f'CPU usage critical: {cpu_usage:.1f}%')
                alerts_generated.append(alert)
            elif cpu_usage > self.thresholds['cpu_warning']:
                alert = await self.create_alert('warning', 'cpu', f'CPU usage high: {cpu_usage:.1f}%')
                alerts_generated.append(alert)
            
            # Check memory usage
            memory_usage = metrics.get('memory', {}).get('virtual', {}).get('percentage', 0)
            if memory_usage > self.thresholds['memory_critical']:
                alert = await self.create_alert('critical', 'memory', f'Memory usage critical: {memory_usage:.1f}%')
                alerts_generated.append(alert)
            elif memory_usage > self.thresholds['memory_warning']:
                alert = await self.create_alert('warning', 'memory', f'Memory usage high: {memory_usage:.1f}%')
                alerts_generated.append(alert)
            
            # Check disk usage
            disk_info = metrics.get('disk', {})
            for device, usage in disk_info.items():
                if isinstance(usage, dict) and 'percentage' in usage:
                    disk_usage = usage['percentage']
                    if disk_usage > self.thresholds['disk_critical']:
                        alert = await self.create_alert('critical', 'disk', f'Disk {device} usage critical: {disk_usage:.1f}%')
                        alerts_generated.append(alert)
                    elif disk_usage > self.thresholds['disk_warning']:
                        alert = await self.create_alert('warning', 'disk', f'Disk {device} usage high: {disk_usage:.1f}%')
                        alerts_generated.append(alert)
            
            # Store alerts
            self.alerts.extend(alerts_generated)
            
            # Limit alerts history
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
            
        except Exception as e:
            logger.error(f"Error checking thresholds: {e}")
    
    async def create_alert(self, severity: str, category: str, message: str) -> Dict[str, Any]:
        """Create a system alert"""
        alert = {
            'id': f"alert_{int(time.time() * 1000)}",
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'category': category,
            'message': message,
            'acknowledged': False
        }
        
        logger.warning(f"System Alert [{severity.upper()}]: {message}")
        return alert
    
    async def update_performance_metrics(self, current_metrics: Dict[str, Any]):
        """Update long-term performance metrics"""
        try:
            # Update average CPU usage
            cpu_usage = current_metrics.get('cpu', {}).get('usage_percent', 0)
            if hasattr(self, '_cpu_readings'):
                self._cpu_readings.append(cpu_usage)
                if len(self._cpu_readings) > 100:
                    self._cpu_readings = self._cpu_readings[-100:]
            else:
                self._cpu_readings = [cpu_usage]
            
            self.performance_metrics['average_cpu_usage'] = sum(self._cpu_readings) / len(self._cpu_readings)
            
            # Update peak memory usage
            memory_usage = current_metrics.get('memory', {}).get('virtual', {}).get('used', 0)
            self.performance_metrics['peak_memory_usage'] = max(
                self.performance_metrics['peak_memory_usage'],
                memory_usage
            )
            
            # Update process count
            self.performance_metrics['process_count'] = current_metrics.get('processes', {}).get('total_count', 0)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status summary"""
        try:
            if not self.metrics_history:
                return {'status': 'no_data', 'message': 'No metrics collected yet'}
            
            latest_metrics = self.metrics_history[-1]
            
            # Determine overall system health
            cpu_usage = latest_metrics.get('cpu', {}).get('usage_percent', 0)
            memory_usage = latest_metrics.get('memory', {}).get('virtual', {}).get('percentage', 0)
            
            if cpu_usage > 90 or memory_usage > 95:
                status = 'critical'
            elif cpu_usage > 80 or memory_usage > 85:
                status = 'warning'
            else:
                status = 'healthy'
            
            return {
                'status': status,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'uptime': latest_metrics.get('system', {}).get('uptime', 0),
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'active_alerts': len([a for a in self.alerts if not a['acknowledged']])
            }
            
        except Exception as e:
            logger.error(f"Error getting current status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_metrics = [
                m for m in self.metrics_history
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
            if not recent_metrics:
                return {'message': 'No metrics available for specified period'}
            
            # Calculate averages
            cpu_values = [m.get('cpu', {}).get('usage_percent', 0) for m in recent_metrics]
            memory_values = [m.get('memory', {}).get('virtual', {}).get('percentage', 0) for m in recent_metrics]
            
            summary = {
                'period_hours': hours,
                'data_points': len(recent_metrics),
                'cpu': {
                    'average': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    'min': min(cpu_values) if cpu_values else 0,
                    'max': max(cpu_values) if cpu_values else 0
                },
                'memory': {
                    'average': sum(memory_values) / len(memory_values) if memory_values else 0,
                    'min': min(memory_values) if memory_values else 0,
                    'max': max(memory_values) if memory_values else 0
                },
                'alerts_in_period': len([
                    a for a in self.alerts
                    if datetime.fromisoformat(a['timestamp']) > cutoff_time
                ])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {'error': str(e)}
    
    def get_system_information(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'system_info': self.system_info,
            'performance_metrics': self.performance_metrics,
            'monitoring_status': {
                'active': self.monitoring_active,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'update_interval': self.update_interval,
                'metrics_count': len(self.metrics_history)
            },
            'thresholds': self.thresholds
        }
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a system alert"""
        try:
            for alert in self.alerts:
                if alert['id'] == alert_id:
                    alert['acknowledged'] = True
                    alert['acknowledged_at'] = datetime.now().isoformat()
                    return True
            return False
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def set_thresholds(self, new_thresholds: Dict[str, float]) -> bool:
        """Update monitoring thresholds"""
        try:
            for key, value in new_thresholds.items():
                if key in self.thresholds and isinstance(value, (int, float)):
                    self.thresholds[key] = float(value)
            
            logger.info("Monitoring thresholds updated")
            return True
            
        except Exception as e:
            logger.error(f"Error setting thresholds: {e}")
            return False
    
    async def load_historical_data(self):
        """Load historical monitoring data"""
        try:
            data_file = Path("data/system_metrics.json")
            
            if data_file.exists():
                with open(data_file, 'r') as f:
                    historical_data = json.load(f)
                    self.metrics_history = historical_data.get('metrics', [])[-100:]  # Keep last 100
                    self.alerts = historical_data.get('alerts', [])[-50:]  # Keep last 50
                    
                    if 'thresholds' in historical_data:
                        self.thresholds.update(historical_data['thresholds'])
                
                logger.info("Loaded historical monitoring data")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    async def save_data(self):
        """Save monitoring data to disk"""
        try:
            data_file = Path("data/system_metrics.json")
            data_file.parent.mkdir(exist_ok=True)
            
            data = {
                'metrics': self.metrics_history[-100:],  # Save last 100 metrics
                'alerts': self.alerts[-50:],  # Save last 50 alerts
                'thresholds': self.thresholds,
                'performance_metrics': self.performance_metrics,
                'last_saved': datetime.now().isoformat()
            }
            
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
    
    async def shutdown(self):
        """Shutdown system monitor"""
        logger.info("Shutting down System Monitor...")
        self.monitoring_active = False
        
        try:
            # Save data before shutdown
            await self.save_data()
        except Exception as e:
            logger.error(f"Error saving data during shutdown: {e}")
