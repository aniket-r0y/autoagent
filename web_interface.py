"""
Web Interface for AI Agent Control and Monitoring
Provides a comprehensive dashboard for interacting with the AI agent
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, websocket
from flask_socketio import SocketIO, emit
import threading

logger = logging.getLogger(__name__)

class WebInterface:
    """Web interface for AI Agent control and monitoring"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'ai-agent-secret-key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.setup_routes()
        self.setup_websocket_events()
        
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def get_status():
            """Get current agent status"""
            try:
                status = {
                    'active': self.orchestrator.is_running,
                    'current_model': self.orchestrator.llm_manager.current_model,
                    'tasks_running': len(self.orchestrator.active_tasks),
                    'memory_usage': self.orchestrator.get_memory_usage(),
                    'uptime': self.orchestrator.get_uptime(),
                    'last_activity': self.orchestrator.get_last_activity()
                }
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/command', methods=['POST'])
        def process_command():
            """Process user commands"""
            try:
                data = request.get_json()
                command = data.get('command', '')
                
                if not command:
                    return jsonify({'error': 'No command provided'}), 400
                
                # Queue command for processing
                result = self.orchestrator.queue_command(command)
                return jsonify({'result': result, 'status': 'queued'})
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/models')
        def get_models():
            """Get available AI models"""
            try:
                models = self.orchestrator.llm_manager.get_available_models()
                return jsonify(models)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/models/<model_name>', methods=['POST'])
        def switch_model(model_name):
            """Switch AI model"""
            try:
                success = self.orchestrator.llm_manager.switch_model(model_name)
                return jsonify({'success': success})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/tasks')
        def get_tasks():
            """Get current tasks"""
            try:
                tasks = self.orchestrator.get_active_tasks()
                return jsonify(tasks)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/learning/stats')
        def get_learning_stats():
            """Get learning system statistics"""
            try:
                stats = self.orchestrator.learning_system.get_stats()
                return jsonify(stats)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/memory/summary')
        def get_memory_summary():
            """Get memory system summary"""
            try:
                summary = self.orchestrator.memory_manager.get_summary()
                return jsonify(summary)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/settings', methods=['GET', 'POST'])
        def handle_settings():
            """Get or update settings"""
            try:
                if request.method == 'GET':
                    settings = self.orchestrator.get_settings()
                    return jsonify(settings)
                else:
                    data = request.get_json()
                    success = self.orchestrator.update_settings(data)
                    return jsonify({'success': success})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def setup_websocket_events(self):
        """Setup WebSocket events for real-time communication"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info('Client connected to WebSocket')
            emit('status', {'message': 'Connected to AI Agent'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info('Client disconnected from WebSocket')
        
        @self.socketio.on('command')
        def handle_command(data):
            """Handle real-time commands"""
            try:
                command = data.get('command', '')
                result = self.orchestrator.queue_command(command)
                emit('command_result', {'result': result})
            except Exception as e:
                emit('error', {'message': str(e)})
        
        @self.socketio.on('get_logs')
        def handle_get_logs():
            """Send recent logs to client"""
            try:
                logs = self.get_recent_logs()
                emit('logs', {'logs': logs})
            except Exception as e:
                emit('error', {'message': str(e)})
    
    def get_recent_logs(self, lines=100):
        """Get recent log entries"""
        try:
            log_file = Path('ai_agent.log')
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = f.readlines()
                    return logs[-lines:] if len(logs) > lines else logs
            return []
        except Exception as e:
            logger.error(f"Error reading logs: {e}")
            return []
    
    def broadcast_status_update(self, status):
        """Broadcast status updates to all connected clients"""
        self.socketio.emit('status_update', status)
    
    def broadcast_log(self, log_entry):
        """Broadcast log entries to all connected clients"""
        self.socketio.emit('log', {'entry': log_entry, 'timestamp': datetime.now().isoformat()})
    
    async def run(self):
        """Run the web interface"""
        try:
            logger.info("Starting web interface...")
            
            # Run Flask-SocketIO app in a separate thread
            def run_app():
                self.socketio.run(self.app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
            
            thread = threading.Thread(target=run_app)
            thread.daemon = True
            thread.start()
            
            # Keep the async context alive
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error running web interface: {e}")
