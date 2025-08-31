#!/usr/bin/env python3
"""
Next-Generation AI Agent - Main Entry Point
A comprehensive AI agent with advanced learning, automation, and creative capabilities
"""

import asyncio
import logging
import signal
import sys
import threading
import time
from pathlib import Path

from core.ai_orchestrator import AIOrchestrator
from web_interface import WebInterface
from config.settings import Settings
from utils.system_monitor import SystemMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class NextGenAIAgent:
    """Main AI Agent class that orchestrates all components"""
    
    def __init__(self):
        self.settings = Settings()
        self.orchestrator = None
        self.web_interface = None
        self.system_monitor = None
        self.running = False
        
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("Initializing Next-Generation AI Agent...")
            
            # Initialize core orchestrator
            self.orchestrator = AIOrchestrator(self.settings)
            await self.orchestrator.initialize()
            
            # Initialize web interface
            self.web_interface = WebInterface(self.orchestrator)
            
            # Initialize system monitor
            self.system_monitor = SystemMonitor()
            
            logger.info("AI Agent initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Agent: {e}")
            return False
    
    async def start(self):
        """Start the AI Agent"""
        if not await self.initialize():
            return False
            
        self.running = True
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self.orchestrator.run()),
            asyncio.create_task(self.system_monitor.monitor_loop()),
            asyncio.create_task(self.web_interface.run()),
        ]
        
        logger.info("AI Agent started successfully!")
        logger.info("Web interface available at http://0.0.0.0:5000")
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down AI Agent...")
        self.running = False
        
        if self.orchestrator:
            await self.orchestrator.shutdown()
        
        if self.system_monitor:
            await self.system_monitor.shutdown()
        
        logger.info("AI Agent shutdown complete")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.running = False

async def main():
    """Main entry point"""
    agent = NextGenAIAgent()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, agent.signal_handler)
    signal.signal(signal.SIGTERM, agent.signal_handler)
    
    try:
        await agent.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("temp").mkdir(exist_ok=True)
    
    # Run the agent
    asyncio.run(main())
