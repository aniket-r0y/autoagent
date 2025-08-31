"""
LLM Manager - Handles multiple AI model backends with intelligent switching
"""

import asyncio
import logging
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
import requests

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages multiple LLM backends with intelligent model switching"""
    
    def __init__(self, settings):
        self.settings = settings
        self.current_model = "gemini"  # Default
        self.models = {}
        self.model_stats = {}
        self.conversation_history = []
        
    async def initialize(self):
        """Initialize all available LLM backends"""
        try:
            logger.info("Initializing LLM Manager...")
            
            # Initialize Gemini (Online)
            await self.initialize_gemini()
            
            # Initialize Ollama (Local)
            await self.initialize_ollama()
            
            # Initialize LM Studio (Local)
            await self.initialize_lm_studio()
            
            # Set best available model
            await self.select_best_model()
            
            logger.info(f"LLM Manager initialized with model: {self.current_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM Manager: {e}")
            raise
    
    async def initialize_gemini(self):
        """Initialize Gemini API"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                self.models["gemini"] = {
                    "type": "online",
                    "api_key": api_key,
                    "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
                    "available": True,
                    "capabilities": ["text", "vision", "reasoning"],
                    "cost_per_token": 0.001,
                    "speed_rating": 8
                }
                
                # Test connection
                await self.test_gemini_connection()
                logger.info("Gemini API initialized successfully")
            else:
                logger.warning("Gemini API key not found")
                
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.models["gemini"] = {"available": False, "error": str(e)}
    
    async def initialize_ollama(self):
        """Initialize Ollama local instance"""
        try:
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            
            # Test connection
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{ollama_url}/api/version") as response:
                    if response.status == 200:
                        version_info = await response.json()
                        
                        # Get available models
                        models_response = await session.get(f"{ollama_url}/api/tags")
                        if models_response.status == 200:
                            models_data = await models_response.json()
                            available_models = [model["name"] for model in models_data.get("models", [])]
                            
                            self.models["ollama"] = {
                                "type": "local",
                                "endpoint": ollama_url,
                                "available": True,
                                "version": version_info,
                                "models": available_models,
                                "capabilities": ["text", "reasoning", "code"],
                                "cost_per_token": 0.0,
                                "speed_rating": 6
                            }
                            logger.info(f"Ollama initialized with models: {available_models}")
                    else:
                        raise Exception(f"Ollama not responding: {response.status}")
                        
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama: {e}")
            self.models["ollama"] = {"available": False, "error": str(e)}
    
    async def initialize_lm_studio(self):
        """Initialize LM Studio local instance"""
        try:
            lm_studio_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234")
            
            # Test connection
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{lm_studio_url}/v1/models") as response:
                    if response.status == 200:
                        models_data = await response.json()
                        available_models = [model["id"] for model in models_data.get("data", [])]
                        
                        self.models["lm_studio"] = {
                            "type": "local",
                            "endpoint": lm_studio_url,
                            "available": True,
                            "models": available_models,
                            "capabilities": ["text", "reasoning", "code"],
                            "cost_per_token": 0.0,
                            "speed_rating": 7
                        }
                        logger.info(f"LM Studio initialized with models: {available_models}")
                    else:
                        raise Exception(f"LM Studio not responding: {response.status}")
                        
        except Exception as e:
            logger.warning(f"Failed to initialize LM Studio: {e}")
            self.models["lm_studio"] = {"available": False, "error": str(e)}
    
    async def test_gemini_connection(self):
        """Test Gemini API connection"""
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{
                    "parts": [{"text": "Hello, test connection"}]
                }]
            }
            
            url = f"{self.models['gemini']['endpoint']}?key={self.models['gemini']['api_key']}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status != 200:
                        raise Exception(f"Gemini API test failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Gemini connection test failed: {e}")
            raise
    
    async def select_best_model(self):
        """Intelligently select the best available model"""
        available_models = {name: model for name, model in self.models.items() 
                          if model.get("available", False)}
        
        if not available_models:
            raise Exception("No AI models available")
        
        # Prioritize by availability and capabilities
        if "gemini" in available_models:
            self.current_model = "gemini"
        elif "lm_studio" in available_models:
            self.current_model = "lm_studio"
        elif "ollama" in available_models:
            self.current_model = "ollama"
        else:
            self.current_model = list(available_models.keys())[0]
    
    async def process_command(self, command: str, analysis: Dict, context: Dict) -> Dict[str, Any]:
        """Process a command using the current AI model"""
        try:
            # Build comprehensive prompt
            prompt = self.build_command_prompt(command, analysis, context)
            
            # Generate response
            response = await self.generate_response(prompt)
            
            # Parse response into structured format
            structured_response = await self.parse_ai_response(response)
            
            # Update conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "command": command,
                "response": structured_response,
                "model_used": self.current_model
            })
            
            return structured_response
            
        except Exception as e:
            logger.error(f"Error processing command with AI: {e}")
            return {"error": str(e), "actions": []}
    
    def build_command_prompt(self, command: str, analysis: Dict, context: Dict) -> str:
        """Build a comprehensive prompt for the AI model"""
        prompt = f"""
You are an advanced AI agent with comprehensive computer control capabilities. 

USER COMMAND: {command}

COMMAND ANALYSIS: {json.dumps(analysis, indent=2)}

RELEVANT CONTEXT: {json.dumps(context, indent=2)}

SYSTEM CAPABILITIES:
- UI automation and computer vision
- Web browser control and automation  
- Social media management
- File and document processing
- Content generation (text, images, videos)
- Task scheduling and automation
- System monitoring and optimization
- Natural language processing
- Voice interaction
- Translation services

INSTRUCTIONS:
1. Analyze the user's command thoroughly
2. Plan the necessary actions to fulfill the request
3. Consider security, efficiency, and user preferences
4. Generate a structured response with actionable steps

RESPONSE FORMAT (JSON):
{{
    "understanding": "Brief explanation of what you understood",
    "confidence": 0.95,
    "actions": [
        {{
            "type": "action_type",
            "description": "What this action does",
            "parameters": {{"param1": "value1"}},
            "priority": 1
        }}
    ],
    "explanation": "Detailed explanation of the approach",
    "estimated_time": "5 minutes",
    "requires_confirmation": false
}}

Respond only with valid JSON.
"""
        return prompt
    
    async def generate_response(self, prompt: str) -> str:
        """Generate response using the current AI model"""
        model_info = self.models.get(self.current_model)
        if not model_info or not model_info.get("available"):
            raise Exception(f"Model {self.current_model} not available")
        
        if self.current_model == "gemini":
            return await self.generate_gemini_response(prompt)
        elif self.current_model == "ollama":
            return await self.generate_ollama_response(prompt)
        elif self.current_model == "lm_studio":
            return await self.generate_lm_studio_response(prompt)
        else:
            raise Exception(f"Unknown model type: {self.current_model}")
    
    async def generate_gemini_response(self, prompt: str) -> str:
        """Generate response using Gemini API"""
        try:
            model_info = self.models["gemini"]
            url = f"{model_info['endpoint']}?key={model_info['api_key']}"
            
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        raise Exception(f"Gemini API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            raise
    
    async def generate_ollama_response(self, prompt: str) -> str:
        """Generate response using Ollama"""
        try:
            model_info = self.models["ollama"]
            url = f"{model_info['endpoint']}/api/generate"
            
            # Use the first available model
            model_name = model_info["models"][0] if model_info["models"] else "llama2"
            
            data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["response"]
                    else:
                        raise Exception(f"Ollama API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error generating Ollama response: {e}")
            raise
    
    async def generate_lm_studio_response(self, prompt: str) -> str:
        """Generate response using LM Studio"""
        try:
            model_info = self.models["lm_studio"]
            url = f"{model_info['endpoint']}/v1/chat/completions"
            
            data = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 2048
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        raise Exception(f"LM Studio API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error generating LM Studio response: {e}")
            raise
    
    async def parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured format"""
        try:
            # Try to parse as JSON
            parsed = json.loads(response)
            
            # Validate structure
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a dictionary")
            
            # Ensure required fields
            if "actions" not in parsed:
                parsed["actions"] = []
            
            return parsed
            
        except json.JSONDecodeError:
            # If not valid JSON, create a basic structure
            logger.warning("AI response was not valid JSON, creating fallback structure")
            return {
                "understanding": "Could not parse AI response",
                "confidence": 0.5,
                "actions": [],
                "explanation": response,
                "error": "Invalid response format"
            }
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return {
                "understanding": "Error parsing response",
                "confidence": 0.0,
                "actions": [],
                "explanation": str(e),
                "error": str(e)
            }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            name: {
                "available": model.get("available", False),
                "type": model.get("type", "unknown"),
                "capabilities": model.get("capabilities", []),
                "speed_rating": model.get("speed_rating", 0)
            }
            for name, model in self.models.items()
        }
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different AI model"""
        try:
            if model_name in self.models and self.models[model_name].get("available"):
                self.current_model = model_name
                logger.info(f"Switched to model: {model_name}")
                return True
            else:
                logger.warning(f"Model {model_name} not available")
                return False
                
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of LLM manager"""
        try:
            available_models = sum(1 for model in self.models.values() 
                                 if model.get("available", False))
            
            return {
                "healthy": available_models > 0,
                "current_model": self.current_model,
                "available_models": available_models,
                "total_models": len(self.models)
            }
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def shutdown(self):
        """Shutdown LLM manager"""
        logger.info("Shutting down LLM Manager...")
        # Save conversation history
        try:
            with open("data/conversation_history.json", "w") as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")
