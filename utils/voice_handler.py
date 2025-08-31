"""
Voice Handler - Advanced voice interaction and speech processing
"""

import asyncio
import logging
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import threading
import queue
import time

# Speech recognition imports
try:
    import speech_recognition as sr
    import pyaudio
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

# Text-to-speech imports
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Audio processing imports
try:
    import soundfile as sf
    import numpy as np
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

logger = logging.getLogger(__name__)

class VoiceHandler:
    """Advanced voice interaction and speech processing system"""
    
    def __init__(self, settings):
        self.settings = settings
        self.recognizer = None
        self.microphone = None
        self.tts_engine = None
        self.is_listening = False
        self.voice_commands = {}
        self.wake_words = ['hey agent', 'computer', 'assistant']
        self.conversation_history = []
        
        # Audio configuration
        self.sample_rate = 16000
        self.chunk_duration = 1.0  # seconds
        self.silence_timeout = 2.0  # seconds
        self.energy_threshold = 300
        self.dynamic_energy_threshold = True
        
        # Voice settings
        self.voice_speed = 150  # words per minute
        self.voice_volume = 0.8
        self.voice_pitch = 0
        
        # Recognition settings
        self.recognition_timeout = 5
        self.phrase_timeout = 1
        self.supported_languages = ['en-US', 'es-ES', 'fr-FR', 'de-DE', 'it-IT']
        self.default_language = 'en-US'
        
    async def initialize(self):
        """Initialize voice handler"""
        try:
            logger.info("Initializing Voice Handler...")
            
            # Check available capabilities
            self.check_capabilities()
            
            if SPEECH_RECOGNITION_AVAILABLE:
                await self.init_speech_recognition()
            
            if TTS_AVAILABLE:
                await self.init_text_to_speech()
            
            # Load voice commands
            await self.load_voice_commands()
            
            # Initialize audio processing
            await self.init_audio_processing()
            
            logger.info("Voice Handler initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Voice Handler: {e}")
            # Continue with limited functionality
    
    def check_capabilities(self):
        """Check what voice processing capabilities are available"""
        self.capabilities = {
            'speech_recognition': SPEECH_RECOGNITION_AVAILABLE,
            'text_to_speech': TTS_AVAILABLE,
            'audio_processing': AUDIO_PROCESSING_AVAILABLE,
            'real_time_processing': SPEECH_RECOGNITION_AVAILABLE and TTS_AVAILABLE,
            'voice_commands': True,
            'conversation_mode': True
        }
        
        logger.info(f"Voice capabilities: {self.capabilities}")
    
    async def init_speech_recognition(self):
        """Initialize speech recognition system"""
        try:
            self.recognizer = sr.Recognizer()
            
            # Configure recognizer
            self.recognizer.energy_threshold = self.energy_threshold
            self.recognizer.dynamic_energy_threshold = self.dynamic_energy_threshold
            self.recognizer.pause_threshold = self.phrase_timeout
            self.recognizer.phrase_threshold = 0.3
            self.recognizer.non_speaking_duration = self.phrase_timeout
            
            # Initialize microphone
            try:
                self.microphone = sr.Microphone()
                
                # Calibrate for ambient noise
                with self.microphone as source:
                    logger.info("Calibrating microphone for ambient noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    
                logger.info(f"Microphone energy threshold set to: {self.recognizer.energy_threshold}")
                
            except Exception as e:
                logger.warning(f"Could not initialize microphone: {e}")
                self.microphone = None
            
        except Exception as e:
            logger.error(f"Error initializing speech recognition: {e}")
    
    async def init_text_to_speech(self):
        """Initialize text-to-speech system"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            self.tts_engine.setProperty('rate', self.voice_speed)
            self.tts_engine.setProperty('volume', self.voice_volume)
            
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to set a pleasant voice (prefer female voices)
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                
                current_voice = self.tts_engine.getProperty('voice')
                logger.info(f"TTS voice set to: {current_voice}")
            
        except Exception as e:
            logger.error(f"Error initializing text-to-speech: {e}")
            self.tts_engine = None
    
    async def init_audio_processing(self):
        """Initialize audio processing components"""
        try:
            if AUDIO_PROCESSING_AVAILABLE:
                # Initialize audio processing settings
                self.audio_config = {
                    'sample_rate': self.sample_rate,
                    'channels': 1,
                    'format': 'float32'
                }
                
                logger.info("Audio processing initialized")
            
        except Exception as e:
            logger.error(f"Error initializing audio processing: {e}")
    
    async def load_voice_commands(self):
        """Load voice command patterns"""
        try:
            commands_file = Path("data/voice_commands.json")
            
            if commands_file.exists():
                with open(commands_file, 'r') as f:
                    self.voice_commands = json.load(f)
            else:
                # Default voice commands
                self.voice_commands = {
                    "activate_listening": {
                        "patterns": ["start listening", "begin listening", "activate voice"],
                        "action": "start_listening",
                        "response": "Voice recognition activated"
                    },
                    "stop_listening": {
                        "patterns": ["stop listening", "deactivate voice", "silence"],
                        "action": "stop_listening",
                        "response": "Voice recognition deactivated"
                    },
                    "system_status": {
                        "patterns": ["system status", "how are you", "status report"],
                        "action": "get_status",
                        "response": "System is running normally"
                    },
                    "time": {
                        "patterns": ["what time is it", "current time", "tell me the time"],
                        "action": "get_time",
                        "response": "The current time is {time}"
                    },
                    "help": {
                        "patterns": ["help", "what can you do", "voice commands"],
                        "action": "get_help",
                        "response": "I can help you with various tasks using voice commands"
                    }
                }
                
                await self.save_voice_commands()
            
            logger.info(f"Loaded {len(self.voice_commands)} voice command patterns")
            
        except Exception as e:
            logger.error(f"Error loading voice commands: {e}")
    
    async def save_voice_commands(self):
        """Save voice commands to file"""
        try:
            commands_file = Path("data/voice_commands.json")
            commands_file.parent.mkdir(exist_ok=True)
            
            with open(commands_file, 'w') as f:
                json.dump(self.voice_commands, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving voice commands: {e}")
    
    async def start_listening(self, callback: Optional[Callable] = None) -> bool:
        """Start continuous voice listening"""
        try:
            if not self.capabilities['speech_recognition'] or not self.microphone:
                logger.warning("Speech recognition not available")
                return False
            
            if self.is_listening:
                logger.info("Already listening")
                return True
            
            self.is_listening = True
            
            # Start listening in background thread
            listen_thread = threading.Thread(
                target=self._listen_continuously,
                args=(callback,),
                daemon=True
            )
            listen_thread.start()
            
            logger.info("Started continuous voice listening")
            return True
            
        except Exception as e:
            logger.error(f"Error starting voice listening: {e}")
            return False
    
    def _listen_continuously(self, callback: Optional[Callable] = None):
        """Continuous listening loop (runs in background thread)"""
        try:
            while self.is_listening:
                try:
                    # Listen for audio
                    with self.microphone as source:
                        # Listen for wake word or command
                        audio = self.recognizer.listen(
                            source,
                            timeout=1,
                            phrase_time_limit=self.recognition_timeout
                        )
                    
                    # Process audio in background
                    asyncio.run_coroutine_threadsafe(
                        self._process_audio(audio, callback),
                        asyncio.get_event_loop()
                    )
                    
                except sr.WaitTimeoutError:
                    # No speech detected, continue listening
                    continue
                except Exception as e:
                    logger.error(f"Error in listening loop: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            logger.error(f"Error in continuous listening: {e}")
        finally:
            self.is_listening = False
    
    async def _process_audio(self, audio, callback: Optional[Callable] = None):
        """Process captured audio"""
        try:
            # Recognize speech
            text = await self.recognize_speech(audio)
            
            if text:
                # Check for wake words
                if await self.contains_wake_word(text):
                    await self.speak("Yes, I'm listening")
                    
                    # Wait for command
                    command_audio = await self.listen_for_command()
                    if command_audio:
                        command_text = await self.recognize_speech(command_audio)
                        if command_text:
                            await self.process_voice_command(command_text, callback)
                else:
                    # Process as direct command
                    await self.process_voice_command(text, callback)
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
    
    async def recognize_speech(self, audio, language: str = None) -> Optional[str]:
        """Recognize speech from audio data"""
        try:
            if not audio:
                return None
            
            language = language or self.default_language
            
            # Try different recognition services
            recognition_methods = [
                ('Google', self._recognize_google),
                ('Sphinx', self._recognize_sphinx),
                ('Wit.ai', self._recognize_wit)
            ]
            
            for method_name, method in recognition_methods:
                try:
                    text = await method(audio, language)
                    if text:
                        logger.debug(f"Recognized with {method_name}: {text}")
                        return text.lower().strip()
                except Exception as e:
                    logger.debug(f"{method_name} recognition failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error recognizing speech: {e}")
            return None
    
    async def _recognize_google(self, audio, language: str) -> str:
        """Recognize speech using Google Web Speech API"""
        try:
            return self.recognizer.recognize_google(audio, language=language)
        except Exception as e:
            raise Exception(f"Google recognition error: {e}")
    
    async def _recognize_sphinx(self, audio, language: str) -> str:
        """Recognize speech using CMU Sphinx"""
        try:
            return self.recognizer.recognize_sphinx(audio)
        except Exception as e:
            raise Exception(f"Sphinx recognition error: {e}")
    
    async def _recognize_wit(self, audio, language: str) -> str:
        """Recognize speech using Wit.ai"""
        try:
            wit_key = os.getenv("WIT_AI_KEY")
            if not wit_key:
                raise Exception("Wit.ai key not available")
            
            return self.recognizer.recognize_wit(audio, key=wit_key)
        except Exception as e:
            raise Exception(f"Wit.ai recognition error: {e}")
    
    async def contains_wake_word(self, text: str) -> bool:
        """Check if text contains a wake word"""
        try:
            text_lower = text.lower()
            return any(wake_word in text_lower for wake_word in self.wake_words)
        except Exception as e:
            logger.error(f"Error checking wake word: {e}")
            return False
    
    async def listen_for_command(self, timeout: float = 5.0) -> Optional[any]:
        """Listen for a command after wake word detection"""
        try:
            if not self.microphone:
                return None
            
            with self.microphone as source:
                logger.debug("Listening for command...")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=self.recognition_timeout
                )
                return audio
                
        except sr.WaitTimeoutError:
            logger.debug("Command timeout")
            return None
        except Exception as e:
            logger.error(f"Error listening for command: {e}")
            return None
    
    async def process_voice_command(self, text: str, callback: Optional[Callable] = None):
        """Process a recognized voice command"""
        try:
            logger.info(f"Processing voice command: {text}")
            
            # Record in conversation history
            self.conversation_history.append({
                'type': 'voice_input',
                'text': text,
                'timestamp': datetime.now().isoformat()
            })
            
            # Find matching command
            command_match = await self.match_voice_command(text)
            
            if command_match:
                # Execute matched command
                result = await self.execute_voice_command(command_match, text)
                
                # Provide voice response
                if result.get('response'):
                    await self.speak(result['response'])
                
                # Call callback if provided
                if callback:
                    await callback(text, result)
            else:
                # No specific command matched, pass to general processing
                if callback:
                    await callback(text, {'action': 'general_processing'})
                else:
                    await self.speak("I didn't understand that command")
            
        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            await self.speak("Sorry, I encountered an error processing that command")
    
    async def match_voice_command(self, text: str) -> Optional[Dict[str, Any]]:
        """Match text against voice command patterns"""
        try:
            text_lower = text.lower()
            
            for command_name, command_data in self.voice_commands.items():
                patterns = command_data.get('patterns', [])
                
                for pattern in patterns:
                    if pattern.lower() in text_lower:
                        return {
                            'command_name': command_name,
                            'command_data': command_data,
                            'matched_pattern': pattern
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error matching voice command: {e}")
            return None
    
    async def execute_voice_command(self, command_match: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Execute a matched voice command"""
        try:
            command_data = command_match['command_data']
            action = command_data.get('action')
            
            result = {'action': action, 'success': True}
            
            if action == 'start_listening':
                if not self.is_listening:
                    await self.start_listening()
                result['response'] = command_data.get('response', 'Listening activated')
                
            elif action == 'stop_listening':
                await self.stop_listening()
                result['response'] = command_data.get('response', 'Listening deactivated')
                
            elif action == 'get_status':
                status = await self.get_system_status()
                result['response'] = f"System status: {status}"
                
            elif action == 'get_time':
                current_time = datetime.now().strftime("%I:%M %p")
                result['response'] = command_data.get('response', '').format(time=current_time)
                
            elif action == 'get_help':
                help_text = await self.get_voice_help()
                result['response'] = help_text
                
            else:
                # Custom command
                result['response'] = command_data.get('response', 'Command executed')
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing voice command: {e}")
            return {'action': 'error', 'success': False, 'error': str(e)}
    
    async def speak(self, text: str, interrupt: bool = False) -> bool:
        """Convert text to speech and play it"""
        try:
            if not self.capabilities['text_to_speech'] or not self.tts_engine:
                logger.warning("Text-to-speech not available")
                return False
            
            if not text.strip():
                return False
            
            # Record in conversation history
            self.conversation_history.append({
                'type': 'voice_output',
                'text': text,
                'timestamp': datetime.now().isoformat()
            })
            
            # Speak text
            if interrupt:
                self.tts_engine.stop()
            
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
            logger.debug(f"Spoke: {text}")
            return True
            
        except Exception as e:
            logger.error(f"Error speaking text: {e}")
            return False
    
    async def stop_listening(self):
        """Stop continuous voice listening"""
        try:
            self.is_listening = False
            logger.info("Stopped voice listening")
        except Exception as e:
            logger.error(f"Error stopping voice listening: {e}")
    
    async def set_voice_settings(self, settings: Dict[str, Any]) -> bool:
        """Update voice settings"""
        try:
            if not self.tts_engine:
                return False
            
            if 'speed' in settings:
                self.voice_speed = settings['speed']
                self.tts_engine.setProperty('rate', self.voice_speed)
            
            if 'volume' in settings:
                self.voice_volume = settings['volume']
                self.tts_engine.setProperty('volume', self.voice_volume)
            
            if 'voice_id' in settings:
                self.tts_engine.setProperty('voice', settings['voice_id'])
            
            logger.info("Voice settings updated")
            return True
            
        except Exception as e:
            logger.error(f"Error setting voice settings: {e}")
            return False
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available TTS voices"""
        try:
            if not self.tts_engine:
                return []
            
            voices = self.tts_engine.getProperty('voices')
            voice_list = []
            
            for voice in voices:
                voice_info = {
                    'id': voice.id,
                    'name': voice.name,
                    'languages': getattr(voice, 'languages', []),
                    'gender': getattr(voice, 'gender', 'unknown')
                }
                voice_list.append(voice_info)
            
            return voice_list
            
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return []
    
    async def add_voice_command(self, command_name: str, patterns: List[str], action: str, response: str) -> bool:
        """Add a new voice command"""
        try:
            self.voice_commands[command_name] = {
                'patterns': patterns,
                'action': action,
                'response': response
            }
            
            await self.save_voice_commands()
            logger.info(f"Added voice command: {command_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding voice command: {e}")
            return False
    
    async def process_audio_file(self, file_path: str) -> Optional[str]:
        """Process an audio file and return transcribed text"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"Audio file not found: {file_path}")
                return None
            
            # Load audio file
            with sr.AudioFile(file_path) as source:
                audio = self.recognizer.record(source)
            
            # Recognize speech
            text = await self.recognize_speech(audio)
            return text
            
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            return None
    
    async def save_audio_recording(self, duration: float = 10.0) -> Optional[str]:
        """Record audio and save to file"""
        try:
            if not self.microphone:
                logger.error("Microphone not available")
                return None
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            
            # Record audio
            with self.microphone as source:
                logger.info(f"Recording audio for {duration} seconds...")
                audio = self.recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
            
            # Save audio to file
            with open(temp_file.name, 'wb') as f:
                f.write(audio.get_wav_data())
            
            logger.info(f"Audio saved to: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error saving audio recording: {e}")
            return None
    
    async def get_system_status(self) -> str:
        """Get system status for voice response"""
        try:
            status_parts = []
            
            if self.is_listening:
                status_parts.append("voice recognition active")
            else:
                status_parts.append("voice recognition inactive")
            
            if self.capabilities['text_to_speech']:
                status_parts.append("speech synthesis available")
            
            return ", ".join(status_parts)
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return "status unknown"
    
    async def get_voice_help(self) -> str:
        """Get help text for voice commands"""
        try:
            help_commands = []
            
            for command_name, command_data in self.voice_commands.items():
                patterns = command_data.get('patterns', [])
                if patterns:
                    help_commands.append(f"Say '{patterns[0]}' to {command_name.replace('_', ' ')}")
            
            if help_commands:
                return "Available commands: " + ". ".join(help_commands[:3])
            else:
                return "No voice commands available"
                
        except Exception as e:
            logger.error(f"Error getting voice help: {e}")
            return "Help not available"
    
    def get_conversation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:] if self.conversation_history else []
    
    def get_voice_statistics(self) -> Dict[str, Any]:
        """Get voice interaction statistics"""
        try:
            total_interactions = len(self.conversation_history)
            voice_inputs = len([h for h in self.conversation_history if h['type'] == 'voice_input'])
            voice_outputs = len([h for h in self.conversation_history if h['type'] == 'voice_output'])
            
            return {
                'total_interactions': total_interactions,
                'voice_inputs': voice_inputs,
                'voice_outputs': voice_outputs,
                'is_listening': self.is_listening,
                'commands_available': len(self.voice_commands),
                'capabilities': self.capabilities
            }
            
        except Exception as e:
            logger.error(f"Error getting voice statistics: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Shutdown voice handler"""
        logger.info("Shutting down Voice Handler...")
        
        try:
            # Stop listening
            await self.stop_listening()
            
            # Shutdown TTS engine
            if self.tts_engine:
                self.tts_engine.stop()
            
            # Save conversation history
            history_file = Path("data/voice_conversation_history.json")
            history_file.parent.mkdir(exist_ok=True)
            
            with open(history_file, 'w') as f:
                json.dump(self.conversation_history[-100:], f, indent=2)  # Save last 100 interactions
                
        except Exception as e:
            logger.error(f"Error during voice handler shutdown: {e}")
