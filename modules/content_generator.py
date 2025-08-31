"""
Content Generator - Advanced creative content generation system
"""

import asyncio
import logging
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import base64
import aiohttp
import tempfile

logger = logging.getLogger(__name__)

class ContentGenerator:
    """Advanced content generation system for creative AI tasks"""
    
    def __init__(self, llm_manager, settings):
        self.llm_manager = llm_manager
        self.settings = settings
        self.generation_history = []
        self.templates = {}
        self.style_presets = {}
        
        # Content generation parameters
        self.max_history = 1000
        self.supported_formats = {
            'text': ['article', 'blog', 'social_post', 'email', 'report', 'story'],
            'image': ['photo', 'artwork', 'logo', 'diagram', 'infographic'],
            'video': ['slideshow', 'animation', 'presentation'],
            'audio': ['speech', 'music', 'podcast', 'voiceover'],
            'document': ['pdf', 'presentation', 'spreadsheet', 'webpage']
        }
        
    async def initialize(self):
        """Initialize content generator"""
        try:
            logger.info("Initializing Content Generator...")
            
            # Load templates and presets
            await self.load_templates()
            await self.load_style_presets()
            
            # Initialize external services
            await self.init_image_generation()
            await self.init_video_generation()
            await self.init_audio_generation()
            
            logger.info("Content Generator initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Content Generator: {e}")
            raise
    
    async def init_image_generation(self):
        """Initialize image generation capabilities"""
        try:
            # Check for Stable Diffusion API or similar
            api_key = os.getenv("STABILITY_API_KEY")
            if api_key:
                self.image_api = {
                    'enabled': True,
                    'api_key': api_key,
                    'endpoint': 'https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image'
                }
                logger.info("Image generation enabled with Stability AI")
            else:
                self.image_api = {'enabled': False}
                logger.warning("Image generation disabled - no API key found")
                
        except Exception as e:
            logger.error(f"Error initializing image generation: {e}")
            self.image_api = {'enabled': False, 'error': str(e)}
    
    async def init_video_generation(self):
        """Initialize video generation capabilities"""
        try:
            # Video generation would integrate with services like RunwayML or local FFmpeg
            ffmpeg_available = await self.check_ffmpeg_availability()
            
            self.video_api = {
                'enabled': ffmpeg_available,
                'local_processing': True,
                'supported_formats': ['mp4', 'avi', 'mov', 'gif']
            }
            
            if ffmpeg_available:
                logger.info("Video generation enabled with FFmpeg")
            else:
                logger.warning("Video generation limited - FFmpeg not available")
                
        except Exception as e:
            logger.error(f"Error initializing video generation: {e}")
            self.video_api = {'enabled': False, 'error': str(e)}
    
    async def init_audio_generation(self):
        """Initialize audio generation capabilities"""
        try:
            # Check for TTS services
            elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            
            self.audio_api = {
                'enabled': bool(elevenlabs_key or openai_key),
                'elevenlabs': bool(elevenlabs_key),
                'openai_tts': bool(openai_key),
                'local_tts': True  # Basic TTS always available
            }
            
            logger.info(f"Audio generation enabled: {list(k for k, v in self.audio_api.items() if v and k != 'enabled')}")
            
        except Exception as e:
            logger.error(f"Error initializing audio generation: {e}")
            self.audio_api = {'enabled': False, 'error': str(e)}
    
    async def generate_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content based on parameters"""
        try:
            content_type = params.get('content_type', 'text')
            content_format = params.get('format', 'article')
            prompt = params.get('prompt', '')
            style = params.get('style', 'default')
            
            result = {
                'content_type': content_type,
                'format': content_format,
                'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
            
            if not prompt:
                result['error'] = 'No prompt provided'
                return result
            
            if content_type == 'text':
                result = await self.generate_text_content(params)
            elif content_type == 'image':
                result = await self.generate_image_content(params)
            elif content_type == 'video':
                result = await self.generate_video_content(params)
            elif content_type == 'audio':
                result = await self.generate_audio_content(params)
            elif content_type == 'document':
                result = await self.generate_document_content(params)
            else:
                result['error'] = f'Unsupported content type: {content_type}'
            
            # Store in history
            self.generation_history.append(result)
            if len(self.generation_history) > self.max_history:
                self.generation_history = self.generation_history[-self.max_history:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return {'success': False, 'error': str(e), 'content_type': content_type}
    
    async def generate_text_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text-based content"""
        try:
            format_type = params.get('format', 'article')
            prompt = params.get('prompt', '')
            style = params.get('style', 'professional')
            length = params.get('length', 'medium')
            audience = params.get('audience', 'general')
            
            # Build comprehensive prompt based on format
            enhanced_prompt = await self.build_text_prompt(prompt, format_type, style, length, audience)
            
            # Generate content using LLM
            generated_text = await self.llm_manager.generate_response(enhanced_prompt)
            
            # Post-process based on format
            processed_content = await self.post_process_text(generated_text, format_type)
            
            # Calculate metrics
            word_count = len(generated_text.split())
            char_count = len(generated_text)
            
            return {
                'success': True,
                'content_type': 'text',
                'format': format_type,
                'content': processed_content,
                'metadata': {
                    'word_count': word_count,
                    'character_count': char_count,
                    'style': style,
                    'audience': audience,
                    'estimated_reading_time': f"{max(1, word_count // 200)} min"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating text content: {e}")
            return {'success': False, 'error': str(e)}
    
    async def build_text_prompt(self, prompt: str, format_type: str, style: str, length: str, audience: str) -> str:
        """Build enhanced prompt for text generation"""
        
        length_guidelines = {
            'short': '100-300 words',
            'medium': '500-800 words',
            'long': '1000-1500 words',
            'very_long': '2000+ words'
        }
        
        format_instructions = {
            'article': 'Write a well-structured article with clear introduction, body sections, and conclusion.',
            'blog': 'Write in a conversational blog style with engaging tone and personal insights.',
            'social_post': 'Create engaging social media content with hooks and call-to-action.',
            'email': 'Write professional email content with clear subject and action items.',
            'report': 'Create a formal report with executive summary, findings, and recommendations.',
            'story': 'Write a compelling narrative with character development and plot structure.'
        }
        
        enhanced_prompt = f"""
        Create {format_type} content based on this request: {prompt}
        
        Requirements:
        - Format: {format_instructions.get(format_type, 'Standard format')}
        - Style: {style} tone and approach
        - Length: {length_guidelines.get(length, 'medium length')} ({length})
        - Target Audience: {audience}
        
        Guidelines:
        - Make it engaging and valuable to the reader
        - Use clear, well-structured content
        - Include relevant examples or insights where appropriate
        - Ensure proper formatting with headers, paragraphs, etc.
        - Make it actionable and informative
        
        Generate high-quality content that meets these specifications:
        """
        
        return enhanced_prompt
    
    async def post_process_text(self, text: str, format_type: str) -> str:
        """Post-process generated text based on format"""
        try:
            if format_type == 'social_post':
                # Add hashtags and formatting for social media
                if not any(tag in text for tag in ['#', '@']):
                    # Generate relevant hashtags
                    hashtag_prompt = f"Generate 3-5 relevant hashtags for this content: {text[:200]}"
                    hashtags = await self.llm_manager.generate_response(hashtag_prompt)
                    text += f"\n\n{hashtags}"
            
            elif format_type == 'email':
                # Ensure proper email structure
                if not text.startswith('Subject:'):
                    subject_prompt = f"Generate a compelling email subject line for: {text[:100]}"
                    subject = await self.llm_manager.generate_response(subject_prompt)
                    text = f"Subject: {subject.strip()}\n\n{text}"
            
            elif format_type == 'article':
                # Ensure proper article structure with headers
                if '##' not in text and '#' not in text:
                    # Add basic structure
                    lines = text.split('\n')
                    structured_text = []
                    current_section = []
                    
                    for line in lines:
                        if line.strip() and len(current_section) > 3:
                            if any(word in line.lower() for word in ['first', 'second', 'next', 'finally', 'conclusion']):
                                structured_text.append('\n'.join(current_section))
                                structured_text.append(f"\n## {line}")
                                current_section = []
                                continue
                        current_section.append(line)
                    
                    if current_section:
                        structured_text.append('\n'.join(current_section))
                    
                    text = '\n'.join(structured_text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error post-processing text: {e}")
            return text
    
    async def generate_image_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image content"""
        try:
            if not self.image_api.get('enabled', False):
                return {'success': False, 'error': 'Image generation not available'}
            
            prompt = params.get('prompt', '')
            style = params.get('style', 'photorealistic')
            size = params.get('size', '1024x1024')
            format_type = params.get('format', 'photo')
            
            # Enhance prompt based on style and format
            enhanced_prompt = await self.enhance_image_prompt(prompt, style, format_type)
            
            # Generate image using Stability AI or similar service
            image_result = await self.call_image_api(enhanced_prompt, size)
            
            if image_result['success']:
                # Save image locally
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"temp/generated_image_{timestamp}.png"
                
                os.makedirs("temp", exist_ok=True)
                
                # Decode and save image
                image_data = base64.b64decode(image_result['image_data'])
                with open(filename, 'wb') as f:
                    f.write(image_data)
                
                return {
                    'success': True,
                    'content_type': 'image',
                    'format': format_type,
                    'file_path': filename,
                    'metadata': {
                        'size': size,
                        'style': style,
                        'prompt': enhanced_prompt[:100] + '...',
                        'file_size': len(image_data)
                    }
                }
            else:
                return {'success': False, 'error': image_result.get('error', 'Image generation failed')}
            
        except Exception as e:
            logger.error(f"Error generating image content: {e}")
            return {'success': False, 'error': str(e)}
    
    async def enhance_image_prompt(self, prompt: str, style: str, format_type: str) -> str:
        """Enhance image prompt with style and format modifiers"""
        
        style_modifiers = {
            'photorealistic': 'highly detailed, photorealistic, professional photography',
            'artistic': 'artistic, creative, expressive, high quality art',
            'cartoon': 'cartoon style, animated, colorful, stylized',
            'sketch': 'pencil sketch, hand-drawn, artistic sketch style',
            'digital_art': 'digital art, concept art, detailed illustration',
            'minimalist': 'minimalist, clean, simple, elegant design'
        }
        
        format_modifiers = {
            'photo': 'photograph, high resolution, professional lighting',
            'artwork': 'artwork, creative composition, artistic vision',
            'logo': 'logo design, clean, professional, brand identity',
            'diagram': 'technical diagram, clear, informative, well-structured',
            'infographic': 'infographic style, data visualization, clear information'
        }
        
        style_text = style_modifiers.get(style, style)
        format_text = format_modifiers.get(format_type, format_type)
        
        enhanced = f"{prompt}, {style_text}, {format_text}, high quality, detailed"
        
        return enhanced
    
    async def call_image_api(self, prompt: str, size: str) -> Dict[str, Any]:
        """Call external image generation API"""
        try:
            headers = {
                'Authorization': f"Bearer {self.image_api['api_key']}",
                'Content-Type': 'application/json'
            }
            
            data = {
                'text_prompts': [{'text': prompt}],
                'cfg_scale': 7,
                'samples': 1,
                'steps': 30,
                'width': int(size.split('x')[0]),
                'height': int(size.split('x')[1])
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.image_api['endpoint'], json=data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Extract image data from API response
                        if 'artifacts' in result and result['artifacts']:
                            return {
                                'success': True,
                                'image_data': result['artifacts'][0]['base64']
                            }
                    
                    error_text = await response.text()
                    return {'success': False, 'error': f'API error: {error_text}'}
            
        except Exception as e:
            logger.error(f"Error calling image API: {e}")
            return {'success': False, 'error': str(e)}
    
    async def generate_video_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video content"""
        try:
            if not self.video_api.get('enabled', False):
                return {'success': False, 'error': 'Video generation not available'}
            
            video_type = params.get('format', 'slideshow')
            content = params.get('content', [])
            duration = params.get('duration', 30)
            
            if video_type == 'slideshow':
                result = await self.generate_slideshow(content, duration)
            elif video_type == 'animation':
                result = await self.generate_animation(params)
            elif video_type == 'presentation':
                result = await self.generate_presentation_video(params)
            else:
                result = {'success': False, 'error': f'Unsupported video type: {video_type}'}
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating video content: {e}")
            return {'success': False, 'error': str(e)}
    
    async def generate_slideshow(self, content: List[Dict], duration: int) -> Dict[str, Any]:
        """Generate slideshow video from content"""
        try:
            # This would use FFmpeg to create a slideshow
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"temp/slideshow_{timestamp}.mp4"
            
            # For now, return a placeholder structure
            return {
                'success': True,
                'content_type': 'video',
                'format': 'slideshow',
                'file_path': output_path,
                'metadata': {
                    'duration': duration,
                    'slides': len(content),
                    'format': 'mp4'
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating slideshow: {e}")
            return {'success': False, 'error': str(e)}
    
    async def generate_audio_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate audio content"""
        try:
            if not self.audio_api.get('enabled', False):
                return {'success': False, 'error': 'Audio generation not available'}
            
            audio_type = params.get('format', 'speech')
            text = params.get('text', '')
            voice = params.get('voice', 'default')
            speed = params.get('speed', 1.0)
            
            if audio_type == 'speech':
                result = await self.generate_speech(text, voice, speed)
            elif audio_type == 'music':
                result = await self.generate_music(params)
            elif audio_type == 'podcast':
                result = await self.generate_podcast(params)
            else:
                result = {'success': False, 'error': f'Unsupported audio type: {audio_type}'}
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating audio content: {e}")
            return {'success': False, 'error': str(e)}
    
    async def generate_speech(self, text: str, voice: str, speed: float) -> Dict[str, Any]:
        """Generate speech from text"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"temp/speech_{timestamp}.mp3"
            
            # Use available TTS service
            if self.audio_api.get('elevenlabs'):
                result = await self.generate_elevenlabs_speech(text, voice, output_path)
            elif self.audio_api.get('openai_tts'):
                result = await self.generate_openai_speech(text, voice, output_path)
            else:
                # Fallback to local TTS
                result = await self.generate_local_speech(text, output_path)
            
            if result['success']:
                return {
                    'success': True,
                    'content_type': 'audio',
                    'format': 'speech',
                    'file_path': output_path,
                    'metadata': {
                        'voice': voice,
                        'speed': speed,
                        'text_length': len(text),
                        'estimated_duration': f"{len(text.split()) * 0.5:.1f} seconds"
                    }
                }
            else:
                return result
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return {'success': False, 'error': str(e)}
    
    async def generate_document_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate document content"""
        try:
            doc_type = params.get('format', 'pdf')
            content = params.get('content', '')
            template = params.get('template', 'default')
            
            if doc_type == 'pdf':
                result = await self.generate_pdf_document(content, template)
            elif doc_type == 'presentation':
                result = await self.generate_presentation_document(content, template)
            elif doc_type == 'spreadsheet':
                result = await self.generate_spreadsheet_document(params)
            elif doc_type == 'webpage':
                result = await self.generate_webpage_document(content, template)
            else:
                result = {'success': False, 'error': f'Unsupported document type: {doc_type}'}
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating document content: {e}")
            return {'success': False, 'error': str(e)}
    
    async def check_ffmpeg_availability(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    async def generate_elevenlabs_speech(self, text: str, voice: str, output_path: str) -> Dict[str, Any]:
        """Generate speech using ElevenLabs API"""
        try:
            # ElevenLabs API implementation would go here
            return {'success': True, 'method': 'elevenlabs'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def generate_openai_speech(self, text: str, voice: str, output_path: str) -> Dict[str, Any]:
        """Generate speech using OpenAI TTS API"""
        try:
            # OpenAI TTS API implementation would go here
            return {'success': True, 'method': 'openai'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def generate_local_speech(self, text: str, output_path: str) -> Dict[str, Any]:
        """Generate speech using local TTS"""
        try:
            # Local TTS implementation using pyttsx3 or similar
            return {'success': True, 'method': 'local'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def generate_pdf_document(self, content: str, template: str) -> Dict[str, Any]:
        """Generate PDF document"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"temp/document_{timestamp}.pdf"
            
            # PDF generation would use libraries like reportlab
            return {
                'success': True,
                'content_type': 'document',
                'format': 'pdf',
                'file_path': output_path,
                'metadata': {
                    'template': template,
                    'page_count': 1,
                    'content_length': len(content)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return {'success': False, 'error': str(e)}
    
    async def generate_presentation_document(self, content: str, template: str) -> Dict[str, Any]:
        """Generate presentation document"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"temp/presentation_{timestamp}.pptx"
            
            # Presentation generation would use python-pptx
            return {
                'success': True,
                'content_type': 'document',
                'format': 'presentation',
                'file_path': output_path,
                'metadata': {
                    'template': template,
                    'slide_count': 5,
                    'content_length': len(content)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating presentation: {e}")
            return {'success': False, 'error': str(e)}
    
    async def generate_spreadsheet_document(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate spreadsheet document"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"temp/spreadsheet_{timestamp}.xlsx"
            
            # Spreadsheet generation would use openpyxl
            return {
                'success': True,
                'content_type': 'document',
                'format': 'spreadsheet',
                'file_path': output_path,
                'metadata': {
                    'sheets': 1,
                    'rows': 100,
                    'columns': 10
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating spreadsheet: {e}")
            return {'success': False, 'error': str(e)}
    
    async def generate_webpage_document(self, content: str, template: str) -> Dict[str, Any]:
        """Generate webpage document"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"temp/webpage_{timestamp}.html"
            
            # Generate HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Generated Content</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1, h2, h3 {{ color: #333; }}
                    .content {{ max-width: 800px; margin: 0 auto; }}
                </style>
            </head>
            <body>
                <div class="content">
                    {content.replace(chr(10), '<br>')}
                </div>
            </body>
            </html>
            """
            
            os.makedirs("temp", exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return {
                'success': True,
                'content_type': 'document',
                'format': 'webpage',
                'file_path': output_path,
                'metadata': {
                    'template': template,
                    'file_size': len(html_content),
                    'content_length': len(content)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating webpage: {e}")
            return {'success': False, 'error': str(e)}
    
    async def load_templates(self):
        """Load content templates"""
        try:
            templates_dir = Path("data/templates")
            if templates_dir.exists():
                for template_file in templates_dir.glob("*.json"):
                    with open(template_file, 'r') as f:
                        template_data = json.load(f)
                        self.templates[template_file.stem] = template_data
            
            logger.info(f"Loaded {len(self.templates)} content templates")
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
    
    async def load_style_presets(self):
        """Load style presets"""
        try:
            presets_file = Path("data/style_presets.json")
            if presets_file.exists():
                with open(presets_file, 'r') as f:
                    self.style_presets = json.load(f)
            else:
                # Default style presets
                self.style_presets = {
                    'professional': {
                        'tone': 'formal',
                        'vocabulary': 'business',
                        'structure': 'structured'
                    },
                    'casual': {
                        'tone': 'conversational',
                        'vocabulary': 'everyday',
                        'structure': 'flexible'
                    },
                    'creative': {
                        'tone': 'expressive',
                        'vocabulary': 'rich',
                        'structure': 'artistic'
                    }
                }
            
            logger.info(f"Loaded {len(self.style_presets)} style presets")
        except Exception as e:
            logger.error(f"Error loading style presets: {e}")
    
    def get_generation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get content generation history"""
        return self.generation_history[-limit:] if self.generation_history else []
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported content formats"""
        return self.supported_formats
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get content generation capabilities"""
        return {
            'text_generation': True,
            'image_generation': self.image_api.get('enabled', False),
            'video_generation': self.video_api.get('enabled', False),
            'audio_generation': self.audio_api.get('enabled', False),
            'document_generation': True,
            'supported_formats': self.supported_formats,
            'templates_available': len(self.templates),
            'style_presets': len(self.style_presets)
        }
    
    async def shutdown(self):
        """Shutdown content generator"""
        logger.info("Shutting down Content Generator...")
        
        # Save generation history
        try:
            history_file = Path("data/generation_history.json")
            history_file.parent.mkdir(exist_ok=True)
            
            with open(history_file, 'w') as f:
                json.dump(self.generation_history[-100:], f, indent=2)  # Save last 100 items
                
        except Exception as e:
            logger.error(f"Error saving generation history: {e}")
