"""
Translation Service - Multi-language support and translation capabilities
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import aiohttp

# Translation imports
try:
    from googletrans import Translator as GoogleTranslator
    GOOGLE_TRANSLATE_AVAILABLE = True
except ImportError:
    GOOGLE_TRANSLATE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class TranslationService:
    """Multi-language support and translation service"""
    
    def __init__(self, settings):
        self.settings = settings
        self.google_translator = None
        self.translation_cache = {}
        self.language_cache = {}
        self.supported_languages = {}
        
        # Service configuration
        self.cache_enabled = True
        self.cache_max_size = 1000
        self.translation_history = []
        self.max_history = 500
        
        # Language detection confidence threshold
        self.detection_threshold = 0.8
        
        # Common languages with their codes and names
        self.common_languages = {
            'en': {'name': 'English', 'native': 'English'},
            'es': {'name': 'Spanish', 'native': 'Español'},
            'fr': {'name': 'French', 'native': 'Français'},
            'de': {'name': 'German', 'native': 'Deutsch'},
            'it': {'name': 'Italian', 'native': 'Italiano'},
            'pt': {'name': 'Portuguese', 'native': 'Português'},
            'ru': {'name': 'Russian', 'native': 'Русский'},
            'ja': {'name': 'Japanese', 'native': '日本語'},
            'ko': {'name': 'Korean', 'native': '한국어'},
            'zh': {'name': 'Chinese', 'native': '中文'},
            'ar': {'name': 'Arabic', 'native': 'العربية'},
            'hi': {'name': 'Hindi', 'native': 'हिन्दी'},
            'nl': {'name': 'Dutch', 'native': 'Nederlands'},
            'sv': {'name': 'Swedish', 'native': 'Svenska'},
            'no': {'name': 'Norwegian', 'native': 'Norsk'},
            'da': {'name': 'Danish', 'native': 'Dansk'},
            'fi': {'name': 'Finnish', 'native': 'Suomi'},
            'pl': {'name': 'Polish', 'native': 'Polski'},
            'tr': {'name': 'Turkish', 'native': 'Türkçe'},
            'th': {'name': 'Thai', 'native': 'ไทย'},
            'vi': {'name': 'Vietnamese', 'native': 'Tiếng Việt'}
        }
        
    async def initialize(self):
        """Initialize translation service"""
        try:
            logger.info("Initializing Translation Service...")
            
            # Check available translation services
            await self.check_translation_services()
            
            # Initialize Google Translate if available
            if GOOGLE_TRANSLATE_AVAILABLE:
                await self.init_google_translate()
            
            # Load translation cache
            await self.load_translation_cache()
            
            # Get supported languages
            await self.get_supported_languages()
            
            logger.info("Translation Service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Translation Service: {e}")
            # Continue with limited functionality
    
    async def check_translation_services(self):
        """Check available translation services"""
        self.available_services = {
            'google_translate': GOOGLE_TRANSLATE_AVAILABLE,
            'openai': OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY")),
            'microsoft': bool(os.getenv("MICROSOFT_TRANSLATE_KEY")),
            'deepl': bool(os.getenv("DEEPL_API_KEY"))
        }
        
        active_services = [service for service, available in self.available_services.items() if available]
        logger.info(f"Available translation services: {active_services}")
    
    async def init_google_translate(self):
        """Initialize Google Translate"""
        try:
            self.google_translator = GoogleTranslator()
            
            # Test the translator
            test_result = self.google_translator.translate("Hello", dest='es')
            if test_result.text:
                logger.info("Google Translate initialized successfully")
            else:
                raise Exception("Google Translate test failed")
                
        except Exception as e:
            logger.warning(f"Google Translate initialization failed: {e}")
            self.google_translator = None
    
    async def get_supported_languages(self):
        """Get list of supported languages from translation services"""
        try:
            if self.google_translator:
                # Get Google Translate supported languages
                try:
                    from googletrans import LANGUAGES
                    self.supported_languages.update(LANGUAGES)
                except:
                    # Fallback to common languages
                    self.supported_languages = {code: info['name'].lower() 
                                               for code, info in self.common_languages.items()}
            else:
                # Use common languages as fallback
                self.supported_languages = {code: info['name'].lower() 
                                           for code, info in self.common_languages.items()}
            
            logger.info(f"Loaded {len(self.supported_languages)} supported languages")
            
        except Exception as e:
            logger.error(f"Error getting supported languages: {e}")
            self.supported_languages = {code: info['name'].lower() 
                                       for code, info in self.common_languages.items()}
    
    async def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect the language of input text"""
        try:
            if not text.strip():
                return {'language': 'unknown', 'confidence': 0.0}
            
            # Check cache first
            cache_key = f"detect_{hash(text)}"
            if self.cache_enabled and cache_key in self.language_cache:
                return self.language_cache[cache_key]
            
            detection_result = None
            
            # Try Google Translate detection
            if self.google_translator:
                try:
                    detected = self.google_translator.detect(text)
                    detection_result = {
                        'language': detected.lang,
                        'confidence': detected.confidence,
                        'service': 'google'
                    }
                except Exception as e:
                    logger.debug(f"Google Translate detection failed: {e}")
            
            # Fallback to simple detection
            if not detection_result:
                detection_result = await self.simple_language_detection(text)
            
            # Cache result
            if self.cache_enabled:
                self.language_cache[cache_key] = detection_result
                await self.manage_cache_size()
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return {'language': 'unknown', 'confidence': 0.0, 'error': str(e)}
    
    async def simple_language_detection(self, text: str) -> Dict[str, Any]:
        """Simple language detection based on character patterns"""
        try:
            text_lower = text.lower()
            
            # Character-based detection
            language_scores = {}
            
            # English detection
            english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
            english_score = sum(1 for word in english_words if word in text_lower)
            if english_score > 0:
                language_scores['en'] = english_score / len(text_lower.split())
            
            # Spanish detection
            spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no']
            spanish_score = sum(1 for word in spanish_words if word in text_lower)
            if spanish_score > 0:
                language_scores['es'] = spanish_score / len(text_lower.split())
            
            # French detection
            french_words = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'en', 'avoir', 'que']
            french_score = sum(1 for word in french_words if word in text_lower)
            if french_score > 0:
                language_scores['fr'] = french_score / len(text_lower.split())
            
            # Character-based detection for non-Latin scripts
            if any('\u4e00' <= char <= '\u9fff' for char in text):  # Chinese characters
                language_scores['zh'] = 0.9
            elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):  # Japanese
                language_scores['ja'] = 0.9
            elif any('\uac00' <= char <= '\ud7af' for char in text):  # Korean
                language_scores['ko'] = 0.9
            elif any('\u0600' <= char <= '\u06ff' for char in text):  # Arabic
                language_scores['ar'] = 0.9
            elif any('\u0400' <= char <= '\u04ff' for char in text):  # Cyrillic (Russian)
                language_scores['ru'] = 0.9
            
            if language_scores:
                detected_lang = max(language_scores, key=language_scores.get)
                confidence = language_scores[detected_lang]
                
                return {
                    'language': detected_lang,
                    'confidence': min(confidence, 0.8),  # Cap confidence for simple detection
                    'service': 'simple'
                }
            else:
                return {
                    'language': 'en',  # Default to English
                    'confidence': 0.3,
                    'service': 'simple'
                }
                
        except Exception as e:
            logger.error(f"Error in simple language detection: {e}")
            return {'language': 'unknown', 'confidence': 0.0}
    
    async def translate_text(self, text: str, target_language: str, source_language: str = None) -> Dict[str, Any]:
        """Translate text to target language"""
        try:
            if not text.strip():
                return {'translated_text': '', 'success': False, 'error': 'Empty text'}
            
            # Validate target language
            if target_language not in self.supported_languages:
                return {'translated_text': '', 'success': False, 'error': f'Unsupported target language: {target_language}'}
            
            # Check cache first
            cache_key = f"translate_{hash(text)}_{source_language}_{target_language}"
            if self.cache_enabled and cache_key in self.translation_cache:
                cached_result = self.translation_cache[cache_key]
                cached_result['from_cache'] = True
                return cached_result
            
            # Detect source language if not provided
            if not source_language:
                detection = await self.detect_language(text)
                source_language = detection['language']
            
            # Skip translation if source and target are the same
            if source_language == target_language:
                return {
                    'translated_text': text,
                    'success': True,
                    'source_language': source_language,
                    'target_language': target_language,
                    'service': 'none_required'
                }
            
            translation_result = None
            
            # Try different translation services
            if self.google_translator:
                translation_result = await self.translate_with_google(text, target_language, source_language)
            
            if not translation_result and self.available_services.get('openai'):
                translation_result = await self.translate_with_openai(text, target_language, source_language)
            
            if not translation_result:
                return {'translated_text': '', 'success': False, 'error': 'No translation service available'}
            
            # Add metadata
            translation_result.update({
                'source_language': source_language,
                'target_language': target_language,
                'timestamp': datetime.now().isoformat()
            })
            
            # Cache result
            if self.cache_enabled and translation_result['success']:
                self.translation_cache[cache_key] = translation_result.copy()
                await self.manage_cache_size()
            
            # Add to history
            self.translation_history.append({
                'text': text[:100],  # First 100 chars
                'source_lang': source_language,
                'target_lang': target_language,
                'success': translation_result['success'],
                'timestamp': datetime.now().isoformat()
            })
            
            # Limit history size
            if len(self.translation_history) > self.max_history:
                self.translation_history = self.translation_history[-self.max_history:]
            
            return translation_result
            
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            return {'translated_text': '', 'success': False, 'error': str(e)}
    
    async def translate_with_google(self, text: str, target_lang: str, source_lang: str) -> Dict[str, Any]:
        """Translate using Google Translate"""
        try:
            result = self.google_translator.translate(
                text, 
                dest=target_lang, 
                src=source_lang if source_lang != 'unknown' else None
            )
            
            return {
                'translated_text': result.text,
                'success': True,
                'service': 'google',
                'confidence': 0.9,
                'detected_source': result.src
            }
            
        except Exception as e:
            logger.error(f"Google Translate error: {e}")
            return None
    
    async def translate_with_openai(self, text: str, target_lang: str, source_lang: str) -> Dict[str, Any]:
        """Translate using OpenAI"""
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            target_lang_name = self.common_languages.get(target_lang, {}).get('name', target_lang)
            source_lang_name = self.common_languages.get(source_lang, {}).get('name', source_lang)
            
            prompt = f"Translate the following text from {source_lang_name} to {target_lang_name}:\n\n{text}\n\nTranslation:"
            
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            translated_text = response.choices[0].text.strip()
            
            return {
                'translated_text': translated_text,
                'success': True,
                'service': 'openai',
                'confidence': 0.85
            }
            
        except Exception as e:
            logger.error(f"OpenAI translation error: {e}")
            return None
    
    async def translate_batch(self, texts: List[str], target_language: str, source_language: str = None) -> List[Dict[str, Any]]:
        """Translate multiple texts"""
        try:
            results = []
            
            for text in texts:
                result = await self.translate_text(text, target_language, source_language)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch translation: {e}")
            return [{'translated_text': '', 'success': False, 'error': str(e)} for _ in texts]
    
    async def auto_translate_conversation(self, conversation: List[Dict[str, Any]], target_language: str) -> List[Dict[str, Any]]:
        """Auto-translate a conversation to target language"""
        try:
            translated_conversation = []
            
            for message in conversation:
                if 'text' in message:
                    translation = await self.translate_text(message['text'], target_language)
                    
                    translated_message = message.copy()
                    if translation['success']:
                        translated_message['translated_text'] = translation['translated_text']
                        translated_message['original_text'] = message['text']
                        translated_message['translation_info'] = {
                            'target_language': target_language,
                            'source_language': translation.get('source_language'),
                            'service': translation.get('service')
                        }
                    else:
                        translated_message['translation_error'] = translation.get('error')
                    
                    translated_conversation.append(translated_message)
                else:
                    translated_conversation.append(message)
            
            return translated_conversation
            
        except Exception as e:
            logger.error(f"Error auto-translating conversation: {e}")
            return conversation
    
    async def get_language_info(self, language_code: str) -> Dict[str, Any]:
        """Get detailed information about a language"""
        try:
            if language_code in self.common_languages:
                lang_info = self.common_languages[language_code].copy()
                lang_info['code'] = language_code
                lang_info['supported'] = language_code in self.supported_languages
                return lang_info
            elif language_code in self.supported_languages:
                return {
                    'code': language_code,
                    'name': self.supported_languages[language_code].title(),
                    'supported': True
                }
            else:
                return {
                    'code': language_code,
                    'name': 'Unknown',
                    'supported': False
                }
                
        except Exception as e:
            logger.error(f"Error getting language info: {e}")
            return {'code': language_code, 'error': str(e)}
    
    async def suggest_language(self, text: str) -> List[Dict[str, Any]]:
        """Suggest possible languages for text"""
        try:
            detection = await self.detect_language(text)
            suggestions = []
            
            # Add detected language as primary suggestion
            if detection['language'] != 'unknown':
                lang_info = await self.get_language_info(detection['language'])
                lang_info['confidence'] = detection['confidence']
                lang_info['primary'] = True
                suggestions.append(lang_info)
            
            # Add other possible languages based on simple detection
            simple_detection = await self.simple_language_detection(text)
            if simple_detection['language'] != detection['language']:
                lang_info = await self.get_language_info(simple_detection['language'])
                lang_info['confidence'] = simple_detection['confidence']
                lang_info['primary'] = False
                suggestions.append(lang_info)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting languages: {e}")
            return []
    
    async def get_translation_quality(self, original: str, translated: str, target_lang: str) -> Dict[str, Any]:
        """Assess translation quality (basic implementation)"""
        try:
            quality_score = 0.8  # Default quality score
            issues = []
            
            # Basic quality checks
            if len(translated) < len(original) * 0.3:
                quality_score -= 0.2
                issues.append("Translation significantly shorter than original")
            
            if len(translated) > len(original) * 3:
                quality_score -= 0.1
                issues.append("Translation significantly longer than original")
            
            if not translated.strip():
                quality_score = 0.0
                issues.append("Empty translation")
            
            # Check for untranslated content (same as original)
            if original.lower() == translated.lower():
                quality_score -= 0.3
                issues.append("Text appears untranslated")
            
            return {
                'quality_score': max(0.0, quality_score),
                'issues': issues,
                'assessment': 'good' if quality_score > 0.7 else 'fair' if quality_score > 0.4 else 'poor'
            }
            
        except Exception as e:
            logger.error(f"Error assessing translation quality: {e}")
            return {'quality_score': 0.0, 'error': str(e)}
    
    async def load_translation_cache(self):
        """Load translation cache from file"""
        try:
            cache_file = Path("data/translation_cache.json")
            
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self.translation_cache = cache_data.get('translations', {})
                    self.language_cache = cache_data.get('detections', {})
                
                logger.info(f"Loaded translation cache with {len(self.translation_cache)} translations")
            
        except Exception as e:
            logger.error(f"Error loading translation cache: {e}")
    
    async def save_translation_cache(self):
        """Save translation cache to file"""
        try:
            cache_file = Path("data/translation_cache.json")
            cache_file.parent.mkdir(exist_ok=True)
            
            cache_data = {
                'translations': self.translation_cache,
                'detections': self.language_cache,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving translation cache: {e}")
    
    async def manage_cache_size(self):
        """Manage cache size to prevent unlimited growth"""
        try:
            if len(self.translation_cache) > self.cache_max_size:
                # Remove oldest entries (simple FIFO)
                items_to_remove = len(self.translation_cache) - self.cache_max_size
                keys_to_remove = list(self.translation_cache.keys())[:items_to_remove]
                
                for key in keys_to_remove:
                    del self.translation_cache[key]
            
            if len(self.language_cache) > self.cache_max_size:
                items_to_remove = len(self.language_cache) - self.cache_max_size
                keys_to_remove = list(self.language_cache.keys())[:items_to_remove]
                
                for key in keys_to_remove:
                    del self.language_cache[key]
                    
        except Exception as e:
            logger.error(f"Error managing cache size: {e}")
    
    def get_supported_languages_list(self) -> List[Dict[str, Any]]:
        """Get list of all supported languages"""
        try:
            languages = []
            
            for code, name in self.supported_languages.items():
                lang_info = {
                    'code': code,
                    'name': name.title(),
                    'native_name': self.common_languages.get(code, {}).get('native', name.title())
                }
                languages.append(lang_info)
            
            return sorted(languages, key=lambda x: x['name'])
            
        except Exception as e:
            logger.error(f"Error getting supported languages list: {e}")
            return []
    
    def get_translation_statistics(self) -> Dict[str, Any]:
        """Get translation service statistics"""
        try:
            total_translations = len(self.translation_history)
            successful_translations = len([h for h in self.translation_history if h['success']])
            
            # Language usage statistics
            source_langs = {}
            target_langs = {}
            
            for history_item in self.translation_history:
                source_lang = history_item['source_lang']
                target_lang = history_item['target_lang']
                
                source_langs[source_lang] = source_langs.get(source_lang, 0) + 1
                target_langs[target_lang] = target_langs.get(target_lang, 0) + 1
            
            return {
                'total_translations': total_translations,
                'successful_translations': successful_translations,
                'success_rate': successful_translations / max(total_translations, 1),
                'cached_translations': len(self.translation_cache),
                'available_services': self.available_services,
                'most_used_source_languages': dict(sorted(source_langs.items(), key=lambda x: x[1], reverse=True)[:5]),
                'most_used_target_languages': dict(sorted(target_langs.items(), key=lambda x: x[1], reverse=True)[:5]),
                'supported_languages_count': len(self.supported_languages)
            }
            
        except Exception as e:
            logger.error(f"Error getting translation statistics: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear translation cache"""
        try:
            self.translation_cache.clear()
            self.language_cache.clear()
            logger.info("Translation cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    async def shutdown(self):
        """Shutdown translation service"""
        logger.info("Shutting down Translation Service...")
        
        try:
            # Save cache
            await self.save_translation_cache()
            
            # Save translation history
            history_file = Path("data/translation_history.json")
            history_file.parent.mkdir(exist_ok=True)
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.translation_history[-100:], f, indent=2, ensure_ascii=False)  # Save last 100 translations
                
        except Exception as e:
            logger.error(f"Error during translation service shutdown: {e}")
