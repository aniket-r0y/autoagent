"""
NLP Processor - Advanced natural language processing capabilities
"""

import asyncio
import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import os

# NLP imports
try:
    import spacy
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# Text processing imports
import string
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class NLPProcessor:
    """Advanced natural language processing system"""
    
    def __init__(self, settings):
        self.settings = settings
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.lemmatizer = None
        self.stop_words = set()
        self.entity_cache = {}
        
        # NLP configuration
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
        self.default_language = 'en'
        
        # Intent categories
        self.intent_patterns = {
            'question': [r'\?', r'^(what|how|when|where|why|who)', r'(tell me|explain)'],
            'command': [r'^(do|run|execute|perform|start|stop)', r'(please|could you).*?(do|run)'],
            'request': [r'^(can you|would you|please)', r'(help|assist|support)'],
            'information': [r'^(show|display|list|find)', r'(information|details|data)'],
            'automation': [r'(automate|schedule|repeat)', r'(every|daily|weekly)'],
            'creation': [r'^(create|make|generate|build)', r'(new|fresh)'],
            'modification': [r'^(change|modify|update|edit)', r'(alter|adjust)'],
            'deletion': [r'^(delete|remove|clear)', r'(get rid of|eliminate)']
        }
        
        # Sentiment analysis thresholds
        self.sentiment_thresholds = {
            'very_positive': 0.6,
            'positive': 0.1,
            'neutral': -0.1,
            'negative': -0.6
        }
        
    async def initialize(self):
        """Initialize NLP processor"""
        try:
            logger.info("Initializing NLP Processor...")
            
            if not NLP_AVAILABLE:
                logger.warning("NLP libraries not available, using basic processing")
                return
            
            # Download required NLTK data
            await self.download_nltk_data()
            
            # Initialize spaCy model
            await self.init_spacy_model()
            
            # Initialize NLTK components
            await self.init_nltk_components()
            
            # Load custom patterns
            await self.load_custom_patterns()
            
            logger.info("NLP Processor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP Processor: {e}")
            # Continue with basic functionality
    
    async def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            required_data = [
                'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                'vader_lexicon', 'maxent_ne_chunker', 'words'
            ]
            
            for data_name in required_data:
                try:
                    nltk.download(data_name, quiet=True)
                except Exception as e:
                    logger.warning(f"Could not download NLTK data {data_name}: {e}")
            
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {e}")
    
    async def init_spacy_model(self):
        """Initialize spaCy NLP model"""
        try:
            # Try to load English model
            model_names = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']
            
            for model_name in model_names:
                try:
                    self.nlp_model = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model: {model_name}")
                    break
                except OSError:
                    continue
            
            if not self.nlp_model:
                logger.warning("No spaCy model available, using NLTK only")
                
        except Exception as e:
            logger.error(f"Error initializing spaCy model: {e}")
    
    async def init_nltk_components(self):
        """Initialize NLTK components"""
        try:
            # Initialize sentiment analyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize lemmatizer
            self.lemmatizer = WordNetLemmatizer()
            
            # Load stop words
            self.stop_words = set(stopwords.words('english'))
            
        except Exception as e:
            logger.error(f"Error initializing NLTK components: {e}")
    
    async def load_custom_patterns(self):
        """Load custom NLP patterns and rules"""
        try:
            patterns_file = Path("data/nlp_patterns.json")
            
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    custom_patterns = json.load(f)
                    
                # Merge with default patterns
                for category, patterns in custom_patterns.items():
                    if category in self.intent_patterns:
                        self.intent_patterns[category].extend(patterns)
                    else:
                        self.intent_patterns[category] = patterns
            
            logger.info(f"Loaded NLP patterns for {len(self.intent_patterns)} categories")
            
        except Exception as e:
            logger.error(f"Error loading custom patterns: {e}")
    
    async def analyze_command(self, command: str) -> Dict[str, Any]:
        """Comprehensive analysis of user command"""
        try:
            analysis = {
                'original_text': command,
                'timestamp': datetime.now().isoformat(),
                'language': await self.detect_language(command),
                'length': len(command),
                'word_count': len(command.split()),
                'intent': await self.classify_intent(command),
                'entities': await self.extract_entities(command),
                'sentiment': await self.analyze_sentiment(command),
                'keywords': await self.extract_keywords(command),
                'complexity': await self.assess_complexity(command),
                'context_clues': await self.extract_context_clues(command),
                'action_items': await self.extract_action_items(command),
                'parameters': await self.extract_parameters(command)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing command: {e}")
            return {'error': str(e), 'original_text': command}
    
    async def detect_language(self, text: str) -> str:
        """Detect the language of input text"""
        try:
            if not text.strip():
                return self.default_language
            
            # Simple language detection based on common words
            language_indicators = {
                'en': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with'],
                'es': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no'],
                'fr': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'en', 'avoir', 'que'],
                'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
                'it': ['il', 'di', 'che', 'e', 'la', 'per', 'un', 'in', 'con', 'del'],
                'pt': ['o', 'de', 'e', 'a', 'em', 'um', 'para', 'é', 'com', 'não']
            }
            
            text_lower = text.lower()
            word_tokens = word_tokenize(text_lower) if NLP_AVAILABLE else text_lower.split()
            
            language_scores = {}
            
            for lang, indicators in language_indicators.items():
                score = sum(1 for word in word_tokens if word in indicators)
                language_scores[lang] = score / len(word_tokens) if word_tokens else 0
            
            detected_lang = max(language_scores, key=language_scores.get)
            confidence = language_scores[detected_lang]
            
            # Return English if confidence is too low
            if confidence < 0.1:
                return self.default_language
            
            return detected_lang
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return self.default_language
    
    async def classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify the intent of the user's input"""
        try:
            text_lower = text.lower()
            intent_scores = {}
            
            for intent_type, patterns in self.intent_patterns.items():
                score = 0
                matched_patterns = []
                
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        score += 1
                        matched_patterns.append(pattern)
                
                if score > 0:
                    intent_scores[intent_type] = {
                        'score': score,
                        'confidence': min(score / len(patterns), 1.0),
                        'matched_patterns': matched_patterns
                    }
            
            if intent_scores:
                primary_intent = max(intent_scores, key=lambda x: intent_scores[x]['score'])
                
                return {
                    'primary_intent': primary_intent,
                    'confidence': intent_scores[primary_intent]['confidence'],
                    'all_intents': intent_scores,
                    'matched_patterns': intent_scores[primary_intent]['matched_patterns']
                }
            else:
                return {
                    'primary_intent': 'general',
                    'confidence': 0.5,
                    'all_intents': {},
                    'matched_patterns': []
                }
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return {'primary_intent': 'unknown', 'confidence': 0.0, 'error': str(e)}
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        try:
            entities = []
            
            if self.nlp_model:
                # Use spaCy for entity extraction
                doc = self.nlp_model(text)
                
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'description': spacy.explain(ent.label_),
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.9  # spaCy doesn't provide confidence scores directly
                    })
            
            elif NLP_AVAILABLE:
                # Fallback to NLTK
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                current_entity = []
                current_label = None
                
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        if current_entity and current_label:
                            entities.append({
                                'text': ' '.join(current_entity),
                                'label': current_label,
                                'description': current_label,
                                'confidence': 0.7
                            })
                        current_entity = [leaf[0] for leaf in chunk.leaves()]
                        current_label = chunk.label()
                    else:
                        if current_entity and current_label:
                            entities.append({
                                'text': ' '.join(current_entity),
                                'label': current_label,
                                'description': current_label,
                                'confidence': 0.7
                            })
                            current_entity = []
                            current_label = None
                
                # Add last entity if exists
                if current_entity and current_label:
                    entities.append({
                        'text': ' '.join(current_entity),
                        'label': current_label,
                        'description': current_label,
                        'confidence': 0.7
                    })
            
            # Add custom entity patterns
            custom_entities = await self.extract_custom_entities(text)
            entities.extend(custom_entities)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    async def extract_custom_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract custom entities using regex patterns"""
        try:
            custom_entities = []
            
            # Email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            for match in re.finditer(email_pattern, text):
                custom_entities.append({
                    'text': match.group(),
                    'label': 'EMAIL',
                    'description': 'Email address',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.95
                })
            
            # URLs
            url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w.*))?)?'
            for match in re.finditer(url_pattern, text):
                custom_entities.append({
                    'text': match.group(),
                    'label': 'URL',
                    'description': 'Website URL',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.95
                })
            
            # Phone numbers
            phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
            for match in re.finditer(phone_pattern, text):
                custom_entities.append({
                    'text': match.group(),
                    'label': 'PHONE',
                    'description': 'Phone number',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.90
                })
            
            # File paths
            file_pattern = r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*|/(?:[^/\0]+/)*[^/\0]*'
            for match in re.finditer(file_pattern, text):
                if len(match.group()) > 5:  # Filter out short matches
                    custom_entities.append({
                        'text': match.group(),
                        'label': 'FILE_PATH',
                        'description': 'File path',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.80
                    })
            
            return custom_entities
            
        except Exception as e:
            logger.error(f"Error extracting custom entities: {e}")
            return []
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the text"""
        try:
            if not NLP_AVAILABLE or not self.sentiment_analyzer:
                return await self.basic_sentiment_analysis(text)
            
            # Use VADER sentiment analyzer
            scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Determine sentiment category
            compound_score = scores['compound']
            
            if compound_score >= self.sentiment_thresholds['very_positive']:
                sentiment_label = 'very_positive'
            elif compound_score >= self.sentiment_thresholds['positive']:
                sentiment_label = 'positive'
            elif compound_score > self.sentiment_thresholds['neutral']:
                sentiment_label = 'neutral'
            elif compound_score > self.sentiment_thresholds['negative']:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'very_negative'
            
            return {
                'sentiment': sentiment_label,
                'confidence': abs(compound_score),
                'scores': {
                    'compound': compound_score,
                    'positive': scores['pos'],
                    'neutral': scores['neu'],
                    'negative': scores['neg']
                },
                'emotional_indicators': await self.extract_emotional_indicators(text)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return await self.basic_sentiment_analysis(text)
    
    async def basic_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Basic sentiment analysis using word lists"""
        try:
            positive_words = [
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied'
            ]
            
            negative_words = [
                'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
                'angry', 'frustrated', 'disappointed', 'upset', 'annoyed'
            ]
            
            words = text.lower().split()
            
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            total_sentiment_words = positive_count + negative_count
            
            if total_sentiment_words == 0:
                sentiment = 'neutral'
                confidence = 0.5
            elif positive_count > negative_count:
                sentiment = 'positive'
                confidence = positive_count / total_sentiment_words
            elif negative_count > positive_count:
                sentiment = 'negative'
                confidence = negative_count / total_sentiment_words
            else:
                sentiment = 'neutral'
                confidence = 0.5
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': {
                    'positive_words': positive_count,
                    'negative_words': negative_count,
                    'total_words': len(words)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in basic sentiment analysis: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0, 'error': str(e)}
    
    async def extract_emotional_indicators(self, text: str) -> List[str]:
        """Extract emotional indicators from text"""
        try:
            emotion_patterns = {
                'excitement': [r'!{2,}', r'amazing', r'fantastic', r'incredible'],
                'urgency': [r'urgent', r'asap', r'immediately', r'quickly'],
                'frustration': [r'frustrated', r'annoying', r'problem', r'issue'],
                'confusion': [r'\?{2,}', r'confused', r'not sure', r'unclear'],
                'satisfaction': [r'perfect', r'exactly', r'great job', r'well done']
            }
            
            indicators = []
            text_lower = text.lower()
            
            for emotion, patterns in emotion_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        indicators.append(emotion)
                        break
            
            return list(set(indicators))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting emotional indicators: {e}")
            return []
    
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[Dict[str, Any]]:
        """Extract important keywords from text"""
        try:
            if not text.strip():
                return []
            
            # Tokenize and clean text
            if NLP_AVAILABLE:
                tokens = word_tokenize(text.lower())
            else:
                tokens = text.lower().split()
            
            # Remove punctuation and stop words
            tokens = [token for token in tokens 
                     if token not in string.punctuation 
                     and token not in self.stop_words 
                     and len(token) > 2]
            
            # Lemmatize if available
            if self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            # Count frequency
            word_freq = Counter(tokens)
            
            # Get top keywords
            top_keywords = word_freq.most_common(max_keywords)
            
            keywords = []
            total_words = len(tokens)
            
            for word, frequency in top_keywords:
                keywords.append({
                    'word': word,
                    'frequency': frequency,
                    'importance': frequency / total_words,
                    'category': await self.categorize_keyword(word)
                })
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    async def categorize_keyword(self, word: str) -> str:
        """Categorize a keyword by type"""
        try:
            categories = {
                'action': ['create', 'delete', 'update', 'run', 'execute', 'start', 'stop'],
                'object': ['file', 'folder', 'document', 'image', 'video', 'data'],
                'attribute': ['big', 'small', 'fast', 'slow', 'new', 'old', 'good', 'bad'],
                'time': ['today', 'tomorrow', 'yesterday', 'now', 'later', 'before'],
                'location': ['here', 'there', 'local', 'remote', 'cloud', 'server']
            }
            
            for category, words in categories.items():
                if word in words:
                    return category
            
            return 'general'
            
        except Exception as e:
            logger.error(f"Error categorizing keyword: {e}")
            return 'general'
    
    async def assess_complexity(self, text: str) -> Dict[str, Any]:
        """Assess the complexity of the input text"""
        try:
            words = text.split()
            sentences = sent_tokenize(text) if NLP_AVAILABLE else text.split('.')
            
            # Basic metrics
            word_count = len(words)
            sentence_count = len(sentences)
            avg_words_per_sentence = word_count / max(sentence_count, 1)
            
            # Vocabulary complexity
            unique_words = set(word.lower() for word in words if word.isalpha())
            vocabulary_diversity = len(unique_words) / max(word_count, 1)
            
            # Sentence length variation
            sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
            length_variance = np.var(sentence_lengths) if sentence_lengths else 0
            
            # Determine complexity level
            complexity_score = 0
            
            if avg_words_per_sentence > 15:
                complexity_score += 1
            if vocabulary_diversity > 0.7:
                complexity_score += 1
            if length_variance > 25:
                complexity_score += 1
            if word_count > 50:
                complexity_score += 1
            
            if complexity_score >= 3:
                complexity_level = 'high'
            elif complexity_score >= 2:
                complexity_level = 'medium'
            else:
                complexity_level = 'low'
            
            return {
                'complexity_level': complexity_level,
                'complexity_score': complexity_score,
                'metrics': {
                    'word_count': word_count,
                    'sentence_count': sentence_count,
                    'avg_words_per_sentence': round(avg_words_per_sentence, 2),
                    'vocabulary_diversity': round(vocabulary_diversity, 2),
                    'length_variance': round(length_variance, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error assessing complexity: {e}")
            return {'complexity_level': 'unknown', 'error': str(e)}
    
    async def extract_context_clues(self, text: str) -> Dict[str, Any]:
        """Extract context clues from the text"""
        try:
            context_clues = {
                'temporal': [],
                'spatial': [],
                'conditional': [],
                'causal': [],
                'comparative': []
            }
            
            # Temporal context
            temporal_patterns = [
                r'\b(now|today|tomorrow|yesterday|next|last|soon|later)\b',
                r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
                r'\b(\d{1,2}:\d{2})\b'  # Time format
            ]
            
            for pattern in temporal_patterns:
                matches = re.findall(pattern, text.lower())
                context_clues['temporal'].extend(matches)
            
            # Spatial context
            spatial_patterns = [
                r'\b(here|there|above|below|left|right|near|far|inside|outside)\b',
                r'\b(local|remote|cloud|server|desktop|mobile)\b'
            ]
            
            for pattern in spatial_patterns:
                matches = re.findall(pattern, text.lower())
                context_clues['spatial'].extend(matches)
            
            # Conditional context
            conditional_patterns = [
                r'\b(if|when|unless|provided|assuming)\b',
                r'\b(in case|depending on)\b'
            ]
            
            for pattern in conditional_patterns:
                matches = re.findall(pattern, text.lower())
                context_clues['conditional'].extend(matches)
            
            # Remove duplicates
            for key in context_clues:
                context_clues[key] = list(set(context_clues[key]))
            
            return context_clues
            
        except Exception as e:
            logger.error(f"Error extracting context clues: {e}")
            return {}
    
    async def extract_action_items(self, text: str) -> List[Dict[str, Any]]:
        """Extract actionable items from text"""
        try:
            action_items = []
            
            # Action verb patterns
            action_patterns = [
                r'\b(create|make|build|generate)\s+([^.!?]*)',
                r'\b(delete|remove|clear)\s+([^.!?]*)',
                r'\b(update|modify|change|edit)\s+([^.!?]*)',
                r'\b(run|execute|start|launch)\s+([^.!?]*)',
                r'\b(stop|pause|halt|end)\s+([^.!?]*)',
                r'\b(find|search|locate|look for)\s+([^.!?]*)',
                r'\b(send|email|message)\s+([^.!?]*)',
                r'\b(schedule|plan|set up)\s+([^.!?]*)'
            ]
            
            for pattern in action_patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    verb = match.group(1)
                    object_text = match.group(2).strip()
                    
                    action_items.append({
                        'action': verb,
                        'object': object_text,
                        'full_match': match.group(0),
                        'priority': await self.assess_action_priority(verb, object_text)
                    })
            
            return action_items
            
        except Exception as e:
            logger.error(f"Error extracting action items: {e}")
            return []
    
    async def assess_action_priority(self, verb: str, object_text: str) -> str:
        """Assess the priority of an action item"""
        try:
            high_priority_verbs = ['delete', 'remove', 'stop', 'halt']
            urgent_keywords = ['urgent', 'immediately', 'asap', 'critical']
            
            if verb in high_priority_verbs:
                return 'high'
            
            if any(keyword in object_text.lower() for keyword in urgent_keywords):
                return 'high'
            
            return 'medium'
            
        except Exception as e:
            logger.error(f"Error assessing action priority: {e}")
            return 'medium'
    
    async def extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract parameters and values from text"""
        try:
            parameters = {}
            
            # Key-value patterns
            kv_patterns = [
                r'(\w+)\s*[:=]\s*([^,\s]+)',  # key: value or key = value
                r'--(\w+)\s+([^-\s]+)',       # --flag value
                r'-(\w)\s+([^-\s]+)',         # -f value
                r'(\w+)\s+is\s+([^,\s]+)',    # key is value
                r'set\s+(\w+)\s+to\s+([^,\s]+)'  # set key to value
            ]
            
            for pattern in kv_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    key = match.group(1).lower()
                    value = match.group(2).strip()
                    
                    # Type conversion
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '').isdigit():
                        value = float(value)
                    elif value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    
                    parameters[key] = value
            
            return parameters
            
        except Exception as e:
            logger.error(f"Error extracting parameters: {e}")
            return {}
    
    async def generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate a summary of the input text"""
        try:
            if not text.strip():
                return ""
            
            sentences = sent_tokenize(text) if NLP_AVAILABLE else text.split('.')
            
            if len(sentences) <= max_sentences:
                return text
            
            # Simple extractive summarization
            sentence_scores = {}
            keywords = await self.extract_keywords(text, max_keywords=15)
            keyword_set = {kw['word'] for kw in keywords}
            
            for i, sentence in enumerate(sentences):
                sentence_words = set(word.lower() for word in sentence.split() if word.isalpha())
                score = len(sentence_words.intersection(keyword_set))
                sentence_scores[i] = score
            
            # Get top sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
            top_sentences.sort(key=lambda x: x[0])  # Sort by original order
            
            summary_sentences = [sentences[i] for i, _ in top_sentences]
            return ' '.join(summary_sentences)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return text[:200] + "..." if len(text) > 200 else text
    
    async def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get information about NLP processing capabilities"""
        return {
            'nlp_available': NLP_AVAILABLE,
            'spacy_model': self.nlp_model is not None,
            'sentiment_analysis': self.sentiment_analyzer is not None,
            'entity_extraction': True,
            'intent_classification': True,
            'keyword_extraction': True,
            'language_detection': True,
            'text_summarization': True,
            'supported_languages': self.supported_languages,
            'intent_categories': list(self.intent_patterns.keys())
        }
    
    async def shutdown(self):
        """Shutdown NLP processor"""
        logger.info("Shutting down NLP Processor...")
        # Cleanup resources if needed
