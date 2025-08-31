"""
Social Media Handler - Comprehensive social media automation and management
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import aiohttp
import re
from urllib.parse import urljoin
import os

logger = logging.getLogger(__name__)

class SocialMediaHandler:
    """Advanced social media automation and management system"""
    
    def __init__(self, llm_manager, settings):
        self.llm_manager = llm_manager
        self.settings = settings
        self.platforms = {}
        self.scheduled_posts = []
        self.content_queue = []
        self.analytics_data = {}
        self.auto_responses = {}
        
    async def initialize(self):
        """Initialize social media handler"""
        try:
            logger.info("Initializing Social Media Handler...")
            
            # Initialize platform handlers
            await self.init_twitter()
            await self.init_facebook()
            await self.init_instagram()
            await self.init_linkedin()
            await self.init_reddit()
            await self.init_discord()
            
            # Load scheduled content
            await self.load_scheduled_content()
            
            # Start monitoring tasks
            asyncio.create_task(self.monitor_mentions())
            asyncio.create_task(self.process_scheduled_posts())
            
            logger.info("Social Media Handler initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Social Media Handler: {e}")
            raise
    
    async def init_twitter(self):
        """Initialize Twitter/X integration"""
        try:
            api_key = os.getenv("TWITTER_API_KEY")
            api_secret = os.getenv("TWITTER_API_SECRET")
            access_token = os.getenv("TWITTER_ACCESS_TOKEN")
            access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
            
            if all([api_key, api_secret, access_token, access_token_secret]):
                self.platforms['twitter'] = {
                    'enabled': True,
                    'api_key': api_key,
                    'api_secret': api_secret,
                    'access_token': access_token,
                    'access_token_secret': access_token_secret,
                    'rate_limits': {'tweets': 300, 'likes': 1000, 'follows': 400},
                    'last_action': datetime.now()
                }
                logger.info("Twitter integration enabled")
            else:
                logger.warning("Twitter credentials not found")
                
        except Exception as e:
            logger.error(f"Error initializing Twitter: {e}")
    
    async def init_facebook(self):
        """Initialize Facebook integration"""
        try:
            access_token = os.getenv("FACEBOOK_ACCESS_TOKEN")
            page_id = os.getenv("FACEBOOK_PAGE_ID")
            
            if access_token and page_id:
                self.platforms['facebook'] = {
                    'enabled': True,
                    'access_token': access_token,
                    'page_id': page_id,
                    'api_base': 'https://graph.facebook.com/v18.0',
                    'rate_limits': {'posts': 200, 'comments': 2000},
                    'last_action': datetime.now()
                }
                logger.info("Facebook integration enabled")
            else:
                logger.warning("Facebook credentials not found")
                
        except Exception as e:
            logger.error(f"Error initializing Facebook: {e}")
    
    async def init_instagram(self):
        """Initialize Instagram integration"""
        try:
            access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
            account_id = os.getenv("INSTAGRAM_ACCOUNT_ID")
            
            if access_token and account_id:
                self.platforms['instagram'] = {
                    'enabled': True,
                    'access_token': access_token,
                    'account_id': account_id,
                    'api_base': 'https://graph.instagram.com',
                    'rate_limits': {'posts': 25, 'stories': 100},
                    'last_action': datetime.now()
                }
                logger.info("Instagram integration enabled")
            else:
                logger.warning("Instagram credentials not found")
                
        except Exception as e:
            logger.error(f"Error initializing Instagram: {e}")
    
    async def init_linkedin(self):
        """Initialize LinkedIn integration"""
        try:
            access_token = os.getenv("LINKEDIN_ACCESS_TOKEN")
            
            if access_token:
                self.platforms['linkedin'] = {
                    'enabled': True,
                    'access_token': access_token,
                    'api_base': 'https://api.linkedin.com/v2',
                    'rate_limits': {'posts': 100, 'messages': 300},
                    'last_action': datetime.now()
                }
                logger.info("LinkedIn integration enabled")
            else:
                logger.warning("LinkedIn credentials not found")
                
        except Exception as e:
            logger.error(f"Error initializing LinkedIn: {e}")
    
    async def init_reddit(self):
        """Initialize Reddit integration"""
        try:
            client_id = os.getenv("REDDIT_CLIENT_ID")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET")
            username = os.getenv("REDDIT_USERNAME")
            password = os.getenv("REDDIT_PASSWORD")
            
            if all([client_id, client_secret, username, password]):
                self.platforms['reddit'] = {
                    'enabled': True,
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'username': username,
                    'password': password,
                    'user_agent': 'AI_Agent/1.0',
                    'rate_limits': {'posts': 60, 'comments': 600},
                    'last_action': datetime.now()
                }
                logger.info("Reddit integration enabled")
            else:
                logger.warning("Reddit credentials not found")
                
        except Exception as e:
            logger.error(f"Error initializing Reddit: {e}")
    
    async def init_discord(self):
        """Initialize Discord integration"""
        try:
            bot_token = os.getenv("DISCORD_BOT_TOKEN")
            
            if bot_token:
                self.platforms['discord'] = {
                    'enabled': True,
                    'bot_token': bot_token,
                    'api_base': 'https://discord.com/api/v10',
                    'rate_limits': {'messages': 50, 'reactions': 300},
                    'last_action': datetime.now()
                }
                logger.info("Discord integration enabled")
            else:
                logger.warning("Discord credentials not found")
                
        except Exception as e:
            logger.error(f"Error initializing Discord: {e}")
    
    async def handle_action(self, action_params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle social media actions"""
        try:
            action_type = action_params.get('action_type')
            platform = action_params.get('platform', 'all')
            
            result = {'action': action_type, 'platform': platform, 'success': False}
            
            if action_type == 'post_content':
                result = await self.post_content(action_params)
            elif action_type == 'schedule_post':
                result = await self.schedule_post(action_params)
            elif action_type == 'respond_to_mentions':
                result = await self.respond_to_mentions(action_params)
            elif action_type == 'analyze_engagement':
                result = await self.analyze_engagement(action_params)
            elif action_type == 'auto_follow':
                result = await self.auto_follow(action_params)
            elif action_type == 'content_curation':
                result = await self.curate_content(action_params)
            elif action_type == 'hashtag_research':
                result = await self.research_hashtags(action_params)
            elif action_type == 'competitor_analysis':
                result = await self.analyze_competitors(action_params)
            else:
                result['error'] = f'Unknown action type: {action_type}'
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling social media action: {e}")
            return {'action': action_type, 'success': False, 'error': str(e)}
    
    async def post_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Post content to social media platforms"""
        try:
            content = params.get('content', '')
            platforms = params.get('platforms', ['twitter'])
            media_urls = params.get('media_urls', [])
            hashtags = params.get('hashtags', [])
            
            if not content:
                return {'success': False, 'error': 'No content provided'}
            
            # Add hashtags to content
            if hashtags:
                content += ' ' + ' '.join(f'#{tag}' for tag in hashtags)
            
            results = {}
            
            for platform in platforms:
                if platform in self.platforms and self.platforms[platform]['enabled']:
                    try:
                        if platform == 'twitter':
                            result = await self.post_to_twitter(content, media_urls)
                        elif platform == 'facebook':
                            result = await self.post_to_facebook(content, media_urls)
                        elif platform == 'instagram':
                            result = await self.post_to_instagram(content, media_urls)
                        elif platform == 'linkedin':
                            result = await self.post_to_linkedin(content)
                        elif platform == 'reddit':
                            result = await self.post_to_reddit(content, params.get('subreddit', 'test'))
                        elif platform == 'discord':
                            result = await self.post_to_discord(content, params.get('channel_id'))
                        else:
                            result = {'success': False, 'error': f'Platform {platform} not supported'}
                        
                        results[platform] = result
                        
                        # Rate limiting
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        results[platform] = {'success': False, 'error': str(e)}
                else:
                    results[platform] = {'success': False, 'error': 'Platform not enabled'}
            
            return {
                'success': any(r.get('success', False) for r in results.values()),
                'platforms': results,
                'content': content[:100] + '...' if len(content) > 100 else content
            }
            
        except Exception as e:
            logger.error(f"Error posting content: {e}")
            return {'success': False, 'error': str(e)}
    
    async def post_to_twitter(self, content: str, media_urls: List[str] = None) -> Dict[str, Any]:
        """Post content to Twitter/X"""
        try:
            # This would use Twitter API v2
            # For now, returning a simulated response
            platform_info = self.platforms['twitter']
            
            # Check rate limits
            if not await self.check_rate_limit('twitter', 'tweets'):
                return {'success': False, 'error': 'Rate limit exceeded'}
            
            # Simulate API call
            post_data = {
                'text': content[:280],  # Twitter character limit
                'timestamp': datetime.now().isoformat()
            }
            
            # Update rate limit tracking
            platform_info['last_action'] = datetime.now()
            
            logger.info(f"Posted to Twitter: {content[:50]}...")
            return {'success': True, 'post_id': f'tweet_{int(time.time())}', 'data': post_data}
            
        except Exception as e:
            logger.error(f"Error posting to Twitter: {e}")
            return {'success': False, 'error': str(e)}
    
    async def post_to_facebook(self, content: str, media_urls: List[str] = None) -> Dict[str, Any]:
        """Post content to Facebook"""
        try:
            platform_info = self.platforms['facebook']
            
            if not await self.check_rate_limit('facebook', 'posts'):
                return {'success': False, 'error': 'Rate limit exceeded'}
            
            # Simulate Facebook Graph API call
            post_data = {
                'message': content,
                'timestamp': datetime.now().isoformat()
            }
            
            platform_info['last_action'] = datetime.now()
            
            logger.info(f"Posted to Facebook: {content[:50]}...")
            return {'success': True, 'post_id': f'fb_{int(time.time())}', 'data': post_data}
            
        except Exception as e:
            logger.error(f"Error posting to Facebook: {e}")
            return {'success': False, 'error': str(e)}
    
    async def post_to_instagram(self, content: str, media_urls: List[str] = None) -> Dict[str, Any]:
        """Post content to Instagram"""
        try:
            platform_info = self.platforms['instagram']
            
            if not await self.check_rate_limit('instagram', 'posts'):
                return {'success': False, 'error': 'Rate limit exceeded'}
            
            # Instagram requires media for posts
            if not media_urls:
                return {'success': False, 'error': 'Instagram posts require media'}
            
            post_data = {
                'caption': content,
                'media_url': media_urls[0] if media_urls else None,
                'timestamp': datetime.now().isoformat()
            }
            
            platform_info['last_action'] = datetime.now()
            
            logger.info(f"Posted to Instagram: {content[:50]}...")
            return {'success': True, 'post_id': f'ig_{int(time.time())}', 'data': post_data}
            
        except Exception as e:
            logger.error(f"Error posting to Instagram: {e}")
            return {'success': False, 'error': str(e)}
    
    async def post_to_linkedin(self, content: str) -> Dict[str, Any]:
        """Post content to LinkedIn"""
        try:
            platform_info = self.platforms['linkedin']
            
            if not await self.check_rate_limit('linkedin', 'posts'):
                return {'success': False, 'error': 'Rate limit exceeded'}
            
            post_data = {
                'text': content,
                'timestamp': datetime.now().isoformat()
            }
            
            platform_info['last_action'] = datetime.now()
            
            logger.info(f"Posted to LinkedIn: {content[:50]}...")
            return {'success': True, 'post_id': f'li_{int(time.time())}', 'data': post_data}
            
        except Exception as e:
            logger.error(f"Error posting to LinkedIn: {e}")
            return {'success': False, 'error': str(e)}
    
    async def post_to_reddit(self, content: str, subreddit: str) -> Dict[str, Any]:
        """Post content to Reddit"""
        try:
            platform_info = self.platforms['reddit']
            
            if not await self.check_rate_limit('reddit', 'posts'):
                return {'success': False, 'error': 'Rate limit exceeded'}
            
            post_data = {
                'title': content[:300],  # Reddit title limit
                'subreddit': subreddit,
                'timestamp': datetime.now().isoformat()
            }
            
            platform_info['last_action'] = datetime.now()
            
            logger.info(f"Posted to Reddit r/{subreddit}: {content[:50]}...")
            return {'success': True, 'post_id': f'reddit_{int(time.time())}', 'data': post_data}
            
        except Exception as e:
            logger.error(f"Error posting to Reddit: {e}")
            return {'success': False, 'error': str(e)}
    
    async def post_to_discord(self, content: str, channel_id: str) -> Dict[str, Any]:
        """Post content to Discord"""
        try:
            platform_info = self.platforms['discord']
            
            if not await self.check_rate_limit('discord', 'messages'):
                return {'success': False, 'error': 'Rate limit exceeded'}
            
            if not channel_id:
                return {'success': False, 'error': 'Channel ID required for Discord'}
            
            post_data = {
                'content': content,
                'channel_id': channel_id,
                'timestamp': datetime.now().isoformat()
            }
            
            platform_info['last_action'] = datetime.now()
            
            logger.info(f"Posted to Discord channel {channel_id}: {content[:50]}...")
            return {'success': True, 'message_id': f'discord_{int(time.time())}', 'data': post_data}
            
        except Exception as e:
            logger.error(f"Error posting to Discord: {e}")
            return {'success': False, 'error': str(e)}
    
    async def schedule_post(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a post for later"""
        try:
            content = params.get('content', '')
            platforms = params.get('platforms', ['twitter'])
            schedule_time = params.get('schedule_time')
            
            if not content or not schedule_time:
                return {'success': False, 'error': 'Content and schedule time required'}
            
            # Parse schedule time
            if isinstance(schedule_time, str):
                schedule_time = datetime.fromisoformat(schedule_time)
            
            scheduled_post = {
                'id': f'scheduled_{int(time.time())}',
                'content': content,
                'platforms': platforms,
                'schedule_time': schedule_time.isoformat(),
                'status': 'scheduled',
                'created_at': datetime.now().isoformat()
            }
            
            self.scheduled_posts.append(scheduled_post)
            
            # Save to persistent storage
            await self.save_scheduled_content()
            
            return {
                'success': True,
                'scheduled_post_id': scheduled_post['id'],
                'schedule_time': scheduled_post['schedule_time']
            }
            
        except Exception as e:
            logger.error(f"Error scheduling post: {e}")
            return {'success': False, 'error': str(e)}
    
    async def process_scheduled_posts(self):
        """Process scheduled posts that are due"""
        while True:
            try:
                current_time = datetime.now()
                
                # Find posts that are due
                due_posts = [
                    post for post in self.scheduled_posts
                    if (post['status'] == 'scheduled' and 
                        datetime.fromisoformat(post['schedule_time']) <= current_time)
                ]
                
                for post in due_posts:
                    try:
                        # Post the content
                        result = await self.post_content({
                            'content': post['content'],
                            'platforms': post['platforms']
                        })
                        
                        # Update post status
                        post['status'] = 'posted' if result['success'] else 'failed'
                        post['posted_at'] = current_time.isoformat()
                        post['result'] = result
                        
                        logger.info(f"Processed scheduled post: {post['id']}")
                        
                    except Exception as e:
                        post['status'] = 'failed'
                        post['error'] = str(e)
                        logger.error(f"Error processing scheduled post {post['id']}: {e}")
                
                # Save updated schedule
                if due_posts:
                    await self.save_scheduled_content()
                
                # Wait before checking again
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scheduled posts processor: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def monitor_mentions(self):
        """Monitor social media for mentions and interactions"""
        while True:
            try:
                for platform_name, platform_info in self.platforms.items():
                    if platform_info.get('enabled', False):
                        await self.check_mentions(platform_name)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring mentions: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    async def check_mentions(self, platform: str):
        """Check for mentions on a specific platform"""
        try:
            # This would implement actual API calls to check for mentions
            # For now, we'll simulate the process
            
            mentions = []  # Would fetch from API
            
            for mention in mentions:
                # Process mention with AI
                response = await self.generate_auto_response(mention, platform)
                
                if response and response.get('should_respond', False):
                    await self.respond_to_mention(mention, response['content'], platform)
            
        except Exception as e:
            logger.error(f"Error checking mentions on {platform}: {e}")
    
    async def generate_auto_response(self, mention: Dict, platform: str) -> Dict[str, Any]:
        """Generate automatic response to mentions using AI"""
        try:
            prompt = f"""
            You received a mention on {platform}:
            
            User: {mention.get('user', 'Unknown')}
            Content: {mention.get('content', '')}
            Context: {mention.get('context', '')}
            
            Generate an appropriate response that is:
            1. Helpful and engaging
            2. Platform-appropriate
            3. Professional but friendly
            4. Under 280 characters for Twitter, longer for other platforms
            
            Also determine if we should respond at all based on the content.
            
            Respond with JSON:
            {{
                "should_respond": true/false,
                "content": "response text",
                "tone": "professional/casual/friendly",
                "confidence": 0.0-1.0
            }}
            """
            
            # Use LLM to generate response
            ai_response = await self.llm_manager.generate_response(prompt)
            
            try:
                parsed_response = json.loads(ai_response)
                return parsed_response
            except json.JSONDecodeError:
                # Fallback if AI doesn't return valid JSON
                return {
                    'should_respond': False,
                    'content': '',
                    'confidence': 0.0
                }
            
        except Exception as e:
            logger.error(f"Error generating auto response: {e}")
            return {'should_respond': False, 'error': str(e)}
    
    async def respond_to_mention(self, mention: Dict, response_content: str, platform: str):
        """Respond to a mention"""
        try:
            # Implement platform-specific response logic
            if platform == 'twitter':
                # Reply to tweet
                pass
            elif platform == 'facebook':
                # Comment on post
                pass
            # Add other platforms as needed
            
            logger.info(f"Responded to mention on {platform}: {response_content[:50]}...")
            
        except Exception as e:
            logger.error(f"Error responding to mention on {platform}: {e}")
    
    async def check_rate_limit(self, platform: str, action_type: str) -> bool:
        """Check if action is within rate limits"""
        try:
            platform_info = self.platforms.get(platform, {})
            rate_limits = platform_info.get('rate_limits', {})
            limit = rate_limits.get(action_type, 100)
            
            # Simple rate limiting - in production, this would be more sophisticated
            last_action = platform_info.get('last_action', datetime.now() - timedelta(hours=1))
            time_diff = datetime.now() - last_action
            
            # Reset if it's been more than an hour
            if time_diff > timedelta(hours=1):
                platform_info['action_count'] = 0
            
            current_count = platform_info.get('action_count', 0)
            
            if current_count < limit:
                platform_info['action_count'] = current_count + 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False
    
    async def analyze_engagement(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social media engagement"""
        try:
            platform = params.get('platform', 'all')
            time_period = params.get('time_period', '7d')
            
            analytics = {
                'platform': platform,
                'time_period': time_period,
                'metrics': {},
                'insights': []
            }
            
            # This would fetch real analytics data from platform APIs
            # For now, we'll provide a structure
            
            if platform == 'all' or platform == 'twitter':
                analytics['metrics']['twitter'] = {
                    'followers': 1000,
                    'tweets': 50,
                    'engagement_rate': 0.05,
                    'impressions': 10000,
                    'clicks': 500
                }
            
            # Add insights using AI
            insights_prompt = f"""
            Analyze these social media metrics and provide insights:
            {json.dumps(analytics['metrics'], indent=2)}
            
            Provide 3-5 actionable insights for improving performance.
            """
            
            ai_insights = await self.llm_manager.generate_response(insights_prompt)
            analytics['insights'] = ai_insights.split('\n') if ai_insights else []
            
            return {'success': True, 'analytics': analytics}
            
        except Exception as e:
            logger.error(f"Error analyzing engagement: {e}")
            return {'success': False, 'error': str(e)}
    
    async def curate_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Curate content for social media"""
        try:
            topic = params.get('topic', 'technology')
            platform = params.get('platform', 'twitter')
            count = params.get('count', 5)
            
            # Use AI to generate content ideas
            curation_prompt = f"""
            Generate {count} engaging social media post ideas for {platform} about {topic}.
            
            Each post should include:
            1. Main content (platform-appropriate length)
            2. Suggested hashtags
            3. Best posting time
            4. Engagement strategy
            
            Make them diverse, engaging, and valuable to the audience.
            """
            
            ai_content = await self.llm_manager.generate_response(curation_prompt)
            
            # Parse and structure the response
            curated_content = {
                'topic': topic,
                'platform': platform,
                'generated_at': datetime.now().isoformat(),
                'content_ideas': ai_content,
                'count': count
            }
            
            return {'success': True, 'curated_content': curated_content}
            
        except Exception as e:
            logger.error(f"Error curating content: {e}")
            return {'success': False, 'error': str(e)}
    
    async def research_hashtags(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Research hashtags for better reach"""
        try:
            topic = params.get('topic', 'technology')
            platform = params.get('platform', 'twitter')
            
            # Use AI to suggest relevant hashtags
            hashtag_prompt = f"""
            Suggest 10-15 relevant hashtags for {platform} posts about {topic}.
            
            Include:
            1. Popular hashtags (high reach)
            2. Niche hashtags (targeted audience)
            3. Trending hashtags (current relevance)
            
            Format as a JSON list with engagement estimates.
            """
            
            ai_hashtags = await self.llm_manager.generate_response(hashtag_prompt)
            
            hashtag_research = {
                'topic': topic,
                'platform': platform,
                'generated_at': datetime.now().isoformat(),
                'hashtags': ai_hashtags
            }
            
            return {'success': True, 'hashtag_research': hashtag_research}
            
        except Exception as e:
            logger.error(f"Error researching hashtags: {e}")
            return {'success': False, 'error': str(e)}
    
    async def analyze_competitors(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitor social media presence"""
        try:
            competitors = params.get('competitors', [])
            platform = params.get('platform', 'twitter')
            
            if not competitors:
                return {'success': False, 'error': 'No competitors specified'}
            
            analysis = {
                'platform': platform,
                'competitors': competitors,
                'analyzed_at': datetime.now().isoformat(),
                'insights': {}
            }
            
            # This would analyze competitor profiles and content
            # For now, we'll provide a structure
            for competitor in competitors:
                analysis['insights'][competitor] = {
                    'followers': 'N/A',
                    'posting_frequency': 'N/A',
                    'engagement_rate': 'N/A',
                    'content_themes': 'N/A',
                    'posting_times': 'N/A'
                }
            
            return {'success': True, 'competitor_analysis': analysis}
            
        except Exception as e:
            logger.error(f"Error analyzing competitors: {e}")
            return {'success': False, 'error': str(e)}
    
    async def save_scheduled_content(self):
        """Save scheduled content to persistent storage"""
        try:
            with open('data/scheduled_posts.json', 'w') as f:
                json.dump(self.scheduled_posts, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving scheduled content: {e}")
    
    async def load_scheduled_content(self):
        """Load scheduled content from persistent storage"""
        try:
            try:
                with open('data/scheduled_posts.json', 'r') as f:
                    self.scheduled_posts = json.load(f)
            except FileNotFoundError:
                self.scheduled_posts = []
        except Exception as e:
            logger.error(f"Error loading scheduled content: {e}")
    
    def get_platform_status(self) -> Dict[str, Any]:
        """Get status of all social media platforms"""
        return {
            platform: {
                'enabled': info.get('enabled', False),
                'last_action': info.get('last_action', 'Never').isoformat() if isinstance(info.get('last_action'), datetime) else info.get('last_action', 'Never'),
                'rate_limit_status': 'OK' if info.get('enabled', False) else 'Disabled'
            }
            for platform, info in self.platforms.items()
        }
    
    async def shutdown(self):
        """Shutdown social media handler"""
        logger.info("Shutting down Social Media Handler...")
        await self.save_scheduled_content()
