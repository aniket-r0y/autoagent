"""
Browser Controller - Advanced web browser automation and control
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
import os

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

logger = logging.getLogger(__name__)

class BrowserController:
    """Advanced browser automation and control system"""
    
    def __init__(self, ui_automation, settings):
        self.ui_automation = ui_automation
        self.settings = settings
        self.drivers = {}
        self.active_sessions = {}
        self.page_cache = {}
        self.automation_scripts = {}
        
    async def initialize(self):
        """Initialize browser controller"""
        try:
            logger.info("Initializing Browser Controller...")
            
            if not SELENIUM_AVAILABLE:
                logger.warning("Selenium not available, browser automation limited")
                return
            
            # Initialize default browser session
            await self.create_browser_session('default')
            
            # Load automation scripts
            await self.load_automation_scripts()
            
            logger.info("Browser Controller initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Browser Controller: {e}")
            # Don't raise here as browser automation is optional
    
    async def create_browser_session(self, session_id: str, browser_type: str = 'chrome', headless: bool = False) -> bool:
        """Create a new browser session"""
        try:
            if not SELENIUM_AVAILABLE:
                logger.warning("Cannot create browser session: Selenium not available")
                return False
            
            # Configure browser options
            options = self.get_browser_options(browser_type, headless)
            
            # Create driver
            if browser_type.lower() == 'chrome':
                driver = webdriver.Chrome(options=options)
            elif browser_type.lower() == 'firefox':
                driver = webdriver.Firefox(options=options)
            else:
                logger.error(f"Unsupported browser type: {browser_type}")
                return False
            
            # Set timeouts
            driver.implicitly_wait(10)
            driver.set_page_load_timeout(30)
            
            # Store session
            self.drivers[session_id] = driver
            self.active_sessions[session_id] = {
                'browser_type': browser_type,
                'headless': headless,
                'created_at': datetime.now().isoformat(),
                'current_url': 'about:blank',
                'page_title': '',
                'tabs': [{'id': 0, 'url': 'about:blank', 'title': 'New Tab'}]
            }
            
            logger.info(f"Created browser session: {session_id} ({browser_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error creating browser session {session_id}: {e}")
            return False
    
    def get_browser_options(self, browser_type: str, headless: bool = False):
        """Get browser options configuration"""
        if browser_type.lower() == 'chrome':
            options = Options()
            if headless:
                options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            # Privacy and security options
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-plugins')
            options.add_argument('--disable-images')  # For faster loading
            
            return options
        
        # Add other browser options as needed
        return None
    
    async def execute_action(self, action_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute browser action"""
        try:
            action_type = action_params.get('action_type')
            session_id = action_params.get('session_id', 'default')
            
            result = {'action': action_type, 'session_id': session_id, 'success': False}
            
            if not SELENIUM_AVAILABLE:
                result['error'] = 'Selenium not available'
                return result
            
            # Ensure session exists
            if session_id not in self.drivers:
                if not await self.create_browser_session(session_id):
                    result['error'] = 'Failed to create browser session'
                    return result
            
            driver = self.drivers[session_id]
            
            if action_type == 'navigate':
                result = await self.navigate_to_url(driver, session_id, action_params)
            elif action_type == 'find_element':
                result = await self.find_element(driver, action_params)
            elif action_type == 'click_element':
                result = await self.click_element(driver, action_params)
            elif action_type == 'input_text':
                result = await self.input_text(driver, action_params)
            elif action_type == 'scroll':
                result = await self.scroll_page(driver, action_params)
            elif action_type == 'screenshot':
                result = await self.take_screenshot(driver, session_id)
            elif action_type == 'extract_data':
                result = await self.extract_page_data(driver, action_params)
            elif action_type == 'execute_script':
                result = await self.execute_javascript(driver, action_params)
            elif action_type == 'manage_tabs':
                result = await self.manage_tabs(driver, session_id, action_params)
            elif action_type == 'form_fill':
                result = await self.fill_form(driver, action_params)
            elif action_type == 'wait_for_element':
                result = await self.wait_for_element(driver, action_params)
            else:
                result['error'] = f'Unknown action type: {action_type}'
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing browser action: {e}")
            return {'action': action_type, 'success': False, 'error': str(e)}
    
    async def navigate_to_url(self, driver, session_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Navigate to a URL"""
        try:
            url = params.get('url', '')
            if not url:
                return {'success': False, 'error': 'No URL provided'}
            
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            driver.get(url)
            
            # Wait for page to load
            await asyncio.sleep(2)
            
            # Update session info
            self.active_sessions[session_id]['current_url'] = driver.current_url
            self.active_sessions[session_id]['page_title'] = driver.title
            
            return {
                'success': True,
                'url': driver.current_url,
                'title': driver.title,
                'load_time': 2.0  # Simplified
            }
            
        except Exception as e:
            logger.error(f"Error navigating to URL: {e}")
            return {'success': False, 'error': str(e)}
    
    async def find_element(self, driver, params: Dict[str, Any]) -> Dict[str, Any]:
        """Find element on page"""
        try:
            selector_type = params.get('selector_type', 'css')
            selector = params.get('selector', '')
            
            if not selector:
                return {'success': False, 'error': 'No selector provided'}
            
            # Map selector types
            by_map = {
                'css': By.CSS_SELECTOR,
                'xpath': By.XPATH,
                'id': By.ID,
                'class': By.CLASS_NAME,
                'tag': By.TAG_NAME,
                'name': By.NAME,
                'text': By.LINK_TEXT,
                'partial_text': By.PARTIAL_LINK_TEXT
            }
            
            by_type = by_map.get(selector_type, By.CSS_SELECTOR)
            
            # Find element
            element = driver.find_element(by_type, selector)
            
            if element:
                return {
                    'success': True,
                    'found': True,
                    'tag_name': element.tag_name,
                    'text': element.text[:100],  # First 100 chars
                    'location': element.location,
                    'size': element.size,
                    'displayed': element.is_displayed(),
                    'enabled': element.is_enabled()
                }
            else:
                return {'success': True, 'found': False}
                
        except NoSuchElementException:
            return {'success': True, 'found': False}
        except Exception as e:
            logger.error(f"Error finding element: {e}")
            return {'success': False, 'error': str(e)}
    
    async def click_element(self, driver, params: Dict[str, Any]) -> Dict[str, Any]:
        """Click on an element"""
        try:
            selector_type = params.get('selector_type', 'css')
            selector = params.get('selector', '')
            
            if not selector:
                return {'success': False, 'error': 'No selector provided'}
            
            by_map = {
                'css': By.CSS_SELECTOR,
                'xpath': By.XPATH,
                'id': By.ID,
                'class': By.CLASS_NAME,
                'tag': By.TAG_NAME,
                'name': By.NAME
            }
            
            by_type = by_map.get(selector_type, By.CSS_SELECTOR)
            
            # Wait for element to be clickable
            wait = WebDriverWait(driver, 10)
            element = wait.until(EC.element_to_be_clickable((by_type, selector)))
            
            # Scroll to element if needed
            driver.execute_script("arguments[0].scrollIntoView();", element)
            
            # Click element
            element.click()
            
            return {
                'success': True,
                'clicked': True,
                'element_text': element.text[:50]
            }
            
        except TimeoutException:
            return {'success': False, 'error': 'Element not found or not clickable'}
        except Exception as e:
            logger.error(f"Error clicking element: {e}")
            return {'success': False, 'error': str(e)}
    
    async def input_text(self, driver, params: Dict[str, Any]) -> Dict[str, Any]:
        """Input text into an element"""
        try:
            selector_type = params.get('selector_type', 'css')
            selector = params.get('selector', '')
            text = params.get('text', '')
            clear_first = params.get('clear_first', True)
            
            if not selector or not text:
                return {'success': False, 'error': 'Selector and text required'}
            
            by_map = {
                'css': By.CSS_SELECTOR,
                'xpath': By.XPATH,
                'id': By.ID,
                'class': By.CLASS_NAME,
                'name': By.NAME
            }
            
            by_type = by_map.get(selector_type, By.CSS_SELECTOR)
            
            # Wait for element
            wait = WebDriverWait(driver, 10)
            element = wait.until(EC.presence_of_element_located((by_type, selector)))
            
            # Clear existing text if requested
            if clear_first:
                element.clear()
            
            # Input text
            element.send_keys(text)
            
            return {
                'success': True,
                'text_entered': text,
                'element_value': element.get_attribute('value')
            }
            
        except TimeoutException:
            return {'success': False, 'error': 'Element not found'}
        except Exception as e:
            logger.error(f"Error inputting text: {e}")
            return {'success': False, 'error': str(e)}
    
    async def scroll_page(self, driver, params: Dict[str, Any]) -> Dict[str, Any]:
        """Scroll the page"""
        try:
            direction = params.get('direction', 'down')
            amount = params.get('amount', 3)
            
            if direction == 'down':
                driver.execute_script(f"window.scrollBy(0, {amount * 100});")
            elif direction == 'up':
                driver.execute_script(f"window.scrollBy(0, {-amount * 100});")
            elif direction == 'top':
                driver.execute_script("window.scrollTo(0, 0);")
            elif direction == 'bottom':
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            return {
                'success': True,
                'direction': direction,
                'amount': amount
            }
            
        except Exception as e:
            logger.error(f"Error scrolling page: {e}")
            return {'success': False, 'error': str(e)}
    
    async def take_screenshot(self, driver, session_id: str) -> Dict[str, Any]:
        """Take a screenshot of the current page"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"temp/screenshot_{session_id}_{timestamp}.png"
            
            # Ensure temp directory exists
            os.makedirs("temp", exist_ok=True)
            
            # Take screenshot
            driver.save_screenshot(filename)
            
            return {
                'success': True,
                'screenshot_path': filename,
                'timestamp': timestamp,
                'page_url': driver.current_url,
                'page_title': driver.title
            }
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return {'success': False, 'error': str(e)}
    
    async def extract_page_data(self, driver, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from the current page"""
        try:
            data_type = params.get('data_type', 'all')
            selectors = params.get('selectors', {})
            
            extracted_data = {
                'url': driver.current_url,
                'title': driver.title,
                'timestamp': datetime.now().isoformat()
            }
            
            if data_type == 'all' or data_type == 'text':
                # Extract all text content
                extracted_data['text_content'] = driver.find_element(By.TAG_NAME, 'body').text
            
            if data_type == 'all' or data_type == 'links':
                # Extract all links
                links = driver.find_elements(By.TAG_NAME, 'a')
                extracted_data['links'] = [
                    {'text': link.text, 'href': link.get_attribute('href')}
                    for link in links if link.get_attribute('href')
                ][:50]  # Limit to 50 links
            
            if data_type == 'all' or data_type == 'images':
                # Extract all images
                images = driver.find_elements(By.TAG_NAME, 'img')
                extracted_data['images'] = [
                    {'alt': img.get_attribute('alt'), 'src': img.get_attribute('src')}
                    for img in images if img.get_attribute('src')
                ][:20]  # Limit to 20 images
            
            if data_type == 'all' or data_type == 'forms':
                # Extract form information
                forms = driver.find_elements(By.TAG_NAME, 'form')
                extracted_data['forms'] = []
                
                for form in forms:
                    inputs = form.find_elements(By.TAG_NAME, 'input')
                    form_data = {
                        'action': form.get_attribute('action'),
                        'method': form.get_attribute('method'),
                        'inputs': [
                            {
                                'type': inp.get_attribute('type'),
                                'name': inp.get_attribute('name'),
                                'placeholder': inp.get_attribute('placeholder')
                            }
                            for inp in inputs
                        ]
                    }
                    extracted_data['forms'].append(form_data)
            
            # Extract custom selectors
            if selectors:
                extracted_data['custom'] = {}
                for name, selector in selectors.items():
                    try:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        extracted_data['custom'][name] = [
                            {'text': elem.text, 'html': elem.get_attribute('outerHTML')[:200]}
                            for elem in elements[:10]  # Limit to 10 elements
                        ]
                    except Exception as e:
                        extracted_data['custom'][name] = f"Error: {e}"
            
            return {
                'success': True,
                'extracted_data': extracted_data
            }
            
        except Exception as e:
            logger.error(f"Error extracting page data: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_javascript(self, driver, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute JavaScript on the page"""
        try:
            script = params.get('script', '')
            args = params.get('args', [])
            
            if not script:
                return {'success': False, 'error': 'No script provided'}
            
            # Execute script
            result = driver.execute_script(script, *args)
            
            return {
                'success': True,
                'result': result,
                'script_executed': script[:100] + '...' if len(script) > 100 else script
            }
            
        except Exception as e:
            logger.error(f"Error executing JavaScript: {e}")
            return {'success': False, 'error': str(e)}
    
    async def fill_form(self, driver, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fill out a form automatically"""
        try:
            form_data = params.get('form_data', {})
            submit = params.get('submit', False)
            
            if not form_data:
                return {'success': False, 'error': 'No form data provided'}
            
            filled_fields = []
            
            for field_name, value in form_data.items():
                try:
                    # Try different selector strategies
                    selectors = [
                        f"input[name='{field_name}']",
                        f"input[id='{field_name}']",
                        f"textarea[name='{field_name}']",
                        f"select[name='{field_name}']"
                    ]
                    
                    element = None
                    for selector in selectors:
                        try:
                            element = driver.find_element(By.CSS_SELECTOR, selector)
                            break
                        except NoSuchElementException:
                            continue
                    
                    if element:
                        if element.tag_name == 'select':
                            # Handle select elements
                            from selenium.webdriver.support.ui import Select
                            select = Select(element)
                            select.select_by_visible_text(str(value))
                        else:
                            # Handle input and textarea elements
                            element.clear()
                            element.send_keys(str(value))
                        
                        filled_fields.append(field_name)
                    
                except Exception as e:
                    logger.warning(f"Could not fill field {field_name}: {e}")
            
            # Submit form if requested
            if submit and filled_fields:
                try:
                    submit_button = driver.find_element(By.CSS_SELECTOR, "input[type='submit'], button[type='submit'], button")
                    submit_button.click()
                except Exception as e:
                    logger.warning(f"Could not submit form: {e}")
            
            return {
                'success': True,
                'filled_fields': filled_fields,
                'submitted': submit and len(filled_fields) > 0
            }
            
        except Exception as e:
            logger.error(f"Error filling form: {e}")
            return {'success': False, 'error': str(e)}
    
    async def wait_for_element(self, driver, params: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for an element to appear"""
        try:
            selector_type = params.get('selector_type', 'css')
            selector = params.get('selector', '')
            timeout = params.get('timeout', 10)
            condition = params.get('condition', 'presence')
            
            if not selector:
                return {'success': False, 'error': 'No selector provided'}
            
            by_map = {
                'css': By.CSS_SELECTOR,
                'xpath': By.XPATH,
                'id': By.ID,
                'class': By.CLASS_NAME,
                'tag': By.TAG_NAME,
                'name': By.NAME
            }
            
            by_type = by_map.get(selector_type, By.CSS_SELECTOR)
            
            wait = WebDriverWait(driver, timeout)
            
            # Different wait conditions
            if condition == 'presence':
                element = wait.until(EC.presence_of_element_located((by_type, selector)))
            elif condition == 'visible':
                element = wait.until(EC.visibility_of_element_located((by_type, selector)))
            elif condition == 'clickable':
                element = wait.until(EC.element_to_be_clickable((by_type, selector)))
            else:
                element = wait.until(EC.presence_of_element_located((by_type, selector)))
            
            return {
                'success': True,
                'element_found': True,
                'condition_met': condition,
                'wait_time': timeout
            }
            
        except TimeoutException:
            return {'success': False, 'error': f'Element not found within {timeout} seconds'}
        except Exception as e:
            logger.error(f"Error waiting for element: {e}")
            return {'success': False, 'error': str(e)}
    
    async def manage_tabs(self, driver, session_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Manage browser tabs"""
        try:
            action = params.get('action', 'list')
            
            if action == 'list':
                # List all tabs
                handles = driver.window_handles
                current_handle = driver.current_window_handle
                
                tabs = []
                for i, handle in enumerate(handles):
                    driver.switch_to.window(handle)
                    tabs.append({
                        'id': i,
                        'handle': handle,
                        'title': driver.title,
                        'url': driver.current_url,
                        'active': handle == current_handle
                    })
                
                # Switch back to original tab
                driver.switch_to.window(current_handle)
                
                return {'success': True, 'tabs': tabs}
                
            elif action == 'new':
                # Open new tab
                driver.execute_script("window.open('');")
                driver.switch_to.window(driver.window_handles[-1])
                
                return {'success': True, 'new_tab_created': True}
                
            elif action == 'close':
                # Close current tab
                if len(driver.window_handles) > 1:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    return {'success': True, 'tab_closed': True}
                else:
                    return {'success': False, 'error': 'Cannot close last tab'}
                    
            elif action == 'switch':
                # Switch to specific tab
                tab_index = params.get('tab_index', 0)
                handles = driver.window_handles
                
                if 0 <= tab_index < len(handles):
                    driver.switch_to.window(handles[tab_index])
                    return {'success': True, 'switched_to_tab': tab_index}
                else:
                    return {'success': False, 'error': 'Invalid tab index'}
            
            return {'success': False, 'error': f'Unknown tab action: {action}'}
            
        except Exception as e:
            logger.error(f"Error managing tabs: {e}")
            return {'success': False, 'error': str(e)}
    
    async def load_automation_scripts(self):
        """Load pre-defined automation scripts"""
        try:
            scripts_dir = Path("data/browser_scripts")
            if scripts_dir.exists():
                for script_file in scripts_dir.glob("*.json"):
                    try:
                        with open(script_file, 'r') as f:
                            script = json.load(f)
                            self.automation_scripts[script_file.stem] = script
                    except Exception as e:
                        logger.warning(f"Failed to load script {script_file}: {e}")
            
            logger.info(f"Loaded {len(self.automation_scripts)} automation scripts")
            
        except Exception as e:
            logger.error(f"Error loading automation scripts: {e}")
    
    def get_session_info(self, session_id: str = None) -> Dict[str, Any]:
        """Get information about browser sessions"""
        if session_id:
            return self.active_sessions.get(session_id, {})
        else:
            return {
                'active_sessions': len(self.active_sessions),
                'sessions': self.active_sessions
            }
    
    async def close_session(self, session_id: str) -> bool:
        """Close a browser session"""
        try:
            if session_id in self.drivers:
                self.drivers[session_id].quit()
                del self.drivers[session_id]
                del self.active_sessions[session_id]
                logger.info(f"Closed browser session: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown browser controller"""
        logger.info("Shutting down Browser Controller...")
        
        # Close all browser sessions
        for session_id in list(self.drivers.keys()):
            await self.close_session(session_id)
