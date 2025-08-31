"""
UI Automation System - Advanced user interface interaction and control
"""

import asyncio
import logging
import pyautogui
import time
import subprocess
import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import platform
import psutil

# Disable pyautogui failsafe for automation
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.1

logger = logging.getLogger(__name__)

class UIAutomation:
    """Advanced UI automation with intelligent interaction capabilities"""
    
    def __init__(self, computer_vision):
        self.computer_vision = computer_vision
        self.interaction_history = []
        self.window_cache = {}
        self.last_screenshot = None
        self.system_platform = platform.system().lower()
        
    async def initialize(self):
        """Initialize UI automation system"""
        try:
            logger.info("Initializing UI Automation system...")
            
            # Set PyAutoGUI settings
            pyautogui.PAUSE = 0.05  # Small pause between actions
            
            # Get screen size
            self.screen_width, self.screen_height = pyautogui.size()
            logger.info(f"Screen resolution: {self.screen_width}x{self.screen_height}")
            
            # Initialize window management
            await self.refresh_window_cache()
            
            logger.info("UI Automation system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize UI Automation: {e}")
            raise
    
    async def perform_action(self, action_params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a UI action based on parameters"""
        try:
            action_type = action_params.get('type', '')
            result = {'action': action_type, 'success': False, 'timestamp': datetime.now().isoformat()}
            
            if action_type == 'click':
                result = await self.click_action(action_params)
            elif action_type == 'type':
                result = await self.type_action(action_params)
            elif action_type == 'key_press':
                result = await self.key_press_action(action_params)
            elif action_type == 'drag':
                result = await self.drag_action(action_params)
            elif action_type == 'scroll':
                result = await self.scroll_action(action_params)
            elif action_type == 'window_control':
                result = await self.window_control_action(action_params)
            elif action_type == 'application_control':
                result = await self.application_control_action(action_params)
            elif action_type == 'find_and_click':
                result = await self.find_and_click_action(action_params)
            else:
                result['error'] = f'Unknown action type: {action_type}'
            
            # Record interaction
            self.interaction_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error performing UI action: {e}")
            return {'action': action_type, 'success': False, 'error': str(e)}
    
    async def click_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform click action"""
        try:
            # Get coordinates
            if 'coordinates' in params:
                x, y = params['coordinates']
            elif 'element' in params:
                element = params['element']
                x, y = await self.computer_vision.get_element_center(element)
            else:
                return {'action': 'click', 'success': False, 'error': 'No coordinates or element specified'}
            
            # Validate coordinates
            if not self.validate_coordinates(x, y):
                return {'action': 'click', 'success': False, 'error': 'Invalid coordinates'}
            
            # Perform click
            click_type = params.get('click_type', 'left')
            if click_type == 'left':
                pyautogui.click(x, y)
            elif click_type == 'right':
                pyautogui.rightClick(x, y)
            elif click_type == 'double':
                pyautogui.doubleClick(x, y)
            else:
                pyautogui.click(x, y)
            
            await asyncio.sleep(0.1)  # Small delay after click
            
            return {
                'action': 'click',
                'success': True,
                'coordinates': (x, y),
                'click_type': click_type
            }
            
        except Exception as e:
            logger.error(f"Error in click action: {e}")
            return {'action': 'click', 'success': False, 'error': str(e)}
    
    async def type_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform typing action"""
        try:
            text = params.get('text', '')
            if not text:
                return {'action': 'type', 'success': False, 'error': 'No text specified'}
            
            # Optional: click on element first
            if 'element' in params:
                element = params['element']
                x, y = await self.computer_vision.get_element_center(element)
                pyautogui.click(x, y)
                await asyncio.sleep(0.1)
            
            # Clear existing text if specified
            if params.get('clear_first', False):
                pyautogui.hotkey('ctrl', 'a')
                await asyncio.sleep(0.05)
            
            # Type text
            typing_speed = params.get('speed', 0.05)
            for char in text:
                pyautogui.write(char)
                await asyncio.sleep(typing_speed)
            
            return {
                'action': 'type',
                'success': True,
                'text': text,
                'length': len(text)
            }
            
        except Exception as e:
            logger.error(f"Error in type action: {e}")
            return {'action': 'type', 'success': False, 'error': str(e)}
    
    async def key_press_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform key press action"""
        try:
            keys = params.get('keys', [])
            if not keys:
                return {'action': 'key_press', 'success': False, 'error': 'No keys specified'}
            
            # Handle single key or key combination
            if len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)
            
            await asyncio.sleep(0.1)
            
            return {
                'action': 'key_press',
                'success': True,
                'keys': keys
            }
            
        except Exception as e:
            logger.error(f"Error in key press action: {e}")
            return {'action': 'key_press', 'success': False, 'error': str(e)}
    
    async def drag_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform drag action"""
        try:
            start_coords = params.get('start_coordinates')
            end_coords = params.get('end_coordinates')
            
            if not start_coords or not end_coords:
                return {'action': 'drag', 'success': False, 'error': 'Start and end coordinates required'}
            
            # Validate coordinates
            if not self.validate_coordinates(*start_coords) or not self.validate_coordinates(*end_coords):
                return {'action': 'drag', 'success': False, 'error': 'Invalid coordinates'}
            
            # Perform drag
            duration = params.get('duration', 0.5)
            pyautogui.drag(end_coords[0] - start_coords[0], end_coords[1] - start_coords[1], 
                          duration=duration, button='left')
            
            return {
                'action': 'drag',
                'success': True,
                'start_coordinates': start_coords,
                'end_coordinates': end_coords,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Error in drag action: {e}")
            return {'action': 'drag', 'success': False, 'error': str(e)}
    
    async def scroll_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform scroll action"""
        try:
            direction = params.get('direction', 'down')
            amount = params.get('amount', 3)
            coordinates = params.get('coordinates', (self.screen_width // 2, self.screen_height // 2))
            
            # Move to coordinates first
            pyautogui.moveTo(coordinates[0], coordinates[1])
            
            # Perform scroll
            if direction == 'up':
                pyautogui.scroll(amount)
            elif direction == 'down':
                pyautogui.scroll(-amount)
            elif direction == 'left':
                pyautogui.hscroll(-amount)
            elif direction == 'right':
                pyautogui.hscroll(amount)
            
            return {
                'action': 'scroll',
                'success': True,
                'direction': direction,
                'amount': amount,
                'coordinates': coordinates
            }
            
        except Exception as e:
            logger.error(f"Error in scroll action: {e}")
            return {'action': 'scroll', 'success': False, 'error': str(e)}
    
    async def window_control_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform window control actions"""
        try:
            control_type = params.get('control_type')
            window_title = params.get('window_title', '')
            
            if control_type == 'focus':
                result = await self.focus_window(window_title)
            elif control_type == 'minimize':
                result = await self.minimize_window(window_title)
            elif control_type == 'maximize':
                result = await self.maximize_window(window_title)
            elif control_type == 'close':
                result = await self.close_window(window_title)
            elif control_type == 'resize':
                result = await self.resize_window(window_title, params.get('width'), params.get('height'))
            elif control_type == 'move':
                result = await self.move_window(window_title, params.get('x'), params.get('y'))
            else:
                return {'action': 'window_control', 'success': False, 'error': f'Unknown control type: {control_type}'}
            
            return {
                'action': 'window_control',
                'control_type': control_type,
                'window_title': window_title,
                **result
            }
            
        except Exception as e:
            logger.error(f"Error in window control action: {e}")
            return {'action': 'window_control', 'success': False, 'error': str(e)}
    
    async def application_control_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform application control actions"""
        try:
            control_type = params.get('control_type')
            application = params.get('application', '')
            
            if control_type == 'launch':
                result = await self.launch_application(application, params.get('arguments', []))
            elif control_type == 'terminate':
                result = await self.terminate_application(application)
            elif control_type == 'restart':
                result = await self.restart_application(application)
            else:
                return {'action': 'application_control', 'success': False, 'error': f'Unknown control type: {control_type}'}
            
            return {
                'action': 'application_control',
                'control_type': control_type,
                'application': application,
                **result
            }
            
        except Exception as e:
            logger.error(f"Error in application control action: {e}")
            return {'action': 'application_control', 'success': False, 'error': str(e)}
    
    async def find_and_click_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Find an element and click on it"""
        try:
            # Take screenshot
            screenshot = await self.computer_vision.capture_screen()
            
            # Find element by text or image
            element = None
            if 'text' in params:
                element = await self.computer_vision.find_element_by_text(params['text'], screenshot)
            elif 'image_path' in params:
                element = await self.computer_vision.find_element_by_image(params['image_path'], screenshot)
            
            if not element:
                return {'action': 'find_and_click', 'success': False, 'error': 'Element not found'}
            
            # Click on the found element
            click_params = {
                'element': element,
                'click_type': params.get('click_type', 'left')
            }
            
            click_result = await self.click_action(click_params)
            
            return {
                'action': 'find_and_click',
                'success': click_result['success'],
                'element_found': True,
                'click_result': click_result
            }
            
        except Exception as e:
            logger.error(f"Error in find and click action: {e}")
            return {'action': 'find_and_click', 'success': False, 'error': str(e)}
    
    async def focus_window(self, window_title: str) -> Dict[str, Any]:
        """Focus on a specific window"""
        try:
            if self.system_platform == 'windows':
                # Use Windows-specific window focusing
                import win32gui
                import win32con
                
                def enum_windows_callback(hwnd, windows):
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        if window_title.lower() in title.lower():
                            windows.append(hwnd)
                
                windows = []
                win32gui.EnumWindows(enum_windows_callback, windows)
                
                if windows:
                    hwnd = windows[0]
                    win32gui.SetForegroundWindow(hwnd)
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    return {'success': True}
                else:
                    return {'success': False, 'error': 'Window not found'}
                    
            else:
                # Generic approach using alt+tab
                pyautogui.hotkey('alt', 'tab')
                await asyncio.sleep(0.5)
                return {'success': True}
                
        except Exception as e:
            logger.error(f"Error focusing window: {e}")
            return {'success': False, 'error': str(e)}
    
    async def launch_application(self, application: str, arguments: List[str] = None) -> Dict[str, Any]:
        """Launch an application"""
        try:
            if arguments is None:
                arguments = []
            
            # Build command
            command = [application] + arguments
            
            # Launch application
            process = subprocess.Popen(command, shell=True if self.system_platform == 'windows' else False)
            
            # Wait a moment for the application to start
            await asyncio.sleep(2)
            
            return {
                'success': True,
                'process_id': process.pid,
                'command': ' '.join(command)
            }
            
        except Exception as e:
            logger.error(f"Error launching application {application}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def terminate_application(self, application: str) -> Dict[str, Any]:
        """Terminate an application"""
        try:
            terminated_count = 0
            
            # Find processes by name
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    if application.lower() in proc.info['name'].lower():
                        proc.terminate()
                        terminated_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            if terminated_count > 0:
                # Wait for processes to terminate
                await asyncio.sleep(2)
                return {'success': True, 'terminated_processes': terminated_count}
            else:
                return {'success': False, 'error': 'No matching processes found'}
                
        except Exception as e:
            logger.error(f"Error terminating application {application}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def minimize_window(self, window_title: str) -> Dict[str, Any]:
        """Minimize a window"""
        try:
            # Focus the window first
            focus_result = await self.focus_window(window_title)
            if focus_result['success']:
                pyautogui.hotkey('alt', 'f9')  # Windows minimize shortcut
                return {'success': True}
            else:
                return focus_result
                
        except Exception as e:
            logger.error(f"Error minimizing window: {e}")
            return {'success': False, 'error': str(e)}
    
    async def maximize_window(self, window_title: str) -> Dict[str, Any]:
        """Maximize a window"""
        try:
            # Focus the window first
            focus_result = await self.focus_window(window_title)
            if focus_result['success']:
                pyautogui.hotkey('alt', 'f10')  # Windows maximize shortcut
                return {'success': True}
            else:
                return focus_result
                
        except Exception as e:
            logger.error(f"Error maximizing window: {e}")
            return {'success': False, 'error': str(e)}
    
    async def close_window(self, window_title: str) -> Dict[str, Any]:
        """Close a window"""
        try:
            # Focus the window first
            focus_result = await self.focus_window(window_title)
            if focus_result['success']:
                pyautogui.hotkey('alt', 'f4')  # Windows close shortcut
                return {'success': True}
            else:
                return focus_result
                
        except Exception as e:
            logger.error(f"Error closing window: {e}")
            return {'success': False, 'error': str(e)}
    
    def validate_coordinates(self, x: int, y: int) -> bool:
        """Validate that coordinates are within screen bounds"""
        return 0 <= x <= self.screen_width and 0 <= y <= self.screen_height
    
    async def refresh_window_cache(self):
        """Refresh the cache of available windows"""
        try:
            self.window_cache = {}
            
            if self.system_platform == 'windows':
                try:
                    import win32gui
                    
                    def enum_windows_callback(hwnd, windows):
                        if win32gui.IsWindowVisible(hwnd):
                            title = win32gui.GetWindowText(hwnd)
                            if title:
                                windows[title] = hwnd
                    
                    win32gui.EnumWindows(enum_windows_callback, self.window_cache)
                except ImportError:
                    logger.warning("win32gui not available, window management limited")
            
            logger.info(f"Found {len(self.window_cache)} windows")
            
        except Exception as e:
            logger.error(f"Error refreshing window cache: {e}")
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        return pyautogui.position()
    
    async def move_mouse(self, x: int, y: int, duration: float = 0.5) -> bool:
        """Move mouse to specific coordinates"""
        try:
            if self.validate_coordinates(x, y):
                pyautogui.moveTo(x, y, duration=duration)
                return True
            return False
        except Exception as e:
            logger.error(f"Error moving mouse: {e}")
            return False
    
    async def take_action_screenshot(self, action_type: str) -> str:
        """Take a screenshot after performing an action for debugging"""
        try:
            screenshot = await self.computer_vision.capture_screen()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"temp/action_{action_type}_{timestamp}.png"
            
            # Ensure temp directory exists
            os.makedirs("temp", exist_ok=True)
            
            import cv2
            cv2.imwrite(filename, screenshot)
            return filename
            
        except Exception as e:
            logger.error(f"Error taking action screenshot: {e}")
            return ""
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of UI automation system"""
        try:
            # Test basic functionality
            mouse_pos = self.get_mouse_position()
            screen_size = (self.screen_width, self.screen_height)
            
            return {
                "healthy": True,
                "screen_size": screen_size,
                "mouse_position": mouse_pos,
                "interactions_performed": len(self.interaction_history),
                "windows_cached": len(self.window_cache)
            }
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def shutdown(self):
        """Shutdown UI automation system"""
        logger.info("Shutting down UI Automation system...")
        
        # Save interaction history
        try:
            import json
            with open("data/ui_interaction_history.json", "w") as f:
                json.dump(self.interaction_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving interaction history: {e}")
