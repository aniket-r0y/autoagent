"""
Computer Vision System - Advanced visual processing and UI element detection
"""

import asyncio
import logging
import numpy as np
import cv2
import os
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image, ImageGrab
import pytesseract
import threading

logger = logging.getLogger(__name__)

class ComputerVision:
    """Advanced computer vision for UI automation and visual processing"""
    
    def __init__(self, settings):
        self.settings = settings
        self.screen_monitor = None
        self.template_cache = {}
        self.ocr_cache = {}
        self.ui_elements_cache = {}
        self.is_monitoring = False
        
    async def initialize(self):
        """Initialize computer vision system"""
        try:
            logger.info("Initializing Computer Vision system...")
            
            # Initialize OCR
            self.initialize_ocr()
            
            # Load UI element templates
            await self.load_ui_templates()
            
            # Initialize screen monitoring
            await self.start_screen_monitoring()
            
            logger.info("Computer Vision system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Computer Vision: {e}")
            raise
    
    def initialize_ocr(self):
        """Initialize OCR capabilities"""
        try:
            # Test OCR installation
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            pytesseract.image_to_string(test_image)
            logger.info("OCR initialized successfully")
        except Exception as e:
            logger.warning(f"OCR initialization failed: {e}")
    
    async def load_ui_templates(self):
        """Load UI element templates for recognition"""
        try:
            template_dir = "templates/ui_elements"
            if os.path.exists(template_dir):
                for filename in os.listdir(template_dir):
                    if filename.endswith(('.png', '.jpg', '.jpeg')):
                        template_path = os.path.join(template_dir, filename)
                        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                        if template is not None:
                            self.template_cache[filename.split('.')[0]] = template
                            
            logger.info(f"Loaded {len(self.template_cache)} UI templates")
            
        except Exception as e:
            logger.warning(f"Error loading UI templates: {e}")
    
    async def start_screen_monitoring(self):
        """Start continuous screen monitoring"""
        self.is_monitoring = True
        logger.info("Screen monitoring started")
    
    async def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture screenshot of entire screen or specific region"""
        try:
            if region:
                # Capture specific region (x, y, width, height)
                screenshot = ImageGrab.grab(bbox=region)
            else:
                # Capture entire screen
                screenshot = ImageGrab.grab()
            
            # Convert to OpenCV format
            screenshot_np = np.array(screenshot)
            screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            
            return screenshot_cv
            
        except Exception as e:
            logger.error(f"Error capturing screen: {e}")
            return np.array([])
    
    async def find_ui_elements(self, screenshot: np.ndarray, element_types: List[str] = None) -> List[Dict]:
        """Find UI elements in screenshot using various methods"""
        try:
            elements = []
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            # Find buttons using template matching
            buttons = await self.find_buttons(gray)
            elements.extend(buttons)
            
            # Find text elements using OCR
            text_elements = await self.find_text_elements(screenshot)
            elements.extend(text_elements)
            
            # Find input fields
            input_fields = await self.find_input_fields(gray)
            elements.extend(input_fields)
            
            # Find clickable elements using contour detection
            clickable = await self.find_clickable_elements(gray)
            elements.extend(clickable)
            
            return elements
            
        except Exception as e:
            logger.error(f"Error finding UI elements: {e}")
            return []
    
    async def find_buttons(self, gray_image: np.ndarray) -> List[Dict]:
        """Find button elements using template matching and contour detection"""
        try:
            buttons = []
            
            # Template matching for common button types
            for template_name, template in self.template_cache.items():
                if 'button' in template_name.lower():
                    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= 0.8)
                    
                    for pt in zip(*locations[::-1]):
                        buttons.append({
                            'type': 'button',
                            'template': template_name,
                            'position': pt,
                            'size': template.shape[::-1],
                            'confidence': float(result[pt[1], pt[0]])
                        })
            
            # Contour-based button detection
            contour_buttons = await self.detect_button_contours(gray_image)
            buttons.extend(contour_buttons)
            
            return buttons
            
        except Exception as e:
            logger.error(f"Error finding buttons: {e}")
            return []
    
    async def detect_button_contours(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect button-like elements using contour analysis"""
        try:
            buttons = []
            
            # Edge detection
            edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter contours by area and aspect ratio
                area = cv2.contourArea(contour)
                if 500 < area < 10000:  # Reasonable button size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Button-like aspect ratios
                    if 0.3 < aspect_ratio < 5.0:
                        buttons.append({
                            'type': 'button',
                            'method': 'contour',
                            'position': (x, y),
                            'size': (w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
            
            return buttons
            
        except Exception as e:
            logger.error(f"Error detecting button contours: {e}")
            return []
    
    async def find_text_elements(self, screenshot: np.ndarray) -> List[Dict]:
        """Find text elements using OCR"""
        try:
            text_elements = []
            
            # Convert to PIL Image for OCR
            pil_image = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
            
            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                conf = int(ocr_data['conf'][i])
                
                if text and conf > 30:  # Filter out low-confidence text
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    text_elements.append({
                        'type': 'text',
                        'text': text,
                        'position': (x, y),
                        'size': (w, h),
                        'confidence': conf / 100.0,
                        'word_num': ocr_data['word_num'][i],
                        'block_num': ocr_data['block_num'][i]
                    })
            
            return text_elements
            
        except Exception as e:
            logger.error(f"Error finding text elements: {e}")
            return []
    
    async def find_input_fields(self, gray_image: np.ndarray) -> List[Dict]:
        """Find input fields and text boxes"""
        try:
            input_fields = []
            
            # Template matching for input fields
            for template_name, template in self.template_cache.items():
                if 'input' in template_name.lower() or 'textbox' in template_name.lower():
                    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= 0.7)
                    
                    for pt in zip(*locations[::-1]):
                        input_fields.append({
                            'type': 'input',
                            'template': template_name,
                            'position': pt,
                            'size': template.shape[::-1],
                            'confidence': float(result[pt[1], pt[0]])
                        })
            
            # Detect rectangular input-like shapes
            rectangles = await self.detect_input_rectangles(gray_image)
            input_fields.extend(rectangles)
            
            return input_fields
            
        except Exception as e:
            logger.error(f"Error finding input fields: {e}")
            return []
    
    async def detect_input_rectangles(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect input field rectangles"""
        try:
            input_fields = []
            
            # Use morphological operations to find rectangular shapes
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            processed = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:  # Input field size range
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Input fields are typically wider than tall
                    if aspect_ratio > 1.5:
                        input_fields.append({
                            'type': 'input',
                            'method': 'rectangle',
                            'position': (x, y),
                            'size': (w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
            
            return input_fields
            
        except Exception as e:
            logger.error(f"Error detecting input rectangles: {e}")
            return []
    
    async def find_clickable_elements(self, gray_image: np.ndarray) -> List[Dict]:
        """Find potentially clickable elements"""
        try:
            clickable = []
            
            # Use corner detection to find interactive elements
            corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=100, qualityLevel=0.3, minDistance=10)
            
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel()
                    clickable.append({
                        'type': 'clickable',
                        'method': 'corner',
                        'position': (int(x), int(y)),
                        'size': (10, 10)  # Small clickable point
                    })
            
            return clickable
            
        except Exception as e:
            logger.error(f"Error finding clickable elements: {e}")
            return []
    
    async def find_element_by_text(self, text: str, screenshot: np.ndarray = None) -> Optional[Dict]:
        """Find UI element by text content"""
        try:
            if screenshot is None:
                screenshot = await self.capture_screen()
            
            text_elements = await self.find_text_elements(screenshot)
            
            # Search for exact match first
            for element in text_elements:
                if element['text'].lower() == text.lower():
                    return element
            
            # Search for partial match
            for element in text_elements:
                if text.lower() in element['text'].lower():
                    return element
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding element by text '{text}': {e}")
            return None
    
    async def find_element_by_image(self, template_path: str, screenshot: np.ndarray = None, threshold: float = 0.8) -> Optional[Dict]:
        """Find UI element by template image matching"""
        try:
            if screenshot is None:
                screenshot = await self.capture_screen()
            
            # Load template
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                logger.error(f"Could not load template image: {template_path}")
                return None
            
            # Convert screenshot to grayscale
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            # Perform template matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= threshold:
                return {
                    'type': 'template_match',
                    'position': max_loc,
                    'size': template.shape[::-1],
                    'confidence': max_val,
                    'template_path': template_path
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding element by image: {e}")
            return None
    
    async def analyze_ui_context(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Analyze the current UI context and identify the application/page"""
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'elements_found': 0,
                'application': 'unknown',
                'context_type': 'unknown',
                'interactable_elements': [],
                'text_content': []
            }
            
            # Find all UI elements
            elements = await self.find_ui_elements(screenshot)
            analysis['elements_found'] = len(elements)
            
            # Separate elements by type
            buttons = [e for e in elements if e['type'] == 'button']
            text_elements = [e for e in elements if e['type'] == 'text']
            input_fields = [e for e in elements if e['type'] == 'input']
            
            analysis['interactable_elements'] = buttons + input_fields
            analysis['text_content'] = [e['text'] for e in text_elements if 'text' in e]
            
            # Attempt to identify application/context
            if any('chrome' in text.lower() for text in analysis['text_content']):
                analysis['application'] = 'chrome'
                analysis['context_type'] = 'browser'
            elif any('firefox' in text.lower() for text in analysis['text_content']):
                analysis['application'] = 'firefox'
                analysis['context_type'] = 'browser'
            elif any('notepad' in text.lower() for text in analysis['text_content']):
                analysis['application'] = 'notepad'
                analysis['context_type'] = 'text_editor'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing UI context: {e}")
            return {'error': str(e)}
    
    async def get_element_center(self, element: Dict) -> Tuple[int, int]:
        """Get the center coordinates of a UI element"""
        try:
            x, y = element['position']
            w, h = element['size']
            center_x = x + w // 2
            center_y = y + h // 2
            return (center_x, center_y)
            
        except Exception as e:
            logger.error(f"Error getting element center: {e}")
            return (0, 0)
    
    async def highlight_elements(self, screenshot: np.ndarray, elements: List[Dict], output_path: str = None) -> np.ndarray:
        """Highlight found elements on screenshot for debugging"""
        try:
            highlighted = screenshot.copy()
            
            for element in elements:
                x, y = element['position']
                w, h = element['size']
                element_type = element['type']
                
                # Choose color based on element type
                if element_type == 'button':
                    color = (0, 255, 0)  # Green
                elif element_type == 'input':
                    color = (255, 0, 0)  # Blue
                elif element_type == 'text':
                    color = (0, 0, 255)  # Red
                else:
                    color = (255, 255, 0)  # Cyan
                
                # Draw rectangle
                cv2.rectangle(highlighted, (x, y), (x + w, y + h), color, 2)
                
                # Add label
                label = f"{element_type}"
                if 'confidence' in element:
                    label += f" ({element['confidence']:.2f})"
                
                cv2.putText(highlighted, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if output_path:
                cv2.imwrite(output_path, highlighted)
            
            return highlighted
            
        except Exception as e:
            logger.error(f"Error highlighting elements: {e}")
            return screenshot
    
    def screenshot_to_base64(self, screenshot: np.ndarray) -> str:
        """Convert screenshot to base64 string"""
        try:
            _, buffer = cv2.imencode('.png', screenshot)
            img_str = base64.b64encode(buffer).decode('utf-8')
            return img_str
        except Exception as e:
            logger.error(f"Error converting screenshot to base64: {e}")
            return ""
    
    async def monitor_screen_changes(self, callback, region: Optional[Tuple[int, int, int, int]] = None):
        """Monitor screen for changes and call callback when detected"""
        try:
            previous_screenshot = await self.capture_screen(region)
            
            while self.is_monitoring:
                current_screenshot = await self.capture_screen(region)
                
                # Compare screenshots
                diff = cv2.absdiff(previous_screenshot, current_screenshot)
                mean_diff = np.mean(diff)
                
                if mean_diff > 10:  # Threshold for significant change
                    await callback(current_screenshot, diff)
                    previous_screenshot = current_screenshot
                
                await asyncio.sleep(0.1)  # Check 10 times per second
                
        except Exception as e:
            logger.error(f"Error monitoring screen changes: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of computer vision system"""
        try:
            # Test screen capture
            screenshot = await self.capture_screen()
            screenshot_ok = screenshot.size > 0
            
            # Test OCR
            ocr_ok = True
            try:
                test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
                pytesseract.image_to_string(test_image)
            except:
                ocr_ok = False
            
            return {
                "healthy": screenshot_ok and ocr_ok,
                "screenshot_capture": screenshot_ok,
                "ocr_available": ocr_ok,
                "templates_loaded": len(self.template_cache),
                "monitoring_active": self.is_monitoring
            }
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def shutdown(self):
        """Shutdown computer vision system"""
        logger.info("Shutting down Computer Vision system...")
        self.is_monitoring = False
