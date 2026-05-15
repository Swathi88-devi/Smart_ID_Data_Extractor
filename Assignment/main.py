
"""
SMART ID DATA EXTRACTOR - MAIN PROCESSOR
==========================================
Processes ID card images to extract Name and Email with visual feedback.
Handles OCR noise, varying text positions, and multiple OCR engines.

Author: Assessment Solution
Date: 2026-05-15
"""

import os
import re
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import OCR libraries (with graceful fallback)
try:
    import easyocr
    OCR_AVAILABLE = {'easyocr': True}
except ImportError:
    OCR_AVAILABLE = {'easyocr': False}
    logger.warning("EasyOCR not installed. Install with: pip install easyocr")

try:
    import pytesseract
    OCR_AVAILABLE['tesseract'] = True
except ImportError:
    OCR_AVAILABLE['tesseract'] = False
    logger.warning("pytesseract not installed. Install with: pip install pytesseract")

try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE['paddle'] = True
except ImportError:
    OCR_AVAILABLE['paddle'] = False
    logger.warning("PaddleOCR not installed. Install with: pip install paddleocr")


class SmartIDExtractor:
    """
    Extracts name and email from ID card images using OCR and Computer Vision.
    
    Features:
    - Multi-stage image preprocessing for various lighting conditions
    - Multiple OCR engine support (EasyOCR, Tesseract, PaddleOCR)
    - Intelligent text parsing with regex and heuristics
    - Visual bounding box highlighting (GREEN for name, RED for email)
    - Confidence scoring and result validation
    """
    
    def __init__(self, ocr_engine='easyocr', debug=False):
        """
        Initialize the extractor with specified OCR engine.
        
        Args:
            ocr_engine (str): 'easyocr', 'tesseract', or 'paddle'
            debug (bool): Enable debug visualizations
        """
        self.ocr_engine = ocr_engine
        self.debug = debug
        self.reader = None
        self.image = None
        self.preprocessed = None
        self.results = {'name': None, 'email': None, 'raw_text': None}
        
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize the selected OCR engine."""
        if self.ocr_engine == 'easyocr' and OCR_AVAILABLE['easyocr']:
            logger.info("Initializing EasyOCR...")
            self.reader = easyocr.Reader(['en'], gpu=False)
        elif self.ocr_engine == 'tesseract' and OCR_AVAILABLE['tesseract']:
            logger.info("Using Tesseract OCR")
            self.reader = 'tesseract'
        elif self.ocr_engine == 'paddle' and OCR_AVAILABLE['paddle']:
            logger.info("Initializing PaddleOCR...")
            self.reader = PaddleOCR(use_angle_cls=True, lang='en')
        else:
            raise ValueError(f"OCR engine '{self.ocr_engine}' not available or not installed")
    
    # ============================================================================
    # STEP 1: IMAGE PREPROCESSING
    # ============================================================================
    
    def load_image(self, image_path: str) -> bool:
        """
        Load and validate image.
        
        Args:
            image_path (str): Path to ID card image
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return False
        
        self.image = cv2.imread(image_path)
        if self.image is None:
            logger.error(f"Failed to load image: {image_path}")
            return False
        
        logger.info(f"✓ Loaded image: {image_path}")
        logger.info(f"  Dimensions: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        return True
    
    def preprocess_image(self) -> np.ndarray:
        """
        Multi-stage image preprocessing pipeline.
        
        Stages:
        1. Grayscale conversion (reduce color complexity)
        2. Gaussian blur (smooth noise while preserving edges)
        3. Adaptive thresholding (handle varying lighting)
        4. Morphological dilation (reconnect text fragments)
        5. Denoising (remove salt-pepper noise)
        
        Returns:
            np.ndarray: Preprocessed image
            
        Technical Justification:
        - Grayscale: Reduces computational load, removes color noise
        - Gaussian Blur (5×5): Balances noise reduction vs text preservation
        - Adaptive Threshold (block=11): Adjusts per-neighborhood for lighting variation
        - Dilation (2×2, 1 iter): Connects broken characters without over-expansion
        - Non-Local Means: Preserves edges while removing noise
        """
        # Stage 1: Grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        logger.info("✓ Converted to grayscale")
        
        # Stage 2: Gaussian Blur (noise reduction)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        logger.info("✓ Applied Gaussian blur (5×5)")
        
        # Stage 3: Adaptive Thresholding (handles varying lighting)
        # blockSize must be odd and > 1
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,  # Window size for local threshold
            C=2            # Constant subtracted from mean
        )
        logger.info("✓ Applied adaptive thresholding")
        
        # Stage 4: Morphological Dilation (reconnect text)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(adaptive_thresh, kernel, iterations=1)
        logger.info("✓ Applied morphological dilation")
        
        # Stage 5: Non-Local Means Denoising (advanced noise removal)
        denoised = cv2.fastNlMeansDenoising(dilated, None, h=10, templateWindowSize=7, searchWindowSize=21)
        logger.info("✓ Applied non-local means denoising")
        
        self.preprocessed = denoised
        
        # Save debug images if enabled
        if self.debug:
            self._save_debug_images(gray, blurred, adaptive_thresh, dilated, denoised)
        
        return denoised
    
    def _save_debug_images(self, gray, blurred, thresh, dilated, denoised):
        """Save preprocessing stages for visual inspection."""
        os.makedirs('debug_output', exist_ok=True)
        cv2.imwrite('debug_output/01_grayscale.jpg', gray)
        cv2.imwrite('debug_output/02_blurred.jpg', blurred)
        cv2.imwrite('debug_output/03_threshold.jpg', thresh)
        cv2.imwrite('debug_output/04_dilated.jpg', dilated)
        cv2.imwrite('debug_output/05_denoised.jpg', denoised)
        logger.info("  Debug images saved to debug_output/")
    
    # ============================================================================
    # STEP 2: OPTICAL CHARACTER RECOGNITION (OCR)
    # ============================================================================
    
    def extract_text_easyocr(self) -> Tuple[str, List[Dict]]:
        """
        Extract text using EasyOCR (recommended for speed + accuracy).
        
        Returns:
            Tuple[str, List[Dict]]: (raw_text, bounding_boxes_with_text)
        """
        if self.reader is None:
            return "", []
        
        # Convert preprocessed (grayscale) to BGR for display overlay
        display_image = cv2.cvtColor(self.preprocessed, cv2.COLOR_GRAY2BGR)
        
        logger.info("Running EasyOCR...")
        results = self.reader.readtext(self.preprocessed)
        
        # Format results: (bbox, text, confidence)
        text_blocks = []
        raw_text = ""
        
        for detection in results:
            bbox = detection[0]
            text = detection[1]
            confidence = detection[2]
            
            if confidence > 0.1:  # Filter low confidence
                text_blocks.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
                raw_text += text + " "
        
        logger.info(f"✓ Extracted {len(text_blocks)} text blocks")
        return raw_text, text_blocks
    
    def extract_text_tesseract(self) -> Tuple[str, List[Dict]]:
        """
        Extract text using Tesseract (fast, lightweight).
        
        Returns:
            Tuple[str, List[Dict]]: (raw_text, bounding_boxes_with_text)
        """
        if not OCR_AVAILABLE['tesseract']:
            logger.warning("Tesseract not available")
            return "", []
        
        logger.info("Running Tesseract OCR...")
        
        # Get data with bounding boxes
        data = pytesseract.image_to_data(
            self.preprocessed,
            output_type=pytesseract.Output.DICT
        )
        
        text_blocks = []
        raw_text = ""
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if conf > 30:  # Tesseract confidence > 30%
                text_blocks.append({
                    'text': text,
                    'confidence': conf / 100.0,
                    'bbox': [
                        (data['left'][i], data['top'][i]),
                        (data['left'][i] + data['width'][i], data['top'][i]),
                        (data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]),
                        (data['left'][i], data['top'][i] + data['height'][i])
                    ]
                })
                raw_text += text + " "
        
        logger.info(f"✓ Extracted {len(text_blocks)} text blocks")
        return raw_text, text_blocks
    
    def extract_text_paddle(self) -> Tuple[str, List[Dict]]:
        """
        Extract text using PaddleOCR (multilingual, good accuracy).
        
        Returns:
            Tuple[str, List[Dict]]: (raw_text, bounding_boxes_with_text)
        """
        if self.reader is None:
            return "", []
        
        logger.info("Running PaddleOCR...")
        results = self.reader.ocr(self.preprocessed, cls=True)
        
        text_blocks = []
        raw_text = ""
        
        if results and results[0]:
            for line in results[0]:
                bbox = line[0]
                text = line[1]
                confidence = line[2]
                
                if confidence > 0.3:
                    text_blocks.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    raw_text += text + " "
        
        logger.info(f"✓ Extracted {len(text_blocks)} text blocks")
        return raw_text, text_blocks
    
    def extract_text(self) -> Tuple[str, List[Dict]]:
        """
        Extract text using configured OCR engine.
        
        Returns:
            Tuple[str, List[Dict]]: (raw_text, text_blocks_with_bboxes)
        """
        if self.ocr_engine == 'easyocr':
            return self.extract_text_easyocr()
        elif self.ocr_engine == 'tesseract':
            return self.extract_text_tesseract()
        elif self.ocr_engine == 'paddle':
            return self.extract_text_paddle()
        else:
            logger.error(f"Unknown OCR engine: {self.ocr_engine}")
            return "", []
    
    # ============================================================================
    # STEP 3: INTELLIGENT TEXT PARSING
    # ============================================================================
    
    @staticmethod
    def extract_email(text: str) -> Optional[str]:
        """
        Extract email address from text using regex.
        
        Pattern matches:
        - firstname.lastname@company.com
        - firstname_lastname@company.co.uk
        - f.lastname@company.com
        
        Args:
            text (str): Raw text to search
            
        Returns:
            Optional[str]: Email address if found, None otherwise
        """
        # RFC 5322 simplified email regex
        email_pattern = r'[a-zA-Z0-9][a-zA-Z0-9._-]*@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        matches = re.findall(email_pattern, text, re.IGNORECASE)
        
        if matches:
            # Return the first (most likely) match
            email = matches[0].lower()
            logger.info(f"  → Found email: {email}")
            return email
        
        return None
    
    @staticmethod
    def extract_name(text: str) -> Optional[str]:
        """
        Extract full name (First Last) from text using heuristics.
        
        Strategy:
        1. Look for "NAME:" label and capture following text
        2. Look for capitalized words (First Last pattern)
        3. Find word pairs that form common name patterns
        
        Args:
            text (str): Raw text to search
            
        Returns:
            Optional[str]: Full name if found, None otherwise
        """
        # Strategy 1: Look for "NAME:" label
        name_label_pattern = r'NAME\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)'
        match = re.search(name_label_pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            logger.info(f"  → Found name (from label): {name}")
            return name
        
        # Strategy 2: Find capitalized word pairs (likely First Last names)
        # Look for pattern: CAPITAL_WORD space CAPITAL_WORD
        capital_pattern = r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'
        matches = re.findall(capital_pattern, text)
        
        if matches:
            # Return the first match that's not common labels
            exclude_words = {'ID', 'CARD', 'EMAIL', 'PHONE', 'COMPANY', 'DEPARTMENT', 'EMP'}
            for first, last in matches:
                if first not in exclude_words and last not in exclude_words:
                    name = f"{first} {last}"
                    logger.info(f"  → Found name (from pattern): {name}")
                    return name
        
        # Strategy 3: OCR might have extracted name without spaces
        # Look for words that appear after "NAME" keyword
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'NAME' in line.upper():
                # Check next line or same line after colon
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        potential_name = parts[1].strip()
                        # Extract first two capitalized words
                        name_match = re.search(r'([A-Z][a-z]+)\s+([A-Z][a-z]+)', potential_name)
                        if name_match:
                            name = f"{name_match.group(1)} {name_match.group(2)}"
                            logger.info(f"  → Found name (from context): {name}")
                            return name
        
        return None
    
    # ============================================================================
    # STEP 4: TEXT BLOCK MATCHING (Find coordinates)
    # ============================================================================
    
    def find_text_block_bbox(self, target_text: str, text_blocks: List[Dict]) -> Optional[Dict]:
        """
        Find bounding box of text block containing target text.
        
        Args:
            target_text (str): Text to find (e.g., "John Smith")
            text_blocks (List[Dict]): List of OCR text blocks with bboxes
            
        Returns:
            Optional[Dict]: Block with bbox if found, None otherwise
        """
        target_lower = target_text.lower()
        
        for block in text_blocks:
            block_text = block['text'].lower()
            
            # Exact match
            if block_text == target_lower:
                return block
            
            # Partial match (in case of extra spaces or characters)
            if target_lower in block_text or block_text in target_lower:
                return block
        
        # Try fuzzy matching: split target and look for adjacent blocks
        parts = target_text.split()
        matching_blocks = []
        
        for block in text_blocks:
            block_text = block['text'].lower()
            for part in parts:
                if part.lower() in block_text:
                    matching_blocks.append(block)
                    break
        
        if matching_blocks:
            # Return the first matching block
            return matching_blocks[0]
        
        return None
    
    # ============================================================================
    # STEP 5: VISUAL FEEDBACK (Bounding Boxes)
    # ============================================================================
    
    def draw_bounding_boxes(self, name: Optional[str], email: Optional[str], 
                           text_blocks: List[Dict]) -> np.ndarray:
        """
        Draw colored bounding boxes on original image.
        
        - GREEN box around detected Name
        - RED box around detected Email
        
        Args:
            name (Optional[str]): Extracted name
            email (Optional[str]): Extracted email
            text_blocks (List[Dict]): OCR text blocks with coordinates
            
        Returns:
            np.ndarray: Image with bounding boxes drawn
        """
        # Convert preprocessed to BGR for display
        display_image = cv2.cvtColor(self.preprocessed, cv2.COLOR_GRAY2BGR)
        
        # Draw green box for NAME
        if name:
            name_block = self.find_text_block_bbox(name, text_blocks)
            if name_block:
                bbox = name_block['bbox']
                pts = np.array(bbox, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                cv2.polylines(display_image, [pts], True, (0, 255, 0), 3)  # Green
                cv2.putText(
                    display_image, 
                    'NAME', 
                    tuple(map(int, bbox[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                logger.info(f"  ✓ Drew GREEN box around: {name}")
            else:
                logger.warning(f"  ✗ Could not find bbox for name: {name}")
        
        # Draw red box for EMAIL
        if email:
            email_block = self.find_text_block_bbox(email, text_blocks)
            if email_block:
                bbox = email_block['bbox']
                pts = np.array(bbox, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                cv2.polylines(display_image, [pts], True, (0, 0, 255), 3)  # Red
                cv2.putText(
                    display_image, 
                    'EMAIL', 
                    tuple(map(int, bbox[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
                logger.info(f"  ✓ Drew RED box around: {email}")
            else:
                logger.warning(f"  ✗ Could not find bbox for email: {email}")
        
        return display_image
    
    def save_output(self, image: np.ndarray, filename: str = 'output_processed.png') -> bool:
        """
        Save processed image with bounding boxes.
        
        Args:
            image (np.ndarray): Image to save
            filename (str): Output filename
            
        Returns:
            bool: True if successful
        """
        success = cv2.imwrite(filename, image)
        if success:
            logger.info(f"✓ Saved output image: {filename}")
        else:
            logger.error(f"✗ Failed to save output image")
        return success
    
    # ============================================================================
    # MAIN PROCESSING PIPELINE
    # ============================================================================
    
    def process(self, image_path: str, output_path: str = 'output_processed.png') -> Dict:
        """
        Full processing pipeline: Load → Preprocess → OCR → Parse → Visualize.
        
        Args:
            image_path (str): Path to ID card image
            output_path (str): Path to save output image
            
        Returns:
            Dict: Results containing {'name': ..., 'email': ..., 'raw_text': ...}
        """
        logger.info("=" * 70)
        logger.info("STARTING ID CARD PROCESSING")
        logger.info("=" * 70)
        
        # Step 1: Load image
        if not self.load_image(image_path):
            return self.results
        
        # Step 2: Preprocess
        logger.info("\n[PREPROCESSING]")
        self.preprocess_image()
        
        # Step 3: Extract text via OCR
        logger.info("\n[OCR EXTRACTION]")
        raw_text, text_blocks = self.extract_text()
        self.results['raw_text'] = raw_text
        
        # Step 4: Parse extracted text
        logger.info("\n[TEXT PARSING]")
        name = self.extract_name(raw_text)
        email = self.extract_email(raw_text)
        
        self.results['name'] = name
        self.results['email'] = email
        
        # Step 5: Draw bounding boxes
        logger.info("\n[VISUAL FEEDBACK]")
        output_image = self.draw_bounding_boxes(name, email, text_blocks)
        
        # Step 6: Save output
        logger.info("\n[OUTPUT]")
        self.save_output(output_image, output_path)
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Full Name : {name if name else '✗ NOT FOUND'}")
        logger.info(f"Email     : {email if email else '✗ NOT FOUND'}")
        logger.info(f"Output    : {output_path}")
        logger.info("=" * 70)
        
        return self.results


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_batch(input_folder: str = 'input_images', 
                  output_folder: str = 'output_results',
                  ocr_engine: str = 'easyocr',
                  debug: bool = False) -> List[Dict]:
    """
    Process all ID card images in a folder.
    
    Args:
        input_folder (str): Folder containing ID card images
        output_folder (str): Folder to save results
        ocr_engine (str): OCR engine to use
        debug (bool): Enable debug mode
        
    Returns:
        List[Dict]: Results for all processed images
    """
    os.makedirs(output_folder, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if not image_files:
        logger.error(f"No images found in {input_folder}")
        return []
    
    logger.info(f"\nProcessing {len(image_files)} images from {input_folder}...")
    logger.info(f"Using OCR engine: {ocr_engine}")
    logger.info("-" * 70)
    
    all_results = []
    extractor = SmartIDExtractor(ocr_engine=ocr_engine, debug=debug)
    
    for idx, filename in enumerate(sorted(image_files), 1):
        image_path = os.path.join(input_folder, filename)
        output_filename = f"processed_{filename}"
        output_path = os.path.join(output_folder, output_filename)
        
        logger.info(f"\n[{idx}/{len(image_files)}] Processing: {filename}")
        
        results = extractor.process(image_path, output_path)
        results['filename'] = filename
        results['output'] = output_path
        all_results.append(results)
    
    # Print summary statistics
    print_batch_summary(all_results)
    
    return all_results


def print_batch_summary(results: List[Dict]):
    """Print summary statistics for batch processing."""
    logger.info("\n" + "=" * 70)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 70)
    
    total = len(results)
    names_found = sum(1 for r in results if r['name'] is not None)
    emails_found = sum(1 for r in results if r['email'] is not None)
    
    logger.info(f"Total Images  : {total}")
    logger.info(f"Names Found   : {names_found}/{total} ({100*names_found//total if total else 0}%)")
    logger.info(f"Emails Found  : {emails_found}/{total} ({100*emails_found//total if total else 0}%)")
    
    logger.info("\nDetailed Results:")
    logger.info("-" * 70)
    
    for i, result in enumerate(results, 1):
        status = "✓" if result['name'] and result['email'] else "✗"
        logger.info(f"{status} {i:2d}. {result['filename']}")
        if result['name']:
            logger.info(f"      Name : {result['name']}")
        if result['email']:
            logger.info(f"      Email: {result['email']}")
    
    logger.info("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if OCR engines are available
    if not any(OCR_AVAILABLE.values()):
        print("\n" + "=" * 70)
        print("ERROR: No OCR engine available!")
        print("=" * 70)
        print("\nInstall at least one OCR engine:")
        print("  pip install easyocr          # Recommended")
        print("  pip install pytesseract      # Fast")
        print("  pip install paddleocr        # Multilingual")
        print("\n" + "=" * 70)
        sys.exit(1)
    
    # Determine which OCR engine to use
    preferred_engine = 'easyocr' if OCR_AVAILABLE['easyocr'] else \
                      'tesseract' if OCR_AVAILABLE['tesseract'] else 'paddle'
    
    logger.info("\n" + "=" * 70)
    logger.info("SMART ID DATA EXTRACTOR")
    logger.info("=" * 70)
    logger.info(f"Available OCR engines: {[k for k, v in OCR_AVAILABLE.items() if v]}")
    logger.info(f"Using OCR engine: {preferred_engine}")
    logger.info("=" * 70)
    
    # Process single image or batch
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'output_processed.png'
        
        extractor = SmartIDExtractor(ocr_engine=preferred_engine, debug=False)
        extractor.process(image_path, output_path)
    else:
        # Batch process all images in input_images folder
        logger.info("\n** No image path provided. Batch processing 'input_images/' folder **\n")
        
        if os.path.exists('input_images'):
            process_batch(
                input_folder='input_images',
                output_folder='output_results',
                ocr_engine=preferred_engine,
                debug=False
            )
        else:
            logger.error("No image path provided and 'input_images' folder not found.")
            logger.info("Usage: python main.py <image_path> [output_path]")
            logger.info("   or: python main.py (processes input_images/ folder)")
