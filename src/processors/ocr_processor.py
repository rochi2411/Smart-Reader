"""
OCR processing functionality for the Document QA System.
"""
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Optional, Union
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)

class OCRProcessor:
    """Handles OCR operations for text extraction from images."""
    
    def __init__(self):
        self.tesseract_config = '--oem 3 --psm 6'
    
    def extract_text_from_image(
        self, 
        image_input: Union[str, Path, Image.Image, np.ndarray],
        preprocess: bool = True
    ) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image_input: Image file path, PIL Image, or numpy array
            preprocess: Whether to preprocess the image for better OCR
            
        Returns:
            Extracted text string
        """
        try:
            # Convert input to PIL Image
            if isinstance(image_input, (str, Path)):
                image = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Preprocess image if requested
            if preprocess:
                image = self._preprocess_image(image)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(image, config=self.tesseract_config)
            
            # Clean up the extracted text
            text = self._clean_extracted_text(text)
            
            logger.info(f"Successfully extracted {len(text)} characters from image")
            return text
            
        except Exception as e:
            logger.error(f"Error during OCR processing: {str(e)}")
            return ""
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(processed)
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Error preprocessing image, using original: {str(e)}")
            return image
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common OCR artifacts
        text = text.replace('|', 'I')  # Common misrecognition
        text = text.replace('0', 'O')  # In some contexts
        
        # Remove very short "words" that are likely artifacts
        words = text.split()
        cleaned_words = [word for word in words if len(word) > 1 or word.isalnum()]
        
        return ' '.join(cleaned_words)
    
    def is_text_rich_image(self, image_input: Union[str, Path, Image.Image]) -> bool:
        """
        Determine if an image contains significant text content.
        
        Args:
            image_input: Image to analyze
            
        Returns:
            True if image appears to contain substantial text
        """
        try:
            text = self.extract_text_from_image(image_input, preprocess=True)
            
            # Simple heuristics to determine if image is text-rich
            word_count = len(text.split())
            char_count = len(text.strip())
            
            # Consider text-rich if it has reasonable word/character counts
            return word_count >= 5 and char_count >= 20
            
        except Exception as e:
            logger.error(f"Error analyzing image text content: {str(e)}")
            return False
