"""
PDF processing functionality for the Document QA System.
"""
import fitz  # PyMuPDF
import camelot
import pandas as pd
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io

from ..processors.ocr_processor import OCRProcessor
from ..processors.web_crawler import WebCrawler
from ..utils.logger import get_logger

logger = get_logger(__name__)

class PDFProcessor:
    """Handles PDF document processing including text, tables, images, and links."""
    
    def __init__(self, use_ocr: bool = True, crawl_links: bool = True, max_links: int = 5):
        self.use_ocr = use_ocr
        self.crawl_links = crawl_links
        self.max_links = max_links
        self.ocr_processor = OCRProcessor() if use_ocr else None
        self.web_crawler = WebCrawler(max_links=max_links) if crawl_links else None
    
    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a PDF file and extract all content.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of content dictionaries with extracted information
        """
        logger.info(f"Processing PDF: {file_path}")
        
        try:
            doc = fitz.open(file_path)
            all_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_content = self._process_page(page, page_num + 1, file_path)
                all_content.extend(page_content)
            
            doc.close()
            
            # Extract tables using Camelot
            table_content = self._extract_tables_camelot(file_path)
            all_content.extend(table_content)
            
            logger.info(f"Successfully processed PDF: {len(all_content)} content items")
            return all_content
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return []
    
    def _process_page(self, page, page_num: int, file_path: str) -> List[Dict[str, Any]]:
        """Process a single PDF page."""
        content_items = []
        
        try:
            # Extract text
            text = page.get_text()
            if text.strip():
                content_items.append({
                    'type': 'text',
                    'content': text,
                    'page': page_num,
                    'source': f"{Path(file_path).name} - Page {page_num}"
                })
            
            # Extract images and apply OCR if enabled
            if self.use_ocr:
                image_content = self._extract_images_from_page(page, page_num, file_path)
                content_items.extend(image_content)
            
            # Extract links
            links = self._extract_links_from_page(page, page_num, file_path)
            content_items.extend(links)
            
        except Exception as e:
            logger.error(f"Error processing page {page_num} of {file_path}: {str(e)}")
        
        return content_items
    
    def _extract_images_from_page(self, page, page_num: int, file_path: str) -> List[Dict[str, Any]]:
        """Extract and OCR images from a PDF page."""
        content_items = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        # Convert to PIL Image
                        img_data = pix.tobytes("ppm")
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        # Apply OCR
                        if self.ocr_processor:
                            ocr_text = self.ocr_processor.extract_text_from_image(pil_image)
                            
                            if ocr_text.strip():
                                content_items.append({
                                    'type': 'image_ocr',
                                    'content': ocr_text,
                                    'page': page_num,
                                    'source': f"{Path(file_path).name} - Page {page_num} - Image {img_index + 1}"
                                })
                    
                    pix = None  # Free memory
                    
                except Exception as e:
                    logger.warning(f"Error processing image {img_index} on page {page_num}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting images from page {page_num}: {str(e)}")
        
        return content_items
    
    def _extract_links_from_page(self, page, page_num: int, file_path: str) -> List[Dict[str, Any]]:
        """Extract and optionally crawl links from a PDF page."""
        content_items = []
        
        try:
            links = page.get_links()
            
            if links and self.web_crawler:
                urls = [link.get('uri') for link in links if link.get('uri')]
                
                if urls:
                    # Crawl the links
                    crawled_content = self.web_crawler.crawl_multiple_urls(urls[:self.max_links])
                    
                    for crawled in crawled_content:
                        content_items.append({
                            'type': 'web_content',
                            'content': f"Title: {crawled['title']}\n\nContent: {crawled['content']}",
                            'page': page_num,
                            'source': f"{Path(file_path).name} - Page {page_num} - Link: {crawled['url']}"
                        })
                        
        except Exception as e:
            logger.error(f"Error extracting links from page {page_num}: {str(e)}")
        
        return content_items
    
    def _extract_tables_camelot(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables using Camelot library."""
        content_items = []
        
        try:
            # Extract tables from all pages
            tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
            
            for i, table in enumerate(tables):
                if table.df is not None and not table.df.empty:
                    # Convert table to string representation
                    table_text = self._table_to_text(table.df)
                    
                    content_items.append({
                        'type': 'table',
                        'content': table_text,
                        'page': table.page,
                        'source': f"{Path(file_path).name} - Table {i + 1} (Page {table.page})"
                    })
                    
        except Exception as e:
            logger.warning(f"Camelot table extraction failed for {file_path}: {str(e)}")
            # Fallback to basic table extraction if Camelot fails
            try:
                content_items.extend(self._extract_tables_basic(file_path))
            except Exception as e2:
                logger.error(f"Basic table extraction also failed: {str(e2)}")
        
        return content_items
    
    def _extract_tables_basic(self, file_path: str) -> List[Dict[str, Any]]:
        """Basic table extraction fallback method."""
        content_items = []
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Look for table-like structures in the text
                text = page.get_text()
                if self._contains_table_structure(text):
                    content_items.append({
                        'type': 'table_text',
                        'content': text,
                        'page': page_num + 1,
                        'source': f"{Path(file_path).name} - Potential Table (Page {page_num + 1})"
                    })
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Basic table extraction failed: {str(e)}")
        
        return content_items
    
    def _contains_table_structure(self, text: str) -> bool:
        """Check if text contains table-like structure."""
        lines = text.split('\n')
        
        # Simple heuristic: look for lines with multiple tab-separated or space-separated values
        table_like_lines = 0
        for line in lines:
            if len(line.split('\t')) > 2 or len([x for x in line.split() if x]) > 3:
                table_like_lines += 1
        
        return table_like_lines > 2
    
    def _table_to_text(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to readable text format."""
        try:
            # Create a formatted string representation
            text_parts = []
            
            # Add headers
            headers = " | ".join(str(col) for col in df.columns)
            text_parts.append(f"Headers: {headers}")
            
            # Add rows
            for idx, row in df.iterrows():
                row_text = " | ".join(str(val) for val in row.values)
                text_parts.append(f"Row {idx + 1}: {row_text}")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error converting table to text: {str(e)}")
            return str(df)
    
    def __del__(self):
        """Cleanup resources."""
        if self.web_crawler:
            self.web_crawler.close()
