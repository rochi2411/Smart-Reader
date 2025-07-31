"""
Web crawling functionality for extracting content from links.
"""
import requests
import time
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
import re

from ..utils.logger import get_logger

logger = get_logger(__name__)

class WebCrawler:
    """Handles web crawling and content extraction from URLs."""
    
    def __init__(self, max_links: int = 5, timeout: int = 10):
        self.max_links = max_links
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.crawled_urls: Set[str] = set()
    
    def extract_links_from_text(self, text: str) -> List[str]:
        """
        Extract URLs from text content.
        
        Args:
            text: Text content to search for URLs
            
        Returns:
            List of found URLs
        """
        # Regular expression to find URLs
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        urls = url_pattern.findall(text)
        
        # Clean and validate URLs
        valid_urls = []
        for url in urls:
            try:
                parsed = urlparse(url)
                if parsed.scheme and parsed.netloc:
                    valid_urls.append(url)
            except Exception:
                continue
        
        logger.info(f"Extracted {len(valid_urls)} valid URLs from text")
        return valid_urls
    
    def crawl_url(self, url: str) -> Optional[Dict[str, str]]:
        """
        Crawl a single URL and extract its content.
        
        Args:
            url: URL to crawl
            
        Returns:
            Dictionary with URL and extracted content, or None if failed
        """
        if url in self.crawled_urls:
            logger.info(f"URL already crawled: {url}")
            return None
        
        try:
            logger.info(f"Crawling URL: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No Title"
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract main content
            content = soup.get_text()
            
            # Clean up the content
            content = self._clean_web_content(content)
            
            self.crawled_urls.add(url)
            
            result = {
                'url': url,
                'title': title_text,
                'content': content,
                'word_count': len(content.split())
            }
            
            logger.info(f"Successfully crawled {url}: {len(content)} characters")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error crawling {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            return None
    
    def crawl_multiple_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Crawl multiple URLs with rate limiting.
        
        Args:
            urls: List of URLs to crawl
            
        Returns:
            List of successfully crawled content dictionaries
        """
        results = []
        crawled_count = 0
        
        for url in urls:
            if crawled_count >= self.max_links:
                logger.info(f"Reached maximum crawl limit: {self.max_links}")
                break
            
            result = self.crawl_url(url)
            if result:
                results.append(result)
                crawled_count += 1
                
                # Rate limiting - wait between requests
                time.sleep(1)
        
        logger.info(f"Successfully crawled {len(results)} URLs")
        return results
    
    def _clean_web_content(self, content: str) -> str:
        """
        Clean and normalize web content.
        
        Args:
            content: Raw web content
            
        Returns:
            Cleaned content
        """
        if not content:
            return ""
        
        # Remove excessive whitespace and newlines
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # Remove common web artifacts
        content = re.sub(r'Cookie Policy|Privacy Policy|Terms of Service', '', content, flags=re.IGNORECASE)
        
        # Limit content length to avoid processing very large pages
        max_length = 10000
        if len(content) > max_length:
            content = content[:max_length] + "... [Content truncated]"
        
        return content
    
    def is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid and accessible.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid and accessible
        """
        try:
            parsed = urlparse(url)
            if not (parsed.scheme and parsed.netloc):
                return False
            
            # Quick HEAD request to check if URL is accessible
            response = self.session.head(url, timeout=5)
            return response.status_code < 400
            
        except Exception:
            return False
    
    def close(self):
        """Close the session."""
        self.session.close()
