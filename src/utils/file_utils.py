"""
File handling utilities for the Document QA System.
"""
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Union
import streamlit as st

from ..utils.logger import get_logger

logger = get_logger(__name__)

class FileManager:
    """Manages file operations and temporary file handling."""
    
    def __init__(self):
        self.temp_dir = None
        self.uploaded_files = []
    
    def create_temp_directory(self) -> Path:
        """Create a temporary directory for file processing."""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp())
            logger.info(f"Created temporary directory: {self.temp_dir}")
        return self.temp_dir
    
    def save_uploaded_files(self, uploaded_files: List) -> List[Path]:
        """
        Save uploaded Streamlit files to temporary directory.
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
            
        Returns:
            List of file paths where files were saved
        """
        if not uploaded_files:
            return []
        
        temp_dir = self.create_temp_directory()
        saved_files = []
        
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(file_path)
                logger.info(f"Saved uploaded file: {file_path}")
            except Exception as e:
                logger.error(f"Error saving file {uploaded_file.name}: {str(e)}")
                st.error(f"Error saving file {uploaded_file.name}: {str(e)}")
        
        self.uploaded_files.extend(saved_files)
        return saved_files
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
                self.temp_dir = None
                self.uploaded_files = []
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {str(e)}")
    
    def get_file_extension(self, file_path: Union[str, Path]) -> str:
        """Get file extension in lowercase."""
        return Path(file_path).suffix.lower()
    
    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """Check if file type is supported."""
        supported_extensions = {
            '.pdf', '.docx', '.pptx', '.xlsx', '.xls', 
            '.csv', '.json', '.txt', '.png', '.jpg', 
            '.jpeg', '.tiff', '.bmp'
        }
        return self.get_file_extension(file_path) in supported_extensions
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup()
