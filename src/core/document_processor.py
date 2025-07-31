"""
Document processor using the latest Google GenAI packages.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import traceback

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from ..config.settings import AppConfig
from ..processors.ocr_processor import OCRProcessor
from ..processors.web_crawler import WebCrawler
from ..utils.file_utils import FileManager
from ..utils.logger import get_logger

logger = get_logger(__name__)

class UpdatedDocumentProcessor:
    """Document processor using latest Google GenAI packages."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.file_manager = FileManager()
        self.models_initialized = False
        
        # Initialize basic processors
        try:
            self.ocr_processor = OCRProcessor() if config.processing.use_ocr else None
            logger.info("OCR processor initialized")
        except Exception as e:
            logger.warning(f"OCR processor initialization failed: {e}")
            self.ocr_processor = None
        
        try:
            self.web_crawler = WebCrawler(max_links=config.processing.max_links_to_crawl) if config.processing.crawl_links else None
            logger.info("Web crawler initialized")
        except Exception as e:
            logger.warning(f"Web crawler initialization failed: {e}")
            self.web_crawler = None
        
        # Initialize node parser
        self.node_parser = SentenceSplitter(
            chunk_size=config.processing.chunk_size,
            chunk_overlap=config.processing.chunk_overlap
        )
        logger.info("Node parser initialized")
        
        # Storage
        self.documents: List[Document] = []
        self.index: Optional[VectorStoreIndex] = None
        
        # Initialize AI models with better error handling
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding and LLM models using latest Google GenAI packages."""
        logger.info("Starting AI model initialization with Google GenAI...")
        
        try:
            # Check API key first
            if not self.config.model.api_key:
                raise ValueError("API key is not set. Please check your .env file.")
            
            if self.config.model.api_key == "your_google_api_key_here":
                raise ValueError("Please replace 'your_google_api_key_here' with your actual Google API key in the .env file.")
            
            logger.info(f"API key found: {self.config.model.api_key[:10]}...")
            
            # Try to import and initialize embedding model using new package
            logger.info("Initializing embedding model with Google GenAI...")
            try:
                from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
                
                embed_model = GoogleGenAIEmbedding(
                    model_name=self.config.model.embedding_model,
                    api_key=self.config.model.api_key
                )
                Settings.embed_model = embed_model
                logger.info(f"Embedding model initialized: {self.config.model.embedding_model}")
                
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                raise Exception(f"Embedding model initialization failed: {str(e)}")
            
            # Try to initialize LLM using new package
            logger.info("Initializing LLM with Google GenAI...")
            try:
                from llama_index.llms.google_genai import GoogleGenAI
                
                self.llm = GoogleGenAI(
                    model=self.config.model.llm_model,
                    api_key=self.config.model.api_key,
                    temperature=self.config.model.temperature
                )
                Settings.llm = self.llm
                logger.info(f"LLM initialized: {self.config.model.llm_model}")
                
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise Exception(f"LLM initialization failed: {str(e)}")
            
            # Set chunk size
            Settings.chunk_size = self.config.processing.chunk_size
            
            self.models_initialized = True
            logger.info("All AI models initialized successfully with Google GenAI")
            
        except Exception as e:
            logger.error(f"Error initializing AI models: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.models_initialized = False
            raise Exception(f"AI model initialization failed: {str(e)}")
    
    def is_ready(self) -> bool:
        """Check if the processor is ready to process documents."""
        return self.models_initialized
    
    def get_initialization_status(self) -> Dict[str, Any]:
        """Get detailed initialization status."""
        return {
            'models_initialized': self.models_initialized,
            'ocr_available': self.ocr_processor is not None,
            'web_crawler_available': self.web_crawler is not None,
            'api_key_set': bool(self.config.model.api_key and self.config.model.api_key != "your_google_api_key_here"),
            'config_valid': self.config is not None,
            'using_latest_genai': True  # Indicates we're using the latest packages
        }
    
    def process_files(self, file_paths: List[Path]) -> bool:
        """Process multiple files and create vector index."""
        if not self.models_initialized:
            logger.error("Cannot process files: AI models not initialized")
            return False
        
        logger.info(f"Processing {len(file_paths)} files")
        
        try:
            all_documents = []
            
            for file_path in file_paths:
                if not self.file_manager.is_supported_file(file_path):
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                
                logger.info(f"Processing file: {file_path}")
                documents = self.process_single_file(str(file_path))
                all_documents.extend(documents)
                logger.info(f"Processed {file_path}: {len(documents)} documents")
            
            if not all_documents:
                logger.warning("No documents were processed successfully")
                return False
            
            self.documents = all_documents
            self._create_vector_index()
            
            logger.info(f"Successfully processed {len(all_documents)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def process_single_file(self, file_path: str) -> List[Document]:
        """Process a single file and return Documents."""
        logger.info(f"Processing file: {file_path}")
        
        try:
            file_extension = self.file_manager.get_file_extension(file_path)
            content_items = []
            
            # Basic file processing
            if file_extension == '.txt':
                content_items = self._process_text_file(file_path)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                content_items = self._process_image_file(file_path)
            elif file_extension == '.pdf':
                content_items = self._process_pdf_basic(file_path)
            else:
                logger.warning(f"File type {file_extension} not supported in this version")
                return []
            
            # Convert to Documents
            documents = self._create_documents_from_content(content_items)
            
            logger.info(f"Created {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def _process_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Text file is empty: {file_path}")
                return []
            
            return [{
                'type': 'text',
                'content': content,
                'source': Path(file_path).name
            }]
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            return []
    
    def _process_image_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process an image file using OCR."""
        try:
            if not self.ocr_processor:
                logger.info(f"OCR not available, skipping image: {file_path}")
                return []
            
            logger.info(f"Extracting text from image: {file_path}")
            text = self.ocr_processor.extract_text_from_image(file_path)
            
            if text.strip():
                return [{
                    'type': 'image_ocr',
                    'content': text,
                    'source': f"{Path(file_path).name} - OCR"
                }]
            else:
                logger.info(f"No text extracted from image: {file_path}")
                return []
                
        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {str(e)}")
            return []
    
    def _process_pdf_basic(self, file_path: str) -> List[Dict[str, Any]]:
        """Basic PDF processing without advanced features."""
        try:
            import fitz  # PyMuPDF
            
            logger.info(f"Processing PDF: {file_path}")
            doc = fitz.open(file_path)
            content_items = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():
                    content_items.append({
                        'type': 'text',
                        'content': text,
                        'page': page_num + 1,
                        'source': f"{Path(file_path).name} - Page {page_num + 1}"
                    })
            
            doc.close()
            logger.info(f"Extracted text from {len(content_items)} pages")
            return content_items
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return []
    
    def _create_documents_from_content(self, content_items: List[Dict[str, Any]]) -> List[Document]:
        """Convert content items to LlamaIndex Documents."""
        documents = []
        
        for item in content_items:
            try:
                metadata = {
                    'source': item.get('source', 'Unknown'),
                    'type': item.get('type', 'text'),
                }
                
                if 'page' in item:
                    metadata['page'] = item['page']
                
                doc = Document(
                    text=item['content'],
                    metadata=metadata
                )
                
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error creating document from content item: {str(e)}")
                continue
        
        return documents
    
    def _create_vector_index(self):
        """Create vector index from processed documents."""
        try:
            if not self.documents:
                logger.warning("No documents available for indexing")
                return
            
            if not self.models_initialized:
                logger.error("Cannot create index: AI models not initialized")
                return
            
            logger.info(f"Creating vector index from {len(self.documents)} documents")
            
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                node_parser=self.node_parser
            )
            
            logger.info("Vector index created successfully")
            
        except Exception as e:
            logger.error(f"Error creating vector index: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise Exception(f"Vector index creation failed: {str(e)}")
    
    def query(self, question: str) -> str:
        """Query the processed documents."""
        if not self.models_initialized:
            return "AI models are not initialized. Please check your configuration and API key."
        
        if not self.index:
            return "No documents have been processed yet. Please upload and process documents first."
        
        try:
            logger.info(f"Processing query: {question[:50]}...")
            
            query_engine = self.index.as_query_engine(
                similarity_top_k=self.config.query.similarity_top_k,
                response_mode="compact"
            )
            
            response = query_engine.query(question)
            
            logger.info("Query processed successfully")
            return str(response)
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"Error processing query: {str(e)}"
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents."""
        return {
            'total_documents': len(self.documents),
            'index_created': self.index is not None,
            'models_initialized': self.models_initialized,
            'ready_to_process': self.is_ready(),
            'using_latest_genai': True
        }
    
    def reset(self):
        """Reset the processor state."""
        self.documents = []
        self.index = None
        self.file_manager.cleanup()
        logger.info("Document processor reset")
    
    def __del__(self):
        """Cleanup resources."""
        self.file_manager.cleanup()
