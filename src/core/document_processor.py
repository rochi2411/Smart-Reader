"""
Main document processor that coordinates all document processing operations.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

from ..config.settings import AppConfig
from ..processors.pdf_processor import PDFProcessor
from ..processors.office_processor import OfficeProcessor
from ..processors.ocr_processor import OCRProcessor
from ..utils.file_utils import FileManager
from ..utils.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessor:
    """Main document processor that handles all file types and creates vector indexes."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.file_manager = FileManager()
        
        # Initialize processors
        self.pdf_processor = PDFProcessor(
            use_ocr=config.processing.use_ocr,
            crawl_links=config.processing.crawl_links,
            max_links=config.processing.max_links_to_crawl
        )
        
        self.office_processor = OfficeProcessor(
            use_ocr=config.processing.use_ocr,
            crawl_links=config.processing.crawl_links,
            max_links=config.processing.max_links_to_crawl
        )
        
        self.ocr_processor = OCRProcessor()
        
        # Initialize AI models
        self._initialize_models()
        
        # Initialize node parser
        self.node_parser = SentenceSplitter(
            chunk_size=config.processing.chunk_size,
            chunk_overlap=config.processing.chunk_overlap
        )
        
        # Storage for processed documents and index
        self.documents: List[Document] = []
        self.index: Optional[VectorStoreIndex] = None
        self.extracted_links: List[str] = []
        self.loaded_dataframes: Dict[str, Any] = {}
    
    def _initialize_models(self):
        """Initialize embedding and LLM models."""
        logger.info("Initializing AI models...")
        
        try:
            # Initialize embedding model
            Settings.embed_model = GeminiEmbedding(
                model_name=self.config.model.embedding_model,
                api_key=self.config.model.api_key
            )
            
            # Initialize LLM
            self.llm = Gemini(
                model=self.config.model.llm_model,
                api_key=self.config.model.api_key,
                temperature=self.config.model.temperature
            )
            Settings.llm = self.llm
            
            # Set chunk size
            Settings.chunk_size = self.config.processing.chunk_size
            
            logger.info("AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AI models: {str(e)}")
            raise
    
    def process_files(self, file_paths: List[Path]) -> bool:
        """
        Process multiple files and create vector index.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            True if processing was successful
        """
        logger.info(f"Processing {len(file_paths)} files")
        
        try:
            all_documents = []
            
            for file_path in file_paths:
                if not self.file_manager.is_supported_file(file_path):
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                
                documents = self.process_single_file(str(file_path))
                all_documents.extend(documents)
            
            if not all_documents:
                logger.warning("No documents were processed successfully")
                return False
            
            # Store documents
            self.documents = all_documents
            
            # Create vector index
            self._create_vector_index()
            
            logger.info(f"Successfully processed {len(all_documents)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            return False
    
    def process_single_file(self, file_path: str) -> List[Document]:
        """
        Process a single file and return LlamaIndex Documents.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of Document objects
        """
        logger.info(f"Processing file: {file_path}")
        
        try:
            file_extension = self.file_manager.get_file_extension(file_path)
            content_items = []
            
            # Route to appropriate processor based on file type
            if file_extension == '.pdf':
                content_items = self.pdf_processor.process_pdf(file_path)
            
            elif file_extension == '.docx':
                content_items = self.office_processor.process_docx(file_path)
            
            elif file_extension == '.pptx':
                content_items = self.office_processor.process_pptx(file_path)
            
            elif file_extension in ['.xlsx', '.xls']:
                content_items = self.office_processor.process_excel(file_path)
            
            elif file_extension == '.csv':
                content_items = self.office_processor.process_csv(file_path)
            
            elif file_extension == '.json':
                content_items = self.office_processor.process_json(file_path)
            
            elif file_extension == '.txt':
                content_items = self._process_text_file(file_path)
            
            elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                content_items = self._process_image_file(file_path)
            
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
            
            # Convert content items to LlamaIndex Documents
            documents = self._create_documents_from_content(content_items)
            
            logger.info(f"Created {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
    
    def _process_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
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
            if not self.config.processing.use_ocr:
                logger.info(f"OCR disabled, skipping image: {file_path}")
                return []
            
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
    
    def _create_documents_from_content(self, content_items: List[Dict[str, Any]]) -> List[Document]:
        """Convert content items to LlamaIndex Documents."""
        documents = []
        
        for item in content_items:
            try:
                # Create metadata
                metadata = {
                    'source': item.get('source', 'Unknown'),
                    'type': item.get('type', 'text'),
                }
                
                # Add page number if available
                if 'page' in item:
                    metadata['page'] = item['page']
                
                # Create Document
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
            
            logger.info(f"Creating vector index from {len(self.documents)} documents")
            
            # Create index
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                node_parser=self.node_parser
            )
            
            logger.info("Vector index created successfully")
            
        except Exception as e:
            logger.error(f"Error creating vector index: {str(e)}")
            raise
    
    def query(self, question: str) -> str:
        """
        Query the processed documents.
        
        Args:
            question: Question to ask
            
        Returns:
            Answer string
        """
        if not self.index:
            return "No documents have been processed yet. Please upload and process documents first."
        
        try:
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=self.config.query.similarity_top_k,
                response_mode="compact"
            )
            
            # Execute query
            response = query_engine.query(question)
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return f"Error processing query: {str(e)}"
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents."""
        return {
            'total_documents': len(self.documents),
            'total_links': len(self.extracted_links),
            'dataframes_loaded': len(self.loaded_dataframes),
            'index_created': self.index is not None
        }
    
    def reset(self):
        """Reset the processor state."""
        self.documents = []
        self.index = None
        self.extracted_links = []
        self.loaded_dataframes = {}
        self.file_manager.cleanup()
        logger.info("Document processor reset")
    
    def __del__(self):
        """Cleanup resources."""
        self.file_manager.cleanup()
