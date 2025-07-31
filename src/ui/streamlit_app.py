"""
Streamlit user interface for the Document QA System.
"""
import streamlit as st
from typing import List, Optional
import time

from ..config.settings import AppConfig, ModelConfig, ProcessingConfig, QueryConfig
from ..core.document_processor import DocumentProcessor
from ..utils.file_utils import FileManager
from ..utils.logger import setup_logger, get_logger

# Setup logging
setup_logger(__name__, log_file="app.log")
logger = get_logger(__name__)

class DocumentQAApp:
    """Streamlit application for document QA system."""
    
    def __init__(self):
        self.config: Optional[AppConfig] = None
        self.processor: Optional[DocumentProcessor] = None
        self.file_manager = FileManager()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'processor_initialized' not in st.session_state:
            st.session_state.processor_initialized = False
        
        if 'documents_processed' not in st.session_state:
            st.session_state.documents_processed = False
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = ""
    
    def run(self):
        """Main application entry point."""
        st.set_page_config(
            page_title="Advanced Document QA System",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📚 Advanced Document QA System")
        st.markdown("Upload documents and ask questions about their content using AI-powered analysis.")
        
        # Sidebar configuration
        self._render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_main_content()
        
        with col2:
            self._render_status_panel()
        
        # Chat interface at the bottom
        self._render_chat_interface()
    
    def _render_sidebar(self):
        """Render the configuration sidebar."""
        st.sidebar.header("⚙️ Configuration")
        
        # Model Configuration
        st.sidebar.subheader("🤖 Model Settings")
        
        embedding_model = st.sidebar.selectbox(
            "Embedding Model",
            ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"],
            index=0,
            help="Model used for document embeddings"
        )
        
        llm_model = st.sidebar.selectbox(
            "Language Model",
            ["models/gemini-1.5-flash", "models/gemini-1.5-pro"],
            index=0,
            help="Model used for generating responses"
        )
        
        temperature = st.sidebar.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Controls randomness in responses (lower = more focused)"
        )
        
        # Processing Configuration
        st.sidebar.subheader("📄 Processing Settings")
        
        chunk_size = st.sidebar.slider(
            "Chunk Size",
            min_value=256,
            max_value=2048,
            value=1024,
            step=128,
            help="Size of text chunks for processing"
        )
        
        chunk_overlap = st.sidebar.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=200,
            value=100,
            step=20,
            help="Overlap between consecutive chunks"
        )
        
        use_ocr = st.sidebar.checkbox(
            "Enable OCR",
            value=True,
            help="Extract text from images and scanned documents"
        )
        
        crawl_links = st.sidebar.checkbox(
            "Crawl Links",
            value=True,
            help="Extract and crawl hyperlinks found in documents"
        )
        
        max_links = st.sidebar.slider(
            "Max Links to Crawl",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum number of links to crawl per document"
        )
        
        # Query Configuration
        st.sidebar.subheader("🔍 Query Settings")
        
        similarity_top_k = st.sidebar.slider(
            "Top K Results",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of similar chunks to retrieve"
        )
        
        similarity_cutoff = st.sidebar.slider(
            "Similarity Cutoff",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum similarity threshold for retrieval"
        )
        
        # Initialize processor button
        if st.sidebar.button("🚀 Initialize Processor", type="primary"):
            self._initialize_processor(
                embedding_model, llm_model, temperature, chunk_size, chunk_overlap,
                use_ocr, crawl_links, max_links, similarity_top_k, similarity_cutoff
            )
        
        # Reset button
        if st.sidebar.button("🔄 Reset System"):
            self._reset_system()
    
    def _render_main_content(self):
        """Render the main content area."""
        st.header("📁 Document Upload")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'pptx', 'xlsx', 'xls', 'csv', 'json', 'txt', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload documents in supported formats"
        )
        
        # Process files button
        if uploaded_files and st.button("📊 Process Files", type="primary"):
            if not st.session_state.processor_initialized:
                st.error("Please initialize the processor first using the sidebar.")
                return
            
            self._process_uploaded_files(uploaded_files)
        
        # Display uploaded files
        if uploaded_files:
            st.subheader("📋 Uploaded Files")
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size} bytes)")
    
    def _render_status_panel(self):
        """Render the status panel."""
        st.header("📊 System Status")
        
        # Processor status
        if st.session_state.processor_initialized:
            st.success("✅ Processor Initialized")
        else:
            st.warning("⚠️ Processor Not Initialized")
        
        # Document processing status
        if st.session_state.documents_processed:
            st.success("✅ Documents Processed")
            
            if self.processor:
                stats = self.processor.get_document_stats()
                st.metric("Total Documents", stats['total_documents'])
                st.metric("Links Extracted", stats['total_links'])
                st.metric("Dataframes Loaded", stats['dataframes_loaded'])
        else:
            st.info("ℹ️ No Documents Processed")
        
        # Processing status messages
        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)
    
    def _render_chat_interface(self):
        """Render the chat interface."""
        st.header("💬 Ask Questions")
        
        if not st.session_state.documents_processed:
            st.info("Please upload and process documents before asking questions.")
            return
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.expander(f"Q{i+1}: {question[:50]}..."):
                    st.write(f"**Question:** {question}")
                    st.write(f"**Answer:** {answer}")
        
        # Question input
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What is the main topic of the document?",
            key="question_input"
        )
        
        if st.button("🔍 Ask Question") and question:
            self._process_question(question)
    
    def _initialize_processor(self, embedding_model, llm_model, temperature, chunk_size, 
                            chunk_overlap, use_ocr, crawl_links, max_links, 
                            similarity_top_k, similarity_cutoff):
        """Initialize the document processor with given configuration."""
        try:
            with st.spinner("Initializing processor..."):
                # Create configuration
                self.config = AppConfig(
                    model=ModelConfig(
                        embedding_model=embedding_model,
                        llm_model=llm_model,
                        temperature=temperature
                    ),
                    processing=ProcessingConfig(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        use_ocr=use_ocr,
                        crawl_links=crawl_links,
                        max_links_to_crawl=max_links
                    ),
                    query=QueryConfig(
                        similarity_top_k=similarity_top_k,
                        similarity_cutoff=similarity_cutoff
                    )
                )
                
                # Initialize processor
                self.processor = DocumentProcessor(self.config)
                st.session_state.processor_initialized = True
                
                st.success("✅ Processor initialized successfully!")
                logger.info("Document processor initialized")
                
        except Exception as e:
            st.error(f"❌ Error initializing processor: {str(e)}")
            logger.error(f"Error initializing processor: {str(e)}")
    
    def _process_uploaded_files(self, uploaded_files):
        """Process the uploaded files."""
        try:
            with st.spinner("Processing files..."):
                # Save uploaded files
                st.session_state.processing_status = "Saving uploaded files..."
                file_paths = self.file_manager.save_uploaded_files(uploaded_files)
                
                if not file_paths:
                    st.error("No files were saved successfully.")
                    return
                
                # Process files
                st.session_state.processing_status = "Processing documents..."
                success = self.processor.process_files(file_paths)
                
                if success:
                    st.session_state.documents_processed = True
                    st.session_state.processing_status = "Documents processed successfully!"
                    st.success("✅ Files processed successfully!")
                    logger.info(f"Successfully processed {len(file_paths)} files")
                else:
                    st.error("❌ Error processing files. Check logs for details.")
                    logger.error("Failed to process files")
                
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            logger.error(f"Error processing files: {str(e)}")
        finally:
            st.session_state.processing_status = ""
    
    def _process_question(self, question: str):
        """Process a user question."""
        try:
            with st.spinner("Generating answer..."):
                answer = self.processor.query(question)
                
                # Add to chat history
                st.session_state.chat_history.append((question, answer))
                
                # Display answer
                st.subheader("Answer:")
                st.write(answer)
                
                logger.info(f"Processed question: {question[:50]}...")
                
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")
            logger.error(f"Error processing question: {str(e)}")
    
    def _reset_system(self):
        """Reset the entire system."""
        try:
            # Reset processor
            if self.processor:
                self.processor.reset()
            
            # Reset session state
            st.session_state.processor_initialized = False
            st.session_state.documents_processed = False
            st.session_state.chat_history = []
            st.session_state.processing_status = ""
            
            # Reset file manager
            self.file_manager.cleanup()
            
            st.success("✅ System reset successfully!")
            logger.info("System reset")
            
        except Exception as e:
            st.error(f"❌ Error resetting system: {str(e)}")
            logger.error(f"Error resetting system: {str(e)}")

def main():
    """Main application entry point."""
    app = DocumentQAApp()
    app.run()

if __name__ == "__main__":
    main()
