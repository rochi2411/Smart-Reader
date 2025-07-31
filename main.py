"""
Advanced Document QA System - Production Ready
A modular document processing and question-answering system using Google AI.
"""
import os
import sys

# Set environment variable for PyTorch
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from typing import List, Optional
import traceback

from src.config.settings import AppConfig, ModelConfig, ProcessingConfig, QueryConfig
from src.core.document_processor import UpdatedDocumentProcessor
from src.utils.file_utils import FileManager
from src.utils.logger import setup_logger, get_logger

# Setup logging
setup_logger(__name__, log_file="app.log")
logger = get_logger(__name__)

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Advanced Document QA System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 Advanced Document QA System")
    st.markdown("**Created by Rochisnu Dutta**")
    
    
    # Initialize session state
    if 'processor_instance' not in st.session_state:
        st.session_state.processor_instance = None
    if 'processor_initialized' not in st.session_state:
        st.session_state.processor_initialized = False
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'initialization_error' not in st.session_state:
        st.session_state.initialization_error = None
    
    # Check API key
    api_key = os.getenv("API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        st.error("🔑 **API Key Required**: Please set your Google API key in the .env file")
        
        with st.expander("📝 How to set up API key"):
            st.markdown("""
            1. Create or edit the `.env` file in the project root
            2. Add your Google API key:
               ```
               API_KEY=your_actual_google_api_key_here
               ```
            3. Restart the application
            
            **Get your API key from**: [Google AI Studio](https://makersuite.google.com/app/apikey)
            """)
        return
    
    # Show features
    with st.expander("🚀 System Features"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📄 Document Processing:**
            - PDF text extraction
            - Image OCR processing
            - Text file processing
            - Intelligent chunking
            """)
        
        with col2:
            st.markdown("""
            **🤖 AI Features:**
            - Google Gemini integration
            - Semantic search
            - Contextual Q&A
            - Chat history
            """)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model settings
    st.sidebar.subheader("🤖 Model Settings")
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["models/embedding-001", "models/text-embedding-004"],
        index=0,
        help="Google AI embedding model for document vectorization"
    )
    
    llm_model = st.sidebar.selectbox(
        "Language Model", 
        [
            "models/gemini-2.5-flash",      # Latest and fastest
            "models/gemini-1.5-flash", 
            "models/gemini-1.5-pro",
            "models/gemini-pro"
        ],
        index=0,  # Default to Gemini 2.5 Flash
        help="Google AI language model for generating responses"
    )
    
    # Show model info
    if llm_model == "models/gemini-2.5-flash":
        st.sidebar.info("🚀 Using Gemini 2.5 Flash - Latest and fastest model!")
    elif llm_model == "models/gemini-1.5-flash":
        st.sidebar.info("⚡ Using Gemini 1.5 Flash - Fast and efficient")
    elif llm_model == "models/gemini-1.5-pro":
        st.sidebar.info("🧠 Using Gemini 1.5 Pro - Most capable model")
    
    temperature = st.sidebar.slider(
        "Temperature", 
        0.0, 1.0, 0.2, 0.1,
        help="Controls randomness in responses (lower = more focused)"
    )
    
    # Processing settings
    st.sidebar.subheader("📄 Processing Settings")
    chunk_size = st.sidebar.slider(
        "Chunk Size", 
        256, 2048, 1024, 128,
        help="Size of text chunks for processing"
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap", 
        0, 200, 100, 20,
        help="Overlap between consecutive chunks"
    )
    use_ocr = st.sidebar.checkbox(
        "Enable OCR", 
        True,
        help="Extract text from images using OCR"
    )
    
    # Query settings
    st.sidebar.subheader("🔍 Query Settings")
    similarity_top_k = st.sidebar.slider(
        "Top K Results", 
        1, 10, 5,
        help="Number of similar chunks to retrieve for each query"
    )
    
    # Initialize button
    if st.sidebar.button("🚀 Initialize Processor", type="primary"):
        try:
            st.session_state.initialization_error = None
            
            with st.spinner(f"Initializing system with {llm_model.split('/')[-1]}..."):
                config = AppConfig(
                    model=ModelConfig(
                        embedding_model=embedding_model,
                        llm_model=llm_model,
                        temperature=temperature
                    ),
                    processing=ProcessingConfig(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        use_ocr=use_ocr,
                        crawl_links=False,
                        max_links_to_crawl=0
                    ),
                    query=QueryConfig(
                        similarity_top_k=similarity_top_k,
                        similarity_cutoff=0.7
                    )
                )
                
                processor = UpdatedDocumentProcessor(config)
                
                if processor.is_ready():
                    st.session_state.processor_instance = processor
                    st.session_state.processor_initialized = True
                    st.success(f"✅ System initialized successfully with {llm_model.split('/')[-1]}!")
                    logger.info(f"Document processor initialized with {llm_model}")
                else:
                    raise Exception("Processor not ready after initialization")
                    
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            st.session_state.initialization_error = traceback.format_exc()
            st.session_state.processor_initialized = False
            st.session_state.processor_instance = None
            logger.error(f"Initialization failed: {str(e)}")
    
    # Reset button
    if st.sidebar.button("🔄 Reset System"):
        if st.session_state.processor_instance:
            st.session_state.processor_instance.reset()
        st.session_state.processor_initialized = False
        st.session_state.processor_instance = None
        st.session_state.documents_processed = False
        st.session_state.chat_history = []
        st.session_state.initialization_error = None
        st.success("✅ System reset successfully!")
        logger.info("System reset")
    
    # Show status
    # if st.session_state.processor_instance:
    #     st.sidebar.subheader("📊 System Status")
    #     status = st.session_state.processor_instance.get_initialization_status()
    #     for key, value in status.items():
    #         if value:
    #             st.sidebar.success(f"✅ {key.replace('_', ' ').title()}")
    #         else:
    #             st.sidebar.error(f"❌ {key.replace('_', ' ').title()}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Document Upload")
        
        # Show initialization error
        if st.session_state.initialization_error:
            st.error("❌ Initialization Error:")
            with st.expander("🔍 Error Details"):
                st.code(st.session_state.initialization_error)
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload documents in supported formats"
        )
        
        # Process files
        if uploaded_files and st.button("📊 Process Files", type="primary"):
            processor = st.session_state.processor_instance
            
            if not processor or not processor.is_ready():
                st.error("Please initialize the processor first using the sidebar.")
                return
            
            try:
                with st.spinner("Processing files..."):
                    file_manager = FileManager()
                    file_paths = file_manager.save_uploaded_files(uploaded_files)
                    
                    if file_paths:
                        success = processor.process_files(file_paths)
                        
                        if success:
                            st.session_state.documents_processed = True
                            st.success("✅ Files processed successfully!")
                            
                            stats = processor.get_document_stats()
                            st.info(f"Created {stats['total_documents']} document chunks")
                            logger.info(f"Successfully processed {len(file_paths)} files")
                        else:
                            st.error("❌ Processing failed. Check logs for details.")
                            logger.error("File processing failed")
                    else:
                        st.error("No files were saved successfully.")
                        
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")
                logger.error(f"Error processing files: {str(e)}")
        
        # Show uploaded files
        if uploaded_files:
            st.subheader("📋 Uploaded Files")
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size:,} bytes)")
    
    with col2:
        st.header("📊 System Status")
        
        processor = st.session_state.processor_instance
        
        if processor and processor.is_ready():
            st.success("✅ System Ready")
            # Show current model
            if hasattr(processor, 'config') and processor.config:
                current_model = processor.config.model.llm_model.split('/')[-1]
                st.info(f"🤖 Using: {current_model}")
        elif processor:
            st.warning("⚠️ System Partially Ready")
        else:
            st.error("❌ System Not Initialized")
        
        if st.session_state.documents_processed:
            st.success("✅ Documents Processed")
            if processor:
                stats = processor.get_document_stats()
                st.metric("Total Documents", stats['total_documents'])
                # if stats['index_created']:
                #     st.success("✅ Vector Index Created")
        else:
            st.info("ℹ️ No Documents Processed")
    
    # Chat interface
    st.header("💬 Ask Questions")
    
    if not st.session_state.documents_processed:
        st.info("Please upload and process documents before asking questions.")
        return
    
    processor = st.session_state.processor_instance
    if not processor or not processor.is_ready():
        st.error("System is not ready for queries. Please check the system status.")
        return
    
    # Chat history
    if st.session_state.chat_history:
        st.subheader("📝 Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {q[:50]}..."):
                st.write(f"**Question:** {q}")
                st.write(f"**Answer:** {a}")
    
    # Question input
    question = st.text_input(
        "Ask a question about your documents:",
        placeholder="What is the main topic of the document?",
        key="question_input"
    )
    
    if st.button("🔍 Search") and question:
        try:
            with st.spinner("Generating answer..."):
                answer = processor.query(question)
                st.session_state.chat_history.append((question, answer))
                
                st.subheader("💡 Answer:")
                st.write(answer)
                
                logger.info(f"Processed question: {question[:50]}...")
                
        except Exception as e:
            st.error(f"❌ Query error: {str(e)}")
            logger.error(f"Query error: {str(e)}")

if __name__ == "__main__":
    main()
