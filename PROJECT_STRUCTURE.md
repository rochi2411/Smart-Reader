# Project Structure Documentation

## 📁 Directory Structure

```
smart-doc/
├── 📁 src/                          # Source code directory
│   ├── 📁 config/                   # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py              # App configuration classes
│   ├── 📁 core/                     # Core business logic
│   │   ├── __init__.py
│   │   └── document_processor.py    # Main document processing orchestrator
│   ├── 📁 processors/               # Document processing modules
│   │   ├── __init__.py
│   │   ├── ocr_processor.py         # OCR text extraction
│   │   ├── pdf_processor.py         # PDF document processing
│   │   ├── office_processor.py      # Office documents (DOCX, PPTX, Excel)
│   │   └── web_crawler.py           # Web content extraction
│   ├── 📁 ui/                       # User interface components
│   │   ├── __init__.py
│   │   └── streamlit_app.py         # Streamlit web application
│   └── 📁 utils/                    # Utility modules
│       ├── __init__.py
│       ├── file_utils.py            # File management utilities
│       └── logger.py                # Logging configuration
├── 📁 tests/                        # Test suite
│   ├── __init__.py
│   ├── 📁 unit/                     # Unit tests
│   │   └── test_config.py
│   ├── 📁 integration/              # Integration tests
│   └── 📁 fixtures/                 # Test data and fixtures
├── 📄 main.py                       # Application entry point
├── 📄 setup.py                      # Package setup configuration
├── 📄 requirements_modular.txt      # Python dependencies
├── 📄 Dockerfile_modular           # Docker configuration
├── 📄 README_MODULAR.md            # Updated documentation
├── 📄 MIGRATION_GUIDE.md           # Migration instructions
├── 📄 PROJECT_STRUCTURE.md         # This file
├── 📄 .env                         # Environment variables (not in git)
├── 📄 .gitignore                   # Git ignore rules
└── 📄 app.py                       # Original monolithic file (for reference)
```

## 🏗️ Module Responsibilities

### 📁 src/config/
**Purpose**: Centralized configuration management
- `settings.py`: Configuration classes using dataclasses
  - `ModelConfig`: AI model settings
  - `ProcessingConfig`: Document processing parameters
  - `QueryConfig`: Query and retrieval settings
  - `AppConfig`: Main application configuration

### 📁 src/core/
**Purpose**: Core business logic and orchestration
- `document_processor.py`: Main processor that coordinates all operations
  - File type routing
  - Vector index creation
  - Query processing
  - Resource management

### 📁 src/processors/
**Purpose**: Specialized document processing modules
- `ocr_processor.py`: OCR functionality
  - Image text extraction
  - Image preprocessing
  - Text cleaning and validation
- `pdf_processor.py`: PDF document handling
  - Text extraction
  - Table extraction (Camelot)
  - Image processing within PDFs
  - Link extraction and crawling
- `office_processor.py`: Microsoft Office documents
  - DOCX text and table extraction
  - PPTX slide processing
  - Excel/CSV data processing
  - JSON data handling
- `web_crawler.py`: Web content extraction
  - URL validation and crawling
  - Content cleaning
  - Rate limiting

### 📁 src/ui/
**Purpose**: User interface components
- `streamlit_app.py`: Complete Streamlit web application
  - Configuration sidebar
  - File upload interface
  - Processing status display
  - Chat interface for Q&A

### 📁 src/utils/
**Purpose**: Utility functions and helpers
- `file_utils.py`: File management operations
  - Temporary file handling
  - File type validation
  - Upload management
- `logger.py`: Logging configuration
  - Logger setup and configuration
  - Centralized logging utilities

### 📁 tests/
**Purpose**: Test suite organization
- `unit/`: Unit tests for individual modules
- `integration/`: Integration tests for module interactions
- `fixtures/`: Test data and mock objects

## 🔄 Data Flow

```
User Upload → FileManager → DocumentProcessor → Specialized Processors → Vector Index → Query Engine → Response
```

### Detailed Flow:
1. **User uploads files** via Streamlit interface
2. **FileManager** saves files to temporary directory
3. **DocumentProcessor** routes files to appropriate processors
4. **Specialized Processors** extract content:
   - PDFProcessor for PDF files
   - OfficeProcessor for Office documents
   - OCRProcessor for images
5. **Content is converted** to LlamaIndex Documents
6. **Vector Index** is created from documents
7. **User queries** are processed through the index
8. **Responses** are generated and displayed

## 🔧 Configuration Flow

```
.env → ModelConfig → AppConfig → DocumentProcessor → Specialized Processors
```

### Configuration Hierarchy:
1. **Environment variables** loaded from `.env`
2. **Configuration classes** instantiated with defaults
3. **AppConfig** combines all configuration sections
4. **DocumentProcessor** receives configuration
5. **Specialized processors** inherit relevant settings

## 🧩 Extension Points

### Adding New Document Types:
1. Create processor in `src/processors/`
2. Implement processing logic
3. Register in `DocumentProcessor.process_single_file()`
4. Add file extension to `FileManager.is_supported_file()`

### Adding New AI Models:
1. Update `ModelConfig` with new model options
2. Modify model initialization in `DocumentProcessor`
3. Update UI dropdown options

### Adding New Features:
1. Create utility functions in `src/utils/`
2. Add configuration options in `src/config/`
3. Implement feature in appropriate processor
4. Update UI if needed

## 📊 Dependencies

### Core Dependencies:
- **LlamaIndex**: Document indexing and AI integration
- **Streamlit**: Web interface framework
- **Google Gemini**: AI models for embeddings and text generation

### Processing Dependencies:
- **PyMuPDF**: PDF processing
- **python-docx/pptx**: Office document processing
- **pandas**: Data manipulation
- **Pillow/OpenCV**: Image processing
- **pytesseract**: OCR functionality
- **camelot**: Table extraction
- **BeautifulSoup**: Web scraping

### Development Dependencies:
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking

## 🔒 Security Considerations

### API Key Management:
- API keys stored in `.env` file
- Environment variables used for configuration
- No hardcoded secrets in source code

### File Handling:
- Temporary files cleaned up automatically
- File type validation before processing
- Size limits on uploaded files

### Web Crawling:
- Rate limiting for external requests
- URL validation before crawling
- Content size limits

## 🚀 Deployment Options

### Development:
```bash
python main.py
```

### Docker:
```bash
docker build -f Dockerfile_modular -t smart-doc .
docker run -p 8501:8501 smart-doc
```

### Production:
```bash
pip install -e .
smart-doc
```

## 📈 Performance Characteristics

### Memory Usage:
- Lazy loading of processors
- Automatic cleanup of temporary files
- Configurable chunk sizes

### Processing Speed:
- Parallel processing where possible
- Efficient vector indexing
- Optimized OCR preprocessing

### Scalability:
- Modular architecture supports horizontal scaling
- Docker containerization
- Configurable resource limits

---

This modular architecture provides a solid foundation for a professional document QA system that's maintainable, extensible, and production-ready.
