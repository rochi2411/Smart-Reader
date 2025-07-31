# Advanced Document QA System - Modular Architecture

## Overview

The Advanced Document QA System has been refactored into a professional, modular architecture that follows software engineering best practices. This system allows users to upload various document formats, process them with advanced AI techniques, and ask questions about their content.

## 🏗️ Architecture

### Modular Structure

```
src/
├── config/
│   └── settings.py          # Configuration management
├── core/
│   └── document_processor.py # Main document processing logic
├── processors/
│   ├── ocr_processor.py     # OCR functionality
│   ├── pdf_processor.py     # PDF processing
│   ├── office_processor.py  # Office documents (DOCX, PPTX, Excel)
│   └── web_crawler.py       # Web content extraction
├── ui/
│   └── streamlit_app.py     # Streamlit user interface
└── utils/
    ├── file_utils.py        # File management utilities
    └── logger.py            # Logging utilities
```

### Key Improvements

- **Separation of Concerns**: Each module has a specific responsibility
- **Configuration Management**: Centralized configuration with dataclasses
- **Error Handling**: Comprehensive error handling and logging
- **Type Hints**: Full type annotations for better code quality
- **Resource Management**: Proper cleanup and resource management
- **Extensibility**: Easy to add new document processors or features

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/rochi2411/smart-doc.git
cd smart-doc
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements_modular.txt
```

4. **Install system dependencies:**
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

5. **Configure API key:**
   Create a `.env` file in the root directory:
   ```
   API_KEY=your_google_api_key_here
   ```

### Running the Application

**Development mode:**
```bash
python main.py
# or
streamlit run main.py --server.port=8501 --server.address=0.0.0.0
```

**Docker deployment:**
```bash
docker build -f Dockerfile_modular -t smart-doc .
docker run -p 8501:8501 smart-doc
```

**Production setup:**
```bash
pip install -e .
smart-doc
```

## 📋 Features

### Document Processing
- **Multi-format Support**: PDF, DOCX, PPTX, Excel, CSV, JSON, images
- **OCR Integration**: Extract text from images and scanned documents
- **Table Extraction**: Advanced table parsing from PDFs and documents
- **Link Crawling**: Extract and crawl hyperlinks for additional context
- **Structured Data**: Special handling for CSV/Excel with data analysis

### AI-Powered Analysis
- **Semantic Search**: Vector-based document retrieval
- **Contextual Responses**: AI-generated answers based on document content
- **Source Attribution**: Track information sources within documents
- **Configurable Models**: Support for different embedding and LLM models

### User Interface
- **Interactive Web App**: Clean, intuitive Streamlit interface
- **Real-time Processing**: Live status updates during document processing
- **Chat Interface**: Conversational Q&A with chat history
- **Configuration Panel**: Adjust processing parameters on-the-fly

## 🔧 Configuration

### Model Configuration
```python
@dataclass
class ModelConfig:
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_model: str = "models/gemini-1.5-flash"
    temperature: float = 0.2
    api_key: Optional[str] = None
```

### Processing Configuration
```python
@dataclass
class ProcessingConfig:
    chunk_size: int = 1024
    chunk_overlap: int = 100
    use_ocr: bool = True
    crawl_links: bool = True
    max_links_to_crawl: int = 5
```

### Query Configuration
```python
@dataclass
class QueryConfig:
    similarity_top_k: int = 5
    similarity_cutoff: float = 0.7
```

## 🧪 Development

### Code Quality Tools

**Install development dependencies:**
```bash
pip install -e ".[dev]"
```

**Code formatting:**
```bash
black src/
```

**Linting:**
```bash
flake8 src/
```

**Type checking:**
```bash
mypy src/
```

**Testing:**
```bash
pytest tests/
```

### Adding New Processors

To add support for new document types:

1. Create a new processor in `src/processors/`
2. Implement the processing logic
3. Register it in `DocumentProcessor.process_single_file()`
4. Add file extension to `FileManager.is_supported_file()`

Example:
```python
# src/processors/new_processor.py
class NewDocumentProcessor:
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        # Implementation here
        pass
```

## 📊 Performance Considerations

### Memory Management
- Automatic cleanup of temporary files
- Efficient document chunking
- Resource pooling for AI models

### Scalability
- Configurable processing parameters
- Batch processing capabilities
- Docker containerization for deployment

### Optimization Tips
- Disable OCR for text-only documents
- Adjust chunk size based on document complexity
- Use smaller models for faster processing
- Limit link crawling for large documents

## 🐛 Troubleshooting

### Common Issues

**1. OCR not working:**
- Ensure Tesseract is installed and in PATH
- Check image quality and format
- Try adjusting OCR preprocessing settings

**2. Memory issues:**
- Reduce chunk size in configuration
- Process fewer documents simultaneously
- Use smaller AI models

**3. API errors:**
- Verify API key in `.env` file
- Check internet connection
- Monitor API usage limits

**4. Import errors:**
- Ensure all dependencies are installed
- Check Python path configuration
- Verify virtual environment activation

### Logging

Logs are written to `app.log` by default. Configure logging level in settings:

```python
config = AppConfig(
    log_level="DEBUG",  # DEBUG, INFO, WARNING, ERROR
    log_file="custom.log"
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes following the coding standards
4. Add tests for new functionality
5. Run quality checks: `black`, `flake8`, `mypy`, `pytest`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- **LlamaIndex**: Document indexing and retrieval framework
- **Google Gemini**: AI models for embeddings and text generation
- **Streamlit**: Web application framework
- **Open Source Community**: Various document processing libraries

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

---

**Version**: 2.0.0 (Modular Architecture)  
**Last Updated**: 2024
