# 📚 Advanced Document QA System

A professional, modular document processing and question-answering system powered by Google AI. Upload documents, process them with advanced AI techniques, and ask questions about their content.

## 🚀 Features

### 📄 Document Processing
- **Multi-format Support**: PDF, TXT, and image files (PNG, JPG, JPEG, TIFF, BMP)
- **OCR Integration**: Extract text from images and scanned documents using Tesseract
- **Intelligent Chunking**: Split documents into semantic chunks for better understanding
- **Vector Indexing**: Create embeddings for semantic search and retrieval

### 🤖 AI-Powered Analysis
- **Google Gemini Integration**: Latest Google AI models including **Gemini 2.5 Flash**
- **Semantic Search**: Find relevant document sections based on meaning, not just keywords
- **Contextual Q&A**: Generate accurate answers based on document content
- **Chat History**: Persistent conversation history for better user experience

### 🏗️ Professional Architecture
- **Modular Design**: Clean separation of concerns with specialized components
- **Error Handling**: Comprehensive error handling and logging
- **Configuration Management**: Centralized, type-safe configuration
- **Resource Management**: Automatic cleanup and memory management

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Tesseract OCR (for image processing)

### System Dependencies

**Linux:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Python Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd smart-doc
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure API key:**
Create a `.env` file in the project root:
```bash
echo "API_KEY=your_google_api_key_here" > .env
```

## 🚀 Usage

### Starting the Application
```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`

### Using the System

1. **Initialize the Processor:**
   - Configure model settings in the sidebar
   - Choose from available Gemini models (including **Gemini 2.5 Flash**)
   - Click "🚀 Initialize Processor"
   - Wait for successful initialization

2. **Upload Documents:**
   - Use the file uploader to select documents
   - Supported formats: PDF, TXT, PNG, JPG, JPEG, TIFF, BMP
   - Click "📊 Process Files"

3. **Ask Questions:**
   - Use the chat interface at the bottom
   - Ask questions about your document content
   - View answers and chat history

### Example Queries
- "What is the main topic of this document?"
- "Summarize the key points"
- "What are the conclusions mentioned?"
- "Extract the important dates and numbers"

## ⚙️ Configuration

### Model Settings
- **Embedding Model**: Choose between `models/embedding-001` or `models/text-embedding-004`
- **Language Model**: Select from:
  - `models/gemini-2.5-flash` ⚡ **Latest and fastest** (Default)
  - `models/gemini-1.5-flash` - Fast and efficient
  - `models/gemini-1.5-pro` - Most capable
  - `models/gemini-pro` - Standard model
- **Temperature**: Control randomness in responses (0.0 = focused, 1.0 = creative)

### Processing Settings
- **Chunk Size**: Size of text chunks for processing (256-2048 characters)
- **Chunk Overlap**: Overlap between consecutive chunks (0-200 characters)
- **Enable OCR**: Extract text from images and scanned documents

### Query Settings
- **Top K Results**: Number of similar chunks to retrieve for each query (1-10)

## 🆕 What's New - Gemini 2.5 Flash

### **🚀 Gemini 2.5 Flash Benefits:**
- **Faster Response Times**: Up to 2x faster than previous models
- **Improved Accuracy**: Better understanding of context and nuance
- **Enhanced Reasoning**: Superior logical reasoning capabilities
- **Better Code Understanding**: Improved performance on technical documents
- **Cost Effective**: Optimized for production use

### **When to Use Each Model:**
- **Gemini 2.5 Flash**: Best for most use cases - fast, accurate, and cost-effective
- **Gemini 1.5 Pro**: Use for complex reasoning tasks requiring maximum capability
- **Gemini 1.5 Flash**: Good balance of speed and capability for standard tasks

## 🏗️ Project Structure

```
smart-doc/
├── main.py                   # Main application with complete UI
├── requirements.txt          # Python dependencies
├── README.md                # This documentation
├── Dockerfile               # Docker configuration
├── .env                     # Environment variables (create this)
├── .gitignore              # Git ignore rules
├── app.log                 # Application logs (auto-generated)
└── src/                    # Core business logic modules
    ├── config/
    │   └── settings.py     # Configuration management
    ├── core/
    │   └── document_processor.py  # Main document processing logic
    ├── processors/
    │   ├── ocr_processor.py       # OCR functionality
    │   ├── pdf_processor.py       # PDF processing
    │   ├── office_processor.py    # Office document processing
    │   └── web_crawler.py         # Web content extraction
    └── utils/
        ├── file_utils.py          # File management utilities
        └── logger.py              # Logging configuration
```

### Architecture Design

- **`main.py`**: Contains the complete Streamlit web interface and application logic
- **`src/config/`**: Centralized configuration management with support for all Gemini models
- **`src/core/`**: Main document processing engine with AI integration
- **`src/processors/`**: Specialized processors for different document types and operations
- **`src/utils/`**: Utility functions for file management and logging

## 🐳 Docker Deployment

### Build and Run
```bash
# Build the Docker image
docker build -t smart-doc .

# Run the container
docker run -p 8501:8501 -e API_KEY=your_google_api_key_here smart-doc
```

### Docker Compose (Optional)
Create a `docker-compose.yml`:
```yaml
version: '3.8'
services:
  smart-doc:
    build: .
    ports:
      - "8501:8501"
    environment:
      - API_KEY=your_google_api_key_here
    volumes:
      - ./logs:/app/logs
```

Run with: `docker-compose up`

## 🔧 Development

### Code Structure
The application follows a clean, modular architecture:

- **Main Application**: `main.py` contains the complete Streamlit interface
- **Configuration Layer**: Centralized settings management in `src/config/`
- **Core Layer**: Main business logic and document processing in `src/core/`
- **Processor Layer**: Specialized document processors in `src/processors/`
- **Utils Layer**: Utility functions and helpers in `src/utils/`

### Adding New Document Types
1. Create a new processor in `src/processors/`
2. Implement the processing logic following existing patterns
3. Register it in `DocumentProcessor.process_single_file()`
4. Add the file extension to `FileManager.is_supported_file()`

### Adding New AI Models
1. Update model options in `src/config/settings.py`
2. Add the model to the dropdown in `main.py`
3. Test the integration

### Extending Functionality
- **New Features**: Add new processors or extend existing ones in `src/processors/`
- **UI Enhancements**: Modify `main.py` to add new interface elements
- **Configuration Options**: Update `src/config/settings.py` for new settings

## 🐛 Troubleshooting

### Common Issues

**1. API Key Errors**
```
Error: API key is not set
```
**Solution**: Ensure your Google API key is correctly set in the `.env` file

**2. OCR Not Working**
```
Error: Tesseract not found
```
**Solution**: Install Tesseract OCR and ensure it's in your system PATH

**3. Memory Issues**
```
Error: Out of memory
```
**Solution**: Reduce chunk size or process fewer documents at once

**4. Import Errors**
```
Error: No module named 'src'
```
**Solution**: Run the application from the project root directory

### Debugging
- Check `app.log` for detailed error messages
- Use the system status panel to verify component initialization
- Enable debug mode by expanding error details in the UI

## 📊 Performance Tips

### Optimization
- **Model Selection**: Use **Gemini 2.5 Flash** for best speed/accuracy balance
- **Chunk Size**: Larger chunks provide more context but use more memory
- **OCR**: Disable OCR if processing only text documents
- **File Size**: Process smaller files for better performance

### Resource Usage
- **Memory**: ~2-4GB RAM recommended for typical usage
- **Storage**: Temporary files are automatically cleaned up
- **Network**: Requires internet connection for Google AI API calls

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes following the existing code style
4. Test your changes thoroughly
5. Submit a pull request with a clear description

### Development Guidelines
- Follow the existing modular structure
- Add comprehensive error handling
- Include type hints for better code quality
- Update documentation for new features
- Test with different document types and models

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google AI**: For providing the Gemini models and APIs, including the latest Gemini 2.5 Flash
- **LlamaIndex**: For the document indexing and retrieval framework
- **Streamlit**: For the web application framework
- **Tesseract**: For OCR capabilities
- **PyMuPDF**: For PDF processing
- **Open Source Community**: For the various libraries and tools used

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review the application logs in `app.log`

---

**Version**: 2.1.0 (Gemini 2.5 Flash Support)  
**Last Updated**: 2024  
**Status**: Production Ready ✅
