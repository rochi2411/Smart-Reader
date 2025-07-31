# Migration Guide: Monolithic to Modular Architecture

## Overview

This guide helps you migrate from the monolithic `app.py` to the new modular architecture. The refactored system provides better maintainability, testability, and extensibility.

## 🔄 What Changed

### Before (Monolithic)
```
smart-doc/
├── app.py                    # ~1000+ lines, everything in one file
├── requirements.txt
├── Dockerfile
├── README.md
└── .env
```

### After (Modular)
```
smart-doc/
├── src/
│   ├── config/
│   │   └── settings.py       # Configuration management
│   ├── core/
│   │   └── document_processor.py  # Main processing logic
│   ├── processors/
│   │   ├── ocr_processor.py       # OCR functionality
│   │   ├── pdf_processor.py       # PDF processing
│   │   ├── office_processor.py    # Office documents
│   │   └── web_crawler.py         # Web crawling
│   ├── ui/
│   │   └── streamlit_app.py       # UI components
│   └── utils/
│       ├── file_utils.py          # File operations
│       └── logger.py              # Logging utilities
├── main.py                        # Entry point
├── setup.py                       # Package setup
├── requirements_modular.txt       # Updated dependencies
├── Dockerfile_modular            # Updated Docker config
└── README_MODULAR.md             # Updated documentation
```

## 🚀 Migration Steps

### Step 1: Backup Current System
```bash
# Create backup of current system
cp -r smart-doc smart-doc-backup
```

### Step 2: Update Dependencies
```bash
# Install new requirements
pip install -r requirements_modular.txt
```

### Step 3: Update Environment Variables
Your `.env` file remains the same:
```
API_KEY=your_google_api_key_here
```

### Step 4: Update Run Commands

**Old way:**
```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

**New way:**
```bash
python main.py
# or
streamlit run main.py --server.port=8501 --server.address=0.0.0.0
```

### Step 5: Update Docker Usage

**Old Dockerfile:**
```dockerfile
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**New Dockerfile:**
```bash
# Build with new Dockerfile
docker build -f Dockerfile_modular -t smart-doc .
docker run -p 8501:8501 smart-doc
```

## 🔧 Configuration Changes

### Old Configuration (Hardcoded)
```python
# In app.py - hardcoded values
chunk_size = 1024
use_ocr = True
temperature = 0.2
```

### New Configuration (Centralized)
```python
# In src/config/settings.py - configurable
@dataclass
class ProcessingConfig:
    chunk_size: int = 1024
    chunk_overlap: int = 100
    use_ocr: bool = True
    # ... more options

# Usage
config = AppConfig.default()
processor = DocumentProcessor(config)
```

## 📝 Code Migration Examples

### Example 1: Custom Document Processor

**Old way (modify app.py):**
```python
# Add to DocumentProcessor class in app.py
def process_custom_format(self, file_path):
    # Custom processing logic
    pass
```

**New way (create separate module):**
```python
# src/processors/custom_processor.py
class CustomProcessor:
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        # Custom processing logic
        return processed_content

# Register in src/core/document_processor.py
elif file_extension == '.custom':
    content_items = self.custom_processor.process_document(file_path)
```

### Example 2: Configuration Updates

**Old way:**
```python
# Modify constructor parameters in app.py
processor = DocumentProcessor(
    chunk_size=512,
    use_ocr=False
)
```

**New way:**
```python
# Use configuration objects
config = AppConfig(
    processing=ProcessingConfig(
        chunk_size=512,
        use_ocr=False
    )
)
processor = DocumentProcessor(config)
```

### Example 3: Logging

**Old way:**
```python
# Basic logging in app.py
import logging
logging.basicConfig(filename='app.log', level=logging.INFO)
logger = logging.getLogger(__name__)
```

**New way:**
```python
# Centralized logging utility
from src.utils.logger import setup_logger, get_logger

setup_logger(__name__, log_file="app.log")
logger = get_logger(__name__)
```

## 🧪 Testing Migration

### Verify Installation
```bash
# Test import
python -c "from src.ui.streamlit_app import main; print('Import successful')"

# Test configuration
python -c "from src.config.settings import AppConfig; print('Config loaded')"
```

### Run Application
```bash
# Start the application
python main.py
```

### Test Functionality
1. Upload a test document
2. Process the document
3. Ask a question
4. Verify the response

## 🔍 Troubleshooting Migration

### Common Issues

**1. Import Errors**
```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solution: Run from project root directory
cd /path/to/smart-doc
python main.py
```

**2. Configuration Errors**
```bash
# Error: Missing API_KEY
# Solution: Ensure .env file exists with API_KEY
echo "API_KEY=your_key_here" > .env
```

**3. Dependency Issues**
```bash
# Error: Module not found
# Solution: Install modular requirements
pip install -r requirements_modular.txt
```

**4. Path Issues**
```bash
# Error: File not found
# Solution: Check working directory
pwd  # Should be in smart-doc root
ls   # Should see main.py and src/ directory
```

## 📊 Performance Comparison

### Memory Usage
- **Monolithic**: All code loaded at startup
- **Modular**: Lazy loading of processors, better memory management

### Maintainability
- **Monolithic**: Single 1000+ line file
- **Modular**: Multiple focused modules, easier to maintain

### Extensibility
- **Monolithic**: Modify large file, risk breaking existing code
- **Modular**: Add new processors without touching existing code

## 🎯 Benefits of Migration

### For Developers
- **Easier debugging**: Isolated modules
- **Better testing**: Unit tests for individual components
- **Code reuse**: Processors can be used independently
- **Team collaboration**: Multiple developers can work on different modules

### For Users
- **Better error handling**: More specific error messages
- **Improved performance**: Optimized resource usage
- **More configuration options**: Fine-tune processing parameters
- **Better logging**: Detailed logs for troubleshooting

## 🔮 Future Enhancements

The modular architecture enables:

1. **Plugin System**: Easy addition of new document processors
2. **API Endpoints**: RESTful API alongside Streamlit UI
3. **Batch Processing**: Process multiple documents in parallel
4. **Cloud Integration**: Easy deployment to cloud platforms
5. **Monitoring**: Health checks and metrics collection

## 📞 Support

If you encounter issues during migration:

1. **Check logs**: Look at `app.log` for detailed error messages
2. **Verify environment**: Ensure all dependencies are installed
3. **Test components**: Test individual modules before full integration
4. **Rollback option**: Keep the backup of your original system

## ✅ Migration Checklist

- [ ] Backup current system
- [ ] Install new dependencies
- [ ] Update run commands
- [ ] Test basic functionality
- [ ] Verify document processing
- [ ] Test Q&A functionality
- [ ] Update deployment scripts
- [ ] Update documentation
- [ ] Train team on new structure

---

**Migration completed successfully!** 🎉

Your Document QA System is now running on a professional, modular architecture that's easier to maintain, extend, and deploy.
