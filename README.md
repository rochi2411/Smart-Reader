# Advanced Document QA System

## Overview

The Advanced Document QA System is a powerful document processing and question-answering application that allows users to upload various document formats, process them with advanced techniques, and ask questions about their content. The system uses modern NLP models and techniques to understand and respond to queries based on the uploaded documents.

## Features

### Document Processing
- **Multi-format support**: Process PDF, DOCX, PPTX, Excel, CSV, JSON, and image files
- **OCR Integration**: Extract text from images and scanned documents
- **Table Extraction**: Identify and parse tables in PDFs and other documents
- **Link Extraction**: Extract and optionally crawl hyperlinks found in documents
- **Image Processing**: Extract and OCR embedded images in documents

### Text Analysis
- **Intelligent Chunking**: Split documents into semantic chunks for better understanding
- **Vector Indexing**: Create embeddings and vector indexes for semantic search
- **Structured Data Analysis**: Special handling for tabular data like CSV and Excel files

### Question Answering
- **Contextual Responses**: Generate answers based on document context
- **Source Attribution**: Identify the source of information within documents
- **Multiple Models**: Support for different LLM and embedding models
- **Relevance Control**: Configure similarity thresholds for more accurate responses

## System Requirements

- Python 3.8+
- PyTorch
- 16GB+ RAM (8GB+ recommended for processing large documents)
- 8 GB + GPU
- Internet connection (for downloading models and crawling links)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/document-qa-system.git
cd document-qa-system
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR for image processing:
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

5. If using PDF table extraction, install additional dependencies:
   - **For Camelot**: `pip install camelot-py[cv]`
   - **For Tabula**: Java Runtime Environment (JRE)

## Configuration

The application allows for extensive configuration via the Streamlit interface. Key configurations include:

### Model Selection
- **Embedding Model**: Choose from various embedding models for document vectorization
- **LLM Model**: Select the language model for generating responses

### Processing Parameters
- **Chunk Size**: Configure text chunk size for processing (larger values provide more context)
- **Chunk Overlap**: Set overlap between chunks to maintain context continuity
- **OCR Usage**: Enable/disable OCR for images and scanned documents
- **Link Crawling**: Enable/disable crawling of hyperlinks found in documents

### Query Parameters
- **Top K Results**: Number of similar chunks to retrieve per query
- **Similarity Cutoff**: Minimum similarity threshold for document retrieval
- **LLM Temperature**: Control randomness in generated responses

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Using the application:
   - Access the web interface at `http://localhost:8501`
   - Configure processing parameters in the sidebar
   - Click "Initialize Processor" to set up the system
   - Upload documents using the file upload interface
   - Click "Process Files" to analyze the documents
   - Ask questions in the chat interface at the bottom

## Example Queries

After uploading and processing documents, try questions like:

- "What is the main topic of the document?"
- "Summarize the key points in the third section."
- "What data is shown in the table on page 4?"
- "Calculate the average value in the sales column."
- "What images are included in the presentation?"
- "What links are referenced in the document?"

## Class Structure

The application consists of two main classes:

### DocumentProcessor
Handles all document processing tasks including:
- Processing different file formats (PDF, DOCX, images, etc.)
- OCR for text extraction from images
- Table extraction from PDFs and other documents
- Hyperlink extraction and crawling
- Vector index creation and querying

### DocumentQAApp
Manages the Streamlit interface including:
- User interface components and layout
- File uploading and handling
- Configuration management
- Chat interface for querying

## Dependencies

Major libraries used include:
- **LlamaIndex**: Document indexing and retrieval
- **PyTorch/Hugging Face**: For embedding and language models
- **PyMuPDF (fitz)**: PDF processing
- **Tesseract**: OCR processing
- **Camelot/Tabula**: Table extraction from PDFs
- **pandas**: Data processing for structured files
- **Beautiful Soup**: Web scraping for link crawling
- **Streamlit**: Web interface

## Customization

### Adding New File Types
To support additional file formats, extend the `process_file` method in the `DocumentProcessor` class.

### Custom Embedding Models
Configure different embedding models by changing the `embedding_model` parameter when initializing the processor.

### Custom LLM Models
Use different language models by changing the `llm_model` parameter and ensuring the model is properly loaded.

## Troubleshooting

### Common Issues

1. **OCR not working properly**:
   - Ensure Tesseract is properly installed and in your PATH
   - Try adjusting image preprocessing settings in the `_ocr_image` method

2. **Memory issues with large files**:
   - Reduce chunk size in the configuration
   - Process fewer documents at once
   - Use a machine with more RAM

3. **Slow processing speed**:
   - Disable OCR and link crawling for faster processing
   - Use smaller embedding models
   - Reduce the number of documents processed simultaneously

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- LlamaIndex for the vector indexing capabilities
- Hugging Face for embedding and language models
- The Streamlit team for the web interface framework
- Various open-source projects for document processing components
