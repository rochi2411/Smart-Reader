"""
Main entry point for the Advanced Document QA System.
"""
import os
import sys

# Set environment variable for PyTorch
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.streamlit_app import main

if __name__ == "__main__":
    main()
