"""
Configure pytest for the HRAP test suite
"""
import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(project_root))
