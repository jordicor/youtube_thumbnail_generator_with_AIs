"""
FastAPI Dependencies

Shared dependencies for route handlers.
"""

from pathlib import Path
import sys

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from database.db import get_db


# Re-export database dependency
__all__ = ['get_db']
