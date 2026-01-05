"""
API Routes Package

Contains all FastAPI route handlers.
"""

from . import videos
from . import analysis
from . import generation
from . import thumbnails
from . import events

__all__ = ['videos', 'analysis', 'generation', 'thumbnails', 'events']
