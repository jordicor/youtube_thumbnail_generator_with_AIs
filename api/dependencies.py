"""
FastAPI Dependencies

Shared dependencies for route handlers.
"""

from pathlib import Path
import sys

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import HTTPException

from database.db import get_db
from i18n.i18n import translate as t
from services.analysis_service import ANALYSIS_ACTIVE_STATES


def video_taken_error(status: str) -> HTTPException:
    """Why an analysis cannot take a video (AnalysisService.claim_analysis refused it).

    Args:
        status: The video's status when it was refused
    """
    if status == 'generating':
        return HTTPException(status_code=409, detail=t('api.errors.generation_in_progress'))
    if status in ANALYSIS_ACTIVE_STATES:
        return HTTPException(status_code=409, detail=t('api.errors.analysis_in_progress'))
    # A cancelled analysis whose worker has not stopped yet
    return HTTPException(status_code=409, detail=t('api.errors.analysis_stopping'))


# Re-export database dependency
__all__ = ['get_db', 'video_taken_error']
