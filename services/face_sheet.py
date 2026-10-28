"""
Face Sheet

A person's face seen from several angles and with several expressions, like
an artist's model sheet. The views are picked from the faces the analysis
detected (faces.json) and composed into one image that generation sends as
one more reference of the person.
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

from PIL import Image

from services.face_index import MIN_IDENTITY_SIMILARITY, FaceMatch, face_crop_box


@dataclass(frozen=True)
class SheetView:
    """Which detected faces fit one cell of the sheet."""
    name: str
    description: str               # What the image model is told the cell shows
    min_yaw: float                 # Head yaw range in degrees
    max_yaw: float
    expressions: Optional[FrozenSet[str]] = None  # None accepts any expression


# Cell order on the sheet: left to right, top to bottom
SHEET_VIEWS: Tuple[SheetView, ...] = (
    SheetView('front', 'front view', -12, 12, frozenset({'neutral', 'mouth_closed'})),
    SheetView('three_quarter_a', 'three-quarter view', 22, 55),
    SheetView('three_quarter_b', 'three-quarter view from the other side', -55, -22),
    SheetView('profile', 'profile view', 55, 90),
    SheetView('smiling', 'smiling', -25, 25, frozenset({'smiling'})),
    SheetView('mouth_open', 'talking with the mouth open', -30, 30, frozenset({'mouth_open'})),
)
VIEW_NAMES = tuple(view.name for view in SHEET_VIEWS)
MIN_SHEET_VIEWS = 2
# Smaller faces give blurry cells
MIN_FACE_SIDE = 48

CELL_SIZE = 320
CELL_GAP = 8
BACKGROUND = (110, 110, 110)


def _fits(view: SheetView, match: FaceMatch) -> bool:
    if match.yaw is None or match.similarity is None or match.similarity < MIN_IDENTITY_SIMILARITY:
        return False
    x1, y1, x2, y2 = match.box
    if min(x2 - x1, y2 - y1) < MIN_FACE_SIDE:
        return False
    yaw = abs(match.yaw) if view.name == 'profile' else match.yaw
    if not view.min_yaw <= yaw <= view.max_yaw:
        return False
    return view.expressions is None or match.expression in view.expressions


def view_candidates(matches: Dict[str, FaceMatch], limit: int = 24) -> Dict[str, List[str]]:
    """Frames whose face fits each view, best quality first."""
    ranked = sorted(matches.items(), key=lambda item: (-item[1].quality_score, item[0]))
    return {
        view.name: [name for name, match in ranked if _fits(view, match)][:limit]
        for view in SHEET_VIEWS
    }


def automatic_views(
    candidates: Dict[str, List[str]],
    usable: Callable[[str], bool]
) -> Dict[str, Optional[str]]:
    """The best usable frame of each view, using every frame at most once.

    usable is asked lazily, best candidates first (it may open the image).
    """
    used = set()
    picks = {}
    for view in SHEET_VIEWS:
        pick = next((name for name in candidates[view.name] if name not in used and usable(name)), None)
        picks[view.name] = pick
        if pick:
            used.add(pick)
    return picks


def resolve_views(
    automatic: Dict[str, Optional[str]],
    overrides: Dict[str, Optional[str]],
    members: set
) -> List[Tuple[str, str]]:
    """Filled cells as (view, frame filename): the person's choice, else the automatic one.

    An override of None empties the cell on purpose.
    """
    cells = []
    for view in SHEET_VIEWS:
        if view.name in overrides:
            frame = overrides[view.name]
            if frame is not None and frame not in members:
                raise ValueError(f"Face sheet view {view.name} uses a frame outside the person: {frame}")
        else:
            frame = automatic[view.name]
        if frame:
            cells.append((view.name, frame))
    return cells


def describe_views(views: List[str]) -> str:
    """Text for the image model: what each cell shows, in reading order."""
    descriptions = {view.name: view.description for view in SHEET_VIEWS}
    return ', '.join(descriptions[name] for name in views)


def render_face_sheet(cells: List[Tuple[Path, Optional[List[float]]]]) -> bytes:
    """Lossless PNG grid of face crops, one cell per (frame path, face box)."""
    count = len(cells)
    columns = 2 if count in (2, 4) else min(count, 3)
    rows = -(-count // columns)
    sheet = Image.new(
        'RGB',
        (columns * CELL_SIZE + (columns + 1) * CELL_GAP, rows * CELL_SIZE + (rows + 1) * CELL_GAP),
        BACKGROUND,
    )
    for position, (frame_path, box) in enumerate(cells):
        with Image.open(frame_path) as image:
            width, height = image.size
            if box:
                region = face_crop_box(box, width, height)
            else:
                side = min(width, height)
                region = ((width - side) // 2, (height - side) // 2, (width + side) // 2, (height + side) // 2)
            cell = image.crop(region).convert('RGB').resize((CELL_SIZE, CELL_SIZE), Image.Resampling.LANCZOS)
        row, column = divmod(position, columns)
        sheet.paste(cell, (CELL_GAP + column * (CELL_SIZE + CELL_GAP), CELL_GAP + row * (CELL_SIZE + CELL_GAP)))

    buffer = io.BytesIO()
    sheet.save(buffer, 'PNG')
    return buffer.getvalue()


def render_selected_faces(images: list[bytes]) -> bytes:
    """Experimental layout of the exact selected crops, without extra cropping."""
    from PIL import ImageOps
    if not images:
        raise ValueError("Select at least one face before making a collage.")
    columns = min(3, len(images))
    rows = -(-len(images) // columns)
    sheet = Image.new("RGB", (columns * (CELL_SIZE + CELL_GAP) + CELL_GAP,
                             rows * (CELL_SIZE + CELL_GAP) + CELL_GAP), BACKGROUND)
    for position, data in enumerate(images):
        with Image.open(io.BytesIO(data)) as source:
            cell = ImageOps.contain(source.convert("RGB"), (CELL_SIZE, CELL_SIZE))
        x = CELL_GAP + (position % columns) * (CELL_SIZE + CELL_GAP) + (CELL_SIZE - cell.width) // 2
        y = CELL_GAP + (position // columns) * (CELL_SIZE + CELL_GAP) + (CELL_SIZE - cell.height) // 2
        sheet.paste(cell, (x, y))
    buffer = io.BytesIO()
    sheet.save(buffer, "PNG")
    return buffer.getvalue()
