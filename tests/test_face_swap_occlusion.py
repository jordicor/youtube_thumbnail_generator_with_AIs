"""Whatever covers a face (a hand, food, glasses) stays in front of its swap."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import cv2
import numpy as np
import pytest

from services.face_swap_service import (
    ARCFACE_128_TEMPLATE,
    FaceSwapAction,
    FaceSwapService,
)

SIZE = 400
OFFSET = (SIZE - 256) // 2  # the face's 256px crop is the image square 72..328


class _Occluder:
    """XSeg stand-in: the face shows except from crop row `covered_from` down."""

    def __init__(self, covered_from=256):
        self.covered_from = covered_from
        self.inputs = []

    def run(self, outputs, feeds):
        self.inputs.append(feeds['input'])
        visible = np.ones((1, 256, 256, 1), np.float32)
        visible[0, self.covered_from:] = 0
        return [visible]


def _face():
    """A face aligned so its crop is the middle square, with 68 landmarks on an
    ellipse centred in the crop (jaw to brows)."""
    angles = np.linspace(0, 2 * np.pi, 68, endpoint=False)
    landmarks = np.stack([128 + 80 * np.cos(angles), 128 + 100 * np.sin(angles), np.zeros(68)], axis=1)
    return SimpleNamespace(
        kps=(ARCFACE_128_TEMPLATE * 256 + OFFSET).astype(np.float32),
        landmark_3d_68=(landmarks + [OFFSET, OFFSET, 0]).astype(np.float32),
    )


def _service(occluders):
    service = FaceSwapService()
    service._load_occluders = lambda: occluders
    return service


def test_the_occluders_read_the_aligned_face_and_any_of_them_covers_it():
    image = np.random.default_rng(1).integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
    # Only one of the three sees the lower half of the crop covered (a hand over the mouth)
    occluders = [_Occluder(), _Occluder(covered_from=128), _Occluder()]

    occlusion = _service(occluders).occlusion(image, _face())

    # The occluders get the aligned crop as BGR in 0-1
    crop = image[OFFSET:OFFSET + 256, OFFSET:OFFSET + 256]
    assert np.allclose(occluders[0].inputs[0][0], crop / 255, atol=1 / 255)
    # Half the face is covered, and the swap is pasted only where it shows
    assert occlusion.covered == pytest.approx(0.5, abs=0.03)
    assert occlusion.mask.shape == (SIZE, SIZE)
    assert occlusion.mask[OFFSET + 70, 200] == pytest.approx(1, abs=0.01)   # forehead: shows
    assert occlusion.mask[OFFSET + 200, 200] == 0                           # mouth: covered
    assert occlusion.mask[10, 10] == 0                                      # outside the face


@pytest.mark.parametrize('restore_faces', [False, True])
def test_the_swap_leaves_whatever_covers_the_face_in_front(restore_faces):
    original = np.full((SIZE, SIZE, 3), 100, np.uint8)
    occluder = _Occluder(covered_from=128)
    service = _service([occluder] * 3)

    def swap_one(image, target_face, source_face):
        swapped = image.copy()
        swapped[OFFSET:OFFSET + 256, OFFSET:OFFSET + 256] = 200
        return swapped

    service.swap_one = swap_one
    service.restore = Mock(side_effect=lambda image: image + 10)  # GFPGAN touches every face

    result, covered = service.apply(original, [(_face(), object())], restore_faces=restore_faces)
    assert service.restore.call_count == int(restore_faces)

    assert covered == [pytest.approx(0.5, abs=0.03)]
    # What covers the face was read before swapping
    assert np.allclose(occluder.inputs[0], 100 / 255)
    assert (result[OFFSET + 70, 200] == (210 if restore_faces else 200)).all()
    assert (result[OFFSET + 200, 200] == 100).all()   # the hand over it stays
    assert (result[10, 10] == 100).all()              # nothing else changes, other faces included


def test_no_pairs_change_nothing():
    image = np.zeros((4, 4, 3), np.uint8)
    result, covered = _service([]).apply(image, [])
    assert result is image and covered == []


@pytest.mark.parametrize('restore_faces', [False, True])
async def test_each_swap_records_how_much_of_its_face_was_covered(monkeypatch, tmp_path, restore_faces):
    from services import face_swap_pipeline

    thumbnail = tmp_path / 'thumbnail.png'
    image = np.full((8, 8, 3), 100, np.uint8)
    cv2.imwrite(str(thumbnail), image)

    def action(index, decision):
        return FaceSwapAction(target_face_index=index, target_bbox=(0, 0, 4, 4), person_id=None,
                              cluster_id=None, similarity=0.7, decision=decision, source_frame=None)

    actions = [action(0, 'skip_low_match'), action(1, 'swap')]
    monkeypatch.setattr(face_swap_pipeline, 'face_swap_available', lambda: (True, []))
    monkeypatch.setattr(face_swap_pipeline, 'face_restoration_available', lambda: (True, []))
    monkeypatch.setattr(face_swap_pipeline, 'match_thumbnail_faces',
                        AsyncMock(return_value=(image, [('target', 'source')], actions)))
    swapper = SimpleNamespace(apply=Mock(side_effect=lambda target, pairs, restore_faces=False: (target + 1, [0.4242])))
    monkeypatch.setattr(face_swap_pipeline, 'FaceSwapService', SimpleNamespace(get=lambda: swapper))

    result = await face_swap_pipeline.apply_face_swap_to_thumbnail(None, thumbnail, [{'cluster_id': 1}], restore_faces=restore_faces)

    assert result.success and result.raw_path.exists()
    assert swapper.apply.call_args.kwargs == {'restore_faces': restore_faces}
    assert result.to_metadata()['restoration_applied'] is restore_faces
    assert (cv2.imread(str(result.raw_path)) == 100).all()
    metadata = result.to_metadata()['actions']
    assert [a['covered'] for a in metadata] == [None, 0.424]
    assert (cv2.imread(str(thumbnail)) == 101).all()


@pytest.mark.parametrize('matching_quality, expected', [(69.9, 'sharp.png'), (70.0, 'expression.png')])
async def test_donor_expression_requires_quality_on_the_100_point_scale(monkeypatch, tmp_path, matching_quality, expected):
    import face_extraction
    from services import face_matcher

    embedding = np.zeros(512, np.float32)
    embedding[0] = 1
    paths = [tmp_path / name for name in ('sharp.png', 'expression.png')]
    for path in paths:
        cv2.imwrite(str(path), np.zeros((16, 16, 3), np.uint8))
    candidates = [
        face_matcher.SourceFaceCandidate(paths[0], 95, 'neutral', embedding, paths[0].read_bytes()),
        face_matcher.SourceFaceCandidate(paths[1], matching_quality, 'smiling', embedding, paths[1].read_bytes()),
    ]
    monkeypatch.setattr(face_matcher, 'list_source_candidates', AsyncMock(return_value=candidates))
    selected = SimpleNamespace(embedding=embedding)
    monkeypatch.setattr(face_extraction, 'get_face_app', lambda: SimpleNamespace(get=lambda image: [selected]))

    result = await face_matcher.pick_source_face(None, 1, target_expression='smiling')

    assert result.source_frame_path == str(tmp_path / expected)


async def test_a_source_identity_mismatch_fails_instead_of_swapping_another_person(monkeypatch, tmp_path):
    import face_extraction
    from services import face_matcher

    embedding = np.zeros(512, np.float32)
    embedding[0] = 1
    other_embedding = np.zeros(512, np.float32)
    other_embedding[1] = 1
    path = tmp_path / 'source.png'
    cv2.imwrite(str(path), np.zeros((16, 16, 3), np.uint8))
    candidate = face_matcher.SourceFaceCandidate(path, 90, 'neutral', embedding, path.read_bytes())
    monkeypatch.setattr(face_matcher, 'list_source_candidates', AsyncMock(return_value=[candidate]))
    monkeypatch.setattr(face_extraction, 'get_face_app', lambda: SimpleNamespace(
        get=lambda image: [SimpleNamespace(embedding=other_embedding)]))

    with pytest.raises(ValueError, match='no longer matches'):
        await face_matcher.pick_source_face(None, 1)
