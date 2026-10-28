"""The facial expression chosen for a generation reaches the prompts and the face photos."""

from pathlib import Path

from pydantic import ValidationError
import pytest

from gransabio_prompt_generator import PromptGenerationConfig, build_concept_prompt
from prompt_generation import _cache_inputs, _cache_keys
from services.generation_service import GenerationService


def concept_prompt(**kwargs):
    return build_concept_prompt(
        transcription='transcript', video_title='Video', image_index=2, total_images=3,
        skeleton={}, prior_concepts=[{'image_index': 1, 'concept_name': 'First'}], **kwargs,
    )


def test_concept_prompt_asks_for_the_chosen_expression():
    prompt = concept_prompt(facial_expression='mouth_closed')
    assert 'FACIAL EXPRESSION (CHOSEN BY THE USER)' in prompt
    assert 'Every person in this thumbnail must be with the mouth closed' in prompt
    assert 'DO NOT REPEAT their angle, scene, mood or pose:' in prompt


def test_concept_prompt_without_expression_lets_concepts_vary_it():
    prompt = concept_prompt()
    assert 'FACIAL EXPRESSION' not in prompt
    assert 'DO NOT REPEAT their angle, scene, mood, pose or expression:' in prompt


def test_unknown_expression_is_rejected():
    with pytest.raises(ValueError):
        concept_prompt(facial_expression='angry')


def test_expression_changes_the_concepts_cache_key():
    def key(expression):
        config = PromptGenerationConfig(facial_expression=expression)
        return _cache_keys(_cache_inputs(None, None, False, None, 3, config, None, 'summary'))[1]

    assert key(None) != key('smiling') != key('neutral')
    assert key('smiling') == key('smiling')


@pytest.mark.parametrize('field, value', [
    ('preferred_expression', 'angry'),
    ('num_images', 0),
    ('num_images', 51),
    ('num_reference_images', 0),
])
def test_generation_request_rejects_out_of_range_values(field, value):
    from api.routes.generation import GenerationRequest

    with pytest.raises(ValidationError):
        GenerationRequest(cluster_ids=[1], **{field: value})
