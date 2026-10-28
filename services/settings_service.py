"""
Settings Service

The app settings, one document for every video: the AI engines (who writes the
thumbnail ideas, who draws them, who writes titles and descriptions) and the
values the Generate form of every video starts with. A video can change those
values for itself; that never touches the settings.

Stored as JSON in the single row of app_settings. Until it is saved the
defaults come from config.py. Removing a model from config.py makes a saved
document that uses it invalid: migrate app_settings in the same change.
"""

import json
from typing import Literal

import aiosqlite
from services.comfyui_service import Connection, profiles
from config import COMFYUI_WORKFLOW
from gransabio_prompt_generator import TextProvider, resolve_text_model
from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import (
    GEMINI_IMAGE_MODEL,
    GEMINI_MODEL_MAX_REFS,
    IMAGE_PROVIDER,
    OPENAI_IMAGE_MODEL,
    OPENAI_MODEL_MAX_REFS,
    POE_IMAGE_MODEL,
    POE_MODEL_MAX_REFS,
    REPLICATE_MODEL_MAX_REFS,
)

# Image models offered per provider, with the images each one takes per thumbnail.
# Replicate has no model choice in the app: generation always sends flux-1.1-pro.
IMAGE_MODELS = {
    "gemini": GEMINI_MODEL_MAX_REFS,
    "openai": OPENAI_MODEL_MAX_REFS,
    "poe": POE_MODEL_MAX_REFS,
    "replicate": {"flux-1.1-pro": REPLICATE_MODEL_MAX_REFS["flux-1.1-pro"]},
}

# Model each image provider starts with when it is chosen
DEFAULT_IMAGE_MODELS = {
    "comfyui": COMFYUI_WORKFLOW,
    "gemini": GEMINI_IMAGE_MODEL,
    "openai": OPENAI_IMAGE_MODEL,
    "poe": POE_IMAGE_MODEL,
    "replicate": "flux-1.1-pro",
}


def image_models():
    return {**IMAGE_MODELS, "comfyui": {key: len(profile.references) for key, profile in profiles().items()}}


class _Strict(BaseModel):
    # A misspelled or leftover field is an error, not something to ignore
    model_config = ConfigDict(extra="forbid")


class TextEngine(_Strict):
    """A text AI reached through Gran Sabio."""
    provider: TextProvider
    model: str  # "" = Gran Sabio's default model of the provider
    thinking_enabled: bool
    thinking_level: Literal["low", "medium", "high"]

    @model_validator(mode="after")
    def _explicit_model(self):
        resolve_text_model(self.provider, self.model)
        return self


class ImageEngine(_Strict):
    """The AI that draws the thumbnails."""
    provider: Literal["gemini", "openai", "poe", "replicate", "comfyui"]
    model: str

    @model_validator(mode="after")
    def _model_of_provider(self):
        if self.model not in image_models()[self.provider]:
            raise ValueError(f"'{self.model}' is not a {self.provider} image model")
        return self

    @property
    def max_refs(self) -> int:
        """Total model limit for faces, bodies, environments, collages, and style."""
        return image_models()[self.provider][self.model]


class GenerationDefaults(_Strict):
    """What the Generate form of every video starts with."""
    num_images: int = Field(ge=1, le=50)
    # Expression of the people in the thumbnails (None: each idea chooses)
    expression: Literal["smiling", "mouth_closed", "neutral"] | None
    face_swap: bool
    face_restoration: bool
    avoid_repeat: bool  # The ideas take the video's earlier thumbnails into account
    face_photos: int = Field(ge=1)  # Total attachment limit; not an automatic selection target
    provider_fallback: bool  # When the image model fails, try the other providers


class AppSettings(_Strict):
    thumbnail_ideas: TextEngine
    texts: TextEngine
    image: ImageEngine
    generation: GenerationDefaults
    comfyui: Connection

    @model_validator(mode="after")
    def _face_photos_fit_model(self):
        if self.image.provider == "comfyui" and self.generation.provider_fallback:
            raise ValueError("ComfyUI does not allow provider fallback")
        if self.generation.face_photos > self.image.max_refs:
            raise ValueError(
                f"{self.image.model} takes up to {self.image.max_refs} images per thumbnail, "
                f"not {self.generation.face_photos} reference images"
            )
        return self


def default_settings() -> AppSettings:
    """Settings before the owner saves any: the image engine of .env, the rest fixed."""
    text_engine = {"provider": "anthropic", "model": "", "thinking_enabled": False, "thinking_level": "medium"}
    image = ImageEngine(provider=IMAGE_PROVIDER, model=DEFAULT_IMAGE_MODELS[IMAGE_PROVIDER])
    return AppSettings(
        thumbnail_ideas=text_engine,
        texts=text_engine,
        image=image,
        comfyui=Connection(),
        generation={
            "num_images": 5,
            "expression": None,
            "face_swap": False,
            "face_restoration": False,
            "avoid_repeat": False,
            "face_photos": image.max_refs,
            "provider_fallback": False,
        },
    )


async def load_settings(db: aiosqlite.Connection) -> AppSettings:
    """The saved settings, or the defaults if none were saved."""
    async with db.execute("SELECT data FROM app_settings WHERE id = 1") as cursor:
        row = await cursor.fetchone()
    if row is None:
        return default_settings()
    return AppSettings.model_validate_json(row[0])


async def save_settings(db: aiosqlite.Connection, settings: AppSettings) -> None:
    await db.execute(
        """
        INSERT INTO app_settings (id, data, updated_at) VALUES (1, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at
        """,
        (json.dumps(settings.model_dump()),),
    )
    await db.commit()
