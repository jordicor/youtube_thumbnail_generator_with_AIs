/**
 * Video Detail Page - Constants and Configuration
 * ================================================
 * Centralized configuration values and magic numbers.
 */

// UI timing constants
export const CONFIG = {
    ANIMATION_DURATION_MS: 300,
    TRANSCRIPTION_CHECK_TIMEOUT_MS: 30000,
    LAZY_LOAD_ROOT_MARGIN: '100px',
    LAZY_LOAD_THRESHOLD: 0.1,
    TOAST_DURATION_MS: 3000,
    COPY_FEEDBACK_DURATION_MS: 2000,
};

// localStorage keys
export const STORAGE_KEYS = {
    AI_CONFIG: 'aiConfigPreferences',
};

// Model reference image limits per provider/model
// Mirrors backend config.py - should ideally be loaded from API
export const MODEL_MAX_REFS = {
    gemini: {
        "gemini-2.5-flash-image": 3,
        "gemini-3-pro-image-preview": 14
    },
    openai: {
        "gpt-image-1.5": 16,
        "gpt-image-1": 16,
        "gpt-image-1-mini": 16,
        "dall-e-3": 0
    },
    poe: {
        "flux2pro": 8,
        "flux2flex": 8,
        "fluxkontextpro": 1,
        "seedream40": 8,
        "nanobananapro": 14,
        "Ideogram-v3": 3
    },
    replicate: {
        "flux-1.1-pro": 1,
        "face-to-many": 1
    }
};

// Provider display names for UI
export const PROVIDER_NAMES = {
    'anthropic': 'Claude',
    'openai': 'GPT',
    'google': 'Gemini',
    'xai': 'Grok'
};

// Short model names for chip display
export const SHORT_MODEL_NAMES = {
    'gemini-2.5-flash-image': 'Gemini Flash',
    'gemini-3-pro-image-preview': 'Gemini Pro 4K',
    'flux2pro': 'FLUX Pro',
    'flux2flex': 'FLUX Flex',
    'fluxkontextpro': 'FLUX Kontext',
    'seedream40': 'Seedream',
    'nanobananapro': 'Nanobana',
    'Ideogram-v3': 'Ideogram',
    'gpt-image-1.5': 'GPT Image 1.5',
    'gpt-image-1': 'GPT Image',
    'gpt-image-1-mini': 'GPT Mini',
    'dall-e-3': 'DALL-E 3'
};

// Fallback models if API fails to load
// Using function to defer translation lookup until i18n is initialized
export function getFallbackPromptModels() {
    const defaultLabel = typeof t === 'function' ? t('generation.config.default') : 'Default';
    return {
        anthropic: [
            { value: "", label: defaultLabel },
            { value: "claude-sonnet-4-5", label: "Claude Sonnet 4.5" }
        ],
        openai: [
            { value: "", label: defaultLabel },
            { value: "gpt-5.2", label: "GPT-5.2" }
        ],
        google: [
            { value: "", label: defaultLabel },
            { value: "gemini-3-flash-preview", label: "Gemini 3 Flash" }
        ],
        xai: [
            { value: "", label: defaultLabel },
            { value: "grok-4-1-fast-non-reasoning", label: "Grok 4.1 Fast" }
        ]
    };
}

// Legacy export for backward compatibility (deprecated)
export const FALLBACK_PROMPT_MODELS = null;
