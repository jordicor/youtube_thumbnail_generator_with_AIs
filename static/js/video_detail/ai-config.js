/**
 * Video Detail Page - AI Configuration Module
 * ============================================
 * The AI engines come from the app settings (header › Settings): who writes
 * the thumbnail ideas, who draws them and who writes titles and descriptions.
 * This module hands them to the requests, writes the one-line engine summaries
 * next to each generate button, keeps the face photos of Advanced within what
 * the image model takes and checks Gran Sabio's availability.
 */

import { state } from './state.js';

// Request field of the model of each image provider that has a choice of models
const IMAGE_MODEL_FIELDS = {
    gemini: 'gemini_model',
    openai: 'openai_model',
    poe: 'poe_model',
    comfyui: 'comfyui_workflow',
};

// =========================================================================
// IMAGE MODEL LIMITS
// =========================================================================

/**
 * Get the maximum number of reference images for the current image model.
 * @returns {number} Maximum reference images allowed
 */
export function getCurrentModelMaxRefs() {
    return AppSettings.maxRefs();
}

/**
 * Keep the face photos of Advanced within what the image model takes.
 */
export function updateRefMax() {
    const local = AppSettings.get().image.provider === 'comfyui';
    const fallback = document.getElementById('imageAiProviderFallback');
    fallback.disabled = local;
    if (local) fallback.checked = false;
    const input = document.getElementById('facePhotosPerThumbnail');
    const max = getCurrentModelMaxRefs();
    input.max = max;
    const current = parseInt(input.value, 10);
    input.value = Number.isInteger(current)
        ? Math.min(Math.max(current, 1), max)
        : AppSettings.get().generation.face_photos;

    const model = AppSettings.imageModelName(AppSettings.get().image.model);
    document.getElementById('facePhotosHint').textContent = t('generate_form.face_photos_hint', { max, model });

    // People chips, the style image hint and the step summaries depend on the number of photos
    document.dispatchEvent(new CustomEvent('imageConfigChanged'));
}

/**
 * Face photos chosen in Advanced (total for one thumbnail, across all people).
 * @returns {number} Chosen number
 */
export function getFacePhotosSetting() {
    return parseInt(document.getElementById('facePhotosPerThumbnail').value, 10);
}

// =========================================================================
// SUMMARIES
// =========================================================================

/**
 * Write the one-line engine summaries of the Generate and Texts tabs.
 */
export function updateSummaries() {
    const settings = AppSettings.get();
    document.getElementById('generateEngineSummary').textContent = t('ai_config.summary_generate', {
        text: AppSettings.textEngineName(settings.thumbnail_ideas),
        image: AppSettings.imageModelName(settings.image.model)
    });
    const texts = t('ai_config.summary_texts', { text: AppSettings.textEngineName(settings.texts) });
    document.querySelectorAll('.texts-engine-summary').forEach(el => {
        el.textContent = texts;
    });
}

// =========================================================================
// CONFIG GETTERS
// =========================================================================

/**
 * Text AI that writes the thumbnail ideas.
 * @returns {Object} {provider, model, thinking_enabled, thinking_level}
 */
export function getIdeasConfig() {
    return { ...AppSettings.get().thumbnail_ideas };
}

/**
 * Text AI that writes titles and descriptions.
 * @returns {Object} {provider, model, thinking_enabled, thinking_level}
 */
export function getTextConfig() {
    return { ...AppSettings.get().texts };
}

/**
 * Image fields of the generation request.
 * @returns {Object} image_provider, its model field, num_reference_images, allow_provider_fallback
 */
export function getImageRequestFields() {
    const { provider, model } = AppSettings.get().image;
    const fields = {
        image_provider: provider,
        num_reference_images: Math.min(getFacePhotosSetting(), getCurrentModelMaxRefs()),
        allow_provider_fallback: provider !== 'comfyui' && document.getElementById('imageAiProviderFallback').checked
    };
    if (provider in IMAGE_MODEL_FIELDS) {
        fields[IMAGE_MODEL_FIELDS[provider]] = model;
    }
    if (provider === 'comfyui') fields.comfyui = { ...AppSettings.get().comfyui };
    return fields;
}

// =========================================================================
// GRAN SABIO
// =========================================================================

/**
 * Check Gran Sabio LLM availability. The "Retry" buttons of its warnings run it again.
 */
export async function checkGranSabioStatus() {
    let available = false;
    try {
        const response = await fetch('/api/generation/gransabio/status');
        const data = await response.json();
        available = data.available === true;
    } catch (e) {
        console.error('Error checking Gran Sabio status:', e);
    }
    setGranSabioAvailable(available);
}

/**
 * Store Gran Sabio availability: while it is unavailable the generate buttons
 * are blocked and a warning above them says why.
 * @param {boolean} available - Whether Gran Sabio LLM answered as available
 */
function setGranSabioAvailable(available) {
    state.granSabioAvailable = available;
    document.querySelectorAll('.requires-gransabio').forEach(btn => {
        btn.classList.toggle('gransabio-blocked', !available);
        btn.title = available ? '' : t('ai_config.gran_sabio.required');
    });
    document.querySelectorAll('.gransabio-banner').forEach(banner => {
        banner.classList.toggle('hidden', available);
    });
}

/**
 * Guard for actions that need Gran Sabio LLM: explains why they cannot run.
 * @returns {boolean} True if Gran Sabio is available
 */
export function ensureGranSabioAvailable() {
    if (state.granSabioAvailable === true) {
        return true;
    }
    ThumbnailApp.showToast(t('ai_config.gran_sabio.required'), 'error');
    return false;
}

// =========================================================================
// INITIALIZATION
// =========================================================================

/**
 * Initialize the engine summaries and the face photos limit; the settings are loaded.
 */
export function init() {
    updateRefMax();
    updateSummaries();
    // Model names replace the model keys once Gran Sabio's list arrives
    AppSettings.loadTextModels().then(updateSummaries).catch(error => {
        console.error('Failed to load text models:', error);
    });

    document.getElementById('facePhotosPerThumbnail').addEventListener('change', updateRefMax);
    document.addEventListener('click', (e) => {
        if (e.target.closest('[data-gransabio-retry]')) {
            checkGranSabioStatus();
        }
    });
}
