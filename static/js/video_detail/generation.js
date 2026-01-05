/**
 * Video Detail Page - Generation Module
 * ======================================
 * Handles thumbnail generation process, SSE updates, and polling fallback.
 */

import { state, resetGenerationState } from './state.js';
import { getTextConfig, getImageConfig } from './ai-config.js';
import { getSelectedArray as getSelectedTitles } from './titles.js';
import { addNewToGallery } from './thumbnails.js';

// Store reference image data
let referenceImageData = null;

// Placeholder texts for custom instructions
const PLACEHOLDER_DEFAULT = 'Ej: Estilo YouTube 2016, borde blanco, fondo kawaii, expresion exagerada...';
const PLACEHOLDER_WITH_REF = 'Ej: Usa el mismo fondo que la imagen de referencia. Manten los colores pero cambia la composicion. Estilo similar pero mas dinamico...';

// =========================================================================
// GENERATION REQUEST
// =========================================================================

/**
 * Build the generation request object from form inputs.
 * @returns {Object} Generation request payload
 */
export function getRequest() {
    const imageAiConfig = getImageConfig();
    const textAiConfig = getTextConfig();

    const request = {
        cluster_index: parseInt(document.getElementById('clusterSelect').value),
        num_prompts: parseInt(document.getElementById('numPrompts').value),
        num_variations: parseInt(document.getElementById('numVariations').value),
        preferred_expression: document.getElementById('expression').value || null,

        // Image AI settings
        image_provider: imageAiConfig.provider,
        num_reference_images: imageAiConfig.num_reference_images
    };

    // Add model based on image provider
    if (imageAiConfig.gemini_model) {
        request.gemini_model = imageAiConfig.gemini_model;
    } else if (imageAiConfig.poe_model) {
        request.poe_model = imageAiConfig.poe_model;
    } else if (imageAiConfig.openai_model) {
        request.openai_model = imageAiConfig.openai_model;
    }

    // Prompt generation AI settings
    request.prompt_provider = textAiConfig.provider;
    if (textAiConfig.model) {
        request.prompt_model = textAiConfig.model;
    }

    request.prompt_thinking_enabled = textAiConfig.thinking_enabled;
    if (textAiConfig.thinking_enabled) {
        request.prompt_thinking_level = textAiConfig.thinking_level;
    }

    const customInstructions = document.getElementById('promptCustomInstructions').value.trim();
    if (customInstructions) {
        request.prompt_custom_instructions = customInstructions;
    }

    request.prompt_include_history = document.getElementById('promptIncludeHistory').checked;

    // Include selected titles to guide image generation
    const selectedTitles = getSelectedTitles();
    if (selectedTitles.length > 0) {
        request.selected_titles = selectedTitles;
    }

    // Include reference image if provided
    if (referenceImageData) {
        const useForPrompts = document.getElementById('useRefForPrompts');
        const includeInImages = document.getElementById('includeRefInImages');

        request.reference_image_base64 = referenceImageData.base64;
        request.reference_image_use_for_prompts = useForPrompts ? useForPrompts.checked : false;
        request.reference_image_include_in_refs = includeInImages ? includeInImages.checked : false;
    }

    return request;
}

// =========================================================================
// FORM SETUP
// =========================================================================

/**
 * Setup generation form event handlers.
 */
export function setupForm() {
    // Setup reference image handler
    setupReferenceImageHandler();

    document.getElementById('generateForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const clusterIndex = document.getElementById('clusterSelect').value;
        if (!clusterIndex) {
            ThumbnailApp.showToast('Selecciona un persona/personaje', 'error');
            return;
        }

        // Reset UI state before starting new generation
        document.getElementById('progressFill').style.width = '0%';
        document.getElementById('progressText').textContent = 'Iniciando...';
        document.getElementById('thumbnailsGrid').innerHTML = '';
        document.getElementById('resultsSection').classList.add('hidden');

        // Close any existing SSE connection
        resetGenerationState();

        const request = getRequest();

        try {
            const response = await fetch(`/api/generation/${state.videoId}/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(request)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Error starting generation');
            }

            const data = await response.json();
            state.currentJobId = data.job_id;

            document.getElementById('generationStatus').classList.remove('hidden');
            pollStatus();
        } catch (error) {
            ThumbnailApp.showToast('Error: ' + error.message, 'error');
        }
    });
}

// =========================================================================
// SSE STREAMING
// =========================================================================

/**
 * Start SSE connection for real-time generation updates.
 */
export function startSSE() {
    if (!state.currentJobId) return;

    // Close any existing connection
    if (state.currentEventSource) {
        state.currentEventSource.close();
    }

    state.currentEventSource = ThumbnailApp.createSSEConnection(
        `/api/events/generation/${state.currentJobId}`,
        {
            progress: (data) => {
                document.getElementById('progressFill').style.width = `${data.progress}%`;
                document.getElementById('progressText').textContent =
                    `${data.current_step || data.status} - ${data.progress}% (${data.thumbnails_generated}/${data.total_thumbnails})`;
            },
            thumbnail: (data) => {
                appendThumbnail(data);
            },
            complete: (data) => {
                document.getElementById('progressFill').style.width = '100%';
                document.getElementById('progressText').textContent = 'Completado!';
                ThumbnailApp.showToast('Thumbnails generados correctamente', 'success');
                loadResults();
            },
            error: (data) => {
                document.getElementById('progressText').textContent = 'Error!';
                ThumbnailApp.showToast('Error: ' + (data.error_message || data.error), 'error');
            },
            cancelled: (data) => {
                document.getElementById('progressText').textContent = 'Cancelado';
                ThumbnailApp.showToast('Generacion cancelada', 'info');
            }
        }
    );
}

// =========================================================================
// POLLING FALLBACK
// =========================================================================

/**
 * Poll for generation status updates.
 * Falls back to polling if SSE is not supported.
 */
export async function pollStatus() {
    if (!state.currentJobId) return;

    // Prefer SSE if supported
    if (ThumbnailApp.isSSESupported()) {
        startSSE();
        return;
    }

    // Fallback to polling
    const interval = setInterval(async () => {
        try {
            const response = await fetch(`/api/generation/jobs/${state.currentJobId}/status`);
            const data = await response.json();

            document.getElementById('progressFill').style.width = `${data.progress}%`;
            document.getElementById('progressText').textContent =
                `${data.status} - ${data.progress}% (${data.thumbnails_generated}/${data.total_thumbnails})`;

            if (data.status === 'completed') {
                clearInterval(interval);
                loadResults();
            } else if (data.status === 'error') {
                clearInterval(interval);
                ThumbnailApp.showToast('Error en la generacion: ' + (data.error_message || 'Error desconocido'), 'error');
            }
        } catch (error) {
            clearInterval(interval);
        }
    }, 2000);
}

// =========================================================================
// THUMBNAIL DISPLAY
// =========================================================================

/**
 * Append a thumbnail to the results grid during generation.
 * @param {Object} thumb - Thumbnail data from SSE event
 */
export function appendThumbnail(thumb) {
    const section = document.getElementById('resultsSection');
    const grid = document.getElementById('thumbnailsGrid');

    section.classList.remove('hidden');

    const card = document.createElement('div');
    card.className = 'thumbnail-card';
    card.innerHTML = `
        <img src="/api/thumbnails/${thumb.thumbnail_id}" alt="Thumbnail">
        <div class="thumbnail-info">
            <p><strong>${escapeHtml(thumb.suggested_title) || 'Generando...'}</strong></p>
            <p>${escapeHtml(thumb.text_overlay) || ''}</p>
            <a href="/api/thumbnails/${thumb.thumbnail_id}" download class="btn btn-small">Descargar</a>
        </div>
    `;
    grid.appendChild(card);

    // Also add to the existing thumbnails gallery
    addNewToGallery(thumb);
}

/**
 * Load final results after generation completes.
 */
export async function loadResults() {
    if (!state.currentJobId) return;

    const section = document.getElementById('resultsSection');
    const grid = document.getElementById('thumbnailsGrid');

    try {
        const response = await fetch(`/api/generation/jobs/${state.currentJobId}/thumbnails`);
        const thumbnails = await response.json();

        section.classList.remove('hidden');
        grid.innerHTML = '';

        for (const thumb of thumbnails) {
            const card = document.createElement('div');
            card.className = 'thumbnail-card';
            card.innerHTML = `
                <img src="/api/thumbnails/${thumb.id}" alt="Thumbnail">
                <div class="thumbnail-info">
                    <p><strong>${escapeHtml(thumb.suggested_title) || 'Sin titulo'}</strong></p>
                    <p>${escapeHtml(thumb.text_overlay) || ''}</p>
                    <a href="/api/thumbnails/${thumb.id}" download class="btn btn-small">Descargar</a>
                </div>
            `;
            grid.appendChild(card);
        }
    } catch (error) {
        grid.innerHTML = `<div class="error">Error cargando thumbnails: ${escapeHtml(error.message)}</div>`;
    }
}

// =========================================================================
// REFERENCE IMAGE HANDLING
// =========================================================================

/**
 * Update the placeholder text for custom instructions based on reference image state.
 * @param {boolean} hasReferenceImage - Whether a reference image is selected
 */
function updateInstructionsPlaceholder(hasReferenceImage) {
    const textarea = document.getElementById('promptCustomInstructions');
    if (textarea) {
        textarea.placeholder = hasReferenceImage ? PLACEHOLDER_WITH_REF : PLACEHOLDER_DEFAULT;
    }
}

/**
 * Setup reference image input handler.
 */
function setupReferenceImageHandler() {
    const input = document.getElementById('referenceImageInput');
    if (!input) return;

    input.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            ThumbnailApp.showToast('Imagen demasiado grande (max 10MB)', 'error');
            input.value = '';
            return;
        }

        // Validate file type
        if (!file.type.startsWith('image/')) {
            ThumbnailApp.showToast('El archivo debe ser una imagen', 'error');
            input.value = '';
            return;
        }

        // Read as base64
        const reader = new FileReader();
        reader.onload = (e) => {
            const dataUrl = e.target.result;
            // Remove data:image/...;base64, prefix
            const base64 = dataUrl.split(',')[1];

            referenceImageData = {
                base64: base64,
                filename: file.name
            };

            // Update UI
            document.getElementById('referenceImageName').textContent = file.name;
            document.getElementById('referenceImageImg').src = dataUrl;
            document.getElementById('referenceImagePreview').style.display = 'block';
            document.getElementById('referenceImageOptions').style.display = 'block';

            // Update placeholder to show reference-related examples
            updateInstructionsPlaceholder(true);
        };

        reader.onerror = () => {
            ThumbnailApp.showToast('Error al leer la imagen', 'error');
        };

        reader.readAsDataURL(file);
    });
}

/**
 * Clear the selected reference image.
 */
export function clearReferenceImage() {
    referenceImageData = null;

    const input = document.getElementById('referenceImageInput');
    if (input) input.value = '';

    const nameEl = document.getElementById('referenceImageName');
    if (nameEl) nameEl.textContent = 'Ninguna imagen seleccionada';

    const previewEl = document.getElementById('referenceImagePreview');
    if (previewEl) previewEl.style.display = 'none';

    const optionsEl = document.getElementById('referenceImageOptions');
    if (optionsEl) optionsEl.style.display = 'none';

    // Reset checkboxes to defaults
    const useForPrompts = document.getElementById('useRefForPrompts');
    if (useForPrompts) useForPrompts.checked = true;

    const includeInImages = document.getElementById('includeRefInImages');
    if (includeInImages) includeInImages.checked = false;

    // Restore default placeholder
    updateInstructionsPlaceholder(false);
}
