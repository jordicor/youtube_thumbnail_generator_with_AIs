/**
 * Video Detail Page - Generation Module
 * ======================================
 * Handles thumbnail generation process, SSE updates, and polling fallback.
 */

import { state, resetGenerationState, saveGenerationJobState, clearGenerationJobState } from './state.js';
import { getTextConfig, getImageConfig, getCurrentModelMaxRefs, getCurrentModelName } from './ai-config.js';
import { getSelectedArray as getSelectedTitles } from './titles.js';
import { addNewToGallery } from './thumbnails.js';

// Store reference image data
let referenceImageData = null;

// Store submit button state for restoration
let submitBtn = null;
let originalBtnText = null;

// Store conflict resolution promise
let conflictResolve = null;

/**
 * Wait for TaskQueue to be ready.
 * @param {number} maxWait - Maximum wait time in ms
 * @returns {Promise<boolean>} True if TaskQueue is ready
 */
async function waitForTaskQueue(maxWait = 3000) {
    const start = Date.now();
    while (!window.TaskQueue?.tasks && Date.now() - start < maxWait) {
        await new Promise(r => setTimeout(r, 100));
    }
    return !!window.TaskQueue?.tasks;
}

/**
 * Restore the submit button to its original state.
 */
function restoreSubmitButton() {
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = originalBtnText;
        submitBtn.classList.remove('generating');
    }
    hideCancelButton();
}

/**
 * Show the cancel button next to generate.
 */
function showCancelButton() {
    const cancelBtn = document.getElementById('cancelGenerationBtn');
    if (cancelBtn) {
        cancelBtn.classList.remove('hidden');
    }
}

/**
 * Hide the cancel button.
 */
function hideCancelButton() {
    const cancelBtn = document.getElementById('cancelGenerationBtn');
    if (cancelBtn) {
        cancelBtn.classList.add('hidden');
    }
}

// =========================================================================
// REFERENCE CONFLICT HANDLING
// =========================================================================

/**
 * Check if adding the style reference would exceed the model's limit.
 * Reference images are always used when present (simplified UX).
 * @returns {Promise<{proceed: boolean, adjustedNumRefs?: number, skipStyleRef?: boolean}>}
 */
async function checkReferenceConflict() {
    // No reference image = no conflict possible
    if (!referenceImageData) {
        return { proceed: true };
    }

    const imageConfig = getImageConfig();
    const numClusterRefs = imageConfig.num_reference_images || 0;
    const modelMax = getCurrentModelMaxRefs();
    const totalWithStyle = numClusterRefs + 1;

    if (totalWithStyle <= modelMax) {
        // No conflict, everything fits
        return { proceed: true };
    }

    // Conflict! Show modal and wait for user decision
    const modelName = getCurrentModelName();

    // Update modal message
    const messageEl = document.getElementById('refConflictMessage');
    messageEl.textContent = t('generate_form.ref_conflict_message', {
        model: modelName,
        max: modelMax,
        cluster: numClusterRefs,
        total: totalWithStyle
    });

    // Update button text with the count
    const reduceBtn = document.getElementById('refConflictReduceCluster');
    reduceBtn.textContent = t('generate_form.ref_conflict_reduce_cluster', {
        count: modelMax - 1
    });

    // Show modal
    document.getElementById('refConflictModal').classList.add('visible');

    // Return a promise that resolves when user makes a choice
    return new Promise((resolve) => {
        conflictResolve = resolve;

        // Setup button handlers
        reduceBtn.onclick = () => {
            document.getElementById('refConflictModal').classList.remove('visible');
            // Update the num refs input to model_max - 1
            document.getElementById('imageAiNumRefs').value = modelMax - 1;
            resolve({ proceed: true, adjustedNumRefs: modelMax - 1 });
        };

        document.getElementById('refConflictRemoveStyle').onclick = () => {
            document.getElementById('refConflictModal').classList.remove('visible');
            // User chose to not include the style reference
            resolve({ proceed: true, skipStyleRef: true });
        };
    });
}

/**
 * Setup modal close handler to cancel generation if modal is closed.
 */
function setupRefConflictModalClose() {
    const modal = document.getElementById('refConflictModal');
    const cancelBtn = modal.querySelector('.btn-cancel');

    cancelBtn.onclick = () => {
        modal.classList.remove('visible');
        if (conflictResolve) {
            conflictResolve({ proceed: false });
            conflictResolve = null;
        }
    };
}

// =========================================================================
// GENERATION REQUEST
// =========================================================================

/**
 * Build the generation request object from form inputs.
 * @param {Object} options - Options from conflict resolution
 * @param {boolean} options.skipStyleRef - If true, don't include style ref in image generation
 * @returns {Object} Generation request payload
 */
export function getRequest(options = {}) {
    const imageAiConfig = getImageConfig();
    const textAiConfig = getTextConfig();

    const request = {
        cluster_index: parseInt(document.getElementById('clusterSelect').value),
        num_images: parseInt(document.getElementById('numImages').value),
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

    // Include reference image if provided (always used for prompts, optionally for image gen)
    if (referenceImageData) {
        request.reference_image_base64 = referenceImageData.base64;
        // Always send to prompt LLM for analysis
        request.reference_image_use_for_prompts = true;
        // Include in image generation unless user chose to skip (conflict resolution)
        request.reference_image_include_in_refs = !options.skipStyleRef;
    }

    return request;
}

// =========================================================================
// FORM SETUP
// =========================================================================

/**
 * Setup cancel generation button handler.
 */
function setupCancelButton() {
    const cancelBtn = document.getElementById('cancelGenerationBtn');
    if (!cancelBtn) return;

    cancelBtn.addEventListener('click', async () => {
        if (!state.currentJobId) return;

        const confirmed = await ThumbnailApp.showModal({
            title: t('task_queue.cancel_title'),
            message: t('task_queue.cancel_confirm'),
            type: 'warning',
            confirmText: t('common.yes'),
            cancelText: t('common.no')
        });

        if (confirmed) {
            try {
                await fetch(`/api/tasks/generation/${state.currentJobId}/cancel`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                // SSE will notify of cancellation and update UI
            } catch (error) {
                console.error('Error cancelling generation:', error);
                ThumbnailApp.showToast(t('errors.generic'), 'error');
            }
        }
    });
}

/**
 * Setup generation form event handlers.
 */
export function setupForm() {
    // Setup reference image handler
    setupReferenceImageHandler();

    // Setup conflict modal close handler
    setupRefConflictModalClose();

    // Setup cancel button handler
    setupCancelButton();

    // Listen for image config changes to update warning
    document.addEventListener('imageConfigChanged', updateStyleRefWarning);

    document.getElementById('generateForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const clusterIndex = document.getElementById('clusterSelect').value;
        if (!clusterIndex) {
            ThumbnailApp.showToast(t('generate_form.no_cluster_selected'), 'error');
            return;
        }

        // Check for reference limit conflict before proceeding
        const conflictResult = await checkReferenceConflict();
        if (!conflictResult.proceed) {
            // User cancelled
            return;
        }

        // Disable submit button and change text during generation
        submitBtn = document.getElementById('generateBtn');
        originalBtnText = submitBtn.textContent;
        submitBtn.disabled = true;
        submitBtn.textContent = t('generation.generating');
        submitBtn.classList.add('generating');
        showCancelButton();

        // Reset UI state before starting new generation
        document.getElementById('progressFill').style.width = '0%';
        document.getElementById('progressText').textContent = t('generate_form.progress_starting');
        document.getElementById('thumbnailsGrid').innerHTML = '';
        document.getElementById('resultsSection').classList.add('hidden');

        // Close any existing SSE connection
        resetGenerationState();

        const request = getRequest({ skipStyleRef: conflictResult.skipStyleRef });

        try {
            const response = await fetch(`/api/generation/${state.videoId}/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(request)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || t('errors.generation_failed'));
            }

            const data = await response.json();
            state.currentJobId = data.job_id;
            saveGenerationJobState();  // Persist job ID for page navigation

            document.getElementById('generationStatus').classList.remove('hidden');
            pollStatus();
        } catch (error) {
            ThumbnailApp.showToast(t('errors.generic') + ': ' + error.message, 'error');
            restoreSubmitButton();
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
                document.getElementById('progressText').textContent = t('generation.complete');
                ThumbnailApp.showToast(t('generation.complete'), 'success');
                clearGenerationJobState();  // Clear persisted state
                restoreSubmitButton();
                loadResults();
            },
            error: (data) => {
                document.getElementById('progressText').textContent = t('common.error');
                ThumbnailApp.showToast(t('errors.generic') + ': ' + (data.error_message || data.error), 'error');
                clearGenerationJobState();  // Clear persisted state
                restoreSubmitButton();
            },
            cancelled: (data) => {
                document.getElementById('progressText').textContent = t('common.cancel');
                ThumbnailApp.showToast(t('common.cancel'), 'info');
                clearGenerationJobState();  // Clear persisted state
                restoreSubmitButton();
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
                clearGenerationJobState();
                restoreSubmitButton();
                loadResults();
            } else if (data.status === 'error' || data.status === 'cancelled') {
                clearInterval(interval);
                clearGenerationJobState();
                restoreSubmitButton();
                if (data.status === 'error') {
                    ThumbnailApp.showToast(t('errors.generation_failed') + ': ' + (data.error_message || t('errors.generic')), 'error');
                }
            }
        } catch (error) {
            clearInterval(interval);
            clearGenerationJobState();
            restoreSubmitButton();
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
            <p><strong>${escapeHtml(thumb.suggested_title) || t('common.loading')}</strong></p>
            <p>${escapeHtml(thumb.text_overlay) || ''}</p>
            <a href="/api/thumbnails/${thumb.thumbnail_id}" download class="btn btn-small">${t('results.thumbnails.download')}</a>
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
                    <p><strong>${escapeHtml(thumb.suggested_title) || t('empty_state.no_thumbnails')}</strong></p>
                    <p>${escapeHtml(thumb.text_overlay) || ''}</p>
                    <a href="/api/thumbnails/${thumb.id}" download class="btn btn-small">${t('results.thumbnails.download')}</a>
                </div>
            `;
            grid.appendChild(card);
        }
    } catch (error) {
        grid.innerHTML = `<div class="error">${t('errors.load_thumbnails')}: ${escapeHtml(error.message)}</div>`;
    }
}

// =========================================================================
// JOB THUMBNAIL LOADING
// =========================================================================

/**
 * Check if a thumbnail is already displayed in the gallery.
 * @param {number} thumbnailId - The thumbnail database ID
 * @returns {boolean} True if thumbnail is in gallery
 */
function isThumbnailInGallery(thumbnailId) {
    const card = document.querySelector(
        `.thumbnail-card[data-thumbnail-id="${thumbnailId}"]`
    );
    return !!card;
}

/**
 * Load thumbnails already generated by an active job.
 * Used when restoring state after page navigation to show
 * thumbnails generated while user was away.
 * @param {number} jobId - The generation job ID
 */
async function loadJobThumbnails(jobId) {
    try {
        const response = await fetch(`/api/generation/jobs/${jobId}/thumbnails`);
        if (!response.ok) return;

        const thumbnails = await response.json();
        let loadedCount = 0;

        for (const thumb of thumbnails) {
            // Check if already in gallery to avoid duplicates
            if (!isThumbnailInGallery(thumb.id)) {
                addNewToGallery({
                    thumbnail_id: thumb.id,
                    suggested_title: thumb.suggested_title,
                    text_overlay: thumb.text_overlay
                });
                loadedCount++;
            }
        }

        if (loadedCount > 0) {
            console.log(`Loaded ${loadedCount} thumbnails from job ${jobId}`);
        }
    } catch (error) {
        console.warn('Error loading job thumbnails:', error);
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
        textarea.placeholder = hasReferenceImage
            ? t('generate_form.instructions_placeholder_with_ref')
            : t('generate_form.instructions_placeholder');
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
            ThumbnailApp.showToast(t('errors.generic'), 'error');
            input.value = '';
            return;
        }

        // Validate file type
        if (!file.type.startsWith('image/')) {
            ThumbnailApp.showToast(t('errors.generic'), 'error');
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

            // Check for conflicts
            updateStyleRefWarning();
        };

        reader.onerror = () => {
            ThumbnailApp.showToast(t('errors.generic'), 'error');
        };

        reader.readAsDataURL(file);
    });
}

/**
 * Update the warning message for style reference conflicts.
 * Called when reference image is added or num refs/model changes.
 * Shows warning if adding the style ref would exceed model limits.
 */
export function updateStyleRefWarning() {
    const warningEl = document.getElementById('styleRefWarning');
    if (!warningEl) return;

    // No warning if no reference image
    if (!referenceImageData) {
        warningEl.style.display = 'none';
        return;
    }

    const imageConfig = getImageConfig();
    const numClusterRefs = imageConfig.num_reference_images || 0;
    const modelMax = getCurrentModelMaxRefs();
    const totalWithStyle = numClusterRefs + 1;

    if (totalWithStyle > modelMax) {
        warningEl.textContent = t('generate_form.ref_conflict_warning', { max: modelMax });
        warningEl.style.display = 'block';
    } else {
        warningEl.style.display = 'none';
    }
}

/**
 * Clear the selected reference image.
 */
export function clearReferenceImage() {
    referenceImageData = null;

    const input = document.getElementById('referenceImageInput');
    if (input) input.value = '';

    const nameEl = document.getElementById('referenceImageName');
    if (nameEl) nameEl.textContent = t('generate_form.no_image_selected');

    const previewEl = document.getElementById('referenceImagePreview');
    if (previewEl) previewEl.style.display = 'none';

    const optionsEl = document.getElementById('referenceImageOptions');
    if (optionsEl) optionsEl.style.display = 'none';

    // Hide warning if visible
    const warningEl = document.getElementById('styleRefWarning');
    if (warningEl) warningEl.style.display = 'none';

    // Restore default placeholder
    updateInstructionsPlaceholder(false);
}

// =========================================================================
// UI STATE RESTORATION
// =========================================================================

/**
 * Show the generating state in the UI.
 * Used when restoring state after page navigation.
 * @param {Object} jobData - Optional job status data from API
 */
function showGeneratingUI(jobData = null) {
    // Cache button reference
    submitBtn = document.getElementById('generateBtn');
    if (submitBtn) {
        originalBtnText = submitBtn.textContent;
        submitBtn.disabled = true;
        submitBtn.textContent = t('generation.generating');
        submitBtn.classList.add('generating');
    }
    showCancelButton();

    // Show progress section
    document.getElementById('generationStatus').classList.remove('hidden');

    // Update progress if we have data
    if (jobData) {
        document.getElementById('progressFill').style.width = `${jobData.progress || 0}%`;
        const progressText = jobData.current_step
            ? `${jobData.current_step} - ${jobData.progress || 0}% (${jobData.thumbnails_generated || 0}/${jobData.total_thumbnails || 0})`
            : t('generate_form.progress_starting');
        document.getElementById('progressText').textContent = progressText;
    }
}

/**
 * Check if TaskQueueManager has an active generation job for this video.
 * This handles the case where the job was started on a different page.
 * @returns {boolean} True if found and restored
 */
function checkTaskQueueForActiveJob() {
    if (!window.TaskQueue || !window.TaskQueue.tasks) {
        return false;
    }

    for (const task of window.TaskQueue.tasks.values()) {
        if (task.type === 'generation' && task.video_id === state.videoId) {
            console.log(`Found active generation job ${task.id} in TaskQueue for video ${state.videoId}`);
            state.currentJobId = task.id;
            saveGenerationJobState();
            return true;
        }
    }
    return false;
}

/**
 * Restore generation UI state after page load.
 * Checks if there's an active job for this video and reconnects to it.
 * Call this after initState() during page initialization.
 */
export async function restoreGenerationState() {
    // Wait for TaskQueue to be ready before checking for active jobs
    const ready = await waitForTaskQueue();

    // First check TaskQueueManager for any active job for this video
    // (handles case where job was started on another page)
    if (!state.currentJobId && ready) {
        checkTaskQueueForActiveJob();
    }

    // No job to restore
    if (!state.currentJobId) return;

    console.log(`Checking status of generation job ${state.currentJobId}...`);

    try {
        const response = await fetch(`/api/generation/jobs/${state.currentJobId}/status`);

        if (!response.ok) {
            // Job not found or error - clear state
            console.log('Job not found, clearing state');
            clearGenerationJobState();
            state.currentJobId = null;
            return;
        }

        const data = await response.json();

        // Check if job is still active
        if (data.status === 'generating' || data.status === 'pending' || data.status === 'transcribing' || data.status === 'prompting') {
            console.log(`Job ${state.currentJobId} is still active (${data.status}), restoring UI`);

            // Load thumbnails already generated while user was away
            await loadJobThumbnails(state.currentJobId);

            // Show the generating UI with current progress
            showGeneratingUI(data);

            // Reconnect to SSE stream
            startSSE();
        } else {
            // Job finished while we were away
            console.log(`Job ${state.currentJobId} finished (${data.status}), clearing state`);
            clearGenerationJobState();
            state.currentJobId = null;

            // If completed, we could show a notification
            if (data.status === 'completed') {
                ThumbnailApp.showToast(t('generation.complete'), 'success');
            }
        }
    } catch (error) {
        console.warn('Error checking job status:', error);
        // On error, clear the persisted state
        clearGenerationJobState();
        state.currentJobId = null;
    }
}
