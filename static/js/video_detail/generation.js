/**
 * Video Detail Page - Generation Module
 * ======================================
 * Handles the thumbnail generation form and follows the running job through
 * its SSE stream. The task queue (task-queue.js) is the one that says, with a
 * toast on every page, how the job ended; here the panel shows it.
 */

import { state, resetGenerationState, saveGenerationJobState, clearGenerationJobState } from './state.js';
import { getIdeasConfig, getImageRequestFields, getFacePhotosSetting, ensureGranSabioAvailable } from './ai-config.js';
import * as References from './references.js';
import * as Defaults from './defaults.js';
import { getSelectedArray as getSelectedTitles, toggleSelection as toggleTitleSelection } from './titles.js';
// C2: Import addNewToGallery from thumbnails module
import { addNewToGallery, createResultTile, updateResultsPanel } from './thumbnails.js';
import { getSelectedIds as getSelectedPeopleIds, roleLabel } from './people.js';

// Submit button, kept while a generation runs
let submitBtn = null;

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
 * Number of thumbnails chosen in step 2.
 * @returns {number} Thumbnails to generate
 */
function thumbnailCount() {
    return parseInt(document.getElementById('numImages').value, 10) || 1;
}

/**
 * Label of the generate button: "Generate 5 thumbnails".
 * @returns {string} Label
 */
function generateButtonLabel() {
    return t('generate_form.generate_button', { count: thumbnailCount() });
}

/**
 * Write the label of the generate button unless a generation is running.
 */
function renderGenerateButton() {
    const btn = document.getElementById('generateBtn');
    if (!btn.classList.contains('generating')) {
        btn.textContent = generateButtonLabel();
    }
}

/**
 * Restore the submit button to its original state.
 */
function restoreSubmitButton() {
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.classList.remove('generating');
        submitBtn.textContent = generateButtonLabel();
    }
    hideCancelButton();
}

/**
 * Say next to the results why the video's last generation failed, or clear it.
 * It stays until the next generation starts: the toast fades, and a failure
 * that happened with the page closed would leave nothing else behind.
 * @param {?string} reason - Why it failed; null clears the note
 */
function setFailureNote(reason) {
    const note = document.getElementById('generationFailure');
    note.classList.toggle('hidden', !reason);
    note.textContent = reason ? t('generate_form.last_failed', { error: reason }) : '';
}

/**
 * Show the failure note if the video's newest generation job failed.
 */
async function showLastFailure() {
    try {
        const response = await fetch(`/api/generation/${state.videoId}/history`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const { jobs } = await response.json();
        const last = jobs[0];
        setFailureNote(last && last.status === 'error' ? (last.error_message || t('errors.generic')) : null);
    } catch (error) {
        console.error('Error loading the generation history:', error);
    }
}

/**
 * Restore the submit button if it's still in generating state.
 * Used as fallback when SSE complete event doesn't arrive but TaskQueue does.
 */
export function restoreSubmitButtonIfNeeded() {
    const btn = document.getElementById('generateBtn');
    if (btn && btn.classList.contains('generating')) {
        btn.disabled = false;
        btn.classList.remove('generating');
        btn.textContent = generateButtonLabel();
        hideCancelButton();
    }
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
// GENERATION REQUEST
// =========================================================================

/**
 * Build the generation request object from form inputs.
 * @returns {Object} Generation request payload
 */
export function getRequest() {
    const ideasConfig = getIdeasConfig();

    // Multi-cluster: send the ordered people chosen by the user. Position in
    // the array maps to person_01, person_02, etc.
    const request = {
        cluster_ids: getSelectedPeopleIds(),
        num_images: thumbnailCount(),
        preferred_expression: document.getElementById('expression').value || null,

        // Image AI: provider, its model, face photos and fallback
        ...getImageRequestFields()
    };

    // AI that writes the thumbnail ideas
    request.prompt_provider = ideasConfig.provider;
    if (ideasConfig.model) {
        request.prompt_model = ideasConfig.model;
    }

    request.prompt_thinking_enabled = ideasConfig.thinking_enabled;
    if (ideasConfig.thinking_enabled) {
        request.prompt_thinking_level = ideasConfig.thinking_level;
    }

    const customInstructions = document.getElementById('promptCustomInstructions').value.trim();
    if (customInstructions) {
        request.prompt_custom_instructions = customInstructions;
    }

    request.prompt_include_history = document.getElementById('promptIncludeHistory').checked;

    // Face swap post-processing; a disabled checkbox (models missing) is unchecked
    request.apply_face_swap = document.getElementById('applyFaceSwap').checked;
    const restoration = document.getElementById('applyFaceRestoration');
    request.restore_faces = request.apply_face_swap && !restoration.disabled && restoration.checked;

    // Step 2: the photos of the setting, or a setting the AI imagines
    request.include_scene_reference = References.includeBackground();

    // Include starred titles to guide image generation
    const selectedTitles = getSelectedTitles();
    if (selectedTitles.length > 0) {
        request.selected_titles = selectedTitles;
    }

    // The style image reaches both models and is one of the images the image model gets
    if (state.styleImage) {
        request.reference_image_base64 = state.styleImage.base64;
        request.reference_image_use_for_prompts = true;
        request.reference_image_include_in_refs = true;
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
            message: t('task_queue.cancel_confirm_generation'),
            type: 'warning',
            confirmText: t('common.yes'),
            cancelText: t('common.no')
        });

        if (confirmed) {
            // The task queue says whether it could; the job's stream updates the panel
            await window.TaskQueue.cancelTask('generation', state.currentJobId);
        }
    });
}

/**
 * Typed thumbnail counts out of range snap back when leaving the field.
 */
function setupCountField() {
    document.getElementById('numImages').addEventListener('change', (e) => {
        const input = e.target;
        const value = parseInt(input.value, 10);
        input.value = Number.isInteger(value)
            ? Math.min(Math.max(value, 1), parseInt(input.max, 10))
            : AppSettings.get().generation.num_images;
        renderSummaries();
    });
}

/**
 * Summary above the generate button: "5 thumbnails with Laura (lead) and Leo".
 */
function renderSubmitSummary() {
    const summary = document.getElementById('generateSummary');
    if (state.selectedPeople.length === 0) {
        summary.textContent = t('generate_form.no_people');
        return;
    }
    const names = state.selectedPeople.map((person, index) => {
        const name = ThumbnailApp.getClusterDisplayName(person);
        return index === 0 ? t('generate_form.summary_lead', { name, role: roleLabel(0) }) : name;
    });
    const people = names.length === 1
        ? names[0]
        : t('generate_form.summary_and', { list: names.slice(0, -1).join(', '), last: names[names.length - 1] });
    const thumbnails = t('generate_form.summary_thumbnails', { count: thumbnailCount() });
    summary.innerHTML = t('generate_form.summary', {
        thumbnails: `<strong>${escapeHtml(thumbnails)}</strong>`,
        people: escapeHtml(people)
    });
}

/**
 * A value of the settings' generation defaults as the step summaries write it.
 * @param {string} key - expression, face_swap, avoid_repeat, face_photos or provider_fallback
 * @param {*} value - Value as the settings store it
 * @returns {string} Text
 */
function summaryText(key, value) {
    if (key === 'expression') {
        return document.querySelector(`#expression option[value="${value ?? ''}"]`).textContent;
    }
    if (key === 'face_photos') {
        return t('generate_form.summary_face_photos', { count: value });
    }
    if (key === 'provider_fallback') {
        // "If the model fails: it stops" says more than "If the model fails: No"
        return t(value ? 'generate_form.fallback_tries_others' : 'generate_form.fallback_stops');
    }
    return t(value ? 'generate_form.summary_on' : 'generate_form.summary_off');
}

/**
 * Values of the folded steps, shown on their headers while closed. Values
 * changed for this video stand out and say what the settings have.
 */
function renderStepSummaries() {
    document.getElementById('summaryExpression').textContent =
        summaryText('expression', document.getElementById('expression').value);
    document.getElementById('summaryStyle').textContent = state.styleImage
        ? state.styleImage.filename
        : t('generate_form.summary_no_image');
    const starred = getSelectedTitles().length;
    document.getElementById('summaryTitles').textContent = starred > 0
        ? t('generate_form.summary_titles', { count: starred })
        : t('generate_form.summary_no_titles');
    const instructions = document.getElementById('promptCustomInstructions').value.trim();
    document.getElementById('summaryInstructions').textContent = instructions || t('generate_form.summary_no_instructions');

    const faceSwap = document.getElementById('applyFaceSwap');
    document.getElementById('summaryFaceSwap').textContent = faceSwap.disabled
        ? t('generate_form.summary_unavailable')
        : summaryText('face_swap', faceSwap.checked);
    const restoration = document.getElementById('applyFaceRestoration');
    const showRestoration = faceSwap.checked && !faceSwap.disabled;
    document.getElementById('faceRestorationOption').classList.toggle('hidden', !showRestoration);
    document.querySelector('[data-default="face_restoration"]').classList.toggle('hidden', !showRestoration);
    document.getElementById('summaryFaceRestoration').textContent = restoration.disabled
        ? t('generate_form.summary_unavailable')
        : summaryText('face_restoration', restoration.checked);
    document.getElementById('summaryAvoidRepeat').textContent =
        summaryText('avoid_repeat', document.getElementById('promptIncludeHistory').checked);
    document.getElementById('summaryFacePhotos').textContent = summaryText('face_photos', getFacePhotosSetting());
    document.getElementById('summaryFallback').textContent =
        summaryText('provider_fallback', document.getElementById('imageAiProviderFallback').checked);

    const defaults = AppSettings.get().generation;
    document.querySelectorAll('.step-summary-row[data-default]').forEach(row => {
        const key = row.dataset.default;
        const changed = Defaults.isChanged(key);
        row.classList.toggle('is-changed', changed);
        row.title = changed ? t('generate_form.changed_here', { value: summaryText(key, defaults[key]) }) : '';
    });
    document.querySelectorAll('[data-reset-step]').forEach(button => {
        button.classList.toggle('hidden', !Defaults.stepChanged(button.dataset.resetStep));
    });
}

function renderSummaries() {
    renderGenerateButton();
    renderSubmitSummary();
    renderStepSummaries();
}

/**
 * Setup generation form event handlers.
 */
export function setupForm() {
    setupReferenceImageHandler();
    setupCancelButton();
    setupCountField();

    // Every control of the form shows its value in a summary
    const form = document.getElementById('generateForm');
    form.addEventListener('input', renderSummaries);
    form.addEventListener('change', renderSummaries);
    document.addEventListener('people:changed', renderSubmitSummary);
    document.addEventListener('imageConfigChanged', () => {
        renderStepSummaries();
        updateStyleImageHint();
    });
    // New settings may change what differs from them here
    document.addEventListener('settings:changed', renderSummaries);
    form.addEventListener('click', (e) => {
        const button = e.target.closest('[data-reset-step]');
        if (button) Defaults.reset(button.dataset.resetStep);
    });

    // Step 3 mirrors the titles of the Texts tab and their stars
    document.addEventListener('titles:changed', () => {
        renderGuidingTitles();
        renderStepSummaries();
    });
    document.getElementById('guidingTitles').addEventListener('change', (e) => {
        if (e.target.matches('input[type="checkbox"]')) {
            toggleTitleSelection(parseInt(e.target.dataset.index));
        }
    });
    renderGuidingTitles();
    renderSummaries();

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!ensureGranSabioAvailable()) return;

        if (state.selectedPeople.length === 0) {
            ThumbnailApp.showToast(t('generate_form.no_people'), 'error');
            return;
        }

        // The photos do not fit the model, or someone has no face photo
        if (References.getBlockReason()) {
            ThumbnailApp.showToast(References.getBlockReason(), 'error');
            return;
        }

        // H14: Disable submit button before anything async to prevent double-click races
        submitBtn = document.getElementById('generateBtn');
        submitBtn.disabled = true;
        submitBtn.textContent = t('generation.generating');
        submitBtn.classList.add('generating');
        showCancelButton();

        // Reset UI state before starting new generation
        document.getElementById('progressFill').style.width = '0%';
        document.getElementById('progressText').textContent = t('generate_form.progress_starting');
        document.getElementById('thumbnailsGrid').innerHTML = '';
        setFailureNote(null);

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
                throw new Error(await ThumbnailApp.readApiError(response));
            }

            const data = await response.json();
            state.currentJobId = data.job_id;
            saveGenerationJobState();  // Persist job ID for page navigation

            document.getElementById('generationStatus').classList.remove('hidden');
            document.getElementById('resultsTitle').textContent = t('generate_form.results_generating');
            updateResultsPanel();
            startSSE();
        } catch (error) {
            ThumbnailApp.showToast(t('errors.generic') + ': ' + error.message, 'error');
            restoreSubmitButton();
            updateResultsPanel();
        }
    });
}

/**
 * Render the saved titles as checkboxes: the checked (starred) ones guide the thumbnails.
 */
function renderGuidingTitles() {
    const container = document.getElementById('guidingTitles');
    const titles = state.titles
        .map((title, index) => ({ title, index }))
        .filter(({ title }) => title !== null);

    if (titles.length === 0) {
        container.innerHTML = `
            <p class="step-empty">${escapeHtml(t('generate_form.no_titles'))}
                <button type="button" class="link-btn" data-goto-tab="texts">${escapeHtml(t('generate_form.go_to_titles'))}</button>
            </p>
        `;
        return;
    }

    container.innerHTML = titles.map(({ title, index }) => `
        <label class="guiding-title">
            <input type="checkbox" data-index="${index}" ${state.selectedTitles.has(index) ? 'checked' : ''}>
            <span>${escapeHtml(typeof title === 'string' ? title : title.text)}</span>
        </label>
    `).join('');
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

    const source = ThumbnailApp.createSSEConnection(
        `/api/events/generation/${state.currentJobId}`,
        {
            progress: (data) => {
                if (data.status === 'stopping') {
                    // Cancelled: the worker is finishing the thumbnail it was making
                    hideCancelButton();
                    document.getElementById('progressText').textContent = data.current_step;
                    return;
                }
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
                document.getElementById('resultsTitle').textContent = t('generate_form.results_last');
                clearGenerationJobState();  // Clear persisted state
                restoreSubmitButton();
                loadResults();
            },
            failed: (data) => {
                // The thumbnails the job made before failing stay in the panel;
                // with none, the panel goes back to the video's recent ones
                const made = document.getElementById('thumbnailsGrid').children.length > 0;
                document.getElementById('generationStatus').classList.add('hidden');
                document.getElementById('resultsTitle').textContent =
                    t(made ? 'generate_form.results_last' : 'generate_form.results_recent');
                setFailureNote(data.error_message || data.error);
                clearGenerationJobState();  // Clear persisted state
                restoreSubmitButton();
                updateResultsPanel();
            },
            cancelled: (data) => {
                document.getElementById('progressText').textContent = t('task_queue.cancelled_generation');
                document.getElementById('resultsTitle').textContent = t('generate_form.results_last');
                clearGenerationJobState();  // Clear persisted state
                restoreSubmitButton();
                // The thumbnails the job had made stay
                loadResults();
            },
            closed: () => {
                // The browser gave up on the stream; the job goes on in the worker
                setTimeout(() => {
                    if (state.currentEventSource === source) startSSE();
                }, 3000);
            }
        }
    );
    state.currentEventSource = source;
}

// =========================================================================
// THUMBNAIL DISPLAY
// =========================================================================

/**
 * Append a thumbnail to the results grid during generation.
 * @param {Object} thumb - Thumbnail data from SSE event
 */
export function appendThumbnail(thumb) {
    const grid = document.getElementById('thumbnailsGrid');
    // The stream sends every thumbnail of the job again on each (re)connection
    if (grid.querySelector(`[data-thumbnail-id="${thumb.thumbnail_id}"]`)) return;
    grid.appendChild(createResultTile({
        id: thumb.thumbnail_id,
        suggested_title: thumb.suggested_title
    }));
    updateResultsPanel();
}

/**
 * Load final results after generation completes.
 */
export async function loadResults() {
    if (!state.currentJobId) return;

    const grid = document.getElementById('thumbnailsGrid');

    try {
        const response = await fetch(`/api/generation/jobs/${state.currentJobId}/thumbnails`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const thumbnails = await response.json();
        grid.replaceChildren(...thumbnails.map(createResultTile));
    } catch (error) {
        grid.innerHTML = `<div class="error">${t('errors.load_thumbnails')}: ${escapeHtml(error.message)}</div>`;
    }
    updateResultsPanel();
}

// =========================================================================
// JOB THUMBNAIL LOADING
// =========================================================================

/**
 * Check if the gallery already has a thumbnail. A job's thumbnails are among
 * the newest, which the gallery loads whatever its order.
 * @param {number} thumbnailId - The thumbnail database ID
 * @returns {boolean} True if thumbnail is in gallery
 */
function isThumbnailInGallery(thumbnailId) {
    return state.newestThumbnails.some(thumb => thumb.id === thumbnailId);
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

        // The results panel shows the running job's thumbnails
        document.getElementById('thumbnailsGrid').replaceChildren(...thumbnails.map(createResultTile));
        updateResultsPanel();

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
// STYLE IMAGE
// =========================================================================

/**
 * Update the placeholder text for custom instructions based on the style image.
 * @param {boolean} hasStyleImage - Whether a style image is chosen
 */
function updateInstructionsPlaceholder(hasStyleImage) {
    document.getElementById('promptCustomInstructions').placeholder = hasStyleImage
        ? t('generate_form.instructions_placeholder_with_ref')
        : t('generate_form.instructions_placeholder');
}

const MAX_STYLE_IMAGE_MB = 10;

/**
 * Setup the style image: click the drop zone to pick a file, or drop one on it.
 */
function setupReferenceImageHandler() {
    const input = document.getElementById('referenceImageInput');
    const dropZone = document.getElementById('referenceDropZone');

    dropZone.addEventListener('click', () => input.click());
    input.addEventListener('change', () => {
        const file = input.files[0];
        input.value = '';
        if (file) loadReferenceFile(file);
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file) loadReferenceFile(file);
    });

    document.getElementById('clearReferenceImage').addEventListener('click', clearReferenceImage);
    updateStyleImageHint();
}

/**
 * Validate and read the style image.
 * @param {File} file - Picked or dropped file
 */
function loadReferenceFile(file) {
    if (!file.type.startsWith('image/')) {
        ThumbnailApp.showToast(t('generate_form.style_not_image'), 'error');
        return;
    }
    if (file.size > MAX_STYLE_IMAGE_MB * 1024 * 1024) {
        ThumbnailApp.showToast(t('generate_form.style_too_large', {max: MAX_STYLE_IMAGE_MB}), 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const dataUrl = e.target.result;
        state.styleImage = {
            // Without the data:image/...;base64, prefix
            base64: dataUrl.split(',')[1],
            filename: file.name
        };

        document.getElementById('referenceImageImg').src = dataUrl;
        document.getElementById('referenceImagePreview').classList.remove('hidden');
        document.getElementById('referenceDropZone').classList.add('hidden');
        updateInstructionsPlaceholder(true);
        styleImageChanged();
    };

    reader.onerror = () => {
        ThumbnailApp.showToast(t('errors.generic'), 'error');
    };

    reader.readAsDataURL(file);
}

/**
 * Say under the style image what it is for or, once chosen, that the image
 * model gets it too.
 */
function updateStyleImageHint() {
    document.getElementById('styleImageHint').textContent =
        t(state.styleImage ? 'generate_form.style_counts' : 'generate_form.style_hint');
}

/**
 * The style image is one more image the image model gets: tell the count.
 */
function styleImageChanged() {
    updateStyleImageHint();
    renderStepSummaries();
    document.dispatchEvent(new CustomEvent('imageConfigChanged'));
}

/**
 * Clear the chosen style image.
 */
export function clearReferenceImage() {
    state.styleImage = null;
    document.getElementById('referenceImageInput').value = '';
    document.getElementById('referenceImageImg').removeAttribute('src');
    document.getElementById('referenceImagePreview').classList.add('hidden');
    document.getElementById('referenceDropZone').classList.remove('hidden');
    updateInstructionsPlaceholder(false);
    styleImageChanged();
}

// =========================================================================
// UI STATE RESTORATION
// =========================================================================

/**
 * Show the generating state in the UI.
 * Used when restoring state after page navigation.
 * @param {Object} jobData - Optional job status data from API
 */
function showGeneratingUI(jobData) {
    submitBtn = document.getElementById('generateBtn');
    submitBtn.disabled = true;
    submitBtn.textContent = t('generation.generating');
    submitBtn.classList.add('generating');
    // A job that is already stopping cannot be cancelled again
    if (jobData.stopping) {
        hideCancelButton();
    } else {
        showCancelButton();
    }

    // Show progress section; the stream writes what the job is doing as soon as it connects
    document.getElementById('generationStatus').classList.remove('hidden');
    document.getElementById('resultsTitle').textContent = t('generate_form.results_generating');
    document.getElementById('progressFill').style.width = `${jobData.progress || 0}%`;
    document.getElementById('progressText').textContent = t('generate_form.progress_starting');
    setFailureNote(null);
    updateResultsPanel();
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

let restoring = null;

/**
 * Restore generation UI state after page load.
 * Checks if there's an active job for this video and reconnects to it.
 * Call this after initState() during page initialization. Page init and the
 * task queue's first events can both ask for it at once: they share one run.
 * @returns {Promise<void>}
 */
export function restoreGenerationState() {
    restoring ??= restoreActiveJob().finally(() => { restoring = null; });
    return restoring;
}

async function restoreActiveJob() {
    // Wait for TaskQueue to be ready before checking for active jobs
    const ready = await waitForTaskQueue();

    // First check TaskQueueManager for any active job for this video
    // (handles case where job was started on another page)
    if (!state.currentJobId && ready) {
        checkTaskQueueForActiveJob();
    }

    // No job to restore: the last one may have failed with nobody watching
    if (!state.currentJobId) {
        await showLastFailure();
        return;
    }

    console.log(`Checking status of generation job ${state.currentJobId}...`);

    try {
        const response = await fetch(`/api/generation/jobs/${state.currentJobId}/status`);

        if (!response.ok) {
            // Job not found or error - clear state
            console.log('Job not found, clearing state');
            clearGenerationJobState();
            state.currentJobId = null;
            await showLastFailure();
            return;
        }

        const data = await response.json();

        // Still working, or cancelled with its worker still finishing a thumbnail
        const active = ['pending', 'transcribing', 'prompting', 'generating'].includes(data.status) || data.stopping;
        if (active) {
            console.log(`Job ${state.currentJobId} is still active (${data.status}), restoring UI`);

            // Load thumbnails already generated while user was away
            await loadJobThumbnails(state.currentJobId);

            // Show the generating UI with current progress
            showGeneratingUI(data);

            // Reconnect to SSE stream
            startSSE();
        } else {
            // Job finished while we were away: its thumbnails are among the recent ones
            console.log(`Job ${state.currentJobId} finished (${data.status}), clearing state`);
            clearGenerationJobState();
            state.currentJobId = null;
            await showLastFailure();
        }
    } catch (error) {
        console.warn('Error checking job status:', error);
        // On error, clear the persisted state
        clearGenerationJobState();
        state.currentJobId = null;
    }
}
