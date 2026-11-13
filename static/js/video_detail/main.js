/**
 * Video Detail Page - Main Entry Point
 * =====================================
 * Initializes all modules and binds the page-level controls.
 */

import { initState, state } from './state.js';
import * as AiConfig from './ai-config.js';
import * as Defaults from './defaults.js';
import * as Clusters from './clusters.js';
import * as Titles from './titles.js';
import * as Descriptions from './descriptions.js';
import * as Thumbnails from './thumbnails.js';
import * as Lightbox from './lightbox.js';
import * as Generation from './generation.js';
import * as QueueIndicator from './queue-indicator.js';
import * as People from './people.js';
import * as References from './references.js';

// =========================================================================
// TABS
// =========================================================================

const TABS = ['generate', 'people', 'thumbnails', 'texts', 'transcript'];
const DEFAULT_TAB = 'generate';

/**
 * Show one workspace tab and remember it in the URL hash.
 * @param {string} tab - One of TABS
 */
function switchTab(tab) {
    if (!TABS.includes(tab)) {
        throw new Error(`Unknown tab: ${tab}`);
    }
    // Without an analysis there is nothing to see in the other tabs
    if (document.getElementById(`tab-${tab}`).disabled) {
        tab = DEFAULT_TAB;
    }

    document.getElementById('videoDetailApp').dataset.tab = tab;
    for (const name of TABS) {
        const active = name === tab;
        const button = document.getElementById(`tab-${name}`);
        button.classList.toggle('active', active);
        button.setAttribute('aria-selected', String(active));
        button.tabIndex = active ? 0 : -1;
        document.getElementById(`panel-${name}`).hidden = !active;
    }
    history.replaceState(null, '', `#${tab}`);
    document.dispatchEvent(new CustomEvent('help:context'));
}

function setupTabs() {
    document.querySelectorAll('.workspace-tab').forEach(button => {
        button.addEventListener('click', () => switchTab(button.dataset.tab));
    });

    // Arrow keys move between tabs (WAI-ARIA tabs pattern)
    document.querySelector('.workspace-tabs').addEventListener('keydown', (e) => {
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
        const visible = [...document.querySelectorAll('.workspace-tab:not(.hidden)')];
        const current = visible.indexOf(document.activeElement);
        if (current < 0) return;
        const next = visible[(current + (e.key === 'ArrowRight' ? 1 : visible.length - 1)) % visible.length];
        switchTab(next.dataset.tab);
        next.focus();
    });

    // Any element can jump to a tab with data-goto-tab="name"
    document.addEventListener('click', (e) => {
        const link = e.target.closest('[data-goto-tab]');
        if (link) {
            e.preventDefault();
            switchTab(link.dataset.gotoTab);
        }
    });

    document.getElementById('viewAllThumbnailsBtn').addEventListener('click', () => switchTab('thumbnails'));

    // The hash comes from the URL: an unknown one just opens the default tab
    const fromHash = location.hash.slice(1);
    switchTab(TABS.includes(fromHash) ? fromHash : DEFAULT_TAB);
}

// =========================================================================
// PAGE-LEVEL FUNCTIONS
// =========================================================================

/**
 * Load video information and display it in the page header.
 */
async function loadVideoInfo() {
    const response = await fetch(`/api/videos/${state.videoId}`);
    // H7: Check response before parsing
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const video = await response.json();

    const name = ThumbnailApp.formatVideoName(video.filename);
    const title = document.getElementById('videoTitle');
    title.textContent = name;
    title.title = video.filepath;
    document.title = `${name} - ${t('common.app_name')}`;
    ThumbnailApp.setVideoBreadcrumb(video.filename);

    const view = ThumbnailApp.getVideoStatusView(video.status);
    const pill = document.getElementById('videoStatusPill');
    pill.className = `status-pill tone-${view.tone}`;
    pill.textContent = t(view.label);

    const duration = document.getElementById('videoMetaDuration');
    duration.classList.toggle('hidden', !video.duration_seconds);
    duration.querySelector('span').textContent = ThumbnailApp.formatDuration(video.duration_seconds);

    setAnalyzed(view.kind === 'ready', video);
}

/**
 * Show the workspace of an analyzed video, or what analyzing does for one
 * that has no analysis yet (the other tabs stay off until then).
 * @param {boolean} analyzed - Whether the video is analyzed
 * @param {Object} video - Video from the API
 */
function setAnalyzed(analyzed, video) {
    const app = document.getElementById('videoDetailApp');
    app.classList.toggle('is-not-analyzed', !analyzed);
    document.getElementById('notAnalyzedState').classList.toggle('hidden', analyzed);
    const error = document.getElementById('notAnalyzedError');
    // 'pending' with a reason: the worker stopped in the middle of its analysis
    const failed = !analyzed && ['error', 'pending'].includes(video.status) && video.error_message;
    error.classList.toggle('hidden', !failed);
    error.textContent = failed ? t('video_detail.analysis_failed', { error: video.error_message }) : '';
    document.getElementById('videoMoreBtn').classList.toggle('hidden', !analyzed);

    document.querySelectorAll('.workspace-tab').forEach(button => {
        if (button.dataset.tab !== DEFAULT_TAB) button.disabled = !analyzed;
    });
    if (!analyzed) switchTab(DEFAULT_TAB);
    document.dispatchEvent(new CustomEvent('help:context'));
}

/**
 * Load and display video transcription. The Transcript tab only shows when there is one.
 */
async function loadTranscription() {
    const tab = document.getElementById('tab-transcript');
    const textDiv = document.getElementById('transcriptionText');

    try {
        const response = await fetch(`/api/titles/transcription/${state.videoId}`);
        // H7: Check response before parsing
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        const exists = Boolean(data.exists && data.text);
        tab.classList.toggle('hidden', !exists);
        if (exists) {
            textDiv.innerHTML = formatTranscriptionWithColors(data.text);
        }
    } catch (error) {
        console.error('Error loading transcription:', error);
        tab.classList.add('hidden');
    }

    if (tab.classList.contains('hidden') && document.getElementById('videoDetailApp').dataset.tab === 'transcript') {
        switchTab(DEFAULT_TAB);
    }
}

/**
 * Start the first analysis of a video that has none.
 */
async function analyzeVideo() {
    const button = document.getElementById('analyzeVideoBtn');
    button.disabled = true;
    try {
        const response = await fetch(`/api/analysis/${state.videoId}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        await loadVideoInfo();
        // TaskQueue events show the analysis overlay and reload the page content
    } catch (error) {
        ThumbnailApp.showToast(t('errors.analysis_failed') + ': ' + error.message, 'error');
    } finally {
        button.disabled = false;
    }
}

/**
 * Reset the analysis and analyze the video again (the server queues it).
 */
async function reanalyzeVideo() {
    const count = state.thumbnailsTotal;
    const confirmed = await ThumbnailApp.showModal({
        title: t('analysis.reanalyze_title'),
        htmlMessage: t('analysis.reanalyze_message', {
            thumbnails: t(count === 1 ? 'index.card.thumbnails_one' : 'index.card.thumbnails_other', { count })
        }),
        type: 'danger',
        confirmText: t('video_analyze.reanalyze_confirm'),
        cancelText: t('common.cancel')
    });
    if (!confirmed) return;

    try {
        const response = await fetch(`/api/videos/${state.videoId}/reanalyze`, { method: 'POST' });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        ThumbnailApp.showToast(t('analysis.reanalyze_started'), 'success');
        reloadPageContent();
    } catch (error) {
        ThumbnailApp.showToast(t('errors.analysis_failed') + ': ' + error.message, 'error');
    }
}

function setupHeaderActions() {
    document.getElementById('analyzeVideoBtn').addEventListener('click', analyzeVideo);
    document.querySelectorAll('[data-close-modal]').forEach(button => {
        button.addEventListener('click', () => closeModal(button.dataset.closeModal));
    });
    document.getElementById('contentTabs').addEventListener('click', (e) => {
        const button = e.target.closest('[data-content-tab]');
        if (button) switchContentTab(button.dataset.contentTab);
    });
    switchContentTab('titles');

    const moreBtn = document.getElementById('videoMoreBtn');
    moreBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        ThumbnailApp.showMenu(moreBtn, [
            { label: t('index.card.reanalyze'), icon: 'rotate', danger: true, onSelect: reanalyzeVideo }
        ]);
    });
}

/**
 * Format transcription text with speaker colors.
 * @param {string} text - Raw transcription text
 * @returns {string} Formatted HTML
 */
function formatTranscriptionWithColors(text) {
    const lines = text.split('\n').filter(line => line.trim());
    const lineRegex = /^\[([^\]]+)\]\s*(speaker_\d+):\s*(.*)$/;

    return lines.map(line => {
        const match = line.match(lineRegex);

        if (match) {
            const timestamp = escapeHtml(match[1]);
            const speaker = match[2];
            const content = escapeHtml(match[3]);

            const speakerNum = parseInt(speaker.match(/speaker_(\d+)/)[1], 10);
            const speakerClass = speakerNum <= 5
                ? `transcript-speaker-${speakerNum}`
                : 'transcript-speaker-unknown';

            return `<div class="transcription-line">` +
                `<span class="transcript-timestamp">[${timestamp}]</span> ` +
                `<span class="${speakerClass}">${escapeHtml(t('transcription.speaker', {n: speakerNum + 1}))}:</span> ` +
                `<span class="transcript-text">${content}</span>` +
                `</div>`;
        } else {
            return `<div class="transcription-line"><span class="transcript-text">${escapeHtml(line)}</span></div>`;
        }
    }).join('');
}

// =========================================================================
// ANALYSIS OVERLAY (Phase 2 - Task Queue UX)
// =========================================================================

/**
 * Analysis overlay DOM elements
 */
const analysisOverlay = {
    container: null,
    progressFill: null,
    step: null,
    percent: null,
    cancelBtn: null
};

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
 * Check if there's an active analysis task for this video.
 * @returns {Object|null} The analysis task or null
 */
function getActiveAnalysisTask() {
    if (!window.TaskQueue?.tasks) return null;

    for (const task of window.TaskQueue.tasks.values()) {
        if (task.type === 'analysis' && task.video_id === state.videoId) {
            return task;
        }
    }
    return null;
}

/**
 * Show the analysis overlay with current task data.
 * @param {Object} task - The analysis task
 */
function showAnalysisOverlay(task) {
    if (!analysisOverlay.container) return;

    analysisOverlay.container.classList.remove('hidden');
    updateAnalysisOverlay(task);
}

/**
 * Hide the analysis overlay.
 */
function hideAnalysisOverlay() {
    if (!analysisOverlay.container) return;
    analysisOverlay.container.classList.add('hidden');
}

/**
 * Update the analysis overlay with task progress.
 * @param {Object} task - The analysis task
 */
function updateAnalysisOverlay(task) {
    if (!analysisOverlay.container) return;

    // A cancelled analysis that is still stopping keeps the bar where it was
    const stopping = window.TaskQueue.isStopping(task);
    if (!stopping) {
        const progress = task.progress || 0;
        analysisOverlay.progressFill.style.width = `${progress}%`;
        analysisOverlay.percent.textContent = `${progress}%`;
    }
    analysisOverlay.step.textContent = window.TaskQueue.stepLabel(task);
    analysisOverlay.percent.classList.toggle('hidden', stopping);
    analysisOverlay.cancelBtn.classList.toggle('hidden', stopping);
}

/**
 * Cancel the analysis task.
 */
async function cancelAnalysis() {
    const task = getActiveAnalysisTask();
    if (!task) return;

    const confirmed = await showModal({
        title: t('task_queue.cancel_title'),
        message: t('task_queue.cancel_confirm'),
        type: 'warning',
        confirmText: t('common.cancel'),
        cancelText: t('common.close'),
        showCancel: true
    });

    if (confirmed) {
        // The task queue says whether it could; its events update the overlay
        await window.TaskQueue.cancelTask('analysis', task.id);
    }
}

/**
 * Reload all page content after an analysis ends.
 */
function reloadPageContent() {
    loadVideoInfo().catch(error => {
        ThumbnailApp.showToast(`${t('errors.load_video')}: ${error.message}`, 'error');
    });
    loadTranscription();
    Clusters.load();
    Thumbnails.loadExisting();
}

/**
 * Initialize analysis overlay and check for active analysis.
 */
async function initAnalysisOverlay() {
    // Cache DOM elements
    analysisOverlay.container = document.getElementById('analysisOverlay');
    analysisOverlay.progressFill = document.getElementById('analysisProgressFill');
    analysisOverlay.step = document.getElementById('analysisStep');
    analysisOverlay.percent = document.getElementById('analysisPercent');
    analysisOverlay.cancelBtn = document.getElementById('cancelAnalysisBtn');

    if (!analysisOverlay.container) return;

    // Bind cancel button
    analysisOverlay.cancelBtn?.addEventListener('click', cancelAnalysis);

    // Wait for TaskQueue to be ready
    const ready = await waitForTaskQueue();
    if (!ready) {
        console.warn('TaskQueue not ready, skipping analysis overlay check');
        return;
    }

    // Check if there's an active analysis task for this video
    const task = getActiveAnalysisTask();
    if (task) {
        showAnalysisOverlay(task);
    }

    // Subscribe to TaskQueue events for this video's analysis
    window.TaskQueue.subscribe((eventType, task) => {
        // H6: Handle tasks_snapshot - find relevant task from the array
        if (eventType === 'tasks_snapshot') {
            const tasks = Array.isArray(task?.tasks) ? task.tasks : [];
            const relevantTask = tasks.find(t => t.video_id === state.videoId && t.type === 'analysis');
            if (!relevantTask) return;
            showAnalysisOverlay(relevantTask);
            return;
        }

        if (task.video_id !== state.videoId || task.type !== 'analysis') return;

        switch (eventType) {
            case 'task_started':
                showAnalysisOverlay(task);
                break;

            case 'task_progress':
                updateAnalysisOverlay(task);
                break;

            // However it ended (the task queue says how), the page shows the
            // video as it is now: analyzed, as it was, or with why it failed
            case 'task_completed':
            case 'task_cancelled':
            case 'task_error':
                hideAnalysisOverlay();
                reloadPageContent();
                break;
        }
    });
}

// =========================================================================
// GENERATION TASK SUBSCRIPTION (Phase 3 - Task Queue UX)
// =========================================================================

/**
 * Setup subscription to TaskQueue for generation events.
 * Detects jobs started from other pages and syncs gallery on completion.
 */
function setupGenerationTaskSubscription() {
    if (!window.TaskQueue) {
        console.warn('TaskQueue not available, skipping generation subscription');
        return;
    }

    window.TaskQueue.subscribe((eventType, task) => {
        // H6: Handle tasks_snapshot for generation tasks
        if (eventType === 'tasks_snapshot') {
            const tasks = Array.isArray(task?.tasks) ? task.tasks : [];
            const relevantTask = tasks.find(t => t.video_id === state.videoId && t.type === 'generation');
            if (!relevantTask) return;
            // Adopt the job if we don't have one
            if (!state.currentJobId) {
                console.log(`Detected generation job ${relevantTask.id} from snapshot, adopting`);
                state.currentJobId = relevantTask.id;
                import('./state.js').then(mod => {
                    mod.saveGenerationJobState();
                });
                Generation.restoreGenerationState();
            }
            return;
        }

        if (task.video_id !== state.videoId) return;
        if (task.type !== 'generation') return;

        switch (eventType) {
            case 'task_started':
                // Job started (possibly from another page)
                if (!state.currentJobId) {
                    console.log(`Detected external generation job ${task.id}, adopting`);
                    state.currentJobId = task.id;
                    // Save state and restore UI
                    import('./state.js').then(mod => {
                        mod.saveGenerationJobState();
                    });
                    Generation.restoreGenerationState();
                }
                break;

            // The job ended and let go of the video, however it did: a cancelled
            // or failed one keeps the thumbnails it had made
            case 'task_completed':
            case 'task_cancelled':
            case 'task_error':
                Thumbnails.loadExisting();
                // The job's own stream restores the button too; this covers a cut one
                Generation.restoreSubmitButtonIfNeeded();
                break;
        }
    });
}

// =========================================================================
// UI HELPERS
// =========================================================================

/**
 * Close a modal by ID.
 * @param {string} modalId - Modal element ID
 */
function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('visible');
}

/**
 * Switch the Titles & description tab between titles and descriptions.
 * @param {string} tabName - 'titles' or 'descriptions'
 */
function switchContentTab(tabName) {
    document.querySelectorAll('#contentTabs [data-content-tab]').forEach(button => {
        const active = button.dataset.contentTab === tabName;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
    });
    document.getElementById('tabContentTitles').classList.toggle('hidden', tabName !== 'titles');
    document.getElementById('tabContentDescriptions').classList.toggle('hidden', tabName !== 'descriptions');
}

// =========================================================================
// INITIALIZATION
// =========================================================================

document.addEventListener('DOMContentLoaded', async () => {
    const container = document.getElementById('videoDetailApp');
    if (!container) {
        console.error('Video detail container not found');
        return;
    }

    const videoId = parseInt(container.dataset.videoId, 10);
    if (isNaN(videoId)) {
        console.error('Invalid video ID');
        return;
    }

    initState(videoId);
    state.faceSwapMissing = container.dataset.faceSwapMissing;
    setupTabs();
    setupHeaderActions();

    // The Generate form starts with the app settings (settings.js says if they fail to load)
    await AppSettings.ready;
    Defaults.apply(AppSettings.get().generation);
    AiConfig.init();
    document.addEventListener('settings:changed', (e) => {
        Defaults.apply(e.detail.settings.generation, e.detail.previous.generation);
        AiConfig.updateRefMax();
        AiConfig.updateSummaries();
    });
    People.init({ setView: Clusters.setViewType });
    References.init();
    document.addEventListener('people:changed', Clusters.updatePeopleBadges);
    // Back from the person editor with the browser's button: people may have
    // been renamed, emptied or renumbered there, and the list acts by position
    window.addEventListener('pageshow', (e) => { if (e.persisted) Clusters.load(); });

    // Load all data
    loadVideoInfo().catch(error => {
        ThumbnailApp.showToast(`${t('errors.load_video')}: ${error.message}`, 'error');
    });
    loadTranscription();
    Clusters.setupControls();
    Thumbnails.setupControls();
    Thumbnails.initSort();
    Thumbnails.loadExisting();
    Titles.setupControls();
    Titles.loadSaved();
    Descriptions.setupControls();
    Descriptions.loadSaved();

    // Setup
    Generation.setupForm();
    Thumbnails.setupLazyLoading();
    Lightbox.init();
    AiConfig.checkGranSabioStatus();

    // Restore generation state if returning to page with active job
    Generation.restoreGenerationState();

    // Initialize analysis overlay (Phase 2 - Task Queue UX)
    initAnalysisOverlay();

    // Initialize generation task subscription (Phase 3 - Task Queue UX)
    setupGenerationTaskSubscription();

    // Initialize queue indicator (Phase 4 - Task Queue UX)
    QueueIndicator.init();
});
