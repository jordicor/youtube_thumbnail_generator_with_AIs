/**
 * Video Detail Page - Main Entry Point
 * =====================================
 * Initializes all modules and exposes functions for inline event handlers.
 */

import { initState, state } from './state.js';
import * as AiConfig from './ai-config.js';
import * as Clusters from './clusters.js';
import * as Titles from './titles.js';
import * as Descriptions from './descriptions.js';
import * as Thumbnails from './thumbnails.js';
import * as Generation from './generation.js';
import * as QueueIndicator from './queue-indicator.js';

// =========================================================================
// PAGE-LEVEL FUNCTIONS
// =========================================================================

/**
 * Load video information and display it.
 */
async function loadVideoInfo() {
    try {
        const response = await fetch(`/api/videos/${state.videoId}`);
        const video = await response.json();

        document.getElementById('videoTitle').textContent = video.filename;

        const infoDiv = document.getElementById('videoInfo');
        const duration = video.duration_seconds
            ? `${Math.floor(video.duration_seconds / 60)}:${String(Math.floor(video.duration_seconds % 60)).padStart(2, '0')}`
            : t('common.na');

        infoDiv.innerHTML = `
            <p><strong>${t('video_detail.info.filename')}:</strong> ${escapeHtml(video.filename)}</p>
            <p><strong>${t('video_detail.info.duration')}:</strong> ${escapeHtml(duration)}</p>
            <p><strong>${t('video_detail.info.title')}:</strong> ${escapeHtml(video.status)}</p>
        `;
    } catch (error) {
        document.getElementById('videoInfo').innerHTML =
            `<div class="error">${t('errors.load_video')}: ${escapeHtml(error.message)}</div>`;
    }
}

/**
 * Load and display video transcription.
 */
async function loadTranscription() {
    const section = document.getElementById('transcriptionSection');
    const textDiv = document.getElementById('transcriptionText');

    try {
        const response = await fetch(`/api/titles/transcription/${state.videoId}`);
        const data = await response.json();

        if (data.exists && data.text) {
            section.style.display = 'block';
            textDiv.innerHTML = formatTranscriptionWithColors(data.text);
        } else {
            section.style.display = 'none';
        }
    } catch (error) {
        console.error('Error loading transcription:', error);
        section.style.display = 'none';
    }
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

            const speakerNum = speaker.match(/speaker_(\d+)/);
            const speakerClass = speakerNum && parseInt(speakerNum[1]) <= 5
                ? `transcript-speaker-${speakerNum[1]}`
                : 'transcript-speaker-unknown';

            return `<div class="transcription-line">` +
                `<span class="transcript-timestamp">[${timestamp}]</span> ` +
                `<span class="${speakerClass}">${escapeHtml(speaker)}:</span> ` +
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

    const progress = task.progress || 0;
    const step = task.current_step || t('analysis_overlay.starting');

    analysisOverlay.progressFill.style.width = `${progress}%`;
    analysisOverlay.step.textContent = step;
    analysisOverlay.percent.textContent = `${progress}%`;
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
        try {
            const response = await fetch(`/api/tasks/analysis/${task.id}/cancel`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || 'Failed to cancel');
            }
        } catch (error) {
            console.error('Error cancelling analysis:', error);
            if (window.showToast) {
                showToast(error.message, 'error');
            }
        }
    }
}

/**
 * Reload all page content after analysis completes.
 */
function reloadPageContent() {
    loadVideoInfo();
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
        if (task.video_id !== state.videoId || task.type !== 'analysis') return;

        switch (eventType) {
            case 'task_started':
                showAnalysisOverlay(task);
                break;

            case 'task_progress':
                updateAnalysisOverlay(task);
                break;

            case 'task_completed':
                hideAnalysisOverlay();
                reloadPageContent();
                if (window.showToast) {
                    showToast(t('analysis.complete'), 'success');
                }
                break;

            case 'task_cancelled':
                hideAnalysisOverlay();
                if (window.showToast) {
                    showToast(t('task_queue.cancelled'), 'info');
                }
                break;

            case 'task_error':
                hideAnalysisOverlay();
                if (window.showToast) {
                    const msg = task.error_message || t('errors.analysis_failed');
                    showToast(msg, 'error');
                }
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

            case 'task_completed':
                // Reload full gallery to ensure sync
                console.log(`Generation job ${task.id} completed, reloading gallery`);
                Thumbnails.loadExisting();
                // Restore button if SSE complete event didn't arrive
                Generation.restoreSubmitButtonIfNeeded();
                break;
        }
    });
}

// =========================================================================
// UI HELPERS
// =========================================================================

/**
 * Toggle collapsible section visibility.
 * @param {string} sectionId - Section element ID
 */
function toggleSection(sectionId) {
    const content = document.getElementById(sectionId);
    const icon = document.getElementById(sectionId + 'Icon');

    if (content.classList.contains('collapsed')) {
        content.classList.remove('collapsed');
        icon.classList.remove('collapsed');
    } else {
        content.classList.add('collapsed');
        icon.classList.add('collapsed');
    }
}

/**
 * Close a modal by ID.
 * @param {string} modalId - Modal element ID
 */
function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('visible');
}

/**
 * Switch content tab (titles/descriptions).
 * @param {string} tabName - Tab name ('titles' or 'descriptions')
 */
function switchContentTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    if (tabName === 'titles') {
        document.getElementById('tabTitles').classList.add('active');
        document.getElementById('tabContentTitles').classList.add('active');
    } else if (tabName === 'descriptions') {
        document.getElementById('tabDescriptions').classList.add('active');
        document.getElementById('tabContentDescriptions').classList.add('active');
    }
}

// =========================================================================
// INITIALIZATION
// =========================================================================

document.addEventListener('DOMContentLoaded', () => {
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

    // Load all data
    loadVideoInfo();
    loadTranscription();
    Clusters.load();
    Thumbnails.loadExisting();
    Titles.loadSaved();
    Descriptions.loadSaved();

    // Setup
    Generation.setupForm();
    Thumbnails.setupLazyLoading();
    AiConfig.checkGranSabioStatus();
    AiConfig.loadModelsFromAPI();
    AiConfig.init();

    // Restore generation state if returning to page with active job
    Generation.restoreGenerationState();

    // Initialize analysis overlay (Phase 2 - Task Queue UX)
    initAnalysisOverlay();

    // Initialize generation task subscription (Phase 3 - Task Queue UX)
    setupGenerationTaskSubscription();

    // Initialize queue indicator (Phase 4 - Task Queue UX)
    QueueIndicator.init();
});

// =========================================================================
// GLOBAL EXPORTS (for inline event handlers in HTML)
// =========================================================================

window.VideoDetail = {
    // UI helpers
    toggleSection,
    closeModal,
    switchContentTab,

    // AI Config
    toggleAiPopover: AiConfig.togglePopover,
    closeAiPopover: AiConfig.closePopover,
    closeAllAiPopovers: AiConfig.closeAllPopovers,
    updateTextAiModelOptions: AiConfig.updateTextModelOptions,
    toggleTextAiThinkingLevel: AiConfig.toggleThinkingLevel,
    updateImageAiModelOptions: AiConfig.updateImageModelOptions,
    updateImageAiOpenaiWarning: AiConfig.updateOpenaiWarning,
    updateImageAiRefMax: AiConfig.updateRefMax,

    // Clusters
    toggleClusterSelection: Clusters.toggleSelection,
    toggleSelectAllClusters: Clusters.toggleSelectAll,
    setClusterAsReference: Clusters.setAsReference,
    showDeleteModal: Clusters.showDeleteModal,
    showDeleteSelectedModal: Clusters.showDeleteSelectedModal,
    confirmDeleteCluster: Clusters.confirmDelete,
    showMergeModal: Clusters.showMergeModal,
    confirmMergeClusters: Clusters.confirmMerge,
    changeClusterViewType: Clusters.changeViewType,

    // Titles
    generateTitles: Titles.generate,
    toggleTitleSelection: Titles.toggleSelection,
    toggleSelectAllTitles: Titles.toggleSelectAll,
    deleteTitle: Titles.deleteOne,
    deleteSelectedTitles: Titles.deleteSelected,
    copyTitle: Titles.copy,
    copyAllTitles: Titles.copyAll,
    updateTitleStyleUI: Titles.updateStyleUI,

    // Descriptions
    generateDescriptions: Descriptions.generate,
    deleteDescription: Descriptions.deleteOne,
    deleteAllDescriptions: Descriptions.deleteAll,
    copyDescription: Descriptions.copy,
    copyAllDescriptions: Descriptions.copyAll,
    updateDescStyleUI: Descriptions.updateStyleUI,

    // Thumbnails
    deleteThumbnail: Thumbnails.deleteOne,
    downloadAllThumbnails: Thumbnails.downloadAll,

    // Generation
    clearReferenceImage: Generation.clearReferenceImage,
};
