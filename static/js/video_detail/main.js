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
            : 'N/A';

        infoDiv.innerHTML = `
            <p><strong>Archivo:</strong> ${escapeHtml(video.filename)}</p>
            <p><strong>Duracion:</strong> ${escapeHtml(duration)}</p>
            <p><strong>Estado:</strong> ${escapeHtml(video.status)}</p>
        `;
    } catch (error) {
        document.getElementById('videoInfo').innerHTML =
            `<div class="error">Error cargando video: ${escapeHtml(error.message)}</div>`;
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
