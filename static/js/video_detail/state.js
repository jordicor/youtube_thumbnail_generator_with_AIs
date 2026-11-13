/**
 * Video Detail Page - Shared State
 * =================================
 * Singleton state object shared across all modules.
 * Import this module to access or modify global page state.
 */

// LocalStorage key for persisting generation job state
const STORAGE_KEY_GENERATION_JOB = 'videoDetail_generationJob';

/**
 * Global state object for the video detail page.
 * All modules share this state via import.
 */
export const state = {
    // Core identifiers
    videoId: null,
    currentJobId: null,

    // Clusters state
    clusters: [],                  // Clusters of the current view (Detected people tab)
    personCount: null,             // People of the 'person' view, shown in the page header
    selectedClusters: new Set(),   // Cluster indices checked for merge/delete
    currentClusterType: 'person',
    clusterToDelete: null,
    // People chosen for the thumbnail: ordered cluster objects (any view), kept by
    // database ID. Position maps to the person_XX role: 0 → person_01, and so on.
    selectedPeople: [],
    peopleInitialized: false,      // The default person is only preselected once

    // Titles state
    titles: [],                    // Array of {text: string, dbId: number|null}
    selectedTitles: new Set(),     // Indices of the starred titles, which guide the thumbnails

    // Descriptions state
    descriptions: [],              // Array of {text: string, dbId: number|null}

    // Thumbnails state
    thumbnails: [],                // Thumbnails loaded in the gallery, in its order
    thumbnailsSort: 'recent',      // Gallery order: 'recent' | 'failures_last' | 'match'
    thumbnailsTotal: 0,            // Total thumbnails of the video (the gallery loads them by pages)
    newestThumbnails: [],          // Newest page of thumbnails whatever the gallery order (recent results)
    lazyLoadObserver: null,

    // Gran Sabio LLM availability: null until checked, then true/false.
    // Every text generation (titles, descriptions, thumbnail prompts) needs it.
    granSabioAvailable: null,

    // Why face swap cannot run (its models are missing), from the page; '' when it can
    faceSwapMissing: '',

    // Style image of the next generation: {base64, filename, dataUrl}, or null
    styleImage: null,

    // Generation state
    currentEventSource: null,      // SSE EventSource instance
};

/**
 * Initialize state with video ID.
 * Call this once when the page loads.
 * Also restores any persisted generation job state.
 * @param {number} videoId - The video ID from the URL
 */
export function initState(videoId) {
    state.videoId = videoId;
    restoreGenerationJobState(videoId);
}

/**
 * Restore generation job state from localStorage.
 * Only restores if the stored job belongs to the current video.
 * @param {number} videoId - Current video ID
 */
function restoreGenerationJobState(videoId) {
    try {
        const stored = localStorage.getItem(STORAGE_KEY_GENERATION_JOB);
        if (!stored) return;

        const { videoId: storedVideoId, jobId } = JSON.parse(stored);
        if (storedVideoId === videoId && jobId) {
            state.currentJobId = jobId;
            console.log(`Restored generation job ${jobId} for video ${videoId}`);
        }
    } catch (error) {
        console.warn('Failed to restore generation job state:', error);
        localStorage.removeItem(STORAGE_KEY_GENERATION_JOB);
    }
}

/**
 * Save current generation job state to localStorage.
 * Call this after successfully starting a generation job.
 */
export function saveGenerationJobState() {
    if (!state.currentJobId || !state.videoId) return;

    try {
        localStorage.setItem(STORAGE_KEY_GENERATION_JOB, JSON.stringify({
            videoId: state.videoId,
            jobId: state.currentJobId
        }));
    } catch (error) {
        console.warn('Failed to save generation job state:', error);
    }
}

/**
 * Clear generation job state from localStorage.
 * Call this when a job completes, fails, or is cancelled.
 */
export function clearGenerationJobState() {
    localStorage.removeItem(STORAGE_KEY_GENERATION_JOB);
}

/**
 * Reset generation-related state.
 * Call this before starting a new generation job.
 * @param {boolean} clearStorage - Whether to also clear localStorage (default: false)
 */
export function resetGenerationState(clearStorage = false) {
    state.currentJobId = null;
    if (state.currentEventSource) {
        state.currentEventSource.close();
        state.currentEventSource = null;
    }
    if (clearStorage) {
        clearGenerationJobState();
    }
}

/**
 * Clear all titles from state.
 */
export function clearTitles() {
    state.titles = [];
    state.selectedTitles.clear();
}

/**
 * Clear all descriptions from state.
 */
export function clearDescriptions() {
    state.descriptions = [];
}

/**
 * Clear cluster selection.
 */
export function clearClusterSelection() {
    state.selectedClusters.clear();
}
