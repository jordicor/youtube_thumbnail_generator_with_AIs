/**
 * Video Detail Page - Shared State
 * =================================
 * Singleton state object shared across all modules.
 * Import this module to access or modify global page state.
 */

/**
 * Global state object for the video detail page.
 * All modules share this state via import.
 */
export const state = {
    // Core identifiers
    videoId: null,
    currentJobId: null,

    // Clusters state
    clusters: [],
    selectedClusters: new Set(),
    currentClusterType: 'person',
    clusterToDelete: null,
    referenceClusterIndex: null,  // Cluster selected for AI reference images

    // Titles state
    titles: [],                    // Array of {text: string, dbId: number|null}
    selectedTitles: new Set(),     // Set of selected title indices

    // Descriptions state
    descriptions: [],              // Array of {text: string, dbId: number|null}

    // Thumbnails state
    existingThumbnails: [],
    lazyLoadObserver: null,

    // AI Models state
    promptModels: null,            // Loaded from API
    modelsLoaded: false,

    // Generation state
    currentEventSource: null,      // SSE EventSource instance
};

/**
 * Initialize state with video ID.
 * Call this once when the page loads.
 * @param {number} videoId - The video ID from the URL
 */
export function initState(videoId) {
    state.videoId = videoId;
}

/**
 * Reset generation-related state.
 * Call this before starting a new generation job.
 */
export function resetGenerationState() {
    state.currentJobId = null;
    if (state.currentEventSource) {
        state.currentEventSource.close();
        state.currentEventSource = null;
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

/**
 * Clear reference cluster selection.
 */
export function clearReferenceCluster() {
    state.referenceClusterIndex = null;
}
