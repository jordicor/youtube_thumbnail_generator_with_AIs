/**
 * Video Detail Page - Queue Indicator Module
 * ==========================================
 * Phase 4 - Task Queue UX Improvements
 *
 * Shows a badge and status line for generation tasks on the current video.
 * - Badge: Shows total task count, clickable to open task queue modal
 * - Status line: Shows contextual text (generating progress, queued count, etc.)
 */

import { state } from './state.js';

// DOM element references
let queueBadge = null;
let queueStatus = null;
let queueText = null;
let queueLink = null;

// Track if we're initialized
let initialized = false;

/**
 * Initialize the queue indicator.
 * Call this after DOM is ready and state is initialized.
 */
export function init() {
    if (initialized) return;

    // Cache DOM elements
    queueBadge = document.getElementById('generationQueueBadge');
    queueStatus = document.getElementById('generationQueueStatus');
    queueText = document.getElementById('generationQueueText');
    queueLink = document.getElementById('generationQueueLink');

    if (!queueBadge || !queueStatus) {
        console.warn('Queue indicator elements not found');
        return;
    }

    // Bind click handlers
    queueBadge.addEventListener('click', openTaskQueueModal);
    queueLink?.addEventListener('click', (e) => {
        e.preventDefault();
        openTaskQueueModal();
    });

    // Subscribe to TaskQueue events
    if (window.TaskQueue) {
        window.TaskQueue.subscribe(handleTaskQueueEvent);
        // Initial update with current state
        updateQueueIndicator();
    } else {
        // TaskQueue might not be ready yet, wait for it
        waitForTaskQueue().then(() => {
            window.TaskQueue.subscribe(handleTaskQueueEvent);
            updateQueueIndicator();
        });
    }

    initialized = true;
}

/**
 * Wait for TaskQueue to be available.
 * @param {number} maxWait - Maximum wait time in ms
 * @returns {Promise<boolean>}
 */
async function waitForTaskQueue(maxWait = 3000) {
    const start = Date.now();
    while (!window.TaskQueue?.tasks && Date.now() - start < maxWait) {
        await new Promise(r => setTimeout(r, 100));
    }
    return !!window.TaskQueue?.tasks;
}

/**
 * Handle TaskQueue events.
 * @param {string} eventType - Event type from TaskQueue
 * @param {Object} taskOrData - Task data or snapshot data
 */
function handleTaskQueueEvent(eventType, taskOrData) {
    // Handle snapshot - always update as it contains fresh state
    if (eventType === 'tasks_snapshot') {
        updateQueueIndicator();
        return;
    }

    // For individual events, filter by generation type and this video
    if (taskOrData.type !== 'generation') return;
    if (taskOrData.video_id !== state.videoId) return;

    // Update indicator on any relevant event
    updateQueueIndicator();
}

/**
 * Get all generation tasks for the current video.
 * @returns {Array} Array of generation tasks
 */
function getGenerationTasksForVideo() {
    if (!window.TaskQueue?.tasks) return [];

    const tasks = [];
    for (const task of window.TaskQueue.tasks.values()) {
        if (task.type === 'generation' && task.video_id === state.videoId) {
            tasks.push(task);
        }
    }
    return tasks;
}

/**
 * Update the queue indicator UI based on current tasks.
 */
function updateQueueIndicator() {
    const tasks = getGenerationTasksForVideo();

    if (tasks.length === 0) {
        hideQueueIndicator();
        return;
    }

    // Show badge with task count
    queueBadge.textContent = tasks.length;
    queueBadge.classList.remove('hidden');

    // Generate status text
    const statusText = formatQueueStatusText(tasks);
    queueText.textContent = statusText;
    queueStatus.classList.remove('hidden');
}

/**
 * Hide the queue indicator.
 */
function hideQueueIndicator() {
    queueBadge?.classList.add('hidden');
    queueStatus?.classList.add('hidden');
}

/**
 * Format the status text based on current tasks.
 * @param {Array} tasks - Array of generation tasks
 * @returns {string} Formatted status text
 */
function formatQueueStatusText(tasks) {
    // Categorize tasks
    const running = tasks.filter(t =>
        t.status === 'generating' ||
        t.status === 'prompting' ||
        t.status === 'transcribing'
    );
    const queued = tasks.filter(t => t.status === 'pending');

    // Case: Single task running - show thumbnail progress
    if (tasks.length === 1 && running.length === 1) {
        const task = running[0];
        const done = task.thumbnails_generated || 0;
        const total = task.total_thumbnails || '?';
        return t('queue_indicator.generating_thumbnails', { done, total });
    }

    // Case: Multiple tasks with some running and some queued
    if (running.length > 0 && queued.length > 0) {
        return t('queue_indicator.running_and_queued', {
            running: running.length,
            queued: queued.length
        });
    }

    // Case: Only running tasks
    if (running.length > 0) {
        return t('queue_indicator.tasks_running', { count: running.length });
    }

    // Case: Only queued tasks
    if (queued.length > 0) {
        return t('queue_indicator.tasks_queued', { count: queued.length });
    }

    return '';
}

/**
 * Open the global Task Queue modal.
 */
function openTaskQueueModal() {
    if (window.TaskQueue?.showModal) {
        window.TaskQueue.showModal();
    }
}

/**
 * Force update the indicator (useful for external calls).
 */
export function refresh() {
    updateQueueIndicator();
}
