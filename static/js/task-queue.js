/**
 * Task Queue Manager
 * ==================
 * Global component for viewing and managing running tasks.
 * Connects to SSE endpoint for real-time updates.
 *
 * It is the one that tells, on every page, how each task ended (a toast) and
 * whether tasks can run at all (the banner under the header). Pages only
 * refresh what they show.
 */

class TaskQueueManager {
    constructor() {
        this.tasks = new Map();
        // 'ok', 'redis_down' or 'worker_down'; null until the server says
        this.queueStatus = null;
        this.eventSource = null;
        this.modalVisible = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;

        // Event listeners for external components (e.g., index.html)
        this.listeners = new Set();

        // DOM elements (populated in init)
        this.elements = {
            btn: null,
            badge: null,
            modal: null,
            content: null,
            closeBtn: null
        };
    }

    /**
     * Subscribe to task events.
     * Callback receives (eventType, task) where eventType is one of:
     * 'task_started', 'task_progress', 'task_completed', 'task_cancelled', 'task_error', 'tasks_snapshot'
     * @param {Function} callback - Function to call on task events
     * @returns {Function} Unsubscribe function
     */
    subscribe(callback) {
        this.listeners.add(callback);
        return () => this.listeners.delete(callback);
    }

    /**
     * Notify all listeners of a task event.
     * @param {string} eventType - Type of event
     * @param {Object} task - Task data
     */
    _notifyListeners(eventType, task) {
        for (const callback of this.listeners) {
            try {
                callback(eventType, task);
            } catch (e) {
                console.error('TaskQueue listener error:', e);
            }
        }
    }

    /**
     * Initialize the task queue manager.
     * Call this after DOM is ready.
     */
    init() {
        this.cacheElements();

        if (!this.elements.btn || !this.elements.modal) {
            console.warn('TaskQueue: Required elements not found in DOM');
            return;
        }

        this.bindEvents();
        this.connectSSE();
    }

    /**
     * Cache DOM element references.
     */
    cacheElements() {
        this.elements.btn = document.getElementById('taskQueueBtn');
        this.elements.badge = document.getElementById('taskQueueBadge');
        this.elements.modal = document.getElementById('taskQueueModal');
        this.elements.content = document.getElementById('taskQueueContent');
        this.elements.closeBtn = document.getElementById('taskQueueClose');
    }

    /**
     * Bind event listeners.
     */
    bindEvents() {
        // Toggle modal on button click
        this.elements.btn.addEventListener('click', () => this.toggleModal());

        // Close modal
        this.elements.closeBtn.addEventListener('click', () => this.hideModal());

        // Close on overlay click
        this.elements.modal.addEventListener('click', (e) => {
            if (e.target === this.elements.modal) {
                this.hideModal();
            }
        });

        // M51: Close on Escape key only when task queue modal is actually visible
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.modalVisible && this.elements.modal?.classList.contains('visible')) {
                this.hideModal();
            }
        });

        // Delegate cancel button clicks
        this.elements.content.addEventListener('click', (e) => {
            const cancelBtn = e.target.closest('.task-item-cancel');
            if (cancelBtn) {
                const taskType = cancelBtn.dataset.taskType;
                const taskId = parseInt(cancelBtn.dataset.taskId, 10);
                this.confirmCancelTask(taskType, taskId);
            }
        });
    }

    /**
     * Connect to SSE endpoint for real-time task updates.
     */
    connectSSE() {
        // M34: Clear any pending reconnect timer to prevent duplicates
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }

        if (this.eventSource) {
            this.eventSource.close();
        }

        this.eventSource = new EventSource('/api/events/tasks');

        this.eventSource.onopen = () => {
            console.log('TaskQueue: SSE connected');
            this.reconnectAttempts = 0;
        };

        this.eventSource.onerror = () => {
            console.warn('TaskQueue: SSE error, will reconnect...');
            this.handleSSEError();
        };

        // Initial snapshot of all tasks
        this.eventSource.addEventListener('tasks_snapshot', (e) => {
            try {
                const data = JSON.parse(e.data);
                this.handleTasksSnapshot(data);
            } catch (err) {
                console.error('TaskQueue: Error parsing tasks_snapshot', err);
            }
        });

        // New task started
        this.eventSource.addEventListener('task_started', (e) => {
            try {
                const task = JSON.parse(e.data);
                this.handleTaskStarted(task);
            } catch (err) {
                console.error('TaskQueue: Error parsing task_started', err);
            }
        });

        // Task progress update
        this.eventSource.addEventListener('task_progress', (e) => {
            try {
                const task = JSON.parse(e.data);
                this.handleTaskProgress(task);
            } catch (err) {
                console.error('TaskQueue: Error parsing task_progress', err);
            }
        });

        // Task completed
        this.eventSource.addEventListener('task_completed', (e) => {
            try {
                const task = JSON.parse(e.data);
                this.handleTaskCompleted(task);
            } catch (err) {
                console.error('TaskQueue: Error parsing task_completed', err);
            }
        });

        // Task cancelled
        this.eventSource.addEventListener('task_cancelled', (e) => {
            try {
                const task = JSON.parse(e.data);
                this.handleTaskCancelled(task);
            } catch (err) {
                console.error('TaskQueue: Error parsing task_cancelled', err);
            }
        });

        // Task error
        this.eventSource.addEventListener('task_error', (e) => {
            try {
                const task = JSON.parse(e.data);
                this.handleTaskError(task);
            } catch (err) {
                console.error('TaskQueue: Error parsing task_error', err);
            }
        });

        // Whether queued tasks can run: Redis answers and a worker is alive
        this.eventSource.addEventListener('queue_status', (e) => {
            try {
                this.queueStatus = JSON.parse(e.data).status;
                this.renderQueueBanner();
            } catch (err) {
                console.error('TaskQueue: Error parsing queue_status', err);
            }
        });
    }

    /**
     * Handle SSE connection error with reconnection logic.
     * Uses exponential backoff with a maximum delay of 30 seconds.
     * Never stops trying to reconnect.
     */
    handleSSEError() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }

        this.reconnectAttempts++;
        // Exponential backoff: 2s, 4s, 8s, 16s, 30s (max)
        const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);
        console.warn(`TaskQueue: SSE error, reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
        // M34: Store timer ID so it can be cleaned up
        this._reconnectTimer = setTimeout(() => this.connectSSE(), delay);
    }

    /**
     * Generate unique key for a task.
     */
    getTaskKey(task) {
        return `${task.type}:${task.id}`;
    }

    /**
     * Whether a task is still waiting for the worker to take it.
     */
    isQueued(task) {
        return task.status === 'pending' || (task.type === 'analysis' && task.status === 'analyzing');
    }

    /**
     * Whether a task was cancelled and its worker has not stopped yet.
     */
    isStopping(task) {
        return task.status === 'stopping';
    }

    /**
     * What a task is doing right now, in words.
     */
    stepLabel(task) {
        if (this.isQueued(task)) {
            return t('task_queue.waiting');
        }
        const step = this.isStopping(task) ? `stopping_${task.type}` : task.status;
        return t(`task_queue.steps.${step}`);
    }

    /**
     * Handle the snapshot the stream sends on every (re)connection.
     */
    handleTasksSnapshot(data) {
        const previous = this.tasks;
        this.tasks = new Map();

        if (data.tasks && Array.isArray(data.tasks)) {
            for (const task of data.tasks) {
                const key = this.getTaskKey(task);
                this.tasks.set(key, task);
            }
        }

        this.render();
        this._notifyListeners('tasks_snapshot', data);

        // Tasks that ended while the stream was cut: nobody said how
        for (const [key, task] of previous) {
            if (!this.tasks.has(key)) {
                this.reportEndedTask(task);
            }
        }
    }

    /**
     * Ask how a task ended and tell it as the stream would have.
     */
    async reportEndedTask(task) {
        let outcome;
        try {
            const response = await fetch(`/api/tasks/${task.type}/${task.id}/outcome`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            outcome = await response.json();
        } catch (error) {
            console.error('TaskQueue: could not read how a task ended', error);
            return;
        }

        // The video got a new task meanwhile: the stream tells that one
        if (this.tasks.has(this.getTaskKey(task))) return;

        const ended = { ...task, ...outcome };
        if (outcome.status === 'completed') {
            this.handleTaskCompleted(ended);
        } else if (outcome.status === 'error') {
            this.handleTaskError(ended);
        } else {
            this.handleTaskCancelled(ended);
        }
    }

    /**
     * Handle new task started event.
     */
    handleTaskStarted(task) {
        const key = this.getTaskKey(task);
        this.tasks.set(key, task);
        this.render();
        this._notifyListeners('task_started', task);
    }

    /**
     * Handle task progress update.
     */
    handleTaskProgress(task) {
        const key = this.getTaskKey(task);
        this.tasks.set(key, task);
        this.render();
        this._notifyListeners('task_progress', task);
    }

    /**
     * Handle task completed event.
     */
    handleTaskCompleted(task) {
        const key = this.getTaskKey(task);
        this.tasks.delete(key);
        this.render();
        this._notifyListeners('task_completed', task);

        const name = ThumbnailApp.formatVideoName(task.video_name);
        const message = task.type === 'analysis' ? 'task_queue.analysis_complete' : 'task_queue.generation_complete';
        ThumbnailApp.showToast(t(message, { name }), 'success');
    }

    /**
     * Handle task cancelled event: the task was cancelled and has stopped, or
     * never got queued. Whoever cancelled it was told when they did.
     */
    handleTaskCancelled(task) {
        const key = this.getTaskKey(task);
        this.tasks.delete(key);
        this.render();
        this._notifyListeners('task_cancelled', task);
    }

    /**
     * Handle task error event.
     */
    handleTaskError(task) {
        const key = this.getTaskKey(task);
        this.tasks.delete(key);
        this.render();
        this._notifyListeners('task_error', task);

        const name = ThumbnailApp.formatVideoName(task.video_name);
        ThumbnailApp.showToast(`${name}: ${task.error_message || t('errors.generic')}`, 'error');
    }

    /**
     * Show cancel confirmation and perform cancellation.
     */
    async confirmCancelTask(taskType, taskId) {
        const confirmed = await showModal({
            title: t('task_queue.cancel_title'),
            message: t(taskType === 'generation' ? 'task_queue.cancel_confirm_generation' : 'task_queue.cancel_confirm'),
            type: 'warning',
            confirmText: t('common.cancel'),
            cancelText: t('common.close'),
            showCancel: true
        });

        if (confirmed) {
            await this.cancelTask(taskType, taskId);
        }
    }

    /**
     * Cancel a task via API and say whether it could. A running task shows as
     * stopping until its worker lets go; the stream then removes it.
     * @returns {Promise<boolean>} Whether the task was cancelled
     */
    async cancelTask(taskType, taskId) {
        try {
            const response = await fetch(`/api/tasks/${taskType}/${taskId}/cancel`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }

            ThumbnailApp.showToast(t('task_queue.cancelled'), 'info');
            return true;
        } catch (error) {
            console.error('TaskQueue: Cancel error', error);
            ThumbnailApp.showToast(error.message, 'error');
            return false;
        }
    }

    /**
     * Show under the header why tasks cannot run, while they cannot.
     */
    renderQueueBanner() {
        const banner = document.getElementById('queueBanner');
        const down = this.queueStatus === 'redis_down' || this.queueStatus === 'worker_down';
        banner.classList.toggle('hidden', !down);
        if (down) {
            document.getElementById('queueBannerText').textContent = t(`task_queue.${this.queueStatus}`);
        }
    }

    /**
     * Toggle modal visibility.
     */
    toggleModal() {
        if (this.modalVisible) {
            this.hideModal();
        } else {
            this.showModal();
        }
    }

    /**
     * Show the modal.
     */
    showModal() {
        this.modalVisible = true;
        this.elements.modal.classList.add('visible');
        this.render();
    }

    /**
     * Hide the modal.
     */
    hideModal() {
        this.modalVisible = false;
        this.elements.modal.classList.remove('visible');
    }

    /**
     * Render the task list and update badge.
     */
    render() {
        this.updateBadge();

        if (this.modalVisible) {
            this.renderTaskList();
        }
    }

    /**
     * Update the badge counter in header.
     */
    updateBadge() {
        const count = this.tasks.size;

        if (count > 0) {
            this.elements.badge.textContent = count > 9 ? '9+' : count;
            this.elements.badge.classList.remove('hidden');
            this.elements.btn.classList.add('has-tasks');
        } else {
            this.elements.badge.classList.add('hidden');
            this.elements.btn.classList.remove('has-tasks');
        }
    }

    /**
     * Render the task list inside the modal.
     */
    renderTaskList() {
        if (this.tasks.size === 0) {
            this.elements.content.innerHTML = `
                <div class="task-queue-empty">
                    <div class="task-queue-empty-icon">✓</div>
                    <div class="task-queue-empty-text">${t('task_queue.empty')}</div>
                </div>
            `;
            return;
        }

        // Separate active and pending tasks
        const activeTasks = [];
        const pendingTasks = [];

        for (const task of this.tasks.values()) {
            if (this.isQueued(task)) {
                pendingTasks.push(task);
            } else {
                activeTasks.push(task);
            }
        }

        let html = '';

        // Active tasks section
        if (activeTasks.length > 0) {
            html += `
                <div class="task-section">
                    <div class="task-section-header">
                        ${t('task_queue.active_tasks')}
                        <span class="task-section-count">${activeTasks.length}</span>
                    </div>
                    ${activeTasks.map(task => this.renderTaskItem(task, true)).join('')}
                </div>
            `;
        }

        // Pending tasks section
        if (pendingTasks.length > 0) {
            html += `
                <div class="task-section">
                    <div class="task-section-header">
                        ${t('task_queue.pending_tasks')}
                        <span class="task-section-count">${pendingTasks.length}</span>
                    </div>
                    ${pendingTasks.map(task => this.renderTaskItem(task, false)).join('')}
                </div>
            `;
        }

        this.elements.content.innerHTML = html;
    }

    /**
     * Render a single task item.
     */
    renderTaskItem(task, isActive) {
        const icon = this.getTaskIcon(task);
        const typeLabel = this.getTaskTypeLabel(task);
        const stopping = this.isStopping(task);
        const progressHtml = isActive && !stopping ? this.renderProgress(task) : this.renderStepOnly(task);
        // A task that is already stopping cannot be cancelled again
        const cancelHtml = stopping ? '' : `
                    <button class="task-item-cancel"
                            data-task-type="${task.type}"
                            data-task-id="${task.id}"
                            title="${t('task_queue.cancel')}">
                        ${t('task_queue.cancel')}
                    </button>`;

        return `
            <div class="task-item ${isActive ? 'active' : ''}">
                <div class="task-item-header">
                    <div class="task-item-info">
                        <span class="task-item-icon ${task.type} ${isActive ? '' : 'pending'}">${icon}</span>
                        <div class="task-item-details">
                            <a href="/video/${task.video_id}" class="task-item-link" onclick="window.TaskQueue.hideModal()" title="${escapeHtml(ThumbnailApp.formatVideoName(task.video_name))}">${escapeHtml(ThumbnailApp.formatVideoName(task.video_name))}</a>
                            <div class="task-item-type">${typeLabel}</div>
                        </div>
                    </div>${cancelHtml}
                </div>
                ${progressHtml}
            </div>
        `;
    }

    /**
     * Get icon for task type.
     */
    getTaskIcon(task) {
        if (this.isQueued(task)) {
            return '⏳';
        }
        return task.type === 'analysis' ? '🔍' : '🎨';
    }

    /**
     * Get localized label for task type.
     */
    getTaskTypeLabel(task) {
        return task.type === 'analysis'
            ? t('task_queue.analysis')
            : t('task_queue.generation');
    }

    /**
     * Render progress bar and text.
     */
    renderProgress(task) {
        const progress = task.progress || 0;

        let progressText = '';
        if (task.type === 'generation' && task.status === 'generating' && task.total_thumbnails) {
            progressText = t('task_queue.thumbnails_progress', {
                done: task.thumbnails_generated || 0,
                total: task.total_thumbnails
            });
        } else {
            progressText = this.stepLabel(task);
        }

        return `
            <div class="task-item-progress">
                <div class="task-progress-bar">
                    <div class="task-progress-fill ${task.type}" style="width: ${progress}%"></div>
                </div>
                <div class="task-progress-text">
                    <span>${progressText}</span>
                    <span>${progress}%</span>
                </div>
            </div>
        `;
    }

    /**
     * Render what a task without a progress bar is doing: queued, or stopping.
     */
    renderStepOnly(task) {
        return `
            <div class="task-item-progress">
                <div class="task-progress-text">
                    <span>${this.stepLabel(task)}</span>
                </div>
            </div>
        `;
    }

    /**
     * Cleanup when page unloads.
     */
    destroy() {
        // M34: Clear reconnect timer
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }
}

// Create global instance
window.TaskQueue = new TaskQueueManager();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.TaskQueue.init();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    window.TaskQueue.destroy();
});
