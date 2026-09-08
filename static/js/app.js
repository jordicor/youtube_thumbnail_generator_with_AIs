/**
 * YouTube Thumbnail Generator - Frontend JavaScript
 * =================================================
 * Common utilities and shared functionality.
 */

// API base URL
const API_BASE = '/api';

/**
 * Make an API request
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;

    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers,
        },
    };

    try {
        const response = await fetch(url, mergedOptions);

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `HTTP error ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`API request failed: ${endpoint}`, error);
        throw error;
    }
}

/**
 * Format duration in seconds to MM:SS
 */
function formatDuration(seconds) {
    if (!seconds) return t('common.na');
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${String(secs).padStart(2, '0')}`;
}

/**
 * Format date string
 */
function formatDate(dateString) {
    if (!dateString) return t('common.na');
    const date = new Date(dateString);
    const locale = getLanguage() === 'es' ? 'es-ES' : 'en-US';
    return date.toLocaleDateString(locale, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * Get localized status label
 */
function getStatusLabel(status) {
    // Valid statuses that have translations
    const validStatuses = ['pending', 'analyzing', 'analyzed', 'generating', 'completed', 'cancelled', 'error'];
    if (validStatuses.includes(status)) {
        return t(`status.${status}`);
    }
    return status;
}

// M35: Track active toasts for vertical stacking
let _activeToasts = [];

/**
 * Show a toast notification
 */
function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    // M35: Calculate vertical offset based on active toasts
    const bottomOffset = 20 + (_activeToasts.length * 60);

    // Add styles
    toast.style.cssText = `
        position: fixed;
        bottom: ${bottomOffset}px;
        right: 20px;
        padding: 15px 25px;
        background-color: ${type === 'error' ? '#ff4444' : type === 'success' ? '#44ff44' : '#333'};
        color: ${type === 'success' ? '#000' : '#fff'};
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;

    document.body.appendChild(toast);
    _activeToasts.push(toast);

    // Remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            const idx = _activeToasts.indexOf(toast);
            if (idx !== -1) _activeToasts.splice(idx, 1);
            toast.remove();
            // Reposition remaining toasts
            _activeToasts.forEach((t, i) => {
                t.style.bottom = `${20 + (i * 60)}px`;
            });
        }, 300);
    }, 3000);
}

// Close the active modal through the same cleanup path as user cancellation.
let _currentModalCleanup = null;

/**
 * Modal dialog system
 * @param {Object} options - Modal options
 * @param {string} options.title - Modal title
 * @param {string} options.message - Modal message (text only, use htmlMessage for HTML)
 * @param {string} [options.htmlMessage] - Modal message as HTML (only use when HTML is explicitly needed)
 * @param {string} [options.type='info'] - Modal type: 'info', 'warning', 'danger'
 * @param {string} [options.confirmText] - Confirm button text (default: localized 'Accept')
 * @param {string} [options.cancelText] - Cancel button text (default: localized 'Cancel')
 * @param {boolean} [options.showCancel=true] - Whether to show cancel button
 * @returns {Promise<boolean>} - Resolves to true if confirmed, false if cancelled
 */
function showModal(options) {
    if (_currentModalCleanup) {
        _currentModalCleanup(false);
    }

    return new Promise((resolve) => {
        const {
            title = t('modal.confirm_title'),
            message = '',
            htmlMessage = null,
            type = 'info',
            confirmText = t('common.accept'),
            cancelText = t('common.cancel'),
            showCancel = true
        } = options;

        // Get or create modal overlay
        let overlay = document.getElementById('modalOverlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'modalOverlay';
            overlay.className = 'modal-overlay';
            overlay.innerHTML = `
                <div class="modal-dialog">
                    <div class="modal-header">
                        <span class="modal-icon"></span>
                        <h3 class="modal-title"></h3>
                    </div>
                    <div class="modal-body"></div>
                    <div class="modal-footer">
                        <button class="btn btn-cancel"></button>
                        <button class="btn btn-confirm"></button>
                    </div>
                </div>
            `;
            document.body.appendChild(overlay);
        }

        const dialog = overlay.querySelector('.modal-dialog');
        const iconEl = overlay.querySelector('.modal-icon');
        const titleEl = overlay.querySelector('.modal-title');
        const bodyEl = overlay.querySelector('.modal-body');
        const cancelBtn = overlay.querySelector('.btn-cancel');
        const confirmBtn = overlay.querySelector('.btn-confirm');

        // C1: Set content - use textContent by default (XSS safe), innerHTML only when explicitly requested
        titleEl.textContent = title;
        if (htmlMessage !== null) {
            bodyEl.innerHTML = htmlMessage;
        } else {
            bodyEl.textContent = message;
        }
        confirmBtn.textContent = confirmText;
        cancelBtn.textContent = cancelText;

        // Set type and icon
        dialog.className = 'modal-dialog';
        const icons = {
            info: 'ℹ️',
            warning: '⚠️',
            danger: '⚠️'
        };
        iconEl.textContent = icons[type] || icons.info;
        if (type !== 'info') {
            dialog.classList.add(`modal-${type}`);
        }

        // Show/hide cancel button
        cancelBtn.style.display = showCancel ? '' : 'none';

        // Show modal
        overlay.classList.add('visible');

        let settled = false;
        const cleanup = (result) => {
            // Events already queued for a dismissed modal must not affect its replacement.
            if (settled) return;
            settled = true;
            overlay.classList.remove('visible');
            document.removeEventListener('keydown', handleKeydown);
            confirmBtn.removeEventListener('click', handleConfirm);
            cancelBtn.removeEventListener('click', handleCancel);
            overlay.removeEventListener('click', handleOverlayClick);
            if (_currentModalCleanup === cleanup) {
                _currentModalCleanup = null;
            }
            resolve(result);
        };

        // Event handlers
        const handleConfirm = () => cleanup(true);
        const handleCancel = () => cleanup(false);
        const handleOverlayClick = (e) => {
            if (e.target === overlay) {
                cleanup(false);
            }
        };
        const handleKeydown = (e) => {
            if (e.key === 'Escape') {
                e.preventDefault();
                cleanup(false);
            } else if (e.key === 'Enter') {
                e.preventDefault();
                cleanup(true);
            }
        };

        _currentModalCleanup = cleanup;
        confirmBtn.addEventListener('click', handleConfirm);
        cancelBtn.addEventListener('click', handleCancel);
        overlay.addEventListener('click', handleOverlayClick);
        document.addEventListener('keydown', handleKeydown);
        confirmBtn.focus();
    });
}

/**
 * Confirm dialog (legacy wrapper using modal)
 */
function confirmAction(message) {
    // For synchronous code compatibility, we still use native confirm
    // Use showModal() for async/await code
    return confirm(message);
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Poll for status updates (legacy fallback)
 */
function pollStatus(endpoint, callback, interval = 3000, maxAttempts = 100) {
    let attempts = 0;

    const poll = async () => {
        if (attempts >= maxAttempts) {
            console.warn('Max polling attempts reached');
            return;
        }

        attempts++;

        try {
            const data = await apiRequest(endpoint);
            const shouldContinue = callback(data);

            if (shouldContinue !== false) {
                setTimeout(poll, interval);
            }
        } catch (error) {
            console.error('Polling error:', error);
            // Continue polling on error
            setTimeout(poll, interval);
        }
    };

    poll();
}

/**
 * SSE (Server-Sent Events) helper for real-time updates
 *
 * @param {string} endpoint - SSE endpoint (e.g., '/api/events/analysis/1')
 * @param {Object} handlers - Event handlers { progress, complete, error, thumbnail }
 * @returns {EventSource} - The EventSource object (call .close() to stop)
 */
function createSSEConnection(endpoint, handlers = {}) {
    const eventSource = new EventSource(endpoint);

    // Connection opened
    eventSource.onopen = () => {
        console.log('SSE connected:', endpoint);
    };

    // General error handling
    eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        if (handlers.error) {
            handlers.error({ error: 'Connection error' });
        }
        // Close on persistent errors
        if (eventSource.readyState === EventSource.CLOSED) {
            console.log('SSE connection closed');
        }
    };

    // Progress event
    if (handlers.progress) {
        eventSource.addEventListener('progress', (event) => {
            try {
                const data = JSON.parse(event.data);
                handlers.progress(data);
            } catch (e) {
                console.error('Error parsing progress event:', e);
            }
        });
    }

    // Complete event
    if (handlers.complete) {
        eventSource.addEventListener('complete', (event) => {
            try {
                const data = JSON.parse(event.data);
                handlers.complete(data);
                eventSource.close();
            } catch (e) {
                console.error('Error parsing complete event:', e);
            }
        });
    }

    // Error event (from server)
    if (handlers.error) {
        eventSource.addEventListener('error', (event) => {
            try {
                const data = JSON.parse(event.data);
                handlers.error(data);
                eventSource.close();
            } catch (e) {
                // Ignore parse errors for connection errors
            }
        });
    }

    // Thumbnail event (for generation)
    if (handlers.thumbnail) {
        eventSource.addEventListener('thumbnail', (event) => {
            try {
                const data = JSON.parse(event.data);
                handlers.thumbnail(data);
            } catch (e) {
                console.error('Error parsing thumbnail event:', e);
            }
        });
    }

    // Cancelled event
    if (handlers.cancelled) {
        eventSource.addEventListener('cancelled', (event) => {
            try {
                const data = JSON.parse(event.data);
                handlers.cancelled(data);
                eventSource.close();
            } catch (e) {
                console.error('Error parsing cancelled event:', e);
            }
        });
    }

    // Update event (for video list)
    if (handlers.update) {
        eventSource.addEventListener('update', (event) => {
            try {
                const data = JSON.parse(event.data);
                handlers.update(data);
            } catch (e) {
                console.error('Error parsing update event:', e);
            }
        });
    }

    return eventSource;
}

/**
 * Check if SSE is supported by the browser
 */
function isSSESupported() {
    return typeof EventSource !== 'undefined';
}

// Add animation keyframes
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Export for use in other scripts
window.ThumbnailApp = {
    apiRequest,
    formatDuration,
    formatDate,
    getStatusLabel,
    showToast,
    showModal,
    confirmAction,
    debounce,
    pollStatus,
    createSSEConnection,
    isSSESupported
};
