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

// How each video status is presented: label key, colour tone and what the video
// is ready for ('pending': needs analysis, 'processing': being analyzed,
// 'ready': analyzed, can generate thumbnails)
const VIDEO_STATUS_VIEW = {
    pending:          { label: 'video_status.pending',    tone: 'neutral', kind: 'pending' },
    cancelled:        { label: 'video_status.cancelled',  tone: 'neutral', kind: 'pending' },
    error:            { label: 'video_status.error',      tone: 'danger',  kind: 'pending' },
    analyzing:        { label: 'video_status.analyzing',  tone: 'info',    kind: 'processing' },
    analyzing_scenes: { label: 'video_status.analyzing',  tone: 'info',    kind: 'processing' },
    analyzing_faces:  { label: 'video_status.analyzing',  tone: 'info',    kind: 'processing' },
    clustering:       { label: 'video_status.analyzing',  tone: 'info',    kind: 'processing' },
    transcribing:     { label: 'video_status.analyzing',  tone: 'info',    kind: 'processing' },
    analyzed:         { label: 'video_status.analyzed',   tone: 'success', kind: 'ready' },
    generating:       { label: 'video_status.generating', tone: 'info',    kind: 'ready' },
    completed:        { label: 'video_status.completed',  tone: 'primary', kind: 'ready' }
};

/**
 * Get how a video status is presented.
 * @param {string} status - Video status
 * @returns {{label: string, tone: string, kind: string}} Presentation
 */
function getVideoStatusView(status) {
    const view = VIDEO_STATUS_VIEW[status];
    if (!view) {
        throw new Error(`Unknown video status: ${status}`);
    }
    return view;
}

// Extensions of the video files the library scans (config.py VIDEO_EXTENSIONS)
const VIDEO_EXTENSION_PATTERN = /\.(?:mp4|mkv|avi|mov|webm|m4v|wmv|flv|mpg|mpeg)$/i;

/**
 * Format a video filename for display: without its extension and with the
 * percent-escapes left in downloaded names decoded (e.g. "%28" -> "(").
 * A lone "%", or escapes that are not valid UTF-8 (e.g. "50%CA"), are kept as is.
 */
function formatVideoName(filename) {
    return filename
        .replace(VIDEO_EXTENSION_PATTERN, '')
        .replace(/(?:%[0-9A-Fa-f]{2})+/g, (escapes) => {
            try {
                return decodeURIComponent(escapes);
            } catch {
                return escapes;
            }
        });
}

/**
 * Get the display name of a cluster as plain text (not HTML-escaped).
 * person_scene clusters are named after the person cluster that holds them.
 */
function getClusterDisplayName(cluster) {
    if (cluster.label) {
        return cluster.label;
    }

    if (cluster.view_mode === 'person_scene' && cluster.parent_cluster_index !== null) {
        const person = cluster.parent_label
            || t('clusters.person_label', {index: cluster.parent_cluster_index + 1});
        if (cluster.scene_index === null) {
            return t('video_detail.clusters.person_no_scene', {person});
        }
        return t('video_detail.clusters.person_scene', {person, scene: cluster.scene_index});
    }

    return t('clusters.person_label', {index: cluster.cluster_index + 1});
}

/**
 * URL of the face crop of a person (cluster). It changes with the person's
 * representative frame, so browsers can cache it.
 * @param {number} videoId - Video ID
 * @param {Object} cluster - Cluster from the clusters API
 * @returns {string} Image URL
 */
function personFaceUrl(videoId, cluster) {
    return `/api/people/${videoId}/${cluster.id}/face?v=${cluster.representative_frame_id}`;
}

/**
 * Get the markup of an icon from the SVG sprite (templates/components/icons.html).
 * @param {string} name - Icon name without the "i-" prefix
 * @param {string} [className] - Extra classes (e.g. 'icon-sm')
 * @returns {string} SVG markup
 */
function icon(name, className = '') {
    return `<svg class="icon${className ? ' ' + className : ''}" aria-hidden="true"><use href="#i-${name}"></use></svg>`;
}

/**
 * Show a toast notification.
 * @param {string} message - Text to show
 * @param {string} [type='info'] - 'info', 'success', 'warning' or 'error'
 * @param {{label: string, onClick: Function}} [action] - Optional button, e.g. "Undo"
 */
function showToast(message, type = 'info', action = null) {
    let stack = document.getElementById('toastStack');
    if (!stack) {
        stack = document.createElement('div');
        stack.id = 'toastStack';
        stack.className = 'toast-stack';
        document.body.appendChild(stack);
    }

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.setAttribute('role', type === 'error' ? 'alert' : 'status');

    const text = document.createElement('span');
    text.textContent = message;
    toast.appendChild(text);

    let dismissed = false;
    const dismiss = () => {
        if (dismissed) return;
        dismissed = true;
        toast.classList.add('leaving');
        setTimeout(() => toast.remove(), 200);
    };

    if (action) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'toast-action';
        button.textContent = action.label;
        button.addEventListener('click', () => {
            dismiss();
            action.onClick();
        });
        toast.appendChild(button);
    }

    stack.appendChild(toast);
    // Toasts with an action stay longer so there is time to use it
    setTimeout(dismiss, action ? 6000 : 3500);
}

// Currently open popover menu: {anchor, cleanup}
let _openMenu = null;

/**
 * Open a popover menu below an anchor element. Calling it again with the same
 * anchor closes the menu (toggle).
 * @param {HTMLElement} anchor - Element that opens the menu (usually a "more" button)
 * @param {Array<{label: string, icon?: string, danger?: boolean, onSelect: Function}>} items
 */
function showMenu(anchor, items) {
    const wasOpenHere = _openMenu?.anchor === anchor;
    closeMenu();
    if (wasOpenHere) return;

    const menu = document.createElement('div');
    menu.className = 'menu-popover';
    menu.setAttribute('role', 'menu');

    for (const item of items) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `menu-item${item.danger ? ' danger' : ''}`;
        button.setAttribute('role', 'menuitem');
        button.innerHTML = `${item.icon ? icon(item.icon) : ''}<span></span>`;
        button.querySelector('span').textContent = item.label;
        button.addEventListener('click', (e) => {
            e.stopPropagation();
            closeMenu();
            item.onSelect();
        });
        menu.appendChild(button);
    }

    document.body.appendChild(menu);

    // Right-aligned below the anchor, or above it when there is no room
    const anchorRect = anchor.getBoundingClientRect();
    const menuRect = menu.getBoundingClientRect();
    const below = anchorRect.bottom + 6;
    const top = below + menuRect.height > window.innerHeight ? anchorRect.top - 6 - menuRect.height : below;
    menu.style.top = `${Math.max(8, top)}px`;
    menu.style.left = `${Math.max(8, anchorRect.right - menuRect.width)}px`;

    const onPointer = (e) => {
        if (!menu.contains(e.target) && !anchor.contains(e.target)) closeMenu();
    };
    const onKey = (e) => {
        if (e.key === 'Escape') {
            closeMenu();
            anchor.focus();
        }
    };
    document.addEventListener('click', onPointer, true);
    document.addEventListener('keydown', onKey);
    window.addEventListener('scroll', closeMenu, true);
    window.addEventListener('resize', closeMenu);
    anchor.setAttribute('aria-expanded', 'true');

    _openMenu = {
        anchor,
        cleanup: () => {
            document.removeEventListener('click', onPointer, true);
            document.removeEventListener('keydown', onKey);
            window.removeEventListener('scroll', closeMenu, true);
            window.removeEventListener('resize', closeMenu);
            anchor.setAttribute('aria-expanded', 'false');
            menu.remove();
        }
    };

    menu.querySelector('.menu-item')?.focus();
}

/**
 * Close the open popover menu, if any.
 */
function closeMenu() {
    if (!_openMenu) return;
    const { cleanup } = _openMenu;
    _openMenu = null;
    cleanup();
}

/**
 * Show a video's display name in the page breadcrumbs ([data-breadcrumb-video]).
 * @param {string} filename - Video filename
 */
function setVideoBreadcrumb(filename) {
    const name = formatVideoName(filename);
    document.querySelectorAll('[data-breadcrumb-video]').forEach(el => {
        el.textContent = name;
        el.title = name;
    });
}

/**
 * Load a video and show its name in the page breadcrumbs.
 * @param {number} videoId - Video ID
 */
async function loadVideoBreadcrumb(videoId) {
    const response = await fetch(`/api/videos/${videoId}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const video = await response.json();
    setVideoBreadcrumb(video.filename);
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
            } else if (e.key === 'Enter' && e.target.tagName !== 'TEXTAREA') {
                // In a textarea Enter starts a new line
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

// Steppers: the − and + buttons around a number input, kept within its min and max
document.addEventListener('click', (e) => {
    const button = e.target.closest('.stepper button[data-step]');
    if (!button) return;
    const input = button.closest('.stepper').querySelector('input');
    const value = (parseInt(input.value, 10) || 0) + parseInt(button.dataset.step, 10);
    input.value = Math.min(Math.max(value, parseInt(input.min, 10)), parseInt(input.max, 10));
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
});

// Export for use in other scripts
window.ThumbnailApp = {
    apiRequest,
    formatDuration,
    formatDate,
    formatVideoName,
    getClusterDisplayName,
    personFaceUrl,
    getVideoStatusView,
    icon,
    showToast,
    showModal,
    showMenu,
    closeMenu,
    setVideoBreadcrumb,
    loadVideoBreadcrumb,
    debounce,
    pollStatus,
    createSSEConnection,
    isSSESupported
};
