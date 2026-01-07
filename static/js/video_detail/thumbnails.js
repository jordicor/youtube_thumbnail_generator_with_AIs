/**
 * Video Detail Page - Thumbnails Module
 * ======================================
 * Handles existing thumbnails gallery, lazy loading, and deletion.
 */

import { CONFIG } from './constants.js';
import { state } from './state.js';

// =========================================================================
// THUMBNAIL LOADING
// =========================================================================

/**
 * Load existing thumbnails from the API.
 */
export async function loadExisting() {
    const grid = document.getElementById('existingThumbnailsGrid');
    const container = document.getElementById('existingThumbnailsContainer');
    const noMessage = document.getElementById('noThumbnailsMessage');
    const badge = document.getElementById('thumbnailCountBadge');
    const downloadBtn = document.getElementById('downloadAllBtn');

    try {
        const response = await fetch(`/api/thumbnails/video/${state.videoId}`);
        const data = await response.json();

        state.existingThumbnails = data.thumbnails || [];
        const count = state.existingThumbnails.length;

        badge.textContent = count;
        badge.classList.toggle('has-items', count > 0);

        if (count === 0) {
            container.style.display = 'none';
            noMessage.style.display = 'block';
            downloadBtn.style.display = 'none';
            return;
        }

        container.style.display = 'block';
        noMessage.style.display = 'none';
        downloadBtn.style.display = 'inline-flex';

        grid.innerHTML = '';
        state.existingThumbnails.forEach((thumb, index) => {
            const card = createCard(thumb, index);
            grid.appendChild(card);
        });

        observeImages();
        setupScrollIndicator();

    } catch (error) {
        grid.innerHTML = `<div class="error">${t('errors.load_thumbnails')}: ${escapeHtml(error.message)}</div>`;
    }
}

/**
 * Setup scroll indicator for the thumbnails container.
 * Shows a fade effect when there's more content to scroll.
 */
function setupScrollIndicator() {
    const wrapper = document.getElementById('existingThumbnailsWrapper');
    const container = document.getElementById('existingThumbnailsContainer');

    if (!wrapper || !container) return;

    const updateScrollState = () => {
        const hasScroll = container.scrollHeight > container.clientHeight;
        const scrolledToBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 10;

        wrapper.classList.toggle('has-scroll', hasScroll && !scrolledToBottom);
        wrapper.classList.toggle('scrolled-bottom', scrolledToBottom);
    };

    // Initial check (with slight delay to ensure rendering)
    setTimeout(updateScrollState, 100);

    // Update on scroll
    container.addEventListener('scroll', updateScrollState);

    // Update on resize
    window.addEventListener('resize', updateScrollState);
}

/**
 * Create a thumbnail card element.
 * @param {Object} thumb - Thumbnail data
 * @param {number} index - Thumbnail index
 * @returns {HTMLElement} Card element
 */
export function createCard(thumb, index) {
    const card = document.createElement('div');
    card.className = 'thumbnail-card';
    card.dataset.thumbnailId = thumb.id;

    let dateStr = '';
    if (thumb.created_at) {
        const date = new Date(thumb.created_at);
        const locale = getLanguage() === 'es' ? 'es-ES' : 'en-US';
        dateStr = date.toLocaleDateString(locale, {
            day: '2-digit',
            month: 'short',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    card.innerHTML = `
        <div class="thumbnail-image-wrapper">
            <img class="lazy-image"
                 data-src="/api/thumbnails/${thumb.id}"
                 src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1280 720'%3E%3Crect fill='%232a2a2a' width='1280' height='720'/%3E%3C/svg%3E"
                 alt="Thumbnail ${index + 1}"
                 loading="lazy">
            <div class="thumbnail-loading-overlay">
                <div class="loading-spinner-small"></div>
            </div>
        </div>
        <div class="thumbnail-info">
            <p class="thumbnail-title" title="${escapeHtml(thumb.suggested_title || t('results.no_thumbnails_yet'))}">${escapeHtml(thumb.suggested_title || t('results.no_thumbnails_yet'))}</p>
            ${dateStr ? `<small class="thumbnail-date">${dateStr}</small>` : ''}
            <div class="thumbnail-actions">
                <a href="/api/thumbnails/${thumb.id}" download class="btn btn-small btn-primary" title="${t('results.thumbnails.download')}">&#8595;</a>
                <button class="btn btn-small btn-danger" onclick="VideoDetail.deleteThumbnail(${thumb.id})" title="${t('common.delete')}">&#128465;</button>
            </div>
        </div>
    `;

    return card;
}

// =========================================================================
// LAZY LOADING
// =========================================================================

/**
 * Setup lazy loading observer for thumbnail images.
 */
export function setupLazyLoading() {
    if ('IntersectionObserver' in window) {
        state.lazyLoadObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    if (img.dataset.src) {
                        img.src = img.dataset.src;
                        img.onload = () => {
                            img.classList.add('loaded');
                            const overlay = img.parentElement.querySelector('.thumbnail-loading-overlay');
                            if (overlay) overlay.style.display = 'none';
                        };
                        img.onerror = () => {
                            img.classList.add('error');
                            const overlay = img.parentElement.querySelector('.thumbnail-loading-overlay');
                            if (overlay) overlay.style.display = 'none';
                        };
                        state.lazyLoadObserver.unobserve(img);
                    }
                }
            });
        }, {
            root: document.getElementById('existingThumbnailsContainer'),
            rootMargin: CONFIG.LAZY_LOAD_ROOT_MARGIN,
            threshold: CONFIG.LAZY_LOAD_THRESHOLD
        });
    }
}

/**
 * Observe unloaded lazy images.
 */
export function observeImages() {
    if (!state.lazyLoadObserver) return;

    document.querySelectorAll('.lazy-image:not(.loaded)').forEach(img => {
        state.lazyLoadObserver.observe(img);
    });
}

// =========================================================================
// THUMBNAIL DELETION
// =========================================================================

/**
 * Delete a thumbnail.
 * @param {number} thumbnailId - Thumbnail database ID
 */
export async function deleteOne(thumbnailId) {
    if (!confirm(t('common.delete'))) return;

    try {
        const response = await fetch(`/api/thumbnails/${thumbnailId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error(t('errors.delete_failed'));
        }

        const card = document.querySelector(`.thumbnail-card[data-thumbnail-id="${thumbnailId}"]`);
        if (card) {
            card.classList.add('removing');
            setTimeout(() => {
                card.remove();
                updateCount();
            }, CONFIG.ANIMATION_DURATION_MS);
        }

        ThumbnailApp.showToast(t('common.success'), 'success');
    } catch (error) {
        ThumbnailApp.showToast(t('errors.generic') + ': ' + error.message, 'error');
    }
}

/**
 * Update thumbnail count display.
 */
export function updateCount() {
    const grid = document.getElementById('existingThumbnailsGrid');
    const count = grid.querySelectorAll('.thumbnail-card').length;
    const badge = document.getElementById('thumbnailCountBadge');
    const container = document.getElementById('existingThumbnailsContainer');
    const noMessage = document.getElementById('noThumbnailsMessage');
    const downloadBtn = document.getElementById('downloadAllBtn');

    badge.textContent = count;
    badge.classList.toggle('has-items', count > 0);

    if (count === 0) {
        container.style.display = 'none';
        noMessage.style.display = 'block';
        downloadBtn.style.display = 'none';
    }
}

/**
 * Download all thumbnails as a zip.
 */
export function downloadAll() {
    window.location.href = `/api/thumbnails/video/${state.videoId}/download-all`;
}

// =========================================================================
// GALLERY UPDATES
// =========================================================================

/**
 * Add a newly generated thumbnail to the gallery.
 * @param {Object} thumb - Thumbnail data from SSE event
 */
export function addNewToGallery(thumb) {
    const grid = document.getElementById('existingThumbnailsGrid');
    const container = document.getElementById('existingThumbnailsContainer');
    const noMessage = document.getElementById('noThumbnailsMessage');
    const badge = document.getElementById('thumbnailCountBadge');
    const downloadBtn = document.getElementById('downloadAllBtn');

    container.style.display = 'block';
    noMessage.style.display = 'none';
    downloadBtn.style.display = 'inline-flex';

    const thumbData = {
        id: thumb.thumbnail_id,
        suggested_title: thumb.suggested_title,
        text_overlay: thumb.text_overlay,
        created_at: new Date().toISOString()
    };

    const card = createCard(thumbData, 0);
    card.classList.add('new-thumbnail');

    if (grid.firstChild) {
        grid.insertBefore(card, grid.firstChild);
    } else {
        grid.appendChild(card);
    }

    const count = grid.querySelectorAll('.thumbnail-card').length;
    badge.textContent = count;
    badge.classList.add('has-items');

    // Load image immediately for new thumbnails
    const img = card.querySelector('.lazy-image');
    if (img && img.dataset.src) {
        img.src = img.dataset.src;
        img.onload = () => {
            img.classList.add('loaded');
            const overlay = img.parentElement.querySelector('.thumbnail-loading-overlay');
            if (overlay) overlay.style.display = 'none';
        };
    }

    setTimeout(() => card.classList.remove('new-thumbnail'), 600);
}
