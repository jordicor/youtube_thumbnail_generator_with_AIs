/**
 * Video Detail Page - Thumbnails Module
 * ======================================
 * Handles existing thumbnails gallery, lazy loading, and deletion.
 */

import { CONFIG, STORAGE_KEYS } from './constants.js';
import { state } from './state.js';

// Thumbnails shown as "recent" in the Generate tab while no generation runs
const RECENT_RESULTS = 6;

// Gallery orders (GET /api/thumbnails/video/{id}?sort=...)
const SORTS = ['recent', 'failures_last', 'match'];

// Page request in flight, shared so the same page is never fetched twice
let pageLoad = null;

// Latest gallery load: an older one that answers later must not replace it
let galleryLoad = 0;

// =========================================================================
// THUMBNAIL LOADING
// =========================================================================

/**
 * Fetch one page of the video's thumbnails.
 * @param {number} skip - Number of thumbnails to skip
 * @param {string} sort - Order of the pages (one of SORTS)
 * @returns {Promise<{thumbnails: Object[], total: number}>} Page and total count
 */
async function fetchPage(skip, sort = state.thumbnailsSort) {
    const response = await fetch(
        `/api/thumbnails/video/${state.videoId}?limit=${CONFIG.THUMBNAILS_PAGE_SIZE}&skip=${skip}&sort=${sort}`
    );
    // H7: Check response before parsing
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
}

/**
 * Load the first page of existing thumbnails from the API, and the newest
 * page for the recent results when the gallery shows another order.
 */
export async function loadExisting() {
    const grid = document.getElementById('existingThumbnailsGrid');
    const load = ++galleryLoad;

    try {
        const [data, newest] = await Promise.all([
            fetchPage(0),
            state.thumbnailsSort === 'recent' ? null : fetchPage(0, 'recent'),
        ]);
        if (load !== galleryLoad) return;
        state.thumbnails = data.thumbnails;
        state.newestThumbnails = [...(newest ?? data).thumbnails];
        state.thumbnailsTotal = data.total;

        grid.replaceChildren(...data.thumbnails.map((thumb, index) => createCard(thumb, index)));
        renderRecent();

        updateCount();
        observeImages();
        updateResultsPanel();

    } catch (error) {
        if (load !== galleryLoad) return;
        grid.innerHTML = `<div class="error">${t('errors.load_thumbnails')}: ${escapeHtml(error.message)}</div>`;
    }
}

/**
 * Bind the gallery toolbar and its "load more" button.
 */
export function setupControls() {
    document.getElementById('thumbnailsSort').addEventListener('change', changeSort);
    document.getElementById('downloadAllBtn').addEventListener('click', downloadAll);
    document.getElementById('thumbnailsLoadMoreBtn').addEventListener('click', loadMore);
}

/**
 * Show the gallery in the order last chosen on this browser.
 * Call it before the first loadExisting().
 */
export function initSort() {
    let stored = null;
    try {
        stored = localStorage.getItem(STORAGE_KEYS.THUMBNAILS_SORT);
    } catch {
        // Storage may be unavailable (private mode); the gallery starts newest first
    }
    if (SORTS.includes(stored)) state.thumbnailsSort = stored;
    document.getElementById('thumbnailsSort').value = state.thumbnailsSort;
}

/**
 * Reload the gallery in the order picked in its selector.
 */
export function changeSort() {
    state.thumbnailsSort = document.getElementById('thumbnailsSort').value;
    try {
        localStorage.setItem(STORAGE_KEYS.THUMBNAILS_SORT, state.thumbnailsSort);
    } catch {
        // Storage may be unavailable (private mode); the order just is not remembered
    }
    loadExisting();
}

/**
 * Append the next page of thumbnails to the gallery. Calls made while a page
 * is loading share that request.
 * @returns {Promise<void>} Resolves when the page is shown
 */
function appendNextPage() {
    pageLoad ??= (async () => {
        // New thumbnails are added only where the order puts them first (see
        // addNewToGallery) and deleted ones removed, so the loaded ones are
        // always the first ones of the order: skip exactly that many.
        const loaded = state.thumbnails;
        const shown = loaded.length;
        const data = await fetchPage(shown);
        // The gallery was reloaded meanwhile: this page no longer follows it
        if (state.thumbnails !== loaded) return;

        state.thumbnailsTotal = data.total;
        state.thumbnails.push(...data.thumbnails);

        const grid = document.getElementById('existingThumbnailsGrid');
        data.thumbnails.forEach((thumb, index) => {
            grid.appendChild(createCard(thumb, shown + index));
        });

        updateCount();
        observeImages();
    })().finally(() => {
        pageLoad = null;
    });
    return pageLoad;
}

/**
 * Load pages until the first `count` thumbnails (or all, if there are fewer) are loaded.
 * @param {number} count - Number of thumbnails needed, first ones of the gallery order
 */
export async function ensureLoaded(count) {
    while (state.thumbnails.length < Math.min(count, state.thumbnailsTotal)) {
        const loaded = state.thumbnails;
        const before = loaded.length;
        await appendNextPage();
        // A reload meanwhile replaces the list; the loop then goes on from it
        if (state.thumbnails === loaded && loaded.length === before) {
            throw new Error(`No thumbnails after the first ${before} of ${state.thumbnailsTotal}`);
        }
    }
}

/**
 * Append the next page of thumbnails to the gallery ("load more" button).
 */
export async function loadMore() {
    const btn = document.getElementById('thumbnailsLoadMoreBtn');
    btn.disabled = true;

    try {
        await appendNextPage();
    } catch (error) {
        ThumbnailApp.showToast(`${t('errors.load_thumbnails')}: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
    }
}

/**
 * Show the newest thumbnails as the Generate tab's recent results.
 */
function renderRecent() {
    document.getElementById('recentThumbnailsGrid').replaceChildren(
        ...state.newestThumbnails.slice(0, RECENT_RESULTS).map(createResultTile)
    );
}

function pageLocale() {
    return getLanguage() === 'es' ? 'es-ES' : 'en-US';
}

// Cards and tiles show a thumbnail about 320 px wide: they load a JPEG copy
// twice that (for sharp screens) instead of its PNG of 1-3 MB. The viewer and
// the downloads use the file itself.
const CARD_IMAGE_WIDTH = 640;

/**
 * Address of a thumbnail's small copy.
 * @param {number} thumbnailId - Thumbnail ID
 * @returns {string} URL
 */
function cardImageUrl(thumbnailId) {
    return `/api/thumbnails/${thumbnailId}?width=${CARD_IMAGE_WIDTH}`;
}

/**
 * Format a thumbnail's creation date for display.
 * @param {string} createdAt - ISO date
 * @returns {string} Localized day, month and time
 */
export function formatDate(createdAt) {
    return ThumbnailApp.parseServerDate(createdAt).toLocaleDateString(pageLocale(), {
        day: '2-digit',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * Create a compact result tile (Generate tab results panel).
 * @param {Object} thumb - Thumbnail data with id, suggested_title and maybe created_at
 * @returns {HTMLElement} Tile element
 */
export function createResultTile(thumb) {
    const title = thumb.suggested_title || t('thumbnails.untitled');
    const tile = document.createElement('a');
    tile.className = 'result-tile';
    tile.dataset.thumbnailId = thumb.id;
    if (thumb.created_at) tile.dataset.createdAt = thumb.created_at;
    // Plain clicks open the viewer; the link keeps "open in new tab"
    tile.href = `/api/thumbnails/${thumb.id}`;
    tile.target = '_blank';
    tile.rel = 'noopener';
    tile.title = title;
    tile.innerHTML = `
        <img src="${cardImageUrl(thumb.id)}" alt="${escapeHtml(title)}" loading="lazy">
        <span class="result-tile-title">${escapeHtml(title)}</span>
    `;
    return tile;
}

/**
 * Show in the Generate tab either the running/last job's results, the most
 * recent thumbnails, or the empty hint.
 */
export function updateResultsPanel() {
    const live = document.getElementById('thumbnailsGrid');
    const recent = document.getElementById('recentThumbnailsGrid');
    const running = !document.getElementById('generationStatus').classList.contains('hidden');
    const showLive = running || live.children.length > 0;

    live.classList.toggle('hidden', !showLive);
    recent.classList.toggle('hidden', showLive || recent.children.length === 0);
    document.getElementById('resultsEmpty').classList.toggle('hidden', showLive || recent.children.length > 0);
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

    const dateStr = thumb.created_at ? formatDate(thumb.created_at) : '';

    const hasSwap = thumb.face_swap_applied === 1 || thumb.face_swap_applied === true;
    const swapBadge = hasSwap
        ? `<span class="face-swap-badge" title="${escapeHtml(t('thumbnails.face_swap_applied'))}">${ThumbnailApp.icon('check', 'icon-sm')}${escapeHtml(t('thumbnails.face_swap_badge'))}</span>`
        : '';
    const action = (name, icon, label, extra = '') =>
        `<button type="button" class="thumbnail-action ${extra}" data-action="${name}" title="${escapeHtml(label)}" aria-label="${escapeHtml(label)}">${ThumbnailApp.icon(icon, 'icon-sm')}</button>`;
    const title = thumb.suggested_title || t('thumbnails.untitled');

    // Plain clicks on the image open the viewer; the link keeps "open in new tab"
    card.innerHTML = `
        <a class="thumbnail-image-wrapper" href="/api/thumbnails/${thumb.id}" target="_blank" rel="noopener">
            <img class="lazy-image"
                 data-src="${cardImageUrl(thumb.id)}"
                 src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1280 720'%3E%3Crect fill='%232a2a2a' width='1280' height='720'/%3E%3C/svg%3E"
                 alt="${escapeHtml(title)}"
                 loading="lazy">
            <div class="thumbnail-loading-overlay">
                <div class="loading-spinner-small"></div>
            </div>
            ${swapBadge}
        </a>
        <div class="thumbnail-actions">
            <a href="/api/thumbnails/${thumb.id}" download class="thumbnail-action"
               title="${escapeHtml(t('thumbnails.download'))}" aria-label="${escapeHtml(t('thumbnails.download'))}">${ThumbnailApp.icon('download', 'icon-sm')}</a>
            ${hasSwap ? action('before-after', 'eye', t('thumbnails.show_original'), 'face-swap-toggle') : ''}
            ${state.faceSwapMissing ? '' : action('fix-face', 'wand', t('thumbnails.fix_face'), 'face-swap-fix')}
            ${action('delete', 'trash', t('common.delete'), 'danger')}
        </div>
        <div class="thumbnail-info">
            <span class="thumbnail-eyebrow" title="${escapeHtml(t('thumbnails.suggested_title_hint'))}">${escapeHtml(t('thumbnails.suggested_title'))}</span>
            <p class="thumbnail-title" title="${escapeHtml(title)}">${escapeHtml(title)}</p>
            ${dateStr ? `<small class="thumbnail-date">${escapeHtml(dateStr)}</small>` : ''}
        </div>
    `;
    const swapToggle = card.querySelector('.face-swap-toggle');
    if (swapToggle) swapToggle.dataset.state = 'swapped';

    card.querySelector('.thumbnail-actions').addEventListener('click', (e) => {
        const button = e.target.closest('button[data-action]');
        if (!button) return;
        if (button.dataset.action === 'delete') deleteOne(thumb.id);
        else if (button.dataset.action === 'fix-face') fixFace(thumb.id);
        else if (button.dataset.action === 'before-after') toggleBeforeAfter(thumb.id);
    });

    return card;
}


/**
 * Toggle between the swapped and the original (pre-swap) version of a thumbnail.
 * Looks up the card and swaps the image src between /api/thumbnails/{id} and /api/thumbnails/{id}/raw.
 */
export async function toggleBeforeAfter(thumbId) {
    const card = document.querySelector(`.thumbnail-card[data-thumbnail-id="${thumbId}"]`);
    if (!card) return;
    const img = card.querySelector('.lazy-image');
    const btn = card.querySelector('.face-swap-toggle');
    if (!img || !btn) return;

    const current = btn.dataset.state;
    if (current === 'swapped') {
        // Switch to RAW
        img.src = `/api/thumbnails/${thumbId}/raw`;
        btn.dataset.state = 'raw';
        btn.title = t('thumbnails.show_swapped');
    } else {
        // Switch back to swapped (the canonical filepath)
        img.src = cardImageUrl(thumbId);
        btn.dataset.state = 'swapped';
        btn.title = t('thumbnails.show_original');
    }
    btn.setAttribute('aria-label', btn.title);
    btn.classList.toggle('active', btn.dataset.state === 'raw');
}


/**
 * Re-run face swap on a single thumbnail (idempotent: backend restores raw first).
 */
export async function fixFace(thumbId) {
    const card = document.querySelector(`.thumbnail-card[data-thumbnail-id="${thumbId}"]`);
    if (!card) return;
    const btn = card.querySelector('.face-swap-fix');
    if (btn) {
        btn.disabled = true;
        btn.classList.add('busy');
    }
    try {
        const restoration = document.getElementById('applyFaceRestoration');
        const restoreFaces = document.getElementById('applyFaceSwap').checked
            && !restoration.disabled && restoration.checked;
        const response = await fetch(`/api/thumbnails/${thumbId}/face_swap?restore_faces=${restoreFaces}`, { method: 'POST' });
        if (!response.ok) {
            const reason = await ThumbnailApp.readApiError(response);
            ThumbnailApp.showToast(t('thumbnails.fix_face_failed', {error: reason}), 'error');
            return;
        }
        const result = await response.json();
        if (result.face_swap_applied) {
            ThumbnailApp.showToast(t('thumbnails.face_swap_applied'), 'success');
        } else if (result.success) {
            ThumbnailApp.showToast(t('thumbnails.face_swap_skipped'), 'info');
        } else {
            ThumbnailApp.showToast(t('thumbnails.fix_face_failed', {error: result.error}), 'error');
        }
        // Force-reload the image, busting the browser cache
        const img = card.querySelector('.lazy-image');
        if (img) {
            img.src = `${cardImageUrl(thumbId)}&t=${Date.now()}`;
        }
    } catch (e) {
        ThumbnailApp.showToast(t('thumbnails.fix_face_failed', {error: e.message}), 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.classList.remove('busy');
        }
    }
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
            root: null,
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
 * Delete a thumbnail after confirming, and remove it from the gallery and the results panel.
 * @param {number} thumbnailId - Thumbnail database ID
 * @returns {Promise<boolean>} True if it was deleted
 */
export async function deleteOne(thumbnailId) {
    const confirmed = await ThumbnailApp.showModal({
        title: t('thumbnails.delete_title'),
        message: t('thumbnails.delete_message'),
        type: 'danger',
        confirmText: t('common.delete'),
        cancelText: t('common.cancel')
    });
    if (!confirmed) return false;

    try {
        const response = await fetch(`/api/thumbnails/${thumbnailId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error(t('errors.delete_failed'));
        }
    } catch (error) {
        ThumbnailApp.showToast(t('errors.generic') + ': ' + error.message, 'error');
        return false;
    }

    const index = state.thumbnails.findIndex(thumb => thumb.id === thumbnailId);
    if (index >= 0) state.thumbnails.splice(index, 1);
    const newestIndex = state.newestThumbnails.findIndex(thumb => thumb.id === thumbnailId);
    if (newestIndex >= 0) state.newestThumbnails.splice(newestIndex, 1);
    // Thumbnails of a running job are in neither list nor in the total yet
    if (index >= 0 || newestIndex >= 0) state.thumbnailsTotal -= 1;

    const card = document.querySelector(`.thumbnail-card[data-thumbnail-id="${thumbnailId}"]`);
    if (card) {
        card.classList.add('removing');
        setTimeout(() => card.remove(), CONFIG.ANIMATION_DURATION_MS);
    }
    document.querySelector(`#thumbnailsGrid .result-tile[data-thumbnail-id="${thumbnailId}"]`)?.remove();
    renderRecent();

    updateCount();
    updateResultsPanel();
    ThumbnailApp.showToast(t('thumbnails.deleted'), 'success');
    return true;
}

/**
 * Update the gallery counter, empty state and "load more" button.
 */
export function updateCount() {
    const total = state.thumbnailsTotal;
    const remaining = total - state.thumbnails.length;
    const badge = document.getElementById('thumbnailCountBadge');
    const loadMore = document.getElementById('thumbnailsLoadMore');

    badge.textContent = total;
    badge.classList.toggle('has-items', total > 0);
    document.getElementById('thumbnailsTabCount').textContent = total;

    const headerCount = document.getElementById('videoMetaThumbnails');
    headerCount.classList.toggle('hidden', total === 0);
    headerCount.querySelector('span').textContent = t('index.card.thumbnails', { count: total });
    document.getElementById('viewAllThumbnailsLabel').textContent =
        total > 0 ? t('generate_form.results_view_all_count', { count: total }) : t('generate_form.results_view_all');

    document.getElementById('existingThumbnailsContainer').style.display = total > 0 ? 'block' : 'none';
    document.getElementById('noThumbnailsMessage').style.display = total > 0 ? 'none' : 'block';
    document.getElementById('downloadAllBtn').style.display = total > 0 ? 'inline-flex' : 'none';

    loadMore.classList.toggle('hidden', remaining <= 0);
    if (remaining > 0) {
        document.getElementById('thumbnailsLoadMoreBtn').textContent =
            t('thumbnails.load_more', {count: remaining});
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

    const thumbData = {
        id: thumb.thumbnail_id,
        suggested_title: thumb.suggested_title,
        text_overlay: thumb.text_overlay,
        created_at: new Date().toISOString()
    };

    state.newestThumbnails.unshift(thumbData);
    state.thumbnailsTotal += 1;
    renderRecent();

    // The 'match' order puts it by its face match (last until scored): it comes
    // with its page, and the gallery reloads when the job ends
    if (state.thumbnailsSort === 'match') {
        updateCount();
        return;
    }

    const card = createCard(thumbData, 0);
    card.classList.add('new-thumbnail');

    if (grid.firstChild) {
        grid.insertBefore(card, grid.firstChild);
    } else {
        grid.appendChild(card);
    }

    state.thumbnails.unshift(thumbData);
    updateCount();

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
