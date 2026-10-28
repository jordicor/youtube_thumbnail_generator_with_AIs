/**
 * Video Detail Page - Thumbnail Viewer
 * =====================================
 * Shows generated thumbnails large, one at a time, with download and delete.
 * Opened from the Generate tab results it goes through those thumbnails only,
 * then an end slide offers the rest of the video's thumbnails. Opened from the
 * gallery it goes through all of them, loading pages as needed. Both wrap around.
 */

import { state } from './state.js';
import { deleteOne, ensureLoaded, formatDate } from './thumbnails.js';

// Slide after the last recent thumbnail that offers the rest
const END = 'end';

const viewer = {
    mode: null,      // 'recent' | 'all' while open, null when closed
    grid: null,      // Results grid the recent thumbnails come from ('recent' mode)
    current: null,   // Thumbnail ID shown, or END
    opener: null,    // Element that opened the viewer, focused again on close
    busy: false,     // A page load or a deletion is running
};

// =========================================================================
// SEQUENCE
// =========================================================================

function recentTiles() {
    return [...viewer.grid.querySelectorAll('.result-tile')];
}

/**
 * Count the video's thumbnails that are not among the recent ones.
 * @param {number[]} recentIds - IDs of the recent thumbnails
 * @returns {number} Thumbnails left to see
 */
function restCount(recentIds) {
    // The recent thumbnails are among the newest page, loaded whatever the
    // gallery order. Those of a running job are not in it nor in the total yet.
    const loaded = new Set(state.newestThumbnails.map(thumb => thumb.id));
    return state.thumbnailsTotal - recentIds.filter(id => loaded.has(id)).length;
}

/**
 * The 'recent' sequence: the results panel thumbnails, then END if there are more.
 * @returns {Array<number|string>} Thumbnail IDs and maybe END
 */
function recentSequence() {
    const ids = recentTiles().map(tile => Number(tile.dataset.thumbnailId));
    return restCount(ids) > 0 ? [...ids, END] : ids;
}

function sequenceLength() {
    return viewer.mode === 'all' ? state.thumbnailsTotal : recentSequence().length;
}

function positionOf(id) {
    return viewer.mode === 'all'
        ? state.thumbnails.findIndex(thumb => thumb.id === id)
        : recentSequence().indexOf(id);
}

/**
 * Get the item at a position of the sequence, loading gallery pages if needed.
 * @param {number} position - Index in the sequence
 * @returns {Promise<number|string>} Thumbnail ID or END
 */
async function itemAt(position) {
    if (viewer.mode === 'recent') return recentSequence()[position];

    await ensureLoaded(position + 1);
    const thumb = state.thumbnails[position];
    if (!thumb) throw new Error(`No thumbnail at position ${position + 1} of ${state.thumbnailsTotal}`);
    return thumb.id;
}

/**
 * Run a step that may load pages, ignoring input until it ends.
 * @param {Function} task - Async function
 */
async function run(task) {
    if (viewer.busy) return;
    viewer.busy = true;
    try {
        await task();
    } catch (error) {
        ThumbnailApp.showToast(`${t('errors.load_thumbnails')}: ${error.message}`, 'error');
    } finally {
        viewer.busy = false;
    }
}

/**
 * Show the item at a position (wrapping around), or close when nothing is left.
 * @param {number} position - Index in the sequence
 */
async function showAt(position) {
    const length = sequenceLength();
    if (length === 0) {
        close();
        return;
    }
    viewer.current = await itemAt(((position % length) + length) % length);
    render();
}

function step(delta) {
    run(async () => {
        const position = positionOf(viewer.current);
        // The thumbnail shown is gone (e.g. the gallery reloaded): start over
        await showAt(position < 0 ? 0 : position + delta);
    });
}

/**
 * Leave the recent thumbnails and go on through the video's other ones.
 */
function showRest() {
    run(async () => {
        const recent = recentTiles().map(tile => Number(tile.dataset.thumbnailId));
        // Whatever the gallery order, one more than the recent ones holds another one
        await ensureLoaded(recent.length + 1);
        const position = state.thumbnails.findIndex(thumb => !recent.includes(thumb.id));
        if (position < 0) throw new Error('Every loaded thumbnail is a recent one');

        viewer.mode = 'all';
        viewer.grid = null;
        await showAt(position);
    });
}

// =========================================================================
// OPEN / CLOSE / RENDER
// =========================================================================

/**
 * Open the viewer on a thumbnail.
 * @param {'recent'|'all'} mode - Sequence to go through
 * @param {number} thumbnailId - Thumbnail to show first
 * @param {HTMLElement} opener - Clicked element
 */
function open(mode, thumbnailId, opener) {
    viewer.mode = mode;
    viewer.grid = mode === 'recent' ? opener.closest('.results-grid') : null;
    viewer.current = thumbnailId;
    viewer.opener = opener;

    const dialog = document.getElementById('thumbLightbox');
    dialog.classList.remove('hidden');
    document.body.classList.add('lightbox-open');
    render();
    dialog.focus();
}

function close() {
    // Back to the thumbnail last seen when it is on screen, otherwise to the opener
    const seen = viewer.current === END ? null : viewer.mode === 'all'
        ? document.querySelector(`.thumbnail-card[data-thumbnail-id="${viewer.current}"] .thumbnail-image-wrapper`)
        : viewer.grid.querySelector(`.result-tile[data-thumbnail-id="${viewer.current}"]`);
    const target = [seen, viewer.opener].find(el => el?.isConnected && el.offsetParent !== null);

    viewer.mode = null;
    viewer.grid = null;
    viewer.current = null;
    viewer.opener = null;

    document.getElementById('thumbLightbox').classList.add('hidden');
    document.body.classList.remove('lightbox-open');
    target?.focus();
}

/**
 * Describe the thumbnail at a position of the sequence.
 * @param {number} position - Index of a thumbnail (not END)
 * @returns {{title: string, createdAt: string|undefined}}
 */
function describe(position) {
    if (viewer.mode === 'all') {
        const thumb = state.thumbnails[position];
        return { title: thumb.suggested_title || t('thumbnails.untitled'), createdAt: thumb.created_at };
    }
    const tile = recentTiles()[position];
    return { title: tile.title, createdAt: tile.dataset.createdAt };
}

function render() {
    const atEnd = viewer.current === END;
    const sequence = viewer.mode === 'recent' ? recentSequence() : null;

    document.getElementById('thumbLightboxFrame').classList.toggle('hidden', atEnd);
    document.getElementById('thumbLightboxEnd').classList.toggle('hidden', !atEnd);
    document.getElementById('thumbLightboxDownload').classList.toggle('hidden', atEnd);
    document.getElementById('thumbLightboxDelete').classList.toggle('hidden', atEnd);
    const single = sequenceLength() <= 1;
    document.getElementById('thumbLightboxPrev').disabled = single;
    document.getElementById('thumbLightboxNext').disabled = single;

    const info = document.getElementById('thumbLightboxInfo');
    if (atEnd) {
        const recentIds = sequence.filter(id => id !== END);
        document.getElementById('thumbLightboxEndCount').textContent =
            t('thumbnails.viewer_rest', { count: restCount(recentIds) });
        info.textContent = '';
        return;
    }

    const id = viewer.current;
    const position = positionOf(id);
    const count = viewer.mode === 'all' ? state.thumbnailsTotal : sequence.filter(item => item !== END).length;
    const { title, createdAt } = describe(position);
    const url = `/api/thumbnails/${id}`;

    const image = document.getElementById('thumbLightboxImage');
    if (image.dataset.thumbnailId !== String(id)) {
        image.dataset.thumbnailId = id;
        image.src = url;
    }
    image.alt = title;
    document.getElementById('thumbLightboxDownload').href = url;

    const parts = [title];
    if (createdAt) parts.push(formatDate(createdAt));
    info.innerHTML = `<strong>${position + 1} / ${count}</strong> · ${parts.map(escapeHtml).join(' · ')}`;
}

/**
 * Delete the thumbnail shown and show the one that takes its place.
 */
function deleteCurrent() {
    run(async () => {
        const position = positionOf(viewer.current);
        const deleted = await deleteOne(viewer.current);
        // The confirmation dialog took the focus
        document.getElementById('thumbLightbox').focus();
        if (deleted) await showAt(position);
    });
}

// =========================================================================
// SETUP
// =========================================================================

/**
 * Whether a click should open the viewer instead of following the link
 * (modified clicks keep "open in new tab").
 * @param {MouseEvent} e - Click event
 */
function isPlainClick(e) {
    return e.button === 0 && !e.ctrlKey && !e.metaKey && !e.shiftKey && !e.altKey;
}

export function init() {
    for (const gridId of ['thumbnailsGrid', 'recentThumbnailsGrid']) {
        document.getElementById(gridId).addEventListener('click', (e) => {
            const tile = e.target.closest('.result-tile');
            if (!tile || !isPlainClick(e)) return;
            e.preventDefault();
            open('recent', Number(tile.dataset.thumbnailId), tile);
        });
    }

    document.getElementById('existingThumbnailsGrid').addEventListener('click', (e) => {
        const wrapper = e.target.closest('.thumbnail-image-wrapper');
        if (!wrapper || !isPlainClick(e)) return;
        e.preventDefault();
        const card = wrapper.closest('.thumbnail-card');
        open('all', Number(card.dataset.thumbnailId), wrapper);
    });

    document.getElementById('thumbLightboxClose').addEventListener('click', close);
    document.getElementById('thumbLightboxPrev').addEventListener('click', () => step(-1));
    document.getElementById('thumbLightboxNext').addEventListener('click', () => step(1));
    document.getElementById('thumbLightboxDelete').addEventListener('click', deleteCurrent);
    document.getElementById('thumbLightboxRest').addEventListener('click', showRest);
    document.getElementById('thumbLightboxStage').addEventListener('click', (e) => {
        if (e.target.id === 'thumbLightboxStage') close();
    });

    document.addEventListener('keydown', (e) => {
        // A confirmation dialog (delete) is on top and handles its own keys
        if (!viewer.mode || document.querySelector('.modal-overlay.visible')) return;

        if (e.key === 'Escape') close();
        else if (e.key === 'ArrowLeft') step(-1);
        else if (e.key === 'ArrowRight') step(1);
        else if (e.key === 'Enter' && viewer.current === END && !e.target.closest('button, a')) showRest();
        else return;
        e.preventDefault();
    });
}
