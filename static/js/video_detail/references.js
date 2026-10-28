/**
 * Video Detail Page - What the Image Model Gets
 * ==============================================
 * The photos sent with each thumbnail of the Generate tab: the face and body
 * photos of each chosen person (chosen in their editor), the setting of the
 * video (step 2, "Where?") and the style image. It shows them under each
 * person, counts them against the model's limit above the generate button and
 * blocks generating while they do not fit or a person has no face photo.
 *
 * The setting photos belong to the video: they are saved as soon as they
 * change (PUT /api/people/{video}/environment), and so is whether its
 * thumbnails use them (PUT /api/people/{video}/environment/source).
 */

import { state } from './state.js';
import { getCurrentModelMaxRefs, getFacePhotosSetting } from './ai-config.js';
import { openCropDialog } from '../crop_dialog.js';

const REFRESH_DELAY_MS = 250;
const STRIP_THUMBNAIL = 160;
const VIEWER_THUMBNAIL = 420;

let refreshTimer = null;
let refreshRequest = 0;
let documentLoaded = null;      // Promise of the first load of the video's photos

// Photos of the video (GET /api/people/{video}/references): revision, environment_mode, references
let referenceDocument = null;
// Last preview of the chosen people: {count, limit, over_budget, images} or {error}
let preview = null;
// Why generating is blocked by the photos, '' when it is not
let blockReason = '';

// =========================================================================
// PUBLIC
// =========================================================================

export function init() {
    document.querySelectorAll('input[name="backgroundSource"]').forEach(radio => {
        radio.addEventListener('change', () => {
            saveSource(backgroundSource());
            renderBackground();
            scheduleRefresh();
        });
    });
    document.getElementById('backgroundFromVideo').addEventListener('click', openPicker);
    document.getElementById('backgroundUpload').addEventListener('click', () => {
        const input = document.getElementById('backgroundUploadInput');
        input.value = '';
        input.click();
    });
    document.getElementById('backgroundUploadInput').addEventListener('change', (e) => {
        const [file] = e.target.files;
        if (file) uploadBackground(file);
    });
    document.getElementById('backgroundMode').addEventListener('change', (e) => setMode(e.target.value));
    document.getElementById('backgroundList').addEventListener('click', onBackgroundClick);
    document.getElementById('backgroundList').addEventListener('change', onSideChange);
    setupPicker();
    setupViewer();

    document.addEventListener('people:changed', scheduleRefresh);
    document.addEventListener('people:rendered', fillPersonPhotos);
    document.addEventListener('imageConfigChanged', scheduleRefresh);
    document.addEventListener('settings:changed', scheduleRefresh);
    // Photos edited in a person's editor (another tab, or back from it)
    window.addEventListener('pageshow', (e) => { if (e.persisted) reload(); });
    document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') reload(); });

    documentLoaded = reload({ initial: true });
}

/**
 * Whether the setting photos go with the thumbnails (step 2).
 * @returns {boolean}
 */
export function includeBackground() {
    return backgroundSource() === 'photos';
}

/**
 * Why the photos block generating, '' when they do not.
 * @returns {string}
 */
export function getBlockReason() {
    return blockReason;
}

/**
 * Face and body photos a chosen person sends, from the last preview, or null
 * before it arrives.
 * @param {number} clusterId - Cluster database ID
 * @returns {{faces: number, bodies: number, images: Object[]}|null}
 */
export function personPhotos(clusterId) {
    if (!preview?.images) return null;
    const images = preview.images.filter(entry => entry.cluster_id === clusterId && entry.reference_role !== 'environment');
    return {
        faces: images.filter(entry => entry.reference_role === 'face' || entry.reference_role === 'face_sheet').length,
        bodies: images.filter(entry => entry.reference_role === 'body').length,
        images,
    };
}

// =========================================================================
// LOADING AND PREVIEW
// =========================================================================

async function reload({ initial = false } = {}) {
    const response = await fetch(`/api/people/${state.videoId}/references`);
    if (!response.ok) {
        ThumbnailApp.showToast(t('generate_form.references_load_error', { error: await errorMessage(response) }), 'error');
        return;
    }
    referenceDocument = await response.json();
    if (initial) setBackgroundSource(savedSource());
    renderBackground();
    scheduleRefresh();
}

async function errorMessage(response) {
    try {
        const data = await response.json();
        if (typeof data.detail === 'string') return data.detail;
    } catch {
        // Not a JSON error body
    }
    return `HTTP ${response.status}`;
}

function scheduleRefresh() {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refresh, REFRESH_DELAY_MS);
}

function previewBody(thumbnail) {
    const { image } = AppSettings.get();
    return {
        cluster_ids: state.selectedPeople.map(person => person.id),
        image_provider: image.provider,
        image_model: image.model,
        limit: Math.min(getFacePhotosSetting(), getCurrentModelMaxRefs()),
        include_environment: includeBackground(),
        style_base64: state.styleImage?.base64 ?? null,
        thumbnail,
    };
}

async function refresh() {
    await documentLoaded;
    const request = ++refreshRequest;
    if (state.selectedPeople.length === 0 || !referenceDocument) {
        preview = null;
        render();
        return;
    }
    let result;
    try {
        const response = await fetch(`/api/people/${state.videoId}/references/preview`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(previewBody(STRIP_THUMBNAIL)),
        });
        result = response.ok ? await response.json() : { error: await errorMessage(response) };
    } catch (error) {
        result = { error: error.message };
    }
    // A newer change brings its own preview
    if (request !== refreshRequest) return;
    preview = result;
    render();
}

// =========================================================================
// RENDERING
// =========================================================================

function render() {
    fillPersonPhotos();
    renderLine();
    document.dispatchEvent(new CustomEvent('references:changed'));
}

function captionOf(entry) {
    if (entry.reference_role === 'style') return t('generate_form.ref_style');
    if (entry.reference_role === 'environment') {
        return entry.environment_side
            ? t(`generate_form.ref_background_${entry.environment_side}`)
            : t('generate_form.ref_background');
    }
    const person = state.selectedPeople.find(item => item.id === entry.cluster_id);
    const name = person ? ThumbnailApp.getClusterDisplayName(person) : '';
    const role = entry.reference_role === 'face_sheet' ? 'ref_collage'
        : entry.reference_role === 'body' ? 'ref_body' : entry.is_primary ? 'ref_primary' : 'ref_face';
    return t(`generate_form.${role}`, { name });
}

/**
 * The photos under each chosen person (people.js leaves an empty strip).
 */
function fillPersonPhotos() {
    document.querySelectorAll('[data-person-photos]').forEach(strip => {
        const photos = personPhotos(Number(strip.dataset.personPhotos));
        strip.innerHTML = photos ? photos.images.map(entry => `
            <img src="${entry.image}" alt="" title="${escapeHtml(captionOf(entry))}" class="${entry.reference_role === 'body' ? 'body' : ''}">
        `).join('') : '';
    });
}

/**
 * Above the generate button: how many images the AI gets, or what blocks generating.
 */
function renderLine() {
    const line = document.getElementById('generateReferences');
    blockReason = '';
    if (!preview) {
        line.innerHTML = '';
    } else if (includeBackground() && backgroundPhotos().length === 0) {
        blockReason = t('generate_form.background_missing');
    } else if (preview.error) {
        blockReason = preview.error;
    } else if (preview.over_budget) {
        blockReason = t('generate_form.references_over', { count: preview.count, limit: preview.limit });
    }
    if (blockReason) {
        line.innerHTML = `${ThumbnailApp.icon('alert', 'icon-sm')}<span>${escapeHtml(blockReason)}</span>`;
    } else if (preview) {
        line.innerHTML = `${ThumbnailApp.icon('image', 'icon-sm')}<span>${escapeHtml(t('generate_form.references_count', { count: preview.count, limit: preview.limit }))}</span>` +
            ` <button type="button" class="link-btn" data-open-references>${escapeHtml(t('generate_form.references_view'))}</button>`;
    }
    line.classList.toggle('blocked', Boolean(blockReason));
    const button = document.getElementById('generateBtn');
    button.classList.toggle('references-blocked', Boolean(blockReason));
}

// =========================================================================
// VIEWER: every image the AI gets, larger
// =========================================================================

function setupViewer() {
    document.getElementById('generateReferences').addEventListener('click', (e) => {
        if (e.target.closest('[data-open-references]')) openViewer();
    });
    const dialog = document.getElementById('referencesPreview');
    dialog.addEventListener('click', (e) => {
        if (e.target === dialog || e.target.closest('[data-close-references]')) dialog.classList.remove('visible');
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && dialog.classList.contains('visible')) dialog.classList.remove('visible');
    });
}

async function openViewer() {
    const dialog = document.getElementById('referencesPreview');
    const grid = document.getElementById('referencesPreviewGrid');
    grid.innerHTML = `<div class="loading">${escapeHtml(t('common.loading'))}</div>`;
    dialog.classList.add('visible');
    try {
        const response = await fetch(`/api/people/${state.videoId}/references/preview`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(previewBody(VIEWER_THUMBNAIL)),
        });
        if (!response.ok) throw new Error(await errorMessage(response));
        const result = await response.json();
        grid.innerHTML = result.images.map(entry => `
            <figure class="references-preview-item">
                <img src="${entry.image}" alt="">
                <figcaption>${escapeHtml(captionOf(entry))}</figcaption>
            </figure>
        `).join('');
    } catch (error) {
        grid.innerHTML = `<p class="form-hint form-warning">${ThumbnailApp.icon('alert', 'icon-sm')}<span>${escapeHtml(error.message)}</span></p>`;
    }
}

// =========================================================================
// STEP 2: THE SETTING
// =========================================================================

function backgroundSource() {
    return document.querySelector('input[name="backgroundSource"]:checked').value;
}

function setBackgroundSource(value) {
    document.querySelector(`input[name="backgroundSource"][value="${value}"]`).checked = true;
}

/**
 * The choice saved for the video or, until the user makes one, the photos
 * when the video has some.
 * @returns {string} photos or ai
 */
function savedSource() {
    return referenceDocument.environment_source ?? (backgroundPhotos().length > 0 ? 'photos' : 'ai');
}

async function saveSource(source) {
    try {
        const response = await fetch(`/api/people/${state.videoId}/environment/source`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source }),
        });
        if (!response.ok) throw new Error(await errorMessage(response));
        referenceDocument.environment_source = source;
    } catch (error) {
        ThumbnailApp.showToast(t('generate_form.background_save_error', { error: error.message }), 'error');
        // Back to what is saved
        setBackgroundSource(savedSource());
        renderBackground();
        scheduleRefresh();
    }
}

function backgroundPhotos() {
    return referenceDocument ? referenceDocument.references.filter(ref => ref.role === 'environment') : [];
}

function frameImageUrl(name) {
    return `/api/people/${state.videoId}/frames/${encodeURIComponent(name)}/image`;
}

function sourceUrl(ref) {
    return ref.source_type === 'upload'
        ? `/api/people/${state.videoId}/references/uploads/${encodeURIComponent(ref.source)}`
        : frameImageUrl(ref.source);
}

// Crops of the setting photos as the image model gets them
const crops = new Map();

function cropUrl(ref) {
    const key = JSON.stringify([ref.source_type, ref.source, ref.crop]);
    if (!crops.has(key)) {
        const request = fetch(`/api/people/${state.videoId}/references/crop`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(ref),
        }).then(async response => {
            if (!response.ok) throw new Error(await errorMessage(response));
            return URL.createObjectURL(await response.blob());
        });
        request.catch(() => crops.delete(key));
        crops.set(key, request);
    }
    return crops.get(key);
}

function renderBackground() {
    const photos = backgroundPhotos();
    const usePhotos = includeBackground();
    document.getElementById('backgroundPhotos').classList.toggle('hidden', !usePhotos);
    document.querySelectorAll('.background-choice').forEach(choice => {
        choice.classList.toggle('active', choice.querySelector('input').checked);
    });

    const mode = referenceDocument?.environment_mode ?? 'same';
    const list = document.getElementById('backgroundList');
    list.innerHTML = photos.length === 0
        ? `<li class="background-empty">${escapeHtml(t('generate_form.background_none'))}</li>`
        : photos.map(ref => `
            <li class="background-item" data-ref="${escapeHtml(ref.id)}">
                <span class="background-image"><img alt=""></span>
                <span class="background-item-actions">
                    <button type="button" class="pe-slot-action" data-action="crop" title="${escapeHtml(t('generate_form.background_crop'))}" aria-label="${escapeHtml(t('generate_form.background_crop'))}">${ThumbnailApp.icon('crop', 'icon-sm')}</button>
                    <button type="button" class="pe-slot-action" data-action="remove" title="${escapeHtml(t('generate_form.background_remove'))}" aria-label="${escapeHtml(t('generate_form.background_remove'))}">${ThumbnailApp.icon('x', 'icon-sm')}</button>
                </span>
                ${mode === 'split' ? `
                    <select class="select select-sm background-side" aria-label="${escapeHtml(t('generate_form.background_side'))}">
                        <option value="left"${ref.side === 'left' ? ' selected' : ''}>${escapeHtml(t('generate_form.background_left'))}</option>
                        <option value="right"${ref.side === 'right' ? ' selected' : ''}>${escapeHtml(t('generate_form.background_right'))}</option>
                    </select>` : ''}
            </li>
        `).join('');
    for (const ref of photos) {
        const item = list.querySelector(`[data-ref="${CSS.escape(ref.id)}"]`);
        cropUrl(ref).then(url => {
            if (item.isConnected) item.querySelector('img').src = url;
        }).catch(error => {
            if (item.isConnected) item.title = error.message;
        });
    }

    const modeField = document.getElementById('backgroundModeField');
    modeField.classList.toggle('hidden', photos.length < 2);
    document.getElementById('backgroundMode').value = mode;
    document.getElementById('backgroundModeHint').textContent = t(`generate_form.background_mode_${mode}_hint`);
}

/**
 * Save the setting photos of the video, then show them.
 * @param {Object[]} references - Environment references
 * @param {string} mode - same, alternatives or split
 */
async function saveBackground(references, mode) {
    try {
        const response = await fetch(`/api/people/${state.videoId}/environment`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ references, environment_mode: mode, revision: referenceDocument.revision }),
        });
        if (!response.ok) throw new Error(await errorMessage(response));
        referenceDocument = await response.json();
    } catch (error) {
        ThumbnailApp.showToast(t('generate_form.background_save_error', { error: error.message }), 'error');
        // Start again from what is saved
        await reload();
        return;
    }
    renderBackground();
    scheduleRefresh();
}

function newBackground(sourceType, source, crop = null) {
    return {
        id: crypto.randomUUID().replaceAll('-', ''), role: 'environment', person_id: null,
        source_type: sourceType, source, crop, primary: false, view: null, expression: null,
        coverage: 'unknown', posture: 'unknown', side: null, identity_confirmed: false,
    };
}

/**
 * Split screen needs a side for each photo: the first left, the rest right.
 */
function withSides(references, mode) {
    if (mode !== 'split') return references.map(ref => ({ ...ref, side: null }));
    return references.map((ref, index) => ({ ...ref, side: ref.side ?? (index === 0 ? 'left' : 'right') }));
}

function setMode(mode) {
    saveBackground(withSides(backgroundPhotos(), mode), mode);
}

function onSideChange(e) {
    if (!e.target.matches('.background-side')) return;
    const id = e.target.closest('[data-ref]').dataset.ref;
    const references = backgroundPhotos().map(ref => ref.id === id ? { ...ref, side: e.target.value } : ref);
    saveBackground(references, referenceDocument.environment_mode);
}

async function onBackgroundClick(e) {
    const action = e.target.closest('[data-action]');
    if (!action) return;
    const id = action.closest('[data-ref]').dataset.ref;
    const mode = referenceDocument.environment_mode;
    if (action.dataset.action === 'remove') {
        const rest = backgroundPhotos().filter(ref => ref.id !== id);
        saveBackground(withSides(rest, rest.length < 2 ? 'same' : mode), rest.length < 2 ? 'same' : mode);
        return;
    }
    const ref = backgroundPhotos().find(item => item.id === id);
    const result = await openCropDialog({
        title: t('generate_form.background_crop'),
        hint: t('generate_form.background_crop_hint'),
        src: sourceUrl(ref),
        box: ref.crop,
    }).catch(error => {
        ThumbnailApp.showToast(error.message, 'error');
        return null;
    });
    if (!result) return;
    saveBackground(backgroundPhotos().map(item => item.id === id ? { ...item, crop: result.box } : item), mode);
}

async function uploadBackground(file) {
    try {
        const encoded = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(new Error(t('crop.image_error')));
            reader.readAsDataURL(file);
        });
        const response = await fetch(`/api/people/${state.videoId}/references/upload`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: file.name, image_base64: encoded }),
        });
        if (!response.ok) throw new Error(await errorMessage(response));
        const asset = await response.json();
        const mode = referenceDocument.environment_mode;
        await saveBackground(withSides([...backgroundPhotos(), newBackground('upload', asset.source)], mode), mode);
    } catch (error) {
        ThumbnailApp.showToast(t('generate_form.background_save_error', { error: error.message }), 'error');
    }
}

// =========================================================================
// FRAME PICKER
// =========================================================================

let pickerFrames = null;          // Filenames of the video's frames, in video order
let pickerSelected = new Set();   // Filenames chosen in the picker

function setupPicker() {
    const picker = document.getElementById('backgroundPicker');
    picker.addEventListener('click', (e) => {
        const frame = e.target.closest('.background-picker-frame');
        const action = e.target.closest('[data-background-picker]')?.dataset.backgroundPicker;
        if (frame) {
            const name = frame.dataset.name;
            if (pickerSelected.has(name)) pickerSelected.delete(name);
            else pickerSelected.add(name);
            frame.classList.toggle('selected', pickerSelected.has(name));
            frame.setAttribute('aria-pressed', String(pickerSelected.has(name)));
            renderPickerCount();
        } else if (action === 'apply') {
            applyPicker();
        } else if (action === 'cancel' || e.target === picker) {
            picker.classList.remove('visible');
        }
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && picker.classList.contains('visible')) picker.classList.remove('visible');
    });
}

function renderPickerCount() {
    document.getElementById('backgroundPickerCount').textContent =
        t('generate_form.background_pick_count', { count: pickerSelected.size });
}

async function openPicker() {
    const picker = document.getElementById('backgroundPicker');
    const grid = document.getElementById('backgroundPickerGrid');
    pickerSelected = new Set(backgroundPhotos().filter(ref => ref.source_type === 'frame').map(ref => ref.source));
    renderPickerCount();
    picker.classList.add('visible');
    if (!pickerFrames) {
        grid.innerHTML = `<div class="loading">${escapeHtml(t('common.loading'))}</div>`;
        try {
            const response = await fetch(`/api/people/${state.videoId}/frames?view_mode=person`);
            if (!response.ok) throw new Error(await errorMessage(response));
            pickerFrames = (await response.json()).frames.map(frame => frame.filename);
        } catch (error) {
            grid.innerHTML = `<p class="form-hint form-warning">${escapeHtml(error.message)}</p>`;
            return;
        }
    }
    grid.innerHTML = pickerFrames.map(name => `
        <button type="button" class="background-picker-frame${pickerSelected.has(name) ? ' selected' : ''}"
                data-name="${escapeHtml(name)}" aria-pressed="${pickerSelected.has(name)}">
            <img src="${frameImageUrl(name)}" alt="" loading="lazy" decoding="async">
            <span class="pe-check" aria-hidden="true">${ThumbnailApp.icon('check', 'icon-sm')}</span>
        </button>
    `).join('');
}

function applyPicker() {
    document.getElementById('backgroundPicker').classList.remove('visible');
    const kept = backgroundPhotos().filter(ref => ref.source_type !== 'frame' || pickerSelected.has(ref.source));
    const have = new Set(kept.filter(ref => ref.source_type === 'frame').map(ref => ref.source));
    const added = pickerFrames.filter(name => pickerSelected.has(name) && !have.has(name)).map(name => newBackground('frame', name));
    const references = [...kept, ...added];
    const mode = references.length < 2 ? 'same' : referenceDocument.environment_mode;
    setBackgroundSource('photos');
    if (referenceDocument.environment_source !== 'photos') saveSource('photos');
    saveBackground(withSides(references, mode), mode);
}
