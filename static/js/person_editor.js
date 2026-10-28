/**
 * Person Editor
 * =============
 * One page to create a person or to edit one. The frames of the video fill the
 * left side; the person sits on the right with its save button always in view:
 * name, notes and "what the AI sees", the face and body photos the image model
 * gets of them. Actions on selected frames (add, remove, move, delete) apply at
 * once; the panel is a draft saved with one button. While creating, everything
 * waits for "Create" and the face photos are picked as frames are added.
 */

import { openCropDialog } from './crop_dialog.js';

const root = document.getElementById('personEditor');
const videoId = Number(root.dataset.videoId);
const personId = root.dataset.personId ? Number(root.dataset.personId) : null;
const isNew = personId === null;
// Person of the photos in the draft: -1 until a new person is created
const draftPersonId = isNew ? -1 : personId;
// Face photos the automatic choice picks, and the most the panel offers to fill
const FACE_COUNT = Number(root.dataset.faceCount);

// Face crop geometry, shared with services/face_index.py (face_crop_box)
const FACE_CROP_SCALE = 1.8;
const FACE_CROP_LIFT = 0.12;
// Head and shoulders around a face, as services/face_index.py (portrait_crop_box)
const PORTRAIT_WIDTH = 3.0;
const PORTRAIT_HEIGHT = 3.6;
const PORTRAIT_HEADROOM = 0.9;

// Members this far from the person's face are flagged as possibly someone else
// (services/face_index.py MIN_IDENTITY_SIMILARITY)
const SUSPECT_SIMILARITY = 0.3;

const VIEW_STORAGE_KEY = 'personEditor.view';
const MATCH_DELAY_MS = 350;
const PREVIEW_THUMBNAIL = 360;

const state = {
    viewMode: root.dataset.viewMode || null,
    frames: [],              // Frames of the video (GET /api/people/{video}/frames)
    byName: new Map(),       // filename -> frame
    people: [],              // People of the view (summaries)
    person: null,            // Saved person (edit mode)
    members: new Set(),      // Filenames of the person (a draft while creating)
    revision: 0,             // Revision of the video's photos the draft starts from
    settingPhotos: 0,        // Setting photos that go with every thumbnail of the video
    saved: emptyDraft(),
    draft: emptyDraft(),
    facesTouched: false,     // While creating, faces chosen by hand stop the automatic choice
    pendingBodyView: null,   // Optional orientation requested by an empty body slot
    bodyCandidates: null,    // Detected body crops of the person (filename -> candidate)
    scope: isNew ? 'unassigned' : 'members',
    sort: null,              // Chosen sort, or null for the scope's default
    groupScenes: false,
    view: readStoredView(),
    showFaceless: false,
    selected: new Set(),
    anchor: null,            // Last clicked frame, start of shift-click ranges
    visible: [],             // Filenames shown, in display order
    busy: false,
    lightboxIndex: -1,
};

let matchTimer = null;
let automaticFacesRequest = 0;

/**
 * What the panel edits. references holds the face and body photos as the API
 * takes them (VisualReference), faces first with the primary one at the front.
 */
function emptyDraft() {
    return { label: '', description: '', references: [], format: 'photos' };
}

function copyDraft(draft) {
    return { ...draft, references: draft.references.map(ref => ({ ...ref })) };
}

// =========================================================================
// HELPERS
// =========================================================================

function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[char]);
}

function readStoredView() {
    try {
        return localStorage.getItem(VIEW_STORAGE_KEY) === 'frames' ? 'frames' : 'faces';
    } catch {
        return 'faces';
    }
}

function storeView(view) {
    try {
        localStorage.setItem(VIEW_STORAGE_KEY, view);
    } catch {
        // Storage may be unavailable (private mode); the view just is not remembered
    }
}

async function api(url, { method = 'GET', body } = {}) {
    const options = { method };
    if (body !== undefined) {
        options.headers = { 'Content-Type': 'application/json' };
        options.body = JSON.stringify(body);
    }
    const response = await fetch(url, options);
    if (!response.ok) throw new Error(await errorMessage(response));
    return response.json();
}

/**
 * The detail of a failed API response, or its HTTP status.
 */
async function errorMessage(response) {
    try {
        const data = await response.json();
        if (typeof data.detail === 'string') return data.detail;
    } catch {
        // Not a JSON error body
    }
    return `HTTP ${response.status}`;
}

function frameImageUrl(name) {
    return `/api/people/${videoId}/frames/${encodeURIComponent(name)}/image`;
}

function sourceUrl(ref) {
    return ref.source_type === 'upload'
        ? `/api/people/${videoId}/references/uploads/${encodeURIComponent(ref.source)}`
        : frameImageUrl(ref.source);
}

/**
 * Inline style that shows only the face region of a frame inside a square box.
 */
function faceCropStyle(box, width, height) {
    const [x1, y1, x2, y2] = box;
    const side = Math.min(Math.max(x2 - x1, y2 - y1) * FACE_CROP_SCALE, width, height);
    const centerX = (x1 + x2) / 2;
    const centerY = (y1 + y2) / 2 - (y2 - y1) * FACE_CROP_LIFT;
    const left = Math.min(Math.max(centerX - side / 2, 0), width - side);
    const top = Math.min(Math.max(centerY - side / 2, 0), height - side);
    return `width:${(width / side) * 100}%;left:${(-left / side) * 100}%;top:${(-top / side) * 100}%`;
}

/**
 * Markup of a frame thumbnail; face crops are applied once the image loads.
 */
function thumbMarkup(frame, { square }) {
    const box = square && frame.face_box ? ` data-box="${frame.face_box.join(',')}"` : '';
    return `<span class="pe-thumb${square ? ' square' : ''}${square && !frame.face_box ? ' no-face' : ''}">` +
        `<img src="${frameImageUrl(frame.filename)}" alt="" loading="lazy" decoding="async"${box}></span>`;
}

function applyFaceCrop(img) {
    if (!img.dataset.box) {
        img.classList.add('ready');
        return;
    }
    const box = img.dataset.box.split(',').map(Number);
    img.setAttribute('style', faceCropStyle(box, img.naturalWidth, img.naturalHeight));
    img.classList.add('cropped', 'ready');
}

/**
 * Head and shoulders around a face box, as the server crops them (without
 * knowing the other faces of the frame, which the server then checks).
 */
function portraitBox([x1, y1, x2, y2]) {
    const width = x2 - x1;
    const height = y2 - y1;
    const centerX = (x1 + x2) / 2;
    const top = y1 - height * PORTRAIT_HEADROOM;
    return [centerX - width * PORTRAIT_WIDTH / 2, top, centerX + width * PORTRAIT_WIDTH / 2, top + height * PORTRAIT_HEIGHT]
        .map(Math.round);
}

/**
 * A first guess of the body below a face box, to adjust by hand.
 */
function bodyGuessBox([x1, y1, x2, y2]) {
    const width = x2 - x1;
    const height = y2 - y1;
    const centerX = (x1 + x2) / 2;
    return [centerX - width * 2, y1 - height * 0.6, centerX + width * 2, y2 + height * 7].map(Math.round);
}

function personName(person) {
    return ThumbnailApp.getClusterDisplayName(person);
}

function peopleById() {
    return new Map(state.people.map(person => [person.id, person]));
}

function sameList(a, b) {
    return a.length === b.length && a.every((value, index) => value === b[index]);
}

function sameReferences(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
}

function isDirty() {
    if (isNew) {
        return state.members.size > 0 || state.draft.label.trim() !== '' || state.draft.description.trim() !== '';
    }
    return state.draft.label.trim() !== state.saved.label
        || state.draft.description.trim() !== state.saved.description
        || state.draft.format !== state.saved.format
        || !sameReferences(state.draft.references, state.saved.references);
}

function hasPersonContext() {
    return state.frames.some(frame => frame.similarity !== null);
}

function sceneLabel(frame) {
    return frame.scene_index === null
        ? t('person_editor.scene_unknown')
        : t('person_editor.scene_label', { n: frame.scene_index });
}

function newId() {
    return crypto.randomUUID().replaceAll('-', '');
}

/**
 * A new face or body photo of the person (fields as VisualReference takes them).
 */
function newReference(role, sourceType, source, fields = {}) {
    return {
        id: newId(), role, person_id: draftPersonId, source_type: sourceType, source,
        crop: null, primary: false, view: null, expression: null, coverage: 'unknown',
        posture: 'unknown', side: null, identity_confirmed: false, ...fields,
    };
}

/**
 * Angle of a face as services/face_index.py (face_reference_view) names it.
 */
function faceView(yaw) {
    if (yaw === null || yaw === undefined) return null;
    const angle = Math.abs(yaw);
    return angle < 18 ? 'front' : angle < 55 ? 'three_quarter' : 'profile';
}

// =========================================================================
// PHOTOS OF THE PERSON
// =========================================================================

function faces(references = state.draft.references) {
    return references.filter(ref => ref.role === 'face');
}

function bodies(references = state.draft.references) {
    return references.filter(ref => ref.role === 'body');
}

/**
 * Setting photos that go with every thumbnail of the video (Generate, step 2):
 * the choice saved for the video or, until there is one, the photos it has.
 * @param {Object} document - GET /api/people/{video}/references
 * @returns {number}
 */
function settingPhotosOf(document) {
    const photos = document.references.filter(ref => ref.role === 'environment').length;
    const source = document.environment_source ?? (photos > 0 ? 'photos' : 'ai');
    return source === 'photos' ? photos : 0;
}

/**
 * Images a thumbnail has for its people: the limit of Settings within the
 * image model's, less the setting photos. All its people share them.
 * @returns {number}
 */
function imageBudget() {
    return Math.min(AppSettings.get().generation.face_photos, AppSettings.maxRefs()) - state.settingPhotos;
}

/**
 * Images this person sends: their faces (one collage counts as one) and bodies.
 * @returns {number}
 */
function imagesUsed() {
    const faceCount = faces().length;
    return (state.draft.format === 'collage' ? Math.min(faceCount, 1) : faceCount) + bodies().length;
}

/**
 * How many more photos of a role still fit in the thumbnail.
 * @param {string} role - face or body
 * @returns {number}
 */
function freeImages(role) {
    // One more face of a collage stays one image
    if (role === 'face' && state.draft.format === 'collage' && faces().length > 0) return Infinity;
    return Math.max(imageBudget() - imagesUsed(), 0);
}

/**
 * Faces first, the primary one at the front; there is always one primary
 * when there are faces.
 */
function ordered(references) {
    const faceRefs = faces(references).map(ref => ({ ...ref }));
    let primary = faceRefs.findIndex(ref => ref.primary);
    if (primary < 0 && faceRefs.length > 0) primary = 0;
    faceRefs.forEach((ref, index) => { ref.primary = index === primary; });
    if (primary > 0) faceRefs.unshift(...faceRefs.splice(primary, 1));
    return [...faceRefs, ...bodies(references)];
}

function setReferences(references) {
    state.draft.references = ordered(references);
    renderPanel();
    updateBadges();
    renderSelectionBar();
}

// The crop the server makes of each photo: the image and the box it used
const previews = new Map();

function previewKey(ref) {
    return JSON.stringify([ref.role, ref.person_id, ref.source_type, ref.source, ref.crop, ref.identity_confirmed, ref.view]);
}

/**
 * The photo as the image model gets it. Rejected photos (another person's
 * face, a crop without the face...) throw with the reason.
 * @returns {Promise<{url: string, box: number[]|null}>}
 */
function referencePreview(ref) {
    const key = previewKey(ref);
    if (!previews.has(key)) {
        const request = fetch(`/api/people/${videoId}/references/crop`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(ref),
        }).then(async response => {
            if (!response.ok) throw new Error(await errorMessage(response));
            const header = response.headers.get('X-Reference-Crop');
            return { url: URL.createObjectURL(await response.blob()), box: header ? header.split(',').map(Number) : null };
        });
        request.catch(() => previews.delete(key));
        previews.set(key, request);
    }
    return previews.get(key);
}

/**
 * Add photos of frames, checking each with the server first.
 * @param {Object[]} candidates - New references
 */
async function addReferences(candidates) {
    const accepted = [];
    for (const ref of candidates) {
        try {
            await referencePreview(ref);
            accepted.push(ref);
        } catch (error) {
            const frame = state.byName.get(ref.source);
            const where = frame ? sceneLabel(frame) : ref.source;
            ThumbnailApp.showToast(t('person_editor.photo_refused', { where, error: error.message }), 'error');
        }
    }
    if (accepted.length === 0) return;
    if (accepted.some(ref => ref.role === 'face')) state.facesTouched = true;
    setReferences([...state.draft.references, ...accepted]);
}

function setPrimary(id) {
    state.facesTouched = true;
    setReferences(state.draft.references.map(ref => ({ ...ref, primary: ref.id === id })));
}

function removeReference(id) {
    const removed = state.draft.references.find(ref => ref.id === id);
    if (removed?.role === 'face') state.facesTouched = true;
    setReferences(state.draft.references.filter(ref => ref.id !== id));
}

function selectedMemberFrames() {
    return state.visible.filter(name => state.selected.has(name) && state.members.has(name));
}

/**
 * Selected frames of the person as face photos. A new person's crop is
 * explicit; a saved person's is the automatic one, to adjust later.
 */
function useSelectedAsFaces() {
    const used = new Set(faces().filter(ref => ref.source_type === 'frame').map(ref => ref.source));
    const names = selectedMemberFrames().filter(name => !used.has(name) && state.byName.get(name)?.face_box);
    clearSelection();
    if (names.length === 0) {
        ThumbnailApp.showToast(t('person_editor.no_new_faces'), 'info');
        return;
    }
    addReferences(names.map(name => {
        const frame = state.byName.get(name);
        return newReference('face', 'frame', name, {
            view: faceView(frame.yaw),
            expression: frame.expression && frame.expression !== 'unknown' ? frame.expression : null,
            crop: isNew ? portraitBox(frame.face_box) : null,
            identity_confirmed: isNew,
        });
    }));
}

async function loadBodyCandidates() {
    if (isNew || state.bodyCandidates) return;
    const body = await api(`/api/people/${videoId}/${personId}/body`);
    state.bodyCandidates = new Map(body.candidates.map(candidate => [candidate.filename, candidate]));
}

/**
 * The selected frame as a body photo, cropped by hand from the detected body
 * (or a guess below the face).
 */
async function useSelectedAsBody() {
    const requestedView = state.pendingBodyView;
    state.pendingBodyView = null;
    const [name] = requestedView === 'back' ? [...state.selected] : selectedMemberFrames();
    if (!name) return;
    const frame = state.byName.get(name);
    clearSelection();
    try {
        await loadBodyCandidates();
    } catch (error) {
        ThumbnailApp.showToast(t('person_editor.load_error', { error: error.message }), 'error');
        return;
    }
    const candidate = state.bodyCandidates?.get(name);
    const fields = candidate ? { coverage: candidate.coverage, posture: candidate.posture, view: candidate.view } : {};
    const ref = newReference('body', 'frame', name, { ...fields, view: requestedView || fields.view || null, identity_confirmed: requestedView !== 'back' });
    const result = await openCropDialog({
        title: t('person_editor.crop_body_title'),
        hint: t('person_editor.crop_body_hint'),
        src: frameImageUrl(name),
        box: candidate?.box ?? (frame?.face_box ? bodyGuessBox(frame.face_box) : null),
        confirmText: requestedView === 'back' ? t('person_editor.upload_confirm', { name: currentName() }) : null,
        validate: (box, confirmed) => referencePreview({ ...ref, crop: box, identity_confirmed: requestedView !== 'back' || confirmed }),
    }).catch(showCropError);
    if (!result) return;
    setReferences([...state.draft.references, { ...ref, crop: result.box, identity_confirmed: true }]);
}

function showCropError(error) {
    ThumbnailApp.showToast(error.message, 'error');
    return null;
}

/**
 * A photo from the computer, cropped and confirmed as this person.
 * @param {string} role - 'face' or 'body'
 */
function chooseUpload(role, view = null) {
    const input = document.getElementById('peUploadInput');
    input.dataset.role = role;
    input.dataset.view = view || '';
    input.value = '';
    input.click();
}

async function uploadPhoto(file, role, view = null) {
    let encoded;
    try {
        encoded = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(new Error(t('crop.image_error')));
            reader.readAsDataURL(file);
        });
        const asset = await api(`/api/people/${videoId}/references/upload`, {
            method: 'POST', body: { filename: file.name, image_base64: encoded }
        });
        const ref = newReference(role, 'upload', asset.source, { view });
        const result = await openCropDialog({
            title: t(role === 'face' ? 'person_editor.crop_face_title' : 'person_editor.crop_body_title'),
            hint: t(role === 'face' ? 'person_editor.crop_face_hint' : 'person_editor.crop_body_hint'),
            src: sourceUrl(ref),
            confirmText: t('person_editor.upload_confirm', { name: currentName() }),
            validate: (box, confirmed) => referencePreview({ ...ref, crop: box, identity_confirmed: confirmed }),
        });
        if (!result) return;
        if (role === 'face') state.facesTouched = true;
        setReferences([...state.draft.references, { ...ref, crop: result.box, identity_confirmed: true }]);
    } catch (error) {
        ThumbnailApp.showToast(t('person_editor.upload_error', { error: error.message }), 'error');
    }
}

/**
 * Adjust the crop of a photo. A saved person's frame photo can go back to the
 * automatic crop, which is then stored as no crop.
 */
async function adjustCrop(id) {
    const ref = state.draft.references.find(item => item.id === id);
    let autoBox = null;
    try {
        if (ref.source_type === 'frame' && !isNew && ref.role === 'face') {
            ({ box: autoBox } = await referencePreview({ ...ref, crop: null, identity_confirmed: false }));
        }
        const result = await openCropDialog({
            title: t(ref.role === 'face' ? 'person_editor.crop_face_title' : 'person_editor.crop_body_title'),
            hint: t(ref.role === 'face' ? 'person_editor.crop_face_hint' : 'person_editor.crop_body_hint'),
            src: sourceUrl(ref),
            box: ref.crop ?? autoBox,
            autoBox,
            validate: (box) => referencePreview({ ...ref, crop: box, identity_confirmed: true }),
        });
        if (!result) return;
        const automatic = autoBox && sameList(result.box, autoBox);
        if (ref.role === 'face') state.facesTouched = true;
        setReferences(state.draft.references.map(item => item.id !== id ? item : {
            ...item,
            crop: automatic ? null : result.box,
            identity_confirmed: automatic ? item.identity_confirmed : true,
        }));
    } catch (error) {
        ThumbnailApp.showToast(error.message, 'error');
    }
}

/**
 * The server picks clear faces with different angles or expressions,
 * skipping black and white frames.
 */
async function chooseFacesAutomatically({ quiet = false } = {}) {
    const members = [...state.members];
    const request = ++automaticFacesRequest;
    if (members.length === 0) {
        setReferences(bodies());
        return;
    }
    let references;
    try {
        ({ references } = await api(`/api/people/${videoId}/references/automatic`, {
            method: 'POST',
            body: { view_mode: state.viewMode, person_id: personId, members }
        }));
    } catch (error) {
        if (request !== automaticFacesRequest) return;
        if (quiet) {
            setReferences(bodies());
        } else {
            ThumbnailApp.showToast(error.message, 'error');
        }
        return;
    }
    // A newer change brings its own choice
    if (request !== automaticFacesRequest) return;
    setReferences([...references, ...bodies()]);
}

/**
 * The server analyzes the bodies of the frames if it has not yet (it can take
 * a while) and picks the best body in a color frame.
 */
async function chooseBodyAutomatically() {
    const button = document.getElementById('peAutoBody');
    const label = button.querySelector('span');
    button.disabled = true;
    button.classList.add('busy');
    label.textContent = t('person_editor.body_analyzing');
    try {
        const { references } = await api(`/api/people/${videoId}/${personId}/references/automatic-body`, { method: 'POST' });
        state.bodyCandidates = null;
        // Uploaded evidence and manually assigned back views are deliberate choices.
        const manual = bodies().filter(ref => ref.source_type === 'upload' || ref.view === 'back');
        const picked = references.filter(ref => !manual.some(saved =>
            saved.view && saved.view === ref.view || saved.source_type === ref.source_type && saved.source === ref.source));
        setReferences([...faces(), ...picked, ...manual]);
    } catch (error) {
        ThumbnailApp.showToast(error.message, 'error');
    } finally {
        button.disabled = false;
        button.classList.remove('busy');
        label.textContent = t('person_editor.auto_choose');
    }
}

// =========================================================================
// LOADING
// =========================================================================

async function load() {
    if (isNew) {
        const [listing, document] = await Promise.all([
            api(`/api/people/${videoId}/frames?view_mode=${state.viewMode}`),
            api(`/api/people/${videoId}/references`),
        ]);
        state.revision = document.revision;
        state.settingPhotos = settingPhotosOf(document);
        applyListing(listing);
    } else {
        const [person, listing, document] = await Promise.all([
            api(`/api/people/${videoId}/${personId}`),
            api(`/api/people/${videoId}/frames?person_id=${personId}`),
            api(`/api/people/${videoId}/references`),
        ]);
        state.settingPhotos = settingPhotosOf(document);
        applyPerson(person, { resetDraft: state.person === null });
        applyListing(listing);
    }
}

/**
 * Take a saved person from the server. Unsaved names and notes are kept;
 * unsaved photos are kept minus the frames that left the person.
 */
function applyPerson(person, { resetDraft }) {
    const photosWereClean = sameReferences(state.draft.references, state.saved.references);
    const left = new Set([...state.members].filter(name => !person.members.includes(name)));
    state.person = person;
    state.viewMode = person.view_mode;
    state.members = new Set(person.members);
    state.revision = person.revision;
    state.saved = {
        label: person.label || '',
        description: person.description || '',
        references: ordered(person.references),
        format: person.reference_format,
    };
    if (resetDraft) {
        state.draft = copyDraft(state.saved);
        return;
    }
    // The server drops the photos of frames that leave the person; so does the draft
    state.draft.references = photosWereClean
        ? state.saved.references.map(ref => ({ ...ref }))
        : ordered(state.draft.references.filter(ref => ref.source_type !== 'frame' || !left.has(ref.source)));
}

function applyListing(listing) {
    state.viewMode = listing.view_mode;
    state.frames = listing.frames;
    state.byName = new Map(listing.frames.map(frame => [frame.filename, frame]));
    state.people = listing.people;
    state.selected = new Set([...state.selected].filter(name => state.byName.has(name)));
}

/**
 * While creating, recompute which face of each frame is the person being built
 * and, until faces are chosen by hand, which faces the AI gets.
 */
function scheduleMatch() {
    clearTimeout(matchTimer);
    matchTimer = setTimeout(async () => {
        const members = [...state.members];
        if (!state.facesTouched) chooseFacesAutomatically({ quiet: true });
        let matches;
        try {
            ({ matches } = await api(`/api/people/${videoId}/frames/match`, {
                method: 'POST', body: { view_mode: state.viewMode, filenames: members }
            }));
        } catch (error) {
            ThumbnailApp.showToast(t('person_editor.load_error', { error: error.message }), 'error');
            return;
        }
        // A newer change will bring its own matches
        if (!sameList(members, [...state.members])) return;
        for (const frame of state.frames) {
            const match = matches[frame.filename];
            frame.face_box = match ? match.face_box : null;
            frame.similarity = match ? match.similarity : null;
            if (match) {
                frame.quality_score = match.quality_score;
                frame.expression = match.expression;
            }
        }
        renderAll();
    }, MATCH_DELAY_MS);
}

// =========================================================================
// FRAMES: FILTER, SORT, RENDER
// =========================================================================

function inScope(frame, scope) {
    const isMember = state.members.has(frame.filename);
    if (scope === 'members') return isMember;
    if (!state.showFaceless && !frame.face_box) return false;
    if (scope === 'unassigned') return !isMember && frame.people.length === 0;
    return true;
}

function sortOrder() {
    if (state.sort) return state.sort;
    if (state.scope === 'members') return 'quality';
    return hasPersonContext() ? 'similar' : 'quality';
}

function byNullableDesc(key) {
    return (a, b) => (b[key] ?? -Infinity) - (a[key] ?? -Infinity);
}

function sortFrames(frames) {
    const order = sortOrder();
    const byTime = (a, b) => a.filename.localeCompare(b.filename);
    const sorted = [...frames];
    if (order === 'time') {
        sorted.sort(byTime);
    } else if (order === 'quality') {
        sorted.sort((a, b) => byNullableDesc('quality_score')(a, b) || byTime(a, b));
    } else if (order === 'similar') {
        sorted.sort((a, b) => byNullableDesc('similarity')(a, b) || byTime(a, b));
    } else {
        sorted.sort((a, b) => (a.similarity ?? Infinity) - (b.similarity ?? Infinity) || byTime(a, b));
    }
    return sorted;
}

function isSuspect(frame) {
    return state.members.has(frame.filename) && frame.similarity !== null && frame.similarity < SUSPECT_SIMILARITY;
}

function badgesMarkup(frame) {
    const name = frame.filename;
    const faceIndex = faces().findIndex(ref => ref.source_type === 'frame' && ref.source === name);
    const isBody = bodies().some(ref => ref.source_type === 'frame' && ref.source === name);
    const badges = [];
    if (faceIndex >= 0) {
        const title = t(faceIndex === 0 ? 'person_editor.badge_primary' : 'person_editor.badge_face');
        badges.push(`<span class="pe-badge pe-badge-ref" title="${esc(title)}">${ThumbnailApp.icon('star', 'icon-sm')}${faceIndex + 1}</span>`);
    }
    if (isBody) {
        badges.push(`<span class="pe-badge pe-badge-ref" title="${esc(t('person_editor.badge_body'))}">${ThumbnailApp.icon('shirt', 'icon-sm')}</span>`);
    }
    if (isSuspect(frame)) {
        badges.push(`<span class="pe-badge pe-badge-warn" title="${esc(t('person_editor.suspect'))}">${ThumbnailApp.icon('alert', 'icon-sm')}</span>`);
    }
    return badges.join('');
}

/**
 * Refresh the photo badges of the cards without re-rendering the grid.
 */
function updateBadges() {
    document.querySelectorAll('#peGrid .pe-frame').forEach(card => {
        card.querySelector('.pe-badges').innerHTML = badgesMarkup(state.byName.get(card.dataset.name));
    });
}

function frameMarkup(frame, people) {
    const name = frame.filename;
    const isMember = state.members.has(name);
    const others = frame.people.filter(id => id !== personId).map(id => people.get(id)).filter(Boolean);
    const square = state.view === 'faces';

    const classes = ['pe-frame'];
    if (state.selected.has(name)) classes.push('selected');
    if (isMember && state.scope !== 'members') classes.push('member');

    let tag = '';
    if (isMember && state.scope !== 'members') {
        tag = `<span class="pe-tag pe-tag-self">${esc(t('person_editor.this_person'))}</span>`;
    } else if (others.length > 0) {
        // On a frame of this person the other names are extra faces, not its owner
        const key = isMember ? 'person_editor.also_in' : 'person_editor.in_person';
        const title = t(key, { name: others.map(personName).join(', ') });
        const extra = others.length > 1 ? ` +${others.length - 1}` : '';
        tag = `<span class="pe-tag" title="${esc(title)}">${esc(t(key, { name: personName(others[0]) }))}${extra}</span>`;
    }

    const similarity = frame.similarity !== null
        ? `<span class="pe-similarity" title="${esc(t('person_editor.similarity_hint'))}">${esc(t('person_editor.similarity_value', { value: Math.round(Math.max(frame.similarity, 0) * 100) }))}</span>`
        : '';

    return `
        <div class="${classes.join(' ')}" data-name="${esc(name)}">
            <button type="button" class="pe-frame-select" aria-pressed="${state.selected.has(name)}"
                    aria-label="${esc(sceneLabel(frame))}">
                ${thumbMarkup(frame, { square })}
                <span class="pe-check" aria-hidden="true">${ThumbnailApp.icon('check', 'icon-sm')}</span>
                <span class="pe-badges">${badgesMarkup(frame)}</span>
                ${tag}
            </button>
            <div class="pe-frame-meta">
                <span>${esc(sceneLabel(frame))}</span>
                ${similarity}
            </div>
            <button type="button" class="icon-btn pe-frame-zoom" title="${esc(t('person_editor.open_viewer'))}"
                    aria-label="${esc(t('person_editor.open_viewer'))}">${ThumbnailApp.icon('maximize', 'icon-sm')}</button>
        </div>
    `;
}

function renderGrid() {
    const grid = document.getElementById('peGrid');
    const people = peopleById();
    const frames = sortFrames(state.frames.filter(frame => inScope(frame, state.scope)));
    grid.classList.toggle('view-frames', state.view === 'frames');

    if (state.groupScenes) {
        const groups = new Map();
        for (const frame of frames) {
            const key = frame.scene_index ?? -1;
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push(frame);
        }
        const keys = [...groups.keys()].sort((a, b) => (a === -1) - (b === -1) || a - b);
        state.visible = keys.flatMap(key => groups.get(key).map(frame => frame.filename));
        grid.innerHTML = keys.map(key => {
            const sceneFrames = groups.get(key);
            return `
                <section class="pe-scene" data-scene="${key}">
                    <header class="pe-scene-head">
                        <label class="pe-scene-select">
                            <input type="checkbox" aria-label="${esc(t('person_editor.scene_select'))}">
                            <span class="pe-scene-title">${esc(sceneLabel(sceneFrames[0]))}</span>
                        </label>
                        <span class="pe-scene-count">${esc(t('person_editor.frames_count', { count: sceneFrames.length }))}</span>
                    </header>
                    <div class="pe-grid-items">${sceneFrames.map(frame => frameMarkup(frame, people)).join('')}</div>
                </section>
            `;
        }).join('');
    } else {
        state.visible = frames.map(frame => frame.filename);
        grid.innerHTML = `<div class="pe-grid-items">${frames.map(frame => frameMarkup(frame, people)).join('')}</div>`;
    }

    renderEmptyState(frames.length);
    renderFacelessToggle();
    syncSelectionMarks();
}

function renderEmptyState(count) {
    const empty = document.getElementById('peEmpty');
    empty.classList.toggle('hidden', count > 0);
    if (count > 0) return;

    let message;
    let action = '';
    if (state.scope === 'members') {
        message = t(isNew ? 'person_editor.empty_members_new' : 'person_editor.empty_members');
        action = `<button type="button" class="btn btn-secondary btn-small" data-goto-scope="unassigned">${esc(t('person_editor.go_unassigned'))}</button>`;
    } else if (state.scope === 'unassigned') {
        message = t('person_editor.empty_unassigned');
        action = `<button type="button" class="btn btn-secondary btn-small" data-goto-scope="all">${esc(t('person_editor.go_all'))}</button>`;
    } else {
        message = t('person_editor.empty_all');
    }
    empty.innerHTML = `${ThumbnailApp.icon('image')}<p>${esc(message)}</p>${action}`;
}

function renderFacelessToggle() {
    const wrapper = document.getElementById('peFaceless');
    const button = document.getElementById('peShowFaceless');
    const faceless = state.scope === 'members' ? 0 : state.frames.filter(frame => {
        if (frame.face_box) return false;
        return state.scope === 'all' || (!state.members.has(frame.filename) && frame.people.length === 0);
    }).length;
    wrapper.classList.toggle('hidden', faceless === 0);
    button.textContent = state.showFaceless
        ? t('person_editor.faceless_hide')
        : t('person_editor.faceless_show', { count: faceless });
}

function renderToolbar() {
    const counts = {
        members: state.frames.filter(frame => inScope(frame, 'members')).length,
        unassigned: state.frames.filter(frame => inScope(frame, 'unassigned')).length,
        all: state.frames.filter(frame => inScope(frame, 'all')).length,
    };
    document.querySelectorAll('#peScopes button').forEach(button => {
        const active = button.dataset.scope === state.scope;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
        button.querySelector('[data-count]').textContent = counts[button.dataset.scope];
    });

    const sort = document.getElementById('peSort');
    const withPerson = hasPersonContext();
    sort.innerHTML = [
        withPerson ? ['similar', 'person_editor.sort_similar'] : null,
        withPerson ? ['dissimilar', 'person_editor.sort_dissimilar'] : null,
        ['quality', 'person_editor.sort_quality'],
        ['time', 'person_editor.sort_time'],
    ].filter(Boolean).map(([value, key]) => `<option value="${value}">${esc(t(key))}</option>`).join('');
    sort.value = sortOrder();

    const group = document.getElementById('peGroupScenes');
    group.classList.toggle('active', state.groupScenes);
    group.setAttribute('aria-pressed', String(state.groupScenes));

    document.querySelectorAll('#peViewToggle button').forEach(button => {
        const active = button.dataset.view === state.view;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
    });

    document.getElementById('peScopeHint').innerHTML = scopeHint();
}

function scopeHint() {
    if (state.scope === 'members') {
        const suspects = state.frames.filter(isSuspect).length;
        const hint = esc(t(isNew ? 'person_editor.hint_members_new' : 'person_editor.hint_members'));
        if (suspects === 0) return hint;
        return `${hint} <button type="button" class="pe-hint-link" data-show-suspects>${esc(t('person_editor.suspects', { count: suspects }))}</button>`;
    }
    return esc(t(state.scope === 'unassigned' ? 'person_editor.hint_unassigned' : 'person_editor.hint_all'));
}

// =========================================================================
// PANEL: THE PERSON
// =========================================================================

function defaultName() {
    return isNew ? t('person_editor.new_person') : personName(state.person);
}

function currentName() {
    return state.draft.label.trim() || defaultName();
}

function renderName() {
    const name = currentName();
    document.getElementById('pePersonName').textContent = name;
    document.getElementById('breadcrumbPerson').textContent = name;
    document.title = t('person_editor.page_title', { name });
}

function renderPanel() {
    renderName();

    const viewLabel = t(state.viewMode === 'person_scene' ? 'person_editor.view_person_scene' : 'person_editor.view_person');
    document.getElementById('pePersonMeta').textContent =
        `${t('person_editor.in_frames', { count: state.members.size })} · ${viewLabel}`;

    const label = document.getElementById('peLabel');
    label.placeholder = defaultName();
    if (label.value !== state.draft.label) label.value = state.draft.label;
    const notes = document.getElementById('peNotes');
    if (notes.value !== state.draft.description) notes.value = state.draft.description;
    document.getElementById('peNotesPreview').textContent = state.draft.description.trim()
        ? `· ${state.draft.description.trim()}`
        : '';
    document.getElementById('peCollage').checked = state.draft.format === 'collage';

    renderAvatar();
    renderGuide();
    renderSlots();
    renderSaveState();
}

function renderAvatar() {
    const avatar = document.getElementById('peAvatar');
    const primary = faces()[0];
    const key = primary ? previewKey(primary) : '';
    if (avatar.dataset.key === key) return;
    avatar.dataset.key = key;
    avatar.innerHTML = ThumbnailApp.icon('user');
    if (!primary) return;
    referencePreview(primary).then(({ url }) => {
        if (avatar.dataset.key === key) avatar.innerHTML = `<img src="${url}" alt="">`;
    }).catch(() => {
        // The face slot says what is wrong with the photo
    });
}

/**
 * While creating: mark the step of the guide the person is at.
 */
function renderGuide() {
    const guide = document.getElementById('peGuide');
    if (!guide) return;
    const steps = ['find', 'add', 'create'];
    let current = 'find';
    if (state.members.size > 0) current = 'create';
    else if (selectedOutsiders().length > 0) current = 'add';
    guide.querySelectorAll('[data-step]').forEach(item => {
        const position = steps.indexOf(item.dataset.step) - steps.indexOf(current);
        item.classList.toggle('done', position < 0);
        item.classList.toggle('current', position === 0);
    });
}

function tagsText(ref) {
    if (ref.role === 'body') {
        return ref.coverage && ref.coverage !== 'unknown' ? t(`person_editor.coverage_${ref.coverage}`) : '';
    }
    return [
        ref.view ? t(`person_editor.angle_${ref.view}`) : null,
        ref.expression ? t(`person_editor.expression_${ref.expression}`) : null,
    ].filter(Boolean).join(' · ');
}

function slotMarkup(ref) {
    const actions = [];
    if (ref.role === 'face' && !ref.primary) {
        actions.push(['primary', 'star', 'person_editor.make_primary']);
    }
    actions.push(['crop', 'crop', 'person_editor.adjust_crop'], ['remove', 'x', 'person_editor.remove_photo']);
    const tags = tagsText(ref);
    const origin = ref.source_type === 'upload' ? t('person_editor.uploaded') : '';
    const viewControl = ref.role === 'body' ? `
        <select class="pe-body-view" data-body-view-select aria-label="${esc(t('person_editor.body_view'))}">
            <option value="" ${!ref.view ? 'selected' : ''}>${esc(t('person_editor.angle_unknown'))}</option>
            ${['front', 'three_quarter', 'profile', 'back'].map(view =>
                `<option value="${view}" ${ref.view === view ? 'selected' : ''}>${esc(t(`person_editor.angle_${view}`))}</option>`).join('')}
        </select>` : '';
    return `
        <li class="pe-slot filled${ref.primary ? ' primary' : ''}" data-ref="${esc(ref.id)}" data-role="${ref.role}">
            <span class="pe-slot-image"><img alt=""></span>
            ${ref.primary ? `<span class="pe-slot-badge">${ThumbnailApp.icon('star', 'icon-sm')}${esc(t('person_editor.primary'))}</span>` : ''}
            <span class="pe-slot-actions">
                ${actions.map(([action, icon, key]) => `
                    <button type="button" class="pe-slot-action" data-action="${action}" title="${esc(t(key))}" aria-label="${esc(t(key))}">${ThumbnailApp.icon(icon, 'icon-sm')}</button>
                `).join('')}
            </span>
            ${viewControl}
            <span class="pe-slot-tags">${esc([tags, origin].filter(Boolean).join(' · '))}</span>
            <span class="pe-slot-error hidden" role="alert"></span>
        </li>
    `;
}

function emptySlotMarkup(role, view = null) {
    return `
        <li class="pe-slot empty">
            <button type="button" class="pe-slot-add" data-add="${role}" data-body-view="${view || ''}" aria-haspopup="menu"
                    title="${esc(t(role === 'face' ? 'person_editor.add_face_hint' : 'person_editor.add_body_hint'))}">
                ${ThumbnailApp.icon('plus')}<span>${esc(t(view ? `person_editor.angle_${view}` : 'person_editor.add_photo'))}</span>
                ${view ? `<span class="pe-optional">${esc(t('person_editor.optional'))}</span>` : ''}
            </button>
        </li>
    `;
}

function renderSlots() {
    const faceRefs = faces();
    const bodyRefs = bodies();
    const faceList = document.getElementById('peFaceSlots');
    const bodyList = document.getElementById('peBodySlots');
    // Empty slots up to the faces the automatic choice picks, then one more while they fit
    const emptyFaces = Math.min(Math.max(FACE_COUNT - faceRefs.length, 1), freeImages('face'));
    faceList.innerHTML = faceRefs.map(slotMarkup).join('')
        + Array.from({ length: emptyFaces }, () => emptySlotMarkup('face')).join('');
    const bodyRoom = freeImages('body') > 0;
    const extraBodies = [...bodyRefs];
    bodyList.innerHTML = ['front', 'profile', 'back'].map(view => {
        const index = extraBodies.findIndex(ref => ref.view === view || view === 'front' && ref.view === 'three_quarter');
        if (index >= 0) return slotMarkup(extraBodies.splice(index, 1)[0]);
        return bodyRoom ? emptySlotMarkup('body', view) : '';
    }).join('') + extraBodies.map(slotMarkup).join('') + (bodyRoom ? emptySlotMarkup('body') : '');
    renderBudget();

    for (const ref of [...faceRefs, ...bodyRefs]) {
        const slot = document.querySelector(`.pe-slot[data-ref="${CSS.escape(ref.id)}"]`);
        referencePreview(ref).then(({ url }) => {
            if (slot.isConnected) slot.querySelector('img').src = url;
        }).catch(error => {
            if (!slot.isConnected) return;
            slot.classList.add('failed');
            const message = slot.querySelector('.pe-slot-error');
            message.textContent = error.message;
            message.classList.remove('hidden');
        });
    }

    const note = document.getElementById('peFacesNote');
    let text = '';
    if (faceRefs.length === 0) {
        text = t(isNew && state.members.size === 0 ? 'person_editor.faces_after_frames' : 'person_editor.no_faces');
    } else if (isNew && !state.facesTouched) {
        text = t('person_editor.faces_follow_frames');
    }
    note.textContent = text;
    note.classList.toggle('hidden', !text);
    note.classList.toggle('warning', faceRefs.length === 0 && state.members.size > 0);
    document.getElementById('peAutoFaces').disabled = state.members.size === 0;
}

/**
 * "Uses 4 of the 13 images of each thumbnail", a warning when they do not fit.
 */
function renderBudget() {
    const line = document.getElementById('peBudget');
    const count = imagesUsed();
    const max = imageBudget();
    const key = count > max ? 'budget_over' : state.settingPhotos > 0 ? 'budget_setting' : 'budget';
    line.textContent = t(`person_editor.${key}`, { count, max });
    line.classList.toggle('warning', count > max);
}

function saveBlocker() {
    if (isNew && state.members.size === 0) return t('person_editor.create_needs_frames');
    if (faces().length === 0) return t('person_editor.needs_face');
    return '';
}

function renderSaveState() {
    const dirty = isDirty();
    const reason = saveBlocker();
    document.getElementById('peDirty').classList.toggle('hidden', !dirty || isNew);
    document.getElementById('peDiscard').classList.toggle('hidden', !dirty || isNew);
    document.getElementById('peSave').disabled = state.busy || !dirty || reason !== '';
    // Why the button cannot be used, where the user is looking
    const why = document.getElementById('peSaveReason');
    why.classList.toggle('hidden', reason === '' || (!dirty && !isNew));
    why.textContent = reason;
}

async function save() {
    if (state.busy) return;
    state.busy = true;
    renderSaveState();
    const body = {
        label: state.draft.label,
        description: state.draft.description,
        references: state.draft.references,
        reference_format: state.draft.format,
        revision: state.revision,
    };
    try {
        if (isNew) {
            const person = await api(`/api/people/${videoId}`, {
                method: 'POST',
                body: { ...body, view_mode: state.viewMode, members: [...state.members] }
            });
            // Leave without the unsaved-changes prompt
            state.members.clear();
            state.draft = emptyDraft();
            window.location.replace(`/video/${videoId}/person/${person.id}?created=1`);
            return;
        }
        const person = await api(`/api/people/${videoId}/${personId}`, { method: 'PUT', body });
        applyPerson(person, { resetDraft: true });
        ThumbnailApp.showToast(t('person_editor.saved'), 'success');
    } catch (error) {
        ThumbnailApp.showToast(t('person_editor.save_error', { error: error.message }), 'error', {
            label: t('person_editor.reload'),
            onClick: () => window.location.reload(),
        });
    } finally {
        state.busy = false;
        renderAll();
    }
}

function discard() {
    state.draft = copyDraft(state.saved);
    renderAll();
}

// =========================================================================
// WHAT THE IMAGE MODEL GETS
// =========================================================================

function previewCaption(entry) {
    if (entry.reference_role === 'face_sheet') return t('person_editor.preview_collage');
    if (entry.reference_role === 'body') return t('person_editor.preview_body');
    return t(entry.is_primary ? 'person_editor.preview_primary' : 'person_editor.preview_face');
}

async function openPreview() {
    const dialog = document.getElementById('pePreviewDialog');
    const grid = document.getElementById('pePreviewGrid');
    document.getElementById('pePreviewTitle').textContent = t('person_editor.preview_title', { name: currentName() });
    grid.innerHTML = `<div class="loading">${esc(t('common.loading'))}</div>`;
    dialog.classList.add('visible');
    const { image } = AppSettings.get();
    try {
        const result = await api(`/api/people/${videoId}/references/preview`, {
            method: 'POST',
            body: {
                cluster_ids: [draftPersonId],
                references: state.draft.references,
                formats: { [draftPersonId]: state.draft.format },
                include_environment: false,
                image_provider: image.provider,
                image_model: image.model,
                draft_person: isNew,
                thumbnail: PREVIEW_THUMBNAIL,
            }
        });
        grid.innerHTML = result.images.map(entry => `
            <figure class="references-preview-item">
                <img src="${entry.image}" alt="">
                <figcaption>${esc(previewCaption(entry))}</figcaption>
            </figure>
        `).join('');
    } catch (error) {
        grid.innerHTML = `<p class="form-hint form-warning">${ThumbnailApp.icon('alert', 'icon-sm')}<span>${esc(error.message)}</span></p>`;
    }
}

function closePreview() {
    document.getElementById('pePreviewDialog').classList.remove('visible');
}

// =========================================================================
// SELECTION
// =========================================================================

function selectedMembers() {
    return [...state.selected].filter(name => state.members.has(name));
}

function selectedOutsiders() {
    return [...state.selected].filter(name => !state.members.has(name));
}

function setSelected(names, selected) {
    for (const name of names) {
        if (selected) state.selected.add(name);
        else state.selected.delete(name);
    }
    syncSelectionMarks();
}

function clearSelection() {
    state.selected.clear();
    state.anchor = null;
    syncSelectionMarks();
}

function onFrameClick(name, event) {
    if (event.shiftKey && state.anchor && state.visible.includes(state.anchor)) {
        const from = state.visible.indexOf(state.anchor);
        const to = state.visible.indexOf(name);
        const range = state.visible.slice(Math.min(from, to), Math.max(from, to) + 1);
        // The range takes the state of the frame it starts from
        setSelected(range, state.selected.has(state.anchor));
    } else {
        setSelected([name], !state.selected.has(name));
    }
    state.anchor = name;
}

/**
 * Reflect the selection on the cards, the checkboxes and the action bar
 * without re-rendering the grid.
 */
function syncSelectionMarks() {
    document.querySelectorAll('#peGrid .pe-frame').forEach(card => {
        const selected = state.selected.has(card.dataset.name);
        card.classList.toggle('selected', selected);
        card.querySelector('.pe-frame-select').setAttribute('aria-pressed', String(selected));
    });

    document.querySelectorAll('#peGrid .pe-scene').forEach(section => {
        const names = [...section.querySelectorAll('.pe-frame')].map(card => card.dataset.name);
        const count = names.filter(name => state.selected.has(name)).length;
        const checkbox = section.querySelector('.pe-scene-select input');
        checkbox.checked = count > 0 && count === names.length;
        checkbox.indeterminate = count > 0 && count < names.length;
    });

    const all = document.getElementById('peSelectAll');
    const count = state.visible.filter(name => state.selected.has(name)).length;
    all.checked = count > 0 && count === state.visible.length;
    all.indeterminate = count > 0 && count < state.visible.length;
    all.disabled = state.visible.length === 0;

    renderSelectionBar();
    if (state.lightboxIndex >= 0) renderLightbox();
}

function setAction(id, count, key, { visible = count > 0, disabled = false, title = '' } = {}) {
    const button = document.getElementById(id);
    button.classList.toggle('hidden', !visible);
    button.disabled = disabled || state.busy;
    button.title = title;
    button.querySelector('span').textContent = t(key, { count });
}

function renderSelectionBar() {
    const bar = document.getElementById('peSelectionBar');
    const total = state.selected.size;
    bar.classList.toggle('hidden', total === 0);
    document.body.classList.toggle('pe-has-selection', total > 0);
    // Selecting faces is the second step of creating a person
    renderGuide();
    if (total === 0) return;

    document.getElementById('peSelectionCount').textContent = t('person_editor.selected', { count: total });

    const members = selectedMembers();
    const outsiders = selectedOutsiders();
    const emptiesPerson = !isNew && members.length > 0 && members.length === state.members.size;
    const keepOne = emptiesPerson ? t('person_editor.keep_one_frame') : '';

    setAction('peActAdd', outsiders.length, 'person_editor.act_add');
    setAction('peActFace', members.length, 'person_editor.act_face');
    setAction('peActBody', 1, 'person_editor.act_body', { visible: total === 1 && (members.length === 1 || state.pendingBodyView === 'back') });
    setAction('peActMove', members.length, 'person_editor.act_move', {
        visible: !isNew && members.length > 0, disabled: emptiesPerson, title: keepOne
    });
    setAction('peActRemove', members.length, 'person_editor.act_remove', { disabled: emptiesPerson, title: keepOne });
    setAction('peActDelete', total, 'person_editor.act_delete', { disabled: emptiesPerson, title: keepOne });
}

// =========================================================================
// ACTIONS ON FRAMES
// =========================================================================

/**
 * Run an edit against the server, then reload the person and its frames.
 */
async function runEdit(request, onDone) {
    if (state.busy) return;
    state.busy = true;
    renderSelectionBar();
    try {
        const result = await request();
        clearSelection();
        await load();
        onDone(result);
    } catch (error) {
        ThumbnailApp.showToast(t('person_editor.action_error', { error: error.message }), 'error');
    } finally {
        state.busy = false;
        renderAll();
    }
}

function changeMembersLocally(names, add) {
    for (const name of names) {
        if (add) state.members.add(name);
        else state.members.delete(name);
    }
    if (!add) {
        const kept = state.draft.references.filter(ref => ref.source_type !== 'frame' || state.members.has(ref.source));
        state.draft.references = ordered(kept);
    }
    clearSelection();
    renderAll();
    scheduleMatch();
}

function addSelected() {
    const names = selectedOutsiders();
    if (isNew) {
        changeMembersLocally(names, true);
        return;
    }
    runEdit(
        () => api(`/api/people/${videoId}/${personId}/frames/add`, { method: 'POST', body: { filenames: names } }),
        () => ThumbnailApp.showToast(t('person_editor.added', { count: names.length }), 'success')
    );
}

function removeSelected() {
    const names = selectedMembers();
    if (isNew) {
        changeMembersLocally(names, false);
        return;
    }
    runEdit(
        () => api(`/api/people/${videoId}/${personId}/frames/remove`, { method: 'POST', body: { filenames: names } }),
        () => ThumbnailApp.showToast(t('person_editor.removed', { count: names.length }), 'success', {
            label: t('common.undo'),
            onClick: () => runEdit(
                () => api(`/api/people/${videoId}/${personId}/frames/add`, { method: 'POST', body: { filenames: names } }),
                () => ThumbnailApp.showToast(t('person_editor.restored', { count: names.length }), 'success')
            )
        })
    );
}

function openMoveMenu() {
    const names = selectedMembers();
    const targets = state.people.filter(person => person.id !== personId);
    const move = (targetId) => runEdit(
        () => api(`/api/people/${videoId}/${personId}/frames/move`, {
            method: 'POST', body: { filenames: names, target_id: targetId }
        }),
        (result) => {
            const target = result.target;
            ThumbnailApp.showToast(t('person_editor.moved', { count: names.length, name: personName(target) }), 'success', {
                label: t('person_editor.open_person'),
                onClick: () => { window.location.href = `/video/${videoId}/person/${target.id}`; }
            });
        }
    );
    ThumbnailApp.showMenu(document.getElementById('peActMove'), [
        ...targets.map(person => ({
            label: `${personName(person)} · ${t('person_editor.frames_count', { count: person.num_frames })}`,
            icon: 'user',
            onSelect: () => move(person.id)
        })),
        { label: t('person_editor.move_new'), icon: 'plus', onSelect: () => move(null) }
    ]);
}

async function deleteSelected() {
    const names = [...state.selected];
    const confirmed = await ThumbnailApp.showModal({
        title: t('person_editor.delete_title', { count: names.length }),
        message: t('person_editor.delete_message'),
        type: 'danger',
        confirmText: t('person_editor.delete_confirm'),
    });
    if (!confirmed) return;

    if (isNew) {
        // Deleted frames cannot stay in the person being built
        names.forEach(name => state.members.delete(name));
        state.draft.references = ordered(state.draft.references.filter(ref => ref.source_type !== 'frame' || !names.includes(ref.source)));
        scheduleMatch();
    }
    runEdit(
        () => api(`/api/people/${videoId}/frames/delete`, { method: 'POST', body: { filenames: names } }),
        (result) => {
            ThumbnailApp.showToast(t('person_editor.deleted', { count: result.deleted }), 'success');
            if (result.deleted_people.length > 0) {
                ThumbnailApp.showToast(t('person_editor.deleted_people', { count: result.deleted_people.length }), 'info');
            }
        }
    );
}

/**
 * "+" of an empty photo slot: from the person's frames or from the computer.
 */
function openAddMenu(button) {
    const role = button.dataset.add;
    const view = role === 'body' ? button.dataset.bodyView || null : null;
    ThumbnailApp.showMenu(button, [
        {
            label: t('person_editor.add_from_frames'),
            icon: 'image',
            onSelect: () => {
                state.pendingBodyView = view;
                if (view === 'back') {
                    state.showFaceless = true;
                    setScope('all');
                    renderAll();
                } else if (state.scope !== 'members') setScope('members');
                ThumbnailApp.showToast(t(role === 'face' ? 'person_editor.add_face_hint' : 'person_editor.add_body_hint'), 'info');
            }
        },
        { label: t('person_editor.add_upload'), icon: 'upload', onSelect: () => chooseUpload(role, view) },
    ]);
}

// =========================================================================
// LIGHTBOX
// =========================================================================

function openLightbox(name) {
    state.lightboxIndex = state.visible.indexOf(name);
    document.getElementById('peLightbox').classList.remove('hidden');
    document.body.classList.add('lightbox-open');
    renderLightbox();
    // Focus the dialog itself, so Space selects instead of pressing a button
    document.getElementById('peLightbox').focus();
}

function closeLightbox() {
    const name = state.visible[state.lightboxIndex];
    state.lightboxIndex = -1;
    document.getElementById('peLightbox').classList.add('hidden');
    document.body.classList.remove('lightbox-open');
    if (name) {
        document.querySelector(`#peGrid .pe-frame[data-name="${CSS.escape(name)}"] .pe-frame-select`)?.focus();
    }
}

function stepLightbox(delta) {
    const next = state.lightboxIndex + delta;
    if (next < 0 || next >= state.visible.length) return;
    state.lightboxIndex = next;
    renderLightbox();
}

function renderLightbox() {
    const name = state.visible[state.lightboxIndex];
    const frame = state.byName.get(name);
    if (!frame) {
        closeLightbox();
        return;
    }

    const image = document.getElementById('peLightboxImage');
    const face = document.getElementById('peLightboxFace');
    if (image.dataset.name !== name) {
        image.dataset.name = name;
        face.classList.add('hidden');
        image.src = frameImageUrl(name);
    }
    const placeFace = () => {
        if (!frame.face_box || !image.naturalWidth) return;
        const [x1, y1, x2, y2] = frame.face_box;
        face.style.left = `${(x1 / image.naturalWidth) * 100}%`;
        face.style.top = `${(y1 / image.naturalHeight) * 100}%`;
        face.style.width = `${((x2 - x1) / image.naturalWidth) * 100}%`;
        face.style.height = `${((y2 - y1) / image.naturalHeight) * 100}%`;
        face.classList.remove('hidden');
    };
    image.onload = placeFace;
    if (image.complete) placeFace();

    const people = peopleById();
    const holders = frame.people.map(id => people.get(id)).filter(Boolean).map(personName);
    const parts = [sceneLabel(frame), t('person_editor.info_frame', { n: frame.frame_number ?? '–' })];
    if (frame.quality_score !== null) parts.push(t('person_editor.info_quality', { value: Math.round(frame.quality_score) }));
    if (frame.similarity !== null) parts.push(t('person_editor.info_similarity', { value: Math.round(Math.max(frame.similarity, 0) * 100) }));
    parts.push(holders.length ? t('person_editor.in_person', { name: holders.join(', ') }) : t('person_editor.in_nobody'));
    document.getElementById('peLightboxInfo').innerHTML =
        `<strong>${state.lightboxIndex + 1} / ${state.visible.length}</strong> · ${parts.map(esc).join(' · ')}`;

    const selected = state.selected.has(name);
    const select = document.getElementById('peLightboxSelect');
    select.setAttribute('aria-pressed', String(selected));
    select.classList.toggle('btn-primary', selected);
    select.classList.toggle('btn-secondary', !selected);
    select.querySelector('span').textContent = t(selected ? 'person_editor.viewer_selected' : 'person_editor.viewer_select');

    document.getElementById('peLightboxPrev').disabled = state.lightboxIndex === 0;
    document.getElementById('peLightboxNext').disabled = state.lightboxIndex === state.visible.length - 1;
}

// =========================================================================
// SETUP
// =========================================================================

function renderAll() {
    renderToolbar();
    renderGrid();
    renderPanel();
}

function setScope(scope) {
    if (scope === state.scope) return;
    state.scope = scope;
    state.sort = null;
    clearSelection();
    renderAll();
}

function setupToolbar() {
    document.getElementById('peScopes').addEventListener('click', (e) => {
        const button = e.target.closest('button[data-scope]');
        if (button) setScope(button.dataset.scope);
    });

    document.getElementById('peSort').addEventListener('change', (e) => {
        state.sort = e.target.value;
        renderGrid();
    });

    document.getElementById('peGroupScenes').addEventListener('click', () => {
        state.groupScenes = !state.groupScenes;
        renderToolbar();
        renderGrid();
    });

    document.getElementById('peViewToggle').addEventListener('click', (e) => {
        const button = e.target.closest('button[data-view]');
        if (!button || button.dataset.view === state.view) return;
        state.view = button.dataset.view;
        storeView(state.view);
        renderToolbar();
        renderGrid();
    });

    document.getElementById('peSelectAll').addEventListener('change', (e) => {
        setSelected(state.visible, e.target.checked);
    });

    document.getElementById('peShowFaceless').addEventListener('click', () => {
        state.showFaceless = !state.showFaceless;
        renderAll();
    });

    document.getElementById('peScopeHint').addEventListener('click', (e) => {
        if (e.target.closest('[data-show-suspects]')) {
            state.sort = 'dissimilar';
            renderToolbar();
            renderGrid();
        }
    });
}

function setupGrid() {
    const grid = document.getElementById('peGrid');

    grid.addEventListener('click', (e) => {
        const card = e.target.closest('.pe-frame');
        if (card && e.target.closest('.pe-frame-zoom')) {
            openLightbox(card.dataset.name);
        } else if (card && e.target.closest('.pe-frame-select')) {
            onFrameClick(card.dataset.name, e);
        }
    });

    grid.addEventListener('dblclick', (e) => {
        const card = e.target.closest('.pe-frame');
        if (card) {
            openLightbox(card.dataset.name);
        }
    });

    grid.addEventListener('change', (e) => {
        const section = e.target.closest('.pe-scene');
        if (!section || !e.target.matches('.pe-scene-select input')) return;
        const names = [...section.querySelectorAll('.pe-frame')].map(card => card.dataset.name);
        setSelected(names, e.target.checked);
    });

    // Images load lazily; face crops need their natural size
    document.addEventListener('load', (e) => {
        if (e.target.matches?.('.pe-thumb img')) applyFaceCrop(e.target);
    }, true);

    document.getElementById('peEmpty').addEventListener('click', (e) => {
        const button = e.target.closest('[data-goto-scope]');
        if (button) setScope(button.dataset.gotoScope);
    });
}

function setupPanel() {
    document.getElementById('peLabel').addEventListener('input', (e) => {
        state.draft.label = e.target.value;
        renderName();
        renderSaveState();
    });
    document.getElementById('peNotes').addEventListener('input', (e) => {
        state.draft.description = e.target.value;
        document.getElementById('peNotesPreview').textContent = e.target.value.trim() ? `· ${e.target.value.trim()}` : '';
        renderSaveState();
    });
    document.getElementById('peCollage').addEventListener('change', (e) => {
        state.draft.format = e.target.checked ? 'collage' : 'photos';
        renderSlots();
        renderSaveState();
    });
    // The images of each thumbnail come from Settings
    document.addEventListener('settings:changed', renderSlots);
    document.getElementById('peSave').addEventListener('click', save);
    document.getElementById('peDiscard').addEventListener('click', discard);
    document.getElementById('peAutoFaces').addEventListener('click', () => {
        state.facesTouched = !isNew;
        chooseFacesAutomatically();
    });
    document.getElementById('peAutoBody')?.addEventListener('click', chooseBodyAutomatically);
    document.getElementById('pePreview').addEventListener('click', openPreview);

    const slots = (e) => {
        const add = e.target.closest('[data-add]');
        if (add) {
            openAddMenu(add);
            return;
        }
        const action = e.target.closest('[data-action]');
        if (!action) return;
        const id = action.closest('[data-ref]').dataset.ref;
        if (action.dataset.action === 'primary') setPrimary(id);
        else if (action.dataset.action === 'crop') adjustCrop(id);
        else if (action.dataset.action === 'remove') removeReference(id);
    };
    document.getElementById('peFaceSlots').addEventListener('click', slots);
    document.getElementById('peBodySlots').addEventListener('click', slots);
    document.getElementById('peBodySlots').addEventListener('change', (event) => {
        if (!event.target.matches('[data-body-view-select]')) return;
        const id = event.target.closest('[data-ref]').dataset.ref;
        setReferences(state.draft.references.map(ref => ref.id === id ? { ...ref, view: event.target.value || null } : ref));
    });

    document.getElementById('peUploadInput').addEventListener('change', (e) => {
        const [file] = e.target.files;
        if (file) uploadPhoto(file, e.target.dataset.role, e.target.dataset.view || null);
    });

    const preview = document.getElementById('pePreviewDialog');
    preview.addEventListener('click', (e) => {
        if (e.target === preview || e.target.closest('[data-close-preview]')) closePreview();
    });
}

function setupSelectionBar() {
    document.getElementById('peActAdd').addEventListener('click', addSelected);
    document.getElementById('peActFace').addEventListener('click', useSelectedAsFaces);
    document.getElementById('peActBody').addEventListener('click', useSelectedAsBody);
    document.getElementById('peActMove').addEventListener('click', openMoveMenu);
    document.getElementById('peActRemove').addEventListener('click', removeSelected);
    document.getElementById('peActDelete').addEventListener('click', deleteSelected);
    document.getElementById('peActClear').addEventListener('click', clearSelection);
}

function setupLightbox() {
    document.getElementById('peLightboxClose').addEventListener('click', closeLightbox);
    document.getElementById('peLightboxPrev').addEventListener('click', () => stepLightbox(-1));
    document.getElementById('peLightboxNext').addEventListener('click', () => stepLightbox(1));
    document.getElementById('peLightboxSelect').addEventListener('click', () => {
        const name = state.visible[state.lightboxIndex];
        setSelected([name], !state.selected.has(name));
        state.anchor = name;
    });
    document.getElementById('peLightboxStage').addEventListener('click', (e) => {
        if (e.target.id === 'peLightboxStage') closeLightbox();
    });
}

function setupKeyboard() {
    document.addEventListener('keydown', (e) => {
        const typing = e.target.closest?.('input, textarea, select');

        // The crop dialog handles its own keys
        if (document.getElementById('cropDialog').classList.contains('visible')) return;

        if (document.getElementById('pePreviewDialog').classList.contains('visible')) {
            if (e.key === 'Escape') {
                e.preventDefault();
                closePreview();
            }
            return;
        }

        if (state.lightboxIndex >= 0) {
            if (e.key === 'Escape') closeLightbox();
            else if (e.key === 'ArrowLeft') stepLightbox(-1);
            else if (e.key === 'ArrowRight') stepLightbox(1);
            else if (e.key === ' ' && !e.target.closest('button')) document.getElementById('peLightboxSelect').click();
            else return;
            e.preventDefault();
            return;
        }

        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
            e.preventDefault();
            if (!document.getElementById('peSave').disabled) save();
        } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a' && !typing) {
            e.preventDefault();
            setSelected(state.visible, true);
        } else if (e.key === 'Escape' && !typing && state.selected.size > 0
                   && !document.querySelector('.modal-overlay.visible, .menu-popover')) {
            clearSelection();
        }
    });

    window.addEventListener('beforeunload', (e) => {
        if (isDirty()) {
            e.preventDefault();
            e.returnValue = '';
        }
    });
}

async function init() {
    ThumbnailApp.loadVideoBreadcrumb(videoId).catch((error) => {
        ThumbnailApp.showToast(t('person_editor.load_error', { error: error.message }), 'error');
    });
    setupToolbar();
    setupGrid();
    setupPanel();
    setupSelectionBar();
    setupLightbox();
    setupKeyboard();

    try {
        // The preview asks the image model chosen in Settings
        await Promise.all([load(), AppSettings.ready]);
    } catch (error) {
        document.getElementById('peGrid').innerHTML =
            `<div class="error">${esc(t('person_editor.load_error', { error: error.message }))}</div>`;
        return;
    }
    renderAll();

    const params = new URLSearchParams(window.location.search);
    if (params.has('created')) {
        ThumbnailApp.showToast(t('person_editor.created'), 'success');
        history.replaceState(null, '', window.location.pathname);
    }
}

init();
