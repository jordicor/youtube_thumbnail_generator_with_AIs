/**
 * Video Detail Page - People Picker Module
 * =========================================
 * Chooses who appears in the thumbnail (Generate tab). The selection is an
 * ordered list of clusters kept by database ID; the position is the
 * person_XX role (1 → person_01). Reorder by dragging or with Alt+Up/Down.
 * Each chosen person shows the photos the image model gets of them
 * (references.js fills the strip), and the expression field warns about anyone
 * with no face photo showing the chosen expression.
 */

import { state } from './state.js';
import { personPhotos } from './references.js';

// Switches the cluster view; provided by main.js to avoid a circular import
let setView = null;

/**
 * Initialize the picker.
 * @param {Object} options
 * @param {Function} options.setView - Switch the cluster view ('person' or 'person_scene')
 */
export function init(options) {
    setView = options.setView;

    document.getElementById('addPersonBtn').addEventListener('click', (e) => {
        e.stopPropagation();
        togglePicker();
    });
    document.getElementById('peoplePickerDone').addEventListener('click', () => togglePicker(false));

    document.getElementById('pickerViewToggle').querySelectorAll('button').forEach(button => {
        button.addEventListener('click', () => {
            if (button.dataset.view !== state.currentClusterType) {
                setView(button.dataset.view);
            }
        });
    });

    document.getElementById('peoplePickerGrid').addEventListener('click', (e) => {
        const option = e.target.closest('.picker-option');
        if (!option) return;
        // The grid re-renders, detaching the target: keep the outside-click handler out
        e.stopPropagation();
        const cluster = state.clusters.find(c => c.id === parseInt(option.dataset.id));
        toggle(cluster);
    });

    document.getElementById('peopleSelected').addEventListener('click', (e) => {
        const remove = e.target.closest('.person-remove');
        if (remove) {
            toggle(state.selectedPeople[parseInt(remove.dataset.index)]);
        }
    });

    // Close on outside click and on Escape
    document.addEventListener('click', (e) => {
        if (!document.getElementById('peoplePicker').contains(e.target)) {
            togglePicker(false);
        }
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            togglePicker(false);
        }
    });

    // The photos each person sends arrive after the chips
    document.addEventListener('references:changed', render);
    document.getElementById('expression').addEventListener('change', renderExpression);

    setupReordering();
    render();
}

/**
 * Get the IDs of the chosen people, in role order.
 * @returns {number[]} Cluster IDs
 */
export function getSelectedIds() {
    return state.selectedPeople.map(person => person.id);
}

/**
 * Whether a cluster is in the thumbnail.
 * @param {number} clusterId - Cluster database ID
 * @returns {boolean}
 */
export function isSelected(clusterId) {
    return state.selectedPeople.some(person => person.id === clusterId);
}

/**
 * Add a cluster to the thumbnail, or remove it if it is already there.
 * @param {Object} cluster - Cluster object from the clusters API
 */
export function toggle(cluster) {
    if (isSelected(cluster.id)) {
        state.selectedPeople = state.selectedPeople.filter(person => person.id !== cluster.id);
    } else {
        state.selectedPeople = [...state.selectedPeople, cluster];
    }
    changed();
}

/**
 * Bring the selection in line with freshly loaded clusters of the current view:
 * refresh chosen clusters of that view, drop the ones that no longer exist
 * (deleted or merged) and preselect the main person the first time.
 */
export function syncWithClusters() {
    const loadedById = new Map(state.clusters.map(c => [c.id, c]));
    state.selectedPeople = state.selectedPeople
        .filter(person => person.view_mode !== state.currentClusterType || loadedById.has(person.id))
        .map(person => loadedById.get(person.id) || person);

    if (!state.peopleInitialized) {
        state.peopleInitialized = true;
        // The main person: the first one that is not a picture (a painting, a poster)
        const main = state.clusters.find(cluster => !cluster.is_picture);
        if (state.selectedPeople.length === 0 && state.currentClusterType === 'person' && main) {
            state.selectedPeople = [main];
        }
    }

    changed();
    renderPickerGrid();
}

function changed() {
    render();
    renderPickerGrid();
    document.dispatchEvent(new CustomEvent('people:changed'));
}

// =========================================================================
// ROLES AND NAMES
// =========================================================================

/**
 * Role of the person at a position of the thumbnail: "Lead", "2nd", "3rd"...
 * @param {number} index - Position, 0 for the lead
 * @returns {string} Role label
 */
export function roleLabel(index) {
    return index === 0 ? t('generate_form.role_lead') : t('generate_form.role_nth', { n: index + 1 });
}

/**
 * Whether a person still has the name the analysis gave it.
 * @param {Object} cluster - Cluster object from the clusters API
 * @returns {boolean}
 */
export function isUnnamed(cluster) {
    return !cluster.label && !(cluster.view_mode === 'person_scene' && cluster.parent_label);
}

// =========================================================================
// PHOTOS SENT
// =========================================================================

/**
 * What a chosen person sends to the image model, as a short line.
 * @param {Object} person - Cluster object
 * @returns {{text: string, missing: boolean}} Line, and whether they have no face photo
 */
function photosSent(person) {
    const photos = personPhotos(person.id);
    const faces = photos ? photos.faces : person.reference_count;
    const bodies = photos ? photos.bodies : 0;
    if (faces === 0) return { text: t('generate_form.sends_no_face'), missing: true };
    const text = bodies > 0
        ? t('generate_form.sends_faces_body', { count: faces, bodies })
        : t('generate_form.sends_faces', { count: faces });
    return { text, missing: false };
}

function showsExpression(person, expression) {
    return (person.face_photo_expressions[expression] ?? 0) > 0;
}

// =========================================================================
// EXPRESSION
// =========================================================================

/**
 * "Ana", "Ana and Leo", "Ana, Leo and Eva".
 * @param {Object[]} people - Cluster objects
 * @returns {string} Their names
 */
function listNames(people) {
    const names = people.map(person => ThumbnailApp.getClusterDisplayName(person));
    return names.length === 1
        ? names[0]
        : t('generate_form.summary_and', { list: names.slice(0, -1).join(', '), last: names[names.length - 1] });
}

/**
 * Ask the expression question for the people chosen, say what the choice does
 * and warn about anyone the image model will not see with that expression.
 */
function renderExpression() {
    const count = state.selectedPeople.length;
    const expression = document.getElementById('expression').value;
    document.getElementById('expressionLabel').textContent = t('generate_form.expression_label', { count });
    document.getElementById('expressionHint').textContent = expression
        ? t('generate_form.expression_hint_fixed', { count })
        : t('generate_form.expression_hint_auto');

    const missing = expression
        ? state.selectedPeople.filter(person => !showsExpression(person, expression))
        : [];
    const warning = document.getElementById('expressionWarning');
    warning.classList.toggle('hidden', missing.length === 0);
    warning.innerHTML = missing.length === 0 ? '' : ThumbnailApp.icon('alert', 'icon-sm') +
        `<span>${escapeHtml(t(`generate_form.expression_missing_${expression}`, { names: listNames(missing) }))}</span>`;
}

// =========================================================================
// RENDERING
// =========================================================================

function clusterImage(cluster) {
    return ThumbnailApp.personFaceUrl(state.videoId, cluster);
}

function nameMarkup(cluster) {
    const name = escapeHtml(ThumbnailApp.getClusterDisplayName(cluster));
    if (!isUnnamed(cluster)) return `<span class="person-name">${name}</span>`;
    return `<span class="person-name unnamed">${name}</span>` +
        `<a class="link-btn person-name-link" href="/video/${state.videoId}/person/${cluster.id}">${escapeHtml(t('generate_form.name_them'))}</a>`;
}

export function render() {
    const list = document.getElementById('peopleSelected');
    renderExpression();

    if (state.selectedPeople.length === 0) {
        list.innerHTML = `<li class="people-empty">${escapeHtml(t('generate_form.no_people'))}</li>`;
        return;
    }

    list.innerHTML = state.selectedPeople.map((person, index) => {
        const sent = photosSent(person);
        const editor = `/video/${state.videoId}/person/${person.id}`;
        return `
            <li class="person-chip" draggable="true" tabindex="0" data-index="${index}"
                title="${escapeHtml(t('generate_form.reorder_hint'))}">
                <span class="drag-handle" aria-hidden="true">${ThumbnailApp.icon('grip')}</span>
                <img src="${clusterImage(person)}" alt="" loading="lazy" onerror="this.src='/static/img/placeholder.svg'">
                <span class="person-info">
                    <span class="person-title">
                        ${nameMarkup(person)}
                        <span class="person-role${index === 0 ? ' lead' : ''}">${escapeHtml(roleLabel(index))}</span>
                    </span>
                    <span class="person-meta${sent.missing ? ' person-warning' : ''}">
                        ${sent.missing ? ThumbnailApp.icon('alert', 'icon-sm') : ''}${escapeHtml(sent.text)}
                        <a class="link-btn" href="${editor}">${escapeHtml(t('generate_form.change_photos'))}</a>
                    </span>
                    <span class="person-photos" data-person-photos="${person.id}"></span>
                </span>
                <button type="button" class="icon-btn person-remove" data-index="${index}"
                        title="${escapeHtml(t('generate_form.remove_person'))}" aria-label="${escapeHtml(t('generate_form.remove_person'))}">
                    ${ThumbnailApp.icon('x', 'icon-sm')}
                </button>
            </li>
        `;
    }).join('');
    document.dispatchEvent(new CustomEvent('people:rendered'));
}

function renderPickerGrid() {
    const grid = document.getElementById('peoplePickerGrid');

    document.getElementById('pickerViewToggle').querySelectorAll('button').forEach(button => {
        const active = button.dataset.view === state.currentClusterType;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
    });
    document.getElementById('pickerSceneHint').classList.toggle('hidden', state.currentClusterType !== 'person_scene');

    if (state.clusters.length === 0) {
        grid.innerHTML = `<p class="people-empty">${escapeHtml(t('empty_state.no_clusters'))}</p>`;
        return;
    }

    grid.innerHTML = state.clusters.map(cluster => {
        const role = state.selectedPeople.findIndex(person => person.id === cluster.id);
        const name = escapeHtml(ThumbnailApp.getClusterDisplayName(cluster));
        return `
            <button type="button" class="picker-option ${role >= 0 ? 'selected' : ''}" data-id="${cluster.id}" aria-pressed="${role >= 0}">
                <img src="${clusterImage(cluster)}" alt="" loading="lazy" onerror="this.src='/static/img/placeholder.svg'">
                <span class="picker-option-name${isUnnamed(cluster) ? ' unnamed' : ''}" title="${name}">${name}</span>
                <span class="picker-option-meta">${escapeHtml(t('clusters.in_frames', { count: cluster.num_frames }))}${cluster.is_picture ? ` · ${escapeHtml(t('clusters.picture_short'))}` : ''}</span>
                ${role >= 0 ? `<span class="picker-option-role${role === 0 ? ' lead' : ''}">${escapeHtml(roleLabel(role))}</span>` : ''}
            </button>
        `;
    }).join('');
}

/**
 * Open or close the picker panel.
 * @param {boolean} [open] - Desired state (toggles when omitted)
 */
function togglePicker(open) {
    const popover = document.getElementById('peoplePickerPopover');
    const shouldOpen = open ?? popover.classList.contains('hidden');
    popover.classList.toggle('hidden', !shouldOpen);
    document.getElementById('addPersonBtn').setAttribute('aria-expanded', String(shouldOpen));
    if (shouldOpen) {
        renderPickerGrid();
    }
}

// =========================================================================
// REORDERING
// =========================================================================

function move(from, to) {
    if (to < 0 || to >= state.selectedPeople.length || from === to) return;
    const people = [...state.selectedPeople];
    const [person] = people.splice(from, 1);
    people.splice(to, 0, person);
    state.selectedPeople = people;
    changed();
}

function setupReordering() {
    const list = document.getElementById('peopleSelected');
    let dragFrom = null;

    list.addEventListener('dragstart', (e) => {
        const chip = e.target.closest('.person-chip');
        if (!chip) return;
        dragFrom = parseInt(chip.dataset.index);
        chip.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
    });

    list.addEventListener('dragover', (e) => {
        const chip = e.target.closest('.person-chip');
        if (!chip || dragFrom === null) return;
        e.preventDefault();
        list.querySelectorAll('.drag-over').forEach(el => el.classList.remove('drag-over'));
        chip.classList.add('drag-over');
    });

    list.addEventListener('drop', (e) => {
        const chip = e.target.closest('.person-chip');
        if (!chip || dragFrom === null) return;
        e.preventDefault();
        move(dragFrom, parseInt(chip.dataset.index));
    });

    list.addEventListener('dragend', () => {
        dragFrom = null;
        list.querySelectorAll('.dragging, .drag-over').forEach(el => el.classList.remove('dragging', 'drag-over'));
    });

    // Keyboard: Alt+Up / Alt+Down moves the focused person
    list.addEventListener('keydown', (e) => {
        const chip = e.target.closest('.person-chip');
        if (!chip || !e.altKey || (e.key !== 'ArrowUp' && e.key !== 'ArrowDown')) return;
        e.preventDefault();
        const from = parseInt(chip.dataset.index);
        const to = from + (e.key === 'ArrowUp' ? -1 : 1);
        move(from, to);
        list.querySelector(`.person-chip[data-index="${Math.max(0, Math.min(to, state.selectedPeople.length - 1))}"]`)?.focus();
    });
}
