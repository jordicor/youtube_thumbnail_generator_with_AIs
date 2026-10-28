/**
 * Video Detail Page - Clusters Module
 * ====================================
 * Detected people tab: cluster loading, the "in the thumbnail" button of each
 * card, and checking people to merge or delete them (floating action bar).
 * Choosing who appears in the thumbnail lives in people.js.
 */

import { state, clearClusterSelection } from './state.js';
import * as People from './people.js';

// =========================================================================
// CLUSTER DISPLAY HELPERS
// =========================================================================

/**
 * Get the HTML-escaped display name for a cluster.
 * @param {Object} cluster - Cluster object
 * @returns {string} Escaped display name
 */
export function getDisplayName(cluster) {
    return escapeHtml(ThumbnailApp.getClusterDisplayName(cluster));
}

/**
 * Get cluster name by index (plain text).
 * @param {number} clusterIndex - Cluster index
 * @returns {string} Cluster name
 */
export function getName(clusterIndex) {
    const cluster = state.clusters.find(c => c.cluster_index === clusterIndex);
    return ThumbnailApp.getClusterDisplayName(cluster);
}

// =========================================================================
// CLUSTER LOADING
// =========================================================================

/**
 * Show how many people the video has in the page header (the 'person' view).
 */
function updateHeaderCount() {
    const item = document.getElementById('videoMetaPeople');
    item.classList.toggle('hidden', state.personCount === null);
    if (state.personCount !== null) {
        item.querySelector('span').textContent = t('index.card.persons', { count: state.personCount });
    }
}

/**
 * Load the clusters of the current view and render them.
 */
export async function load() {
    const grid = document.getElementById('clustersGrid');
    const viewMode = state.currentClusterType;

    try {
        const response = await fetch(`/api/analysis/${state.videoId}/clusters?view_mode=${viewMode}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        // A newer view switch superseded this request
        if (viewMode !== state.currentClusterType) return;

        state.clusters = data.clusters;
        clearClusterSelection();
        document.getElementById('createPersonLink').href = `/video/${state.videoId}/person/new?view_mode=${viewMode}`;

        document.getElementById('peopleTabCount').textContent = data.clusters.length;
        if (viewMode === 'person') {
            state.personCount = data.clusters.length;
            updateHeaderCount();
        }

        if (data.clusters.length === 0) {
            const emptyMsg = viewMode === 'person_scene'
                ? t('empty_state.no_clusters_scene')
                : t('empty_state.no_clusters');
            grid.innerHTML = `<p class="empty-state">${escapeHtml(emptyMsg)}</p>`;
        } else {
            grid.replaceChildren(...data.clusters.map(createCard));
        }

        updateSelectionUI();
        People.syncWithClusters();
    } catch (error) {
        grid.innerHTML = `<div class="error">${escapeHtml(t('errors.load_clusters'))}: ${escapeHtml(error.message)}</div>`;
    }
}

/**
 * Build the card of one cluster.
 * @param {Object} cluster - Cluster object
 * @returns {HTMLElement} Card element
 */
function createCard(cluster) {
    const displayName = getDisplayName(cluster);
    const editUrl = `/video/${state.videoId}/person/${cluster.id}`;
    const unnamed = People.isUnnamed(cluster);
    const meta = t('clusters.card_meta', {
        frames: t('clusters.in_frames', { count: cluster.num_frames }),
        photos: t('clusters.face_photos', { count: cluster.reference_count })
    });

    const card = document.createElement('div');
    card.className = 'cluster-card';
    card.dataset.clusterIndex = cluster.cluster_index;
    card.dataset.clusterId = cluster.id;
    card.innerHTML = `
        <label class="card-checkbox" title="${escapeHtml(t('clusters.select_for_actions'))}">
            <input type="checkbox" aria-label="${escapeHtml(t('clusters.select_for_actions'))}">
        </label>
        <span class="cluster-role-badge hidden"></span>
        <a class="cluster-image" href="${editUrl}" title="${escapeHtml(t('video_detail.clusters.edit_person'))}">
            <img src="${ThumbnailApp.personFaceUrl(state.videoId, cluster)}"
                 alt="${displayName}"
                 loading="lazy"
                 onerror="this.src='/static/img/placeholder.svg'">
        </a>
        <div class="cluster-info">
            <h4 title="${displayName}"${unnamed ? ' class="unnamed"' : ''}>${displayName}</h4>
            ${unnamed ? `<a class="link-btn cluster-name-link" href="${editUrl}">${escapeHtml(t('generate_form.name_them'))}</a>` : ''}
            <p>${escapeHtml(meta)}</p>
            ${cluster.is_picture ? `<p>${escapeHtml(t('clusters.picture'))}</p>` : ''}
        </div>
        <div class="cluster-card-footer">
            <button type="button" class="btn btn-small cluster-thumb-toggle"></button>
            <a href="${editUrl}" class="icon-btn" title="${escapeHtml(t('video_detail.clusters.edit_person'))}"
               aria-label="${escapeHtml(t('video_detail.clusters.edit_person'))}">${ThumbnailApp.icon('pencil', 'icon-sm')}</a>
            <button type="button" class="icon-btn cluster-more" aria-haspopup="menu" aria-expanded="false"
                    title="${escapeHtml(t('index.card.more'))}" aria-label="${escapeHtml(t('index.card.more'))}">${ThumbnailApp.icon('more')}</button>
        </div>
    `;

    card.querySelector('.card-checkbox input').addEventListener('change', () => toggleSelection(cluster.cluster_index));
    card.querySelector('.cluster-thumb-toggle').addEventListener('click', () => People.toggle(cluster));

    const moreBtn = card.querySelector('.cluster-more');
    moreBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        ThumbnailApp.showMenu(moreBtn, [
            { label: t('video_detail.clusters.edit_person'), icon: 'pencil', onSelect: () => { window.location.href = editUrl; } },
            { label: t('video_detail.clusters.delete_cluster'), icon: 'trash', danger: true, onSelect: () => showDeleteModal(cluster.cluster_index) }
        ]);
    });

    renderThumbnailState(card);
    return card;
}

/**
 * Show on a card whether its person is in the thumbnail, with its role.
 * @param {HTMLElement} card - Cluster card
 */
function renderThumbnailState(card) {
    const role = state.selectedPeople.findIndex(person => person.id === parseInt(card.dataset.clusterId));
    const inThumbnail = role >= 0;
    const badge = card.querySelector('.cluster-role-badge');
    badge.classList.toggle('hidden', !inThumbnail);
    badge.textContent = inThumbnail ? t('clusters.in_thumbnail', { role: People.roleLabel(role) }) : '';

    const button = card.querySelector('.cluster-thumb-toggle');
    button.classList.toggle('btn-soft', inThumbnail);
    button.classList.toggle('btn-secondary', !inThumbnail);
    button.title = inThumbnail ? t('clusters.remove_from_thumbnail') : '';
    button.innerHTML = inThumbnail
        ? `${ThumbnailApp.icon('check', 'icon-sm')}<span>${escapeHtml(t('clusters.in_thumbnail_short'))}</span>`
        : `${ThumbnailApp.icon('plus', 'icon-sm')}<span>${escapeHtml(t('clusters.use_in_thumbnail'))}</span>`;
}

/**
 * Mark on the cards which clusters are in the thumbnail, with their role.
 */
export function updatePeopleBadges() {
    document.querySelectorAll('.cluster-card').forEach(renderThumbnailState);
}

// =========================================================================
// CLUSTER SELECTION (merge / delete)
// =========================================================================

/**
 * Toggle selection of a single cluster.
 * @param {number} clusterIndex - Cluster index to toggle
 */
export function toggleSelection(clusterIndex) {
    if (state.selectedClusters.has(clusterIndex)) {
        state.selectedClusters.delete(clusterIndex);
    } else {
        state.selectedClusters.add(clusterIndex);
    }
    updateSelectionUI();
}

/**
 * Clear the merge/delete selection.
 */
export function clearSelection() {
    state.selectedClusters.clear();
    updateSelectionUI();
}

/**
 * Update the checkboxes and the floating action bar.
 */
export function updateSelectionUI() {
    const count = state.selectedClusters.size;
    document.getElementById('clusterBulkBar').classList.toggle('hidden', count === 0);
    document.getElementById('clusterSelectionCount').textContent = t('clusters.selected', { count });
    const merge = document.getElementById('btnMergeClusters');
    merge.disabled = count < 2;
    merge.title = count < 2 ? t('clusters.merge_needs_two') : '';

    document.querySelectorAll('.cluster-card').forEach(card => {
        const selected = state.selectedClusters.has(parseInt(card.dataset.clusterIndex));
        card.querySelector('.card-checkbox input').checked = selected;
        card.classList.toggle('cluster-selected', selected);
    });
}

/**
 * Switch the cluster view (Detected people tab or the people picker).
 * The people chosen for the thumbnail are kept: they are stored by ID.
 * @param {string} viewMode - 'person' or 'person_scene'
 */
export function setViewType(viewMode) {
    state.currentClusterType = viewMode;
    document.querySelectorAll('#clusterViewToggle button').forEach(button => {
        const active = button.dataset.view === viewMode;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
    });
    clearClusterSelection();
    load();
}

// =========================================================================
// CLUSTER DELETION
// =========================================================================

/**
 * Show delete confirmation modal for a single cluster.
 * @param {number} clusterIndex - Cluster index
 */
export function showDeleteModal(clusterIndex) {
    state.clusterToDelete = clusterIndex;
    document.getElementById('deleteClusterMessage').textContent = t('clusters.delete_message');
    document.getElementById('deleteClusterModal').classList.add('visible');
}

/**
 * Show delete confirmation modal for selected clusters.
 */
export function showDeleteSelectedModal() {
    if (state.selectedClusters.size === 0) return;

    state.clusterToDelete = Array.from(state.selectedClusters);
    document.getElementById('deleteClusterMessage').textContent = t('clusters.delete_message');
    document.getElementById('deleteClusterModal').classList.add('visible');
}

/**
 * Confirm and execute cluster deletion.
 */
export async function confirmDelete() {
    if (state.clusterToDelete === null) return;

    const indicesToDelete = Array.isArray(state.clusterToDelete)
        ? state.clusterToDelete
        : [state.clusterToDelete];

    try {
        const sortedIndices = [...indicesToDelete].sort((a, b) => b - a);
        for (const idx of sortedIndices) {
            const response = await fetch(`/api/analysis/${state.videoId}/clusters/${idx}?view_mode=${state.currentClusterType}`, {
                method: 'DELETE'
            });
            if (!response.ok) {
                throw new Error(t('errors.delete_cluster'));
            }
        }

        ThumbnailApp.showToast(t('api.messages.cluster_deleted'), 'success');
        closeModal('deleteClusterModal');
        load();
    } catch (error) {
        ThumbnailApp.showToast(t('common.error') + ': ' + error.message, 'error');
    }

    state.clusterToDelete = null;
}

// =========================================================================
// CLUSTER MERGING
// =========================================================================

/**
 * Show merge clusters modal.
 */
export function showMergeModal() {
    if (state.selectedClusters.size < 2) return;

    const selected = Array.from(state.selectedClusters).sort((a, b) => a - b);
    document.getElementById('mergeCount').textContent = selected.length;

    const optionsContainer = document.getElementById('mergeTargetOptions');
    optionsContainer.innerHTML = '';

    selected.forEach((idx, i) => {
        const cluster = state.clusters.find(c => c.cluster_index === idx);
        const name = getName(idx);
        const option = document.createElement('label');
        option.className = 'merge-target-option';
        option.innerHTML = `
            <input type="radio" name="mergeTarget" value="${idx}" ${i === 0 ? 'checked' : ''}>
            <img src="${ThumbnailApp.personFaceUrl(state.videoId, cluster)}" alt="${escapeHtml(name)}">
            <span>${escapeHtml(name)} (${escapeHtml(t('clusters.frame_count', { count: cluster.num_frames }))})</span>
        `;
        optionsContainer.appendChild(option);
    });

    document.getElementById('mergeClustersModal').classList.add('visible');
}

/**
 * Confirm and execute cluster merge.
 */
export async function confirmMerge() {
    const selected = Array.from(state.selectedClusters);
    if (selected.length < 2) return;

    const targetInput = document.querySelector('input[name="mergeTarget"]:checked');
    const targetIndex = targetInput ? parseInt(targetInput.value) : selected[0];

    try {
        const response = await fetch(`/api/analysis/${state.videoId}/clusters/merge`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                cluster_indices: selected,
                target_index: targetIndex,
                view_mode: state.currentClusterType
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || t('errors.merge_clusters'));
        }

        ThumbnailApp.showToast(t('api.messages.clusters_merged'), 'success');
        closeModal('mergeClustersModal');
        load();
    } catch (error) {
        ThumbnailApp.showToast(t('common.error') + ': ' + error.message, 'error');
    }
}

// =========================================================================
// MODAL HELPERS
// =========================================================================

/**
 * Close a modal by ID.
 * @param {string} modalId - Modal element ID
 */
export function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('visible');
}

/**
 * Bind the toolbar, the action bar and the modals of the tab.
 */
export function setupControls() {
    document.querySelectorAll('#clusterViewToggle button').forEach(button => {
        button.addEventListener('click', () => {
            if (button.dataset.view !== state.currentClusterType) setViewType(button.dataset.view);
        });
    });
    setViewType(state.currentClusterType);

    document.getElementById('btnMergeClusters').addEventListener('click', showMergeModal);
    document.getElementById('btnDeleteSelectedClusters').addEventListener('click', showDeleteSelectedModal);
    document.getElementById('btnClearClusterSelection').addEventListener('click', clearSelection);
    document.getElementById('confirmDeleteBtn').addEventListener('click', confirmDelete);
    document.getElementById('confirmMergeBtn').addEventListener('click', confirmMerge);
}
