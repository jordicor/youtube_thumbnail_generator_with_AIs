/**
 * Video Detail Page - Clusters Module
 * ====================================
 * Handles cluster loading, selection, merging, and deletion.
 */

import { state, clearClusterSelection, clearReferenceCluster } from './state.js';

// =========================================================================
// CLUSTER DISPLAY HELPERS
// =========================================================================

/**
 * Get display name for a cluster based on its type.
 * @param {Object} cluster - Cluster object
 * @returns {string} Display name
 */
export function getDisplayName(cluster) {
    if (cluster.label) {
        return escapeHtml(cluster.label);
    }

    if (cluster.cluster_type === 'person_scene') {
        const parentLabel = cluster.parent_label || t('video_detail.clusters.person', {index: (cluster.parent_cluster_index || 0) + 1});
        const sceneNum = cluster.scene_index !== null ? cluster.scene_index : '?';
        return `${escapeHtml(parentLabel)} - ${t('video_detail.clusters.scene', {index: sceneNum})}`;
    }

    return t('video_detail.clusters.person', {index: cluster.cluster_index + 1});
}

/**
 * Get cluster name by index.
 * @param {number} clusterIndex - Cluster index
 * @returns {string} Cluster name
 */
export function getName(clusterIndex) {
    const cluster = state.clusters.find(c => c.cluster_index === clusterIndex);
    return cluster?.label || t('video_detail.clusters.person', {index: clusterIndex + 1});
}

// =========================================================================
// CLUSTER LOADING
// =========================================================================

/**
 * Load clusters from the API and render them.
 */
export async function load() {
    const grid = document.getElementById('clustersGrid');
    const select = document.getElementById('clusterSelect');

    try {
        const response = await fetch(`/api/analysis/${state.videoId}/clusters?view_mode=${state.currentClusterType}`);
        const data = await response.json();

        state.clusters = data.clusters;
        clearClusterSelection();
        updateSelectionUI();

        if (data.clusters.length === 0) {
            const emptyMsg = state.currentClusterType === 'person_scene'
                ? t('empty_state.no_clusters_scene')
                : t('empty_state.no_clusters');
            grid.innerHTML = `<p class="empty-state">${emptyMsg}</p>`;
            return;
        }

        grid.innerHTML = '';
        select.innerHTML = `<option value="">${t('common.select')}</option>`;

        for (const cluster of data.clusters) {
            const displayName = getDisplayName(cluster);

            // Add to grid
            const card = document.createElement('div');
            card.className = 'cluster-card';
            card.dataset.clusterIndex = cluster.cluster_index;
            card.innerHTML = `
                <div class="card-checkbox">
                    <input type="checkbox" id="cluster-cb-${cluster.cluster_index}">
                </div>
                <div class="cluster-card-actions">
                    <button class="cluster-reference-btn" onclick="event.stopPropagation(); VideoDetail.setClusterAsReference(${cluster.cluster_index})" title="${t('generate_form.cluster_tooltip')}">
                        &#9734;
                    </button>
                    <a href="/video/${state.videoId}/cluster/${cluster.cluster_index}/frames?view_mode=${state.currentClusterType}" class="cluster-frames-btn" onclick="event.stopPropagation()" title="${t('video_detail.clusters.manage_frames')}">
                        &#9881;
                    </a>
                    <button class="cluster-delete-btn" onclick="event.stopPropagation(); VideoDetail.showDeleteModal(${cluster.cluster_index}, ${cluster.num_frames})" title="${t('video_detail.clusters.delete_cluster')}">
                        &#128465;
                    </button>
                </div>
                <div class="cluster-reference-badge" style="display: none;">
                    &#128247; ${t('clusters.references_title')}
                </div>
                <img src="/api/analysis/${state.videoId}/clusters/by-id/${cluster.id}/image"
                     alt="Cluster ${cluster.cluster_index}"
                     onerror="this.src='/static/img/placeholder.svg'">
                <div class="cluster-info">
                    <h4>${displayName}</h4>
                    <p>${t('clusters.frame_count_other', {count: cluster.num_frames})}</p>
                </div>
            `;
            // Click on card (outside action buttons) toggles checkbox selection
            card.addEventListener('click', (e) => {
                if (!e.target.closest('.cluster-card-actions') &&
                    !e.target.closest('.card-checkbox')) {
                    toggleSelection(cluster.cluster_index);
                }
            });
            // Also handle checkbox click directly
            const checkbox = card.querySelector('.card-checkbox');
            checkbox.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleSelection(cluster.cluster_index);
            });
            grid.appendChild(card);

            // Add to select
            const option = document.createElement('option');
            option.value = cluster.cluster_index;
            option.textContent = `${displayName} (${t('clusters.frame_count_other', {count: cluster.num_frames})})`;
            select.appendChild(option);
        }

        document.getElementById('clusterSelectionBar').classList.add('visible');

        // Setup scroll indicator for the grid container
        setupScrollIndicator();

    } catch (error) {
        grid.innerHTML = `<div class="error">${t('errors.load_clusters')}: ${escapeHtml(error.message)}</div>`;
    }
}

/**
 * Setup scroll indicator for the clusters grid.
 * Shows a fade effect when there's more content to scroll.
 */
function setupScrollIndicator() {
    const container = document.getElementById('clustersGridContainer');
    const grid = document.getElementById('clustersGrid');

    if (!container || !grid) return;

    const updateScrollState = () => {
        const hasScroll = grid.scrollHeight > grid.clientHeight;
        const scrolledToBottom = grid.scrollTop + grid.clientHeight >= grid.scrollHeight - 10;

        container.classList.toggle('has-scroll', hasScroll && !scrolledToBottom);
        container.classList.toggle('scrolled-bottom', scrolledToBottom);
    };

    // Initial check
    updateScrollState();

    // Update on scroll
    grid.addEventListener('scroll', updateScrollState);

    // Update on resize
    window.addEventListener('resize', updateScrollState);
}

// =========================================================================
// CLUSTER SELECTION
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
 * Toggle selection of all clusters.
 */
export function toggleSelectAll() {
    const selectAllCheckbox = document.getElementById('selectAllClusters');
    if (selectAllCheckbox.checked) {
        state.clusters.forEach(c => state.selectedClusters.add(c.cluster_index));
    } else {
        state.selectedClusters.clear();
    }
    updateSelectionUI();
}

/**
 * Update selection UI elements.
 */
export function updateSelectionUI() {
    const count = state.selectedClusters.size;
    const countEl = document.getElementById('clusterSelectionCount');
    const mergeBtn = document.getElementById('btnMergeClusters');
    const deleteBtn = document.getElementById('btnDeleteSelectedClusters');
    const selectAllCb = document.getElementById('selectAllClusters');

    countEl.textContent = t('selection.selected_count', {count: count});
    countEl.className = count > 0 ? 'selection-count has-selection' : 'selection-count';

    mergeBtn.disabled = count < 2;
    deleteBtn.disabled = count < 1;

    selectAllCb.checked = count === state.clusters.length && count > 0;
    selectAllCb.indeterminate = count > 0 && count < state.clusters.length;

    document.querySelectorAll('.cluster-card').forEach(card => {
        const idx = parseInt(card.dataset.clusterIndex);
        const cb = card.querySelector('input[type="checkbox"]');
        if (cb) cb.checked = state.selectedClusters.has(idx);
        card.classList.toggle('cluster-selected', state.selectedClusters.has(idx));
    });

    // Also update reference UI
    updateReferenceUI();
}

/**
 * Update the visual state of reference cluster selection.
 */
export function updateReferenceUI() {
    const select = document.getElementById('clusterSelect');
    const display = document.getElementById('selectedClusterDisplay');

    document.querySelectorAll('.cluster-card').forEach(card => {
        const idx = parseInt(card.dataset.clusterIndex);
        const isReference = state.referenceClusterIndex === idx;
        const refBtn = card.querySelector('.cluster-reference-btn');
        const refBadge = card.querySelector('.cluster-reference-badge');

        // Update card state
        card.classList.toggle('cluster-reference', isReference);

        // Update star button (filled vs outline)
        if (refBtn) {
            refBtn.innerHTML = isReference ? '&#9733;' : '&#9734;';  // ★ vs ☆
            refBtn.classList.toggle('active', isReference);
        }

        // Update badge visibility
        if (refBadge) {
            refBadge.style.display = isReference ? 'block' : 'none';
        }
    });

    // Sync with hidden select
    if (select) {
        select.value = state.referenceClusterIndex !== null ? state.referenceClusterIndex : '';
    }

    // Update the display in generation form
    if (display) {
        if (state.referenceClusterIndex !== null) {
            const cluster = state.clusters.find(c => c.cluster_index === state.referenceClusterIndex);
            if (cluster) {
                const displayName = getDisplayName(cluster);
                display.innerHTML = `
                    <div class="cluster-preview">
                        <img src="/api/analysis/${state.videoId}/clusters/by-id/${cluster.id}/image"
                             alt="${escapeHtml(displayName)}"
                             onerror="this.src='/static/img/placeholder.svg'">
                        <div class="cluster-preview-info">
                            <span class="cluster-preview-name">${displayName}</span>
                            <span class="cluster-preview-frames">${cluster.num_frames} ${t('video_detail.clusters.frames_other')}</span>
                        </div>
                    </div>
                    <span class="cluster-preview-badge">&#9733; ${t('clusters.references_title')}</span>
                `;
                display.classList.add('has-selection');
            }
        } else {
            display.innerHTML = `<span class="no-selection">&#9734; ${t('errors.select_cluster')}</span>`;
            display.classList.remove('has-selection');
        }
    }
}

/**
 * Set a cluster as the reference for AI image generation.
 * Only one cluster can be the reference at a time.
 * @param {number} clusterIndex - Cluster index to set as reference
 */
export function setAsReference(clusterIndex) {
    // Toggle: if already selected, deselect; otherwise select
    if (state.referenceClusterIndex === clusterIndex) {
        state.referenceClusterIndex = null;
    } else {
        state.referenceClusterIndex = clusterIndex;
    }
    updateReferenceUI();
}

/**
 * Change cluster view type (person vs person_scene).
 */
export function changeViewType() {
    state.currentClusterType = document.getElementById('clusterViewType').value;
    clearClusterSelection();
    clearReferenceCluster();
    updateSelectionUI();
    load();
}

// =========================================================================
// CLUSTER DELETION
// =========================================================================

/**
 * Show delete confirmation modal for a single cluster.
 * @param {number} clusterIndex - Cluster index
 * @param {number} numFrames - Number of frames in cluster
 */
export function showDeleteModal(clusterIndex, numFrames) {
    state.clusterToDelete = clusterIndex;
    const name = getName(clusterIndex);
    document.getElementById('deleteClusterMessage').textContent =
        t('clusters.delete_message');
    document.getElementById('deleteClusterModal').classList.add('visible');
}

/**
 * Show delete confirmation modal for selected clusters.
 */
export function showDeleteSelectedModal() {
    const count = state.selectedClusters.size;
    if (count === 0) return;

    const totalFrames = state.clusters
        .filter(c => state.selectedClusters.has(c.cluster_index))
        .reduce((sum, c) => sum + c.num_frames, 0);

    state.clusterToDelete = Array.from(state.selectedClusters);
    document.getElementById('deleteClusterMessage').textContent =
        t('clusters.merge_message', {count: count});
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

        ThumbnailApp.showToast(
            t('clusters.delete_title'),
            'success'
        );
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
            <img src="/api/analysis/${state.videoId}/clusters/by-id/${cluster?.id}/image" alt="${escapeHtml(name)}">
            <span>${escapeHtml(name)} (${t('clusters.frame_count_other', {count: cluster?.num_frames || 0})})</span>
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

        ThumbnailApp.showToast(t('clusters.merge_title'), 'success');
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
