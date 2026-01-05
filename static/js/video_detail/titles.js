/**
 * Video Detail Page - Titles Module
 * ==================================
 * Handles title generation, saving, selection, and deletion.
 */

import { CONFIG } from './constants.js';
import { state, clearTitles } from './state.js';
import { getTextConfig } from './ai-config.js';

// =========================================================================
// TITLE LOADING
// =========================================================================

/**
 * Load saved titles from the database.
 */
export async function loadSaved() {
    try {
        const response = await fetch(`/api/titles/saved/${state.videoId}`);
        const data = await response.json();

        if (data.success && data.titles && data.titles.length > 0) {
            const savedTitles = data.titles.map(t => ({
                text: t.title_text,
                dbId: t.id
            }));

            display(savedTitles, false);
            console.log(`Loaded ${savedTitles.length} saved titles`);
        }
    } catch (error) {
        console.error('Error loading saved titles:', error);
    }
}

/**
 * Save titles to the database.
 * @param {string[]} titles - Array of title texts
 * @param {string} style - Title style
 * @param {string} language - Language code
 * @param {string} provider - AI provider
 * @param {string} model - AI model
 */
export async function saveToDb(titles, style, language, provider, model) {
    try {
        const response = await fetch('/api/titles/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_id: state.videoId,
                titles: titles,
                style: style,
                language: language,
                provider: provider,
                model: model
            })
        });

        const data = await response.json();

        if (data.success) {
            const startIdx = state.titles.length - titles.length;
            data.title_ids.forEach((dbId, i) => {
                const idx = startIdx + i;
                if (state.titles[idx]) {
                    state.titles[idx].dbId = dbId;
                }
            });
            console.log(`Saved ${data.saved_count} titles to database`);
        }

        return data;
    } catch (error) {
        console.error('Error saving titles:', error);
        return { success: false, error: error.message };
    }
}

// =========================================================================
// TITLE GENERATION
// =========================================================================

/**
 * Generate new titles using AI.
 */
export async function generate() {
    const textAiConfig = getTextConfig();
    const style = document.getElementById('titleStyle').value;
    const language = document.getElementById('titleLanguage').value;
    const numTitles = parseInt(document.getElementById('titleQuantity').value);

    let customPrompt = null;
    let customInstructions = null;

    if (style === 'custom') {
        customPrompt = document.getElementById('titleCustomPrompt').value.trim();
        if (!customPrompt) {
            ThumbnailApp.showToast('Por favor, introduce un prompt personalizado', 'error');
            return;
        }
    } else {
        customInstructions = document.getElementById('titleCustomInstructions').value.trim() || null;
    }

    const loadingEl = document.getElementById('titlesLoading');
    const loadingMain = document.getElementById('titlesLoadingMain');
    const loadingHint = document.getElementById('titlesLoadingHint');

    loadingEl.style.display = 'flex';
    loadingMain.textContent = 'Comprobando...';
    loadingHint.textContent = '';
    document.getElementById('titlesResultsSection').style.display = 'none';
    document.getElementById('generateTitlesBtn').disabled = true;

    let transcriptionExists = false;
    let generatingTimeout = null;

    try {
        const checkResponse = await fetch(`/api/titles/check-transcription/${state.videoId}`);
        if (checkResponse.ok) {
            const checkData = await checkResponse.json();
            transcriptionExists = checkData.exists;
        }
    } catch (e) {
        // Ignore check errors
    }

    if (transcriptionExists) {
        loadingMain.textContent = 'Generando titulos...';
        loadingHint.textContent = '';
    } else {
        loadingMain.textContent = 'Transcribiendo audio...';
        loadingHint.textContent = 'Solo se realiza una vez por video';

        generatingTimeout = setTimeout(() => {
            loadingMain.textContent = 'Generando titulos con IA...';
            loadingHint.textContent = '';
        }, CONFIG.TRANSCRIPTION_CHECK_TIMEOUT_MS);
    }

    try {
        const response = await fetch('/api/titles/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_id: state.videoId,
                style: style,
                custom_prompt: customPrompt,
                custom_instructions: customInstructions,
                language: language,
                num_titles: numTitles,
                provider: textAiConfig.provider,
                model: textAiConfig.model,
                thinking_enabled: textAiConfig.thinking_enabled,
                thinking_level: textAiConfig.thinking_level
            })
        });

        if (generatingTimeout) clearTimeout(generatingTimeout);

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Error generando titulos');
        }

        if (!data.success) {
            throw new Error(data.error || 'Error generando titulos');
        }

        const titleObjects = data.titles.map(text => ({ text, dbId: null }));
        display(titleObjects);

        saveToDb(data.titles, style, language, data.provider, data.model);

        if (data.transcription_generated) {
            ThumbnailApp.showToast(`${data.titles.length} titulos generados (audio transcrito)`, 'success');
        } else {
            ThumbnailApp.showToast(`${data.titles.length} titulos generados`, 'success');
        }

    } catch (error) {
        if (generatingTimeout) clearTimeout(generatingTimeout);
        ThumbnailApp.showToast('Error: ' + error.message, 'error');
    } finally {
        document.getElementById('titlesLoading').style.display = 'none';
        document.getElementById('generateTitlesBtn').disabled = false;
    }
}

// =========================================================================
// TITLE DISPLAY
// =========================================================================

/**
 * Display titles in the UI.
 * @param {Object[]} titles - Array of {text, dbId} objects
 * @param {boolean} append - Whether to append to existing titles
 */
export function display(titles, append = true) {
    const resultsSection = document.getElementById('titlesResultsSection');
    const resultsList = document.getElementById('titlesResultsList');

    if (!append) {
        resultsList.innerHTML = '';
        state.titles = [];
        state.selectedTitles.clear();
    }

    const startIndex = state.titles.length;
    state.titles = state.titles.concat(titles);

    titles.forEach((titleObj, i) => {
        const index = startIndex + i;
        const titleText = typeof titleObj === 'string' ? titleObj : titleObj.text;
        const item = document.createElement('div');
        item.className = 'title-item';
        item.dataset.titleIndex = index;
        item.innerHTML = `
            <div class="title-select-wrapper" onclick="VideoDetail.toggleTitleSelection(${index})" title="Seleccionar para orientar imagenes">
                <input type="checkbox" class="title-checkbox" id="title-cb-${index}"
                       ${state.selectedTitles.has(index) ? 'checked' : ''}
                       onclick="event.stopPropagation(); VideoDetail.toggleTitleSelection(${index})">
            </div>
            <span class="title-text" onclick="VideoDetail.toggleTitleSelection(${index})">${escapeHtml(titleText)}</span>
            <div class="title-actions">
                <button class="title-copy-btn" onclick="event.stopPropagation(); VideoDetail.copyTitle(${index})" title="Copiar al portapapeles">
                    &#128203;
                </button>
                <button class="title-delete-btn" onclick="event.stopPropagation(); VideoDetail.deleteTitle(${index})" title="Eliminar">
                    &#128465;
                </button>
            </div>
        `;
        resultsList.appendChild(item);
    });

    resultsSection.style.display = 'block';
    updateSelectionInfo();
    updateSelectAllButton();
    updateCount();
}

// =========================================================================
// TITLE SELECTION
// =========================================================================

/**
 * Toggle selection of a single title.
 * @param {number} index - Title index
 */
export function toggleSelection(index) {
    if (state.selectedTitles.has(index)) {
        state.selectedTitles.delete(index);
    } else {
        state.selectedTitles.add(index);
    }

    const checkbox = document.getElementById(`title-cb-${index}`);
    if (checkbox) checkbox.checked = state.selectedTitles.has(index);

    const item = document.querySelector(`.title-item[data-title-index="${index}"]`);
    if (item) item.classList.toggle('selected', state.selectedTitles.has(index));

    updateSelectionInfo();
    updateSelectAllButton();
}

/**
 * Toggle selection of all titles.
 */
export function toggleSelectAll() {
    const validIndices = state.titles
        .map((t, i) => t !== null ? i : null)
        .filter(i => i !== null);

    const allSelected = validIndices.every(i => state.selectedTitles.has(i));

    if (allSelected) {
        state.selectedTitles.clear();
    } else {
        validIndices.forEach(i => state.selectedTitles.add(i));
    }

    validIndices.forEach(index => {
        const checkbox = document.getElementById(`title-cb-${index}`);
        if (checkbox) checkbox.checked = state.selectedTitles.has(index);

        const item = document.querySelector(`.title-item[data-title-index="${index}"]`);
        if (item) item.classList.toggle('selected', state.selectedTitles.has(index));
    });

    updateSelectionInfo();
    updateSelectAllButton();
}

/**
 * Get array of selected title texts.
 * @returns {string[]} Selected title texts
 */
export function getSelectedArray() {
    return Array.from(state.selectedTitles)
        .filter(idx => state.titles[idx] !== null)
        .map(idx => {
            const t = state.titles[idx];
            return typeof t === 'string' ? t : t.text;
        });
}

/**
 * Update selection info display.
 */
export function updateSelectionInfo() {
    const selectedCount = state.selectedTitles.size;
    const infoEl = document.getElementById('titlesSelectionInfo');
    if (infoEl) {
        if (selectedCount > 0) {
            infoEl.textContent = `${selectedCount} seleccionado${selectedCount > 1 ? 's' : ''}`;
            infoEl.classList.add('has-selection');
        } else {
            infoEl.textContent = 'Selecciona titulos para orientar las imagenes';
            infoEl.classList.remove('has-selection');
        }
    }
    updateDeleteSelectedButton();
}

/**
 * Update select all button text.
 */
export function updateSelectAllButton() {
    const btn = document.getElementById('selectAllTitlesBtn');
    if (!btn) return;

    const validIndices = state.titles
        .map((t, i) => t !== null ? i : null)
        .filter(i => i !== null);

    const allSelected = validIndices.length > 0 && validIndices.every(i => state.selectedTitles.has(i));

    if (allSelected) {
        btn.innerHTML = '&#9745; Ninguno';
        btn.title = 'Deseleccionar todos';
    } else {
        btn.innerHTML = '&#9744; Todos';
        btn.title = 'Seleccionar todos';
    }
}

/**
 * Update delete selected button visibility.
 */
export function updateDeleteSelectedButton() {
    const btn = document.getElementById('deleteSelectedTitlesBtn');
    if (btn) {
        if (state.selectedTitles.size > 0) {
            btn.classList.add('visible');
        } else {
            btn.classList.remove('visible');
        }
    }
}

/**
 * Update titles count display.
 */
export function updateCount() {
    const countEl = document.getElementById('titlesCount');
    if (!countEl) return;

    const validCount = state.titles.filter(t => t !== null).length;
    countEl.textContent = validCount > 0 ? `(${validCount})` : '';
}

// =========================================================================
// TITLE DELETION
// =========================================================================

/**
 * Delete a single title.
 * @param {number} index - Title index
 */
export async function deleteOne(index) {
    const titleObj = state.titles[index];
    if (!titleObj) return;

    if (titleObj.dbId) {
        try {
            const response = await fetch(`/api/titles/delete/${titleObj.dbId}`, {
                method: 'DELETE'
            });
            if (!response.ok) {
                console.error('Error deleting title from database');
            }
        } catch (error) {
            console.error('Error deleting title:', error);
        }
    }

    state.titles[index] = null;
    state.selectedTitles.delete(index);

    const item = document.querySelector(`.title-item[data-title-index="${index}"]`);
    if (item) {
        item.classList.add('removing');
        setTimeout(() => item.remove(), CONFIG.ANIMATION_DURATION_MS);
    }

    updateSelectionInfo();
    updateSelectAllButton();
    updateCount();
    ThumbnailApp.showToast('Titulo eliminado', 'success');
}

/**
 * Delete all selected titles.
 */
export async function deleteSelected() {
    const selectedCount = state.selectedTitles.size;
    if (selectedCount === 0) return;

    const confirmMsg = selectedCount === 1
        ? '¿Eliminar el titulo seleccionado?'
        : `¿Eliminar los ${selectedCount} titulos seleccionados?`;

    if (!confirm(confirmMsg)) return;

    const indicesToDelete = Array.from(state.selectedTitles);

    const dbIds = indicesToDelete
        .filter(idx => state.titles[idx] && state.titles[idx].dbId)
        .map(idx => state.titles[idx].dbId);

    if (dbIds.length > 0) {
        try {
            const response = await fetch('/api/titles/delete-batch', {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ids: dbIds })
            });
            if (!response.ok) {
                console.error('Error deleting titles from database');
            }
        } catch (error) {
            console.error('Error deleting titles:', error);
        }
    }

    indicesToDelete.forEach(index => {
        const item = document.querySelector(`.title-item[data-title-index="${index}"]`);
        if (item) {
            item.classList.add('removing');
        }
        state.titles[index] = null;
    });

    state.selectedTitles.clear();

    setTimeout(() => {
        indicesToDelete.forEach(index => {
            const item = document.querySelector(`.title-item[data-title-index="${index}"]`);
            if (item) item.remove();
        });

        updateSelectionInfo();
        updateSelectAllButton();
        updateCount();
    }, CONFIG.ANIMATION_DURATION_MS);

    ThumbnailApp.showToast(`${selectedCount} titulo${selectedCount > 1 ? 's' : ''} eliminado${selectedCount > 1 ? 's' : ''}`, 'success');
}

// =========================================================================
// TITLE COPYING
// =========================================================================

/**
 * Copy a single title to clipboard.
 * @param {number} index - Title index
 */
export async function copy(index) {
    const titleObj = state.titles[index];
    if (!titleObj) return;

    const titleText = typeof titleObj === 'string' ? titleObj : titleObj.text;

    try {
        await navigator.clipboard.writeText(titleText);

        const item = document.querySelector(`.title-item[data-title-index="${index}"]`);
        const btn = item?.querySelector('.title-copy-btn');
        if (btn) {
            btn.classList.add('copied');
            btn.innerHTML = '&#10003;';

            setTimeout(() => {
                btn.classList.remove('copied');
                btn.innerHTML = '&#128203;';
            }, CONFIG.COPY_FEEDBACK_DURATION_MS);
        }

        ThumbnailApp.showToast('Titulo copiado al portapapeles', 'success');
    } catch (error) {
        ThumbnailApp.showToast('Error al copiar', 'error');
    }
}

/**
 * Copy all titles to clipboard.
 */
export async function copyAll() {
    const validTitles = state.titles
        .filter(t => t !== null)
        .map(t => typeof t === 'string' ? t : t.text);

    if (validTitles.length === 0) return;

    try {
        const allTitles = validTitles.join('\n');
        await navigator.clipboard.writeText(allTitles);
        ThumbnailApp.showToast(`${validTitles.length} titulos copiados`, 'success');
    } catch (error) {
        ThumbnailApp.showToast('Error al copiar', 'error');
    }
}

// =========================================================================
// UI HELPERS
// =========================================================================

/**
 * Update title style UI based on selection.
 */
export function updateStyleUI() {
    const style = document.getElementById('titleStyle').value;
    const customPromptGroup = document.getElementById('customPromptGroup');
    const customInstructionsGroup = document.getElementById('customInstructionsGroup');
    const customPromptTextarea = document.getElementById('titleCustomPrompt');

    if (style === 'custom') {
        customPromptGroup.style.display = 'block';
        customInstructionsGroup.style.display = 'none';

        customPromptTextarea.classList.remove('attention');
        void customPromptTextarea.offsetWidth;
        customPromptTextarea.classList.add('attention');

        setTimeout(() => customPromptTextarea.focus(), 100);
    } else {
        customPromptGroup.style.display = 'none';
        customInstructionsGroup.style.display = 'block';
        customPromptTextarea.classList.remove('attention');
    }
}
