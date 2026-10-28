/**
 * Video Detail Page - Titles Module
 * ==================================
 * Handles title generation, saving, starring and deletion. Starred titles
 * guide the thumbnail ideas (Generate tab).
 */

import { CONFIG } from './constants.js';
import { state } from './state.js';
import { getTextConfig, ensureGranSabioAvailable } from './ai-config.js';

// =========================================================================
// TITLE LOADING
// =========================================================================

/**
 * Load saved titles from the database.
 */
export async function loadSaved() {
    try {
        const response = await fetch(`/api/titles/saved/${state.videoId}`);
        // H7: Check response before parsing
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
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
 * Write the number of titles to generate in the button.
 */
export function updateGenerateLabel() {
    const count = parseInt(document.getElementById('titleQuantity').value, 10) || 1;
    document.getElementById('generateTitlesLabel').textContent = t('titles_form.generate_button', { count });
}

/**
 * Generate new titles using AI.
 */
export async function generate() {
    if (!ensureGranSabioAvailable()) return;

    const textAiConfig = getTextConfig();
    const style = document.getElementById('titleStyle').value;
    const language = document.getElementById('titleLanguage').value;
    const numTitles = parseInt(document.getElementById('titleQuantity').value);

    let customPrompt = null;
    let customInstructions = null;

    if (style === 'custom') {
        customPrompt = document.getElementById('titleCustomPrompt').value.trim();
        if (!customPrompt) {
            ThumbnailApp.showToast(t('errors.custom_prompt_required'), 'error');
            return;
        }
    } else {
        customInstructions = document.getElementById('titleCustomInstructions').value.trim() || null;
    }

    const loadingEl = document.getElementById('titlesLoading');
    const loadingMain = document.getElementById('titlesLoadingMain');
    const loadingHint = document.getElementById('titlesLoadingHint');

    loadingEl.style.display = 'flex';
    loadingMain.textContent = t('common.loading');
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
        loadingMain.textContent = t('titles.generating');
        loadingHint.textContent = '';
    } else {
        loadingMain.textContent = t('analysis.transcribing');
        loadingHint.textContent = t('titles_form.preparing');

        generatingTimeout = setTimeout(() => {
            loadingMain.textContent = t('titles.generating');
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
            throw new Error(data.detail || t('errors.generate_titles'));
        }

        const titleObjects = data.titles.map(text => ({ text, dbId: null }));
        display(titleObjects);

        saveToDb(data.titles, style, language, data.provider, data.model);

        ThumbnailApp.showToast(t('titles.generated', {count: data.titles.length}), 'success');

    } catch (error) {
        if (generatingTimeout) clearTimeout(generatingTimeout);
        ThumbnailApp.showToast(t('common.error') + ': ' + error.message, 'error');
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
        const starred = state.selectedTitles.has(index);
        const item = document.createElement('li');
        item.className = `title-item${starred ? ' starred' : ''}`;
        item.dataset.titleIndex = index;
        item.innerHTML = `
            <button type="button" class="title-star" data-action="star" aria-pressed="${starred}"
                    title="${escapeHtml(t('titles_form.star_tooltip'))}" aria-label="${escapeHtml(t('titles_form.star_tooltip'))}">
                ${ThumbnailApp.icon('star')}
            </button>
            <span class="title-text">${escapeHtml(titleText)}</span>
            <span class="title-actions">
                <button type="button" class="icon-btn" data-action="copy"
                        title="${escapeHtml(t('titles_form.copy_tooltip'))}" aria-label="${escapeHtml(t('titles_form.copy_tooltip'))}">${ThumbnailApp.icon('copy', 'icon-sm')}</button>
                <button type="button" class="icon-btn danger" data-action="delete"
                        title="${escapeHtml(t('titles_form.delete_tooltip'))}" aria-label="${escapeHtml(t('titles_form.delete_tooltip'))}">${ThumbnailApp.icon('trash', 'icon-sm')}</button>
            </span>
        `;
        resultsList.appendChild(item);
    });

    resultsSection.style.display = 'block';
    updateSelectionInfo();
    updateCount();
}

// =========================================================================
// TITLE STARRING (guides the thumbnails)
// =========================================================================

/**
 * Star a title, or remove its star.
 * @param {number} index - Title index
 */
export function toggleSelection(index) {
    if (state.selectedTitles.has(index)) {
        state.selectedTitles.delete(index);
    } else {
        state.selectedTitles.add(index);
    }

    const item = document.querySelector(`.title-item[data-title-index="${index}"]`);
    if (item) {
        item.classList.toggle('starred', state.selectedTitles.has(index));
        item.querySelector('.title-star').setAttribute('aria-pressed', String(state.selectedTitles.has(index)));
    }

    updateSelectionInfo();
}

/**
 * Get array of starred title texts.
 * @returns {string[]} Starred title texts
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
 * Say how many titles guide the thumbnails.
 */
export function updateSelectionInfo() {
    const starred = state.selectedTitles.size;
    document.getElementById('titlesSelectionInfo').textContent = starred > 0
        ? t('titles_form.starred_info', {count: starred})
        : t('titles_form.star_hint');
    notifyChange();
}

/**
 * Tell other parts of the page (the Generate tab's guiding titles) that the
 * titles or their stars changed.
 */
function notifyChange() {
    document.dispatchEvent(new CustomEvent('titles:changed'));
}

/**
 * Update titles count display.
 */
export function updateCount() {
    const validCount = state.titles.filter(t => t !== null).length;
    document.getElementById('titlesCount').textContent = validCount > 0 ? validCount : '';
    document.getElementById('textsTabCount').textContent = validCount > 0 ? validCount : '';
    if (validCount === 0) {
        document.getElementById('titlesResultsSection').style.display = 'none';
    }
    notifyChange();
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

    // M32: Confirm before deleting (matching pattern from thumbnails.js)
    const confirmed = await ThumbnailApp.showModal({
        title: t('titles_form.delete_title'),
        message: t('titles_form.delete_count_message', {count: 1}),
        type: 'danger',
        confirmText: t('common.delete'),
        cancelText: t('common.cancel')
    });
    if (!confirmed) return;

    if (titleObj.dbId) {
        const response = await fetch(`/api/titles/delete/${titleObj.dbId}`, {
            method: 'DELETE'
        });
        if (!response.ok) {
            ThumbnailApp.showToast(t('errors.delete_failed'), 'error');
            return;
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
    updateCount();
    ThumbnailApp.showToast(t('titles_form.deleted'), 'success');
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
        ThumbnailApp.showToast(t('titles.copied'), 'success');
    } catch (error) {
        ThumbnailApp.showToast(t('errors.generic'), 'error');
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
        ThumbnailApp.showToast(t('titles.copied'), 'success');
    } catch (error) {
        ThumbnailApp.showToast(t('errors.generic'), 'error');
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
        customPromptGroup.style.display = '';
        customInstructionsGroup.style.display = 'none';

        customPromptTextarea.classList.remove('attention');
        void customPromptTextarea.offsetWidth;
        customPromptTextarea.classList.add('attention');

        setTimeout(() => customPromptTextarea.focus(), 100);
    } else {
        customPromptGroup.style.display = 'none';
        customInstructionsGroup.style.display = '';
        customPromptTextarea.classList.remove('attention');
    }
}

/**
 * Bind the form and the list of titles.
 */
export function setupControls() {
    document.getElementById('titleStyle').addEventListener('change', updateStyleUI);
    document.getElementById('titleQuantity').addEventListener('input', updateGenerateLabel);
    document.getElementById('generateTitlesBtn').addEventListener('click', generate);
    document.getElementById('copyAllTitlesBtn').addEventListener('click', copyAll);
    document.getElementById('titlesResultsList').addEventListener('click', (e) => {
        const button = e.target.closest('button[data-action]');
        if (!button) return;
        const index = parseInt(button.closest('.title-item').dataset.titleIndex, 10);
        if (button.dataset.action === 'star') toggleSelection(index);
        else if (button.dataset.action === 'copy') copy(index);
        else if (button.dataset.action === 'delete') deleteOne(index);
    });
    updateGenerateLabel();
    updateSelectionInfo();
}
