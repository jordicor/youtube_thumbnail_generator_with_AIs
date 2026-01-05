/**
 * Video Detail Page - Descriptions Module
 * ========================================
 * Handles description generation, saving, and deletion.
 */

import { CONFIG } from './constants.js';
import { state, clearDescriptions } from './state.js';
import { getTextConfig } from './ai-config.js';

// =========================================================================
// DESCRIPTION LOADING
// =========================================================================

/**
 * Load saved descriptions from the database.
 */
export async function loadSaved() {
    try {
        const response = await fetch(`/api/titles/saved-descriptions/${state.videoId}`);
        const data = await response.json();

        if (data.success && data.descriptions && data.descriptions.length > 0) {
            const savedDescriptions = data.descriptions.map(d => ({
                text: d.description_text,
                dbId: d.id
            }));

            display(savedDescriptions, false);
            console.log(`Loaded ${savedDescriptions.length} saved descriptions`);
        }
    } catch (error) {
        console.error('Error loading saved descriptions:', error);
    }
}

/**
 * Save descriptions to the database.
 * @param {string[]} descriptions - Array of description texts
 * @param {string} style - Description style
 * @param {string} language - Language code
 * @param {string} length - Description length
 * @param {string} provider - AI provider
 * @param {string} model - AI model
 * @param {boolean} includeTimestamps - Include timestamps flag
 * @param {boolean} includeHashtags - Include hashtags flag
 * @param {boolean} includeEmojis - Include emojis flag
 * @param {boolean} includeSocialLinks - Include social links flag
 */
export async function saveToDb(descriptions, style, language, length, provider, model,
                               includeTimestamps, includeHashtags, includeEmojis, includeSocialLinks) {
    try {
        const response = await fetch('/api/titles/save-descriptions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_id: state.videoId,
                descriptions: descriptions,
                style: style,
                language: language,
                length: length,
                provider: provider,
                model: model,
                include_timestamps: includeTimestamps,
                include_hashtags: includeHashtags,
                include_emojis: includeEmojis,
                include_social_links: includeSocialLinks
            })
        });

        if (response.ok) {
            const data = await response.json();
            if (data.description_ids && data.description_ids.length > 0) {
                let idIndex = 0;
                for (let i = state.descriptions.length - descriptions.length;
                     i < state.descriptions.length && idIndex < data.description_ids.length;
                     i++) {
                    if (state.descriptions[i] && !state.descriptions[i].dbId) {
                        state.descriptions[i].dbId = data.description_ids[idIndex];
                        idIndex++;
                    }
                }
            }
            console.log(`Saved ${data.saved_count} descriptions to database`);
        }
    } catch (error) {
        console.error('Error saving descriptions to database:', error);
    }
}

// =========================================================================
// DESCRIPTION GENERATION
// =========================================================================

/**
 * Generate new descriptions using AI.
 */
export async function generate() {
    const textAiConfig = getTextConfig();
    const style = document.getElementById('descStyle').value;
    const language = document.getElementById('descLanguage').value;
    const length = document.getElementById('descLength').value;
    const numDescriptions = parseInt(document.getElementById('descQuantity').value);
    const includeTimestamps = document.getElementById('descTimestamps').checked;
    const includeHashtags = document.getElementById('descHashtags').checked;
    const includeEmojis = document.getElementById('descEmojis').checked;
    const includeSocialLinks = document.getElementById('descSocialLinks').checked;

    let customPrompt = null;
    let customInstructions = null;

    if (style === 'custom') {
        customPrompt = document.getElementById('descCustomPrompt').value.trim();
        if (!customPrompt) {
            ThumbnailApp.showToast('Por favor, introduce un prompt personalizado', 'error');
            return;
        }
    } else {
        customInstructions = document.getElementById('descCustomInstructions').value.trim() || null;
    }

    const loadingEl = document.getElementById('descriptionsLoading');
    const loadingMain = document.getElementById('descriptionsLoadingMain');
    const loadingHint = document.getElementById('descriptionsLoadingHint');

    loadingEl.style.display = 'flex';
    loadingMain.textContent = 'Comprobando...';
    loadingHint.textContent = '';
    document.getElementById('generateDescBtn').disabled = true;

    let transcriptionExists = false;
    let generatingTimeout = null;

    try {
        const checkResponse = await fetch(`/api/titles/check-transcription/${state.videoId}`);
        if (checkResponse.ok) {
            const checkData = await checkResponse.json();
            transcriptionExists = checkData.exists;
        }
    } catch (e) {
        // Ignore
    }

    if (transcriptionExists) {
        loadingMain.textContent = 'Generando descripciones...';
        loadingHint.textContent = '';
    } else {
        loadingMain.textContent = 'Transcribiendo audio...';
        loadingHint.textContent = 'Solo se realiza una vez por video';

        generatingTimeout = setTimeout(() => {
            loadingMain.textContent = 'Generando descripciones con IA...';
            loadingHint.textContent = '';
        }, CONFIG.TRANSCRIPTION_CHECK_TIMEOUT_MS);
    }

    try {
        const response = await fetch('/api/titles/generate-description', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_id: state.videoId,
                style: style,
                custom_prompt: customPrompt,
                custom_instructions: customInstructions,
                language: language,
                length: length,
                num_descriptions: numDescriptions,
                include_timestamps: includeTimestamps,
                include_hashtags: includeHashtags,
                include_emojis: includeEmojis,
                include_social_links: includeSocialLinks,
                provider: textAiConfig.provider,
                model: textAiConfig.model,
                thinking_enabled: textAiConfig.thinking_enabled,
                thinking_level: textAiConfig.thinking_level
            })
        });

        if (generatingTimeout) clearTimeout(generatingTimeout);

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Error generando descripciones');
        }

        if (!data.success) {
            throw new Error(data.error || 'Error generando descripciones');
        }

        const descriptionObjects = data.descriptions.map(text => ({ text, dbId: null }));
        display(descriptionObjects);

        saveToDb(data.descriptions, style, language, length, data.provider, data.model,
                 includeTimestamps, includeHashtags, includeEmojis, includeSocialLinks);

        if (data.transcription_generated) {
            ThumbnailApp.showToast(`${data.descriptions.length} descripcion(es) generada(s) (audio transcrito)`, 'success');
        } else {
            ThumbnailApp.showToast(`${data.descriptions.length} descripcion(es) generada(s)`, 'success');
        }

    } catch (error) {
        if (generatingTimeout) clearTimeout(generatingTimeout);
        ThumbnailApp.showToast('Error: ' + error.message, 'error');
    } finally {
        document.getElementById('descriptionsLoading').style.display = 'none';
        document.getElementById('generateDescBtn').disabled = false;
    }
}

// =========================================================================
// DESCRIPTION DISPLAY
// =========================================================================

/**
 * Display descriptions in the UI.
 * @param {Object[]} descriptions - Array of {text, dbId} objects
 * @param {boolean} append - Whether to append to existing descriptions
 */
export function display(descriptions, append = true) {
    const resultsSection = document.getElementById('descriptionsResultsSection');
    const resultsList = document.getElementById('descriptionsResultsList');

    if (!append) {
        resultsList.innerHTML = '';
        state.descriptions = [];
    }

    const startIndex = state.descriptions.length;
    state.descriptions = state.descriptions.concat(descriptions);

    descriptions.forEach((descObj, i) => {
        const index = startIndex + i;
        const descText = typeof descObj === 'string' ? descObj : descObj.text;
        const item = document.createElement('div');
        item.className = 'description-item';
        item.dataset.descriptionIndex = index;
        item.innerHTML = `
            <div class="description-text">${escapeHtml(descText)}</div>
            <div class="description-actions">
                <button class="description-copy-btn" onclick="VideoDetail.copyDescription(${index})" title="Copiar al portapapeles">
                    &#128203; Copiar
                </button>
                <button class="description-delete-btn" onclick="VideoDetail.deleteDescription(${index})" title="Eliminar">
                    &#128465;
                </button>
            </div>
        `;
        resultsList.appendChild(item);
    });

    resultsSection.style.display = 'block';
    updateCount();
}

/**
 * Update descriptions count display.
 */
export function updateCount() {
    const validDescriptions = state.descriptions.filter(d => d !== null);
    const countEl = document.getElementById('descriptionsCount');
    if (countEl) {
        countEl.textContent = validDescriptions.length > 0 ? `(${validDescriptions.length})` : '';
    }
}

// =========================================================================
// DESCRIPTION DELETION
// =========================================================================

/**
 * Delete a single description.
 * @param {number} index - Description index
 */
export async function deleteOne(index) {
    const descObj = state.descriptions[index];
    if (!descObj) return;

    if (descObj.dbId) {
        try {
            const response = await fetch(`/api/titles/delete-description/${descObj.dbId}`, {
                method: 'DELETE'
            });
            if (!response.ok) {
                console.error('Error deleting description from database');
            }
        } catch (error) {
            console.error('Error deleting description:', error);
        }
    }

    state.descriptions[index] = null;

    const item = document.querySelector(`.description-item[data-description-index="${index}"]`);
    if (item) {
        item.classList.add('removing');
        setTimeout(() => {
            item.remove();
            updateCount();

            const validDescriptions = state.descriptions.filter(d => d !== null);
            if (validDescriptions.length === 0) {
                document.getElementById('descriptionsResultsSection').style.display = 'none';
            }
        }, CONFIG.ANIMATION_DURATION_MS);
    }

    ThumbnailApp.showToast('Descripcion eliminada', 'success');
}

/**
 * Delete all descriptions.
 */
export async function deleteAll() {
    const validDescriptions = state.descriptions.filter(d => d !== null);
    if (validDescriptions.length === 0) return;

    const dbIds = validDescriptions
        .filter(d => d.dbId)
        .map(d => d.dbId);

    if (dbIds.length > 0) {
        try {
            const response = await fetch('/api/titles/delete-descriptions-batch', {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ids: dbIds })
            });
            if (!response.ok) {
                console.error('Error deleting descriptions from database');
            }
        } catch (error) {
            console.error('Error deleting descriptions:', error);
        }
    }

    state.descriptions = [];
    document.getElementById('descriptionsResultsList').innerHTML = '';
    document.getElementById('descriptionsResultsSection').style.display = 'none';
    updateCount();

    ThumbnailApp.showToast(`${validDescriptions.length} descripcion(es) eliminada(s)`, 'success');
}

// =========================================================================
// DESCRIPTION COPYING
// =========================================================================

/**
 * Copy a single description to clipboard.
 * @param {number} index - Description index
 */
export async function copy(index) {
    const descObj = state.descriptions[index];
    if (!descObj) return;

    const descText = typeof descObj === 'string' ? descObj : descObj.text;

    try {
        await navigator.clipboard.writeText(descText);

        const item = document.querySelector(`.description-item[data-description-index="${index}"]`);
        const btn = item?.querySelector('.description-copy-btn');
        if (btn) {
            const originalHTML = btn.innerHTML;
            btn.classList.add('copied');
            btn.innerHTML = '&#10003; Copiado';

            setTimeout(() => {
                btn.classList.remove('copied');
                btn.innerHTML = originalHTML;
            }, CONFIG.COPY_FEEDBACK_DURATION_MS);
        }

        ThumbnailApp.showToast('Descripcion copiada al portapapeles', 'success');
    } catch (error) {
        ThumbnailApp.showToast('Error al copiar', 'error');
    }
}

/**
 * Copy all descriptions to clipboard.
 */
export async function copyAll() {
    const validDescriptions = state.descriptions.filter(d => d !== null);
    if (validDescriptions.length === 0) return;

    try {
        const allDescriptions = validDescriptions
            .map(d => typeof d === 'string' ? d : d.text)
            .join('\n\n---\n\n');
        await navigator.clipboard.writeText(allDescriptions);
        ThumbnailApp.showToast(`${validDescriptions.length} descripcion(es) copiada(s)`, 'success');
    } catch (error) {
        ThumbnailApp.showToast('Error al copiar', 'error');
    }
}

// =========================================================================
// UI HELPERS
// =========================================================================

/**
 * Update description style UI based on selection.
 */
export function updateStyleUI() {
    const style = document.getElementById('descStyle').value;
    const customPromptGroup = document.getElementById('descCustomPromptGroup');
    const customInstructionsGroup = document.getElementById('descCustomInstructionsGroup');
    const customPromptTextarea = document.getElementById('descCustomPrompt');

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
