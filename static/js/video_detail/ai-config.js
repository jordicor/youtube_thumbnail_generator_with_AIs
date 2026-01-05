/**
 * Video Detail Page - AI Configuration Module
 * ============================================
 * Handles AI configuration chips, popovers, and localStorage persistence.
 */

import {
    STORAGE_KEYS,
    MODEL_MAX_REFS,
    PROVIDER_NAMES,
    SHORT_MODEL_NAMES,
    FALLBACK_PROMPT_MODELS
} from './constants.js';
import { state } from './state.js';

// =========================================================================
// LOCALSTORAGE PERSISTENCE
// =========================================================================

/**
 * Save current AI configuration to localStorage.
 */
export function saveToLocalStorage() {
    const config = {
        text: {
            provider: document.getElementById('textAiProvider').value,
            model: document.getElementById('textAiModel').value,
            thinkingEnabled: document.getElementById('textAiThinkingEnabled').checked,
            thinkingLevel: document.getElementById('textAiThinkingLevel').value
        },
        image: {
            provider: document.getElementById('imageAiProvider').value,
            geminiModel: document.getElementById('imageAiGeminiModel').value,
            poeModel: document.getElementById('imageAiPoeModel').value,
            openaiModel: document.getElementById('imageAiOpenaiModel').value,
            numRefs: document.getElementById('imageAiNumRefs').value
        }
    };
    localStorage.setItem(STORAGE_KEYS.AI_CONFIG, JSON.stringify(config));
}

/**
 * Load AI configuration from localStorage.
 * @returns {Object|null} The stored configuration or null if not found
 */
export function loadFromLocalStorage() {
    const stored = localStorage.getItem(STORAGE_KEYS.AI_CONFIG);
    if (!stored) return null;
    try {
        return JSON.parse(stored);
    } catch (e) {
        console.error('Error parsing AI config from localStorage:', e);
        return null;
    }
}

/**
 * Apply stored configuration to UI elements.
 */
export function applyStoredConfig() {
    const config = loadFromLocalStorage();
    if (!config) return;

    // Apply Text AI configuration
    if (config.text) {
        if (config.text.provider) {
            document.getElementById('textAiProvider').value = config.text.provider;
        }
        if (config.text.thinkingEnabled !== undefined) {
            document.getElementById('textAiThinkingEnabled').checked = config.text.thinkingEnabled;
            toggleThinkingLevel();
        }
        if (config.text.thinkingLevel) {
            document.getElementById('textAiThinkingLevel').value = config.text.thinkingLevel;
        }
    }

    // Apply Image AI configuration
    if (config.image) {
        if (config.image.provider) {
            document.getElementById('imageAiProvider').value = config.image.provider;
            updateImageModelOptions();
        }
        if (config.image.geminiModel) {
            document.getElementById('imageAiGeminiModel').value = config.image.geminiModel;
        }
        if (config.image.poeModel) {
            document.getElementById('imageAiPoeModel').value = config.image.poeModel;
        }
        if (config.image.openaiModel) {
            document.getElementById('imageAiOpenaiModel').value = config.image.openaiModel;
        }
        if (config.image.numRefs) {
            document.getElementById('imageAiNumRefs').value = config.image.numRefs;
        }
    }

    updateRefMax();
    updateChipDisplays();
}

/**
 * Setup event listeners for configuration persistence.
 */
export function setupPersistence() {
    // Text AI selectors
    document.getElementById('textAiProvider').addEventListener('change', saveToLocalStorage);
    document.getElementById('textAiModel').addEventListener('change', saveToLocalStorage);
    document.getElementById('textAiThinkingEnabled').addEventListener('change', saveToLocalStorage);
    document.getElementById('textAiThinkingLevel').addEventListener('change', saveToLocalStorage);

    // Image AI selectors
    document.getElementById('imageAiProvider').addEventListener('change', saveToLocalStorage);
    document.getElementById('imageAiGeminiModel').addEventListener('change', saveToLocalStorage);
    document.getElementById('imageAiPoeModel').addEventListener('change', saveToLocalStorage);
    document.getElementById('imageAiOpenaiModel').addEventListener('change', saveToLocalStorage);
    document.getElementById('imageAiNumRefs').addEventListener('change', saveToLocalStorage);
    document.getElementById('imageAiNumRefs').addEventListener('input', saveToLocalStorage);
}

// =========================================================================
// POPOVER MANAGEMENT
// =========================================================================

/**
 * Toggle visibility of an AI config popover.
 * @param {string} type - 'textAi' or 'imageAi'
 */
export function togglePopover(type) {
    const popover = document.getElementById(type + 'Popover');
    const chip = document.getElementById(type + 'Chip');
    const overlay = document.getElementById('aiConfigOverlay');
    const isVisible = popover.classList.contains('visible');

    closeAllPopovers();

    if (!isVisible) {
        popover.classList.add('visible');
        chip.classList.add('active');
        overlay.classList.add('visible');
    }
}

/**
 * Close a specific popover.
 * @param {string} type - 'textAi' or 'imageAi'
 */
export function closePopover(type) {
    const popover = document.getElementById(type + 'Popover');
    const chip = document.getElementById(type + 'Chip');
    const overlay = document.getElementById('aiConfigOverlay');

    popover.classList.remove('visible');
    chip.classList.remove('active');
    overlay.classList.remove('visible');

    updateChipDisplays();
}

/**
 * Close all open popovers.
 */
export function closeAllPopovers() {
    document.querySelectorAll('.ai-config-popover').forEach(p => p.classList.remove('visible'));
    document.querySelectorAll('.ai-config-chip').forEach(c => c.classList.remove('active'));
    document.getElementById('aiConfigOverlay').classList.remove('visible');
}

// =========================================================================
// TEXT AI CONFIGURATION
// =========================================================================

/**
 * Update text AI model options based on selected provider.
 */
export function updateTextModelOptions() {
    const provider = document.getElementById('textAiProvider').value;
    const modelSelect = document.getElementById('textAiModel');

    modelSelect.innerHTML = '';

    if (!state.promptModels) {
        modelSelect.innerHTML = '<option value="">Cargando modelos...</option>';
        return;
    }

    const models = state.promptModels[provider] || [{ value: "", label: "Por defecto" }];
    for (const model of models) {
        const option = document.createElement('option');
        option.value = model.value;
        option.textContent = model.label;
        modelSelect.appendChild(option);
    }

    updateChipDisplays();
}

/**
 * Toggle thinking level selector visibility.
 */
export function toggleThinkingLevel() {
    const enabled = document.getElementById('textAiThinkingEnabled').checked;
    const levelGroup = document.getElementById('textAiThinkingLevelGroup');

    if (enabled) {
        levelGroup.classList.remove('hidden');
    } else {
        levelGroup.classList.add('hidden');
    }

    updateChipDisplays();
}

// =========================================================================
// IMAGE AI CONFIGURATION
// =========================================================================

/**
 * Update image AI model options based on selected provider.
 * @param {boolean} setRefsToMax - Whether to set refs to max value
 */
export function updateImageModelOptions(setRefsToMax = false) {
    const provider = document.getElementById('imageAiProvider').value;

    // Hide all model groups
    document.getElementById('imageAiGeminiModelGroup').style.display = 'none';
    document.getElementById('imageAiPoeModelGroup').style.display = 'none';
    document.getElementById('imageAiOpenaiModelGroup').style.display = 'none';

    // Show relevant model group
    if (provider === 'gemini') {
        document.getElementById('imageAiGeminiModelGroup').style.display = 'block';
    } else if (provider === 'poe') {
        document.getElementById('imageAiPoeModelGroup').style.display = 'block';
    } else if (provider === 'openai') {
        document.getElementById('imageAiOpenaiModelGroup').style.display = 'block';
        updateOpenaiWarning();
    }

    updateRefIndicator();
    updateRefMax(setRefsToMax);
    updateChipDisplays();
}

/**
 * Update OpenAI warning message visibility.
 */
export function updateOpenaiWarning() {
    const openaiModel = document.getElementById('imageAiOpenaiModel');
    const selectedOption = openaiModel.options[openaiModel.selectedIndex];
    const supportsRef = selectedOption.dataset.ref === 'true';
    const warningBox = document.getElementById('imageAiOpenaiWarning');

    warningBox.style.display = supportsRef ? 'none' : 'block';
    updateRefIndicator();
}

/**
 * Update reference support indicator text.
 */
export function updateRefIndicator() {
    const provider = document.getElementById('imageAiProvider').value;
    const refIndicator = document.getElementById('imageAiRefIndicator');
    let supportsRef = true;

    if (provider === 'openai') {
        const openaiModel = document.getElementById('imageAiOpenaiModel');
        const selectedOption = openaiModel.options[openaiModel.selectedIndex];
        supportsRef = selectedOption.dataset.ref === 'true';
    }

    if (supportsRef) {
        refIndicator.textContent = 'Soporta imagenes de referencia';
        refIndicator.className = 'ai-config-ref-indicator supported';
    } else {
        refIndicator.textContent = 'NO soporta imagenes de referencia';
        refIndicator.className = 'ai-config-ref-indicator not-supported';
    }
}

/**
 * Update reference images max value based on selected model.
 * @param {boolean} setToMax - Whether to set value to max
 */
export function updateRefMax(setToMax = false) {
    const provider = document.getElementById('imageAiProvider').value;
    const refInput = document.getElementById('imageAiNumRefs');
    const refMaxLabel = document.getElementById('imageAiRefMaxLabel');
    const refGroup = document.getElementById('imageAiRefGroup');

    // Get current model based on provider
    let model = '';
    if (provider === 'gemini') {
        model = document.getElementById('imageAiGeminiModel').value;
    } else if (provider === 'poe') {
        model = document.getElementById('imageAiPoeModel').value;
    } else if (provider === 'openai') {
        model = document.getElementById('imageAiOpenaiModel').value;
    } else if (provider === 'replicate') {
        model = 'flux-1.1-pro';
    }

    // Get max refs for current provider/model
    const providerRefs = MODEL_MAX_REFS[provider] || {};
    const maxRefs = providerRefs[model] || 1;

    // Update label and input
    refMaxLabel.textContent = `(max. ${maxRefs})`;
    refInput.max = maxRefs;

    // Set to max when model changes, otherwise just clamp to valid range
    if (setToMax) {
        refInput.value = maxRefs;
    } else {
        refInput.value = Math.min(parseInt(refInput.value) || maxRefs, maxRefs);
    }

    // Hide/show based on support
    if (maxRefs === 0) {
        refGroup.style.display = 'none';
    } else {
        refGroup.style.display = 'block';
    }

    updateChipDisplays();
}

// =========================================================================
// CHIP DISPLAY UPDATES
// =========================================================================

/**
 * Update the display text in both AI config chips.
 */
export function updateChipDisplays() {
    // Update Text AI chip
    const textProvider = document.getElementById('textAiProvider').value;
    const textModel = document.getElementById('textAiModel');
    const textThinkingEnabled = document.getElementById('textAiThinkingEnabled').checked;
    const textChipValue = document.getElementById('textAiChipValue');

    // Get model short name
    let modelDisplay = textModel.value;
    if (modelDisplay) {
        const parts = modelDisplay.split('-');
        modelDisplay = parts.slice(0, 2).join('-');
    } else {
        modelDisplay = PROVIDER_NAMES[textProvider] || textProvider;
    }

    textChipValue.innerHTML = modelDisplay +
        (textThinkingEnabled ? ' <span class="ai-config-chip-badge" id="textAiThinkingBadge">Think</span>' : '');

    // Update Image AI chip
    const imageProvider = document.getElementById('imageAiProvider').value;
    const imageChipValue = document.getElementById('imageAiChipValue');
    const numRefs = document.getElementById('imageAiNumRefs').value;

    let imageModel = '';
    if (imageProvider === 'gemini') {
        imageModel = document.getElementById('imageAiGeminiModel').value;
    } else if (imageProvider === 'poe') {
        imageModel = document.getElementById('imageAiPoeModel').value;
    } else if (imageProvider === 'openai') {
        imageModel = document.getElementById('imageAiOpenaiModel').value;
    } else {
        imageModel = imageProvider;
    }

    const displayModel = SHORT_MODEL_NAMES[imageModel] || imageModel;
    const maxRefs = parseInt(document.getElementById('imageAiNumRefs').max) || 0;
    const refBadgeText = maxRefs > 0 ? `${numRefs} refs` : 'Sin refs';

    imageChipValue.innerHTML = displayModel +
        ` <span class="ai-config-chip-badge" id="imageAiRefBadge">${refBadgeText}</span>`;
}

// =========================================================================
// CONFIG GETTERS
// =========================================================================

/**
 * Get current text AI configuration.
 * @returns {Object} Text AI config object
 */
export function getTextConfig() {
    return {
        provider: document.getElementById('textAiProvider').value,
        model: document.getElementById('textAiModel').value,
        thinking_enabled: document.getElementById('textAiThinkingEnabled').checked,
        thinking_level: document.getElementById('textAiThinkingLevel').value
    };
}

/**
 * Get current image AI configuration.
 * @returns {Object} Image AI config object
 */
export function getImageConfig() {
    const provider = document.getElementById('imageAiProvider').value;
    const config = {
        provider: provider,
        num_reference_images: parseInt(document.getElementById('imageAiNumRefs').value) || null
    };

    if (provider === 'gemini') {
        config.gemini_model = document.getElementById('imageAiGeminiModel').value;
    } else if (provider === 'poe') {
        config.poe_model = document.getElementById('imageAiPoeModel').value;
    } else if (provider === 'openai') {
        config.openai_model = document.getElementById('imageAiOpenaiModel').value;
    }

    return config;
}

// =========================================================================
// API LOADING
// =========================================================================

/**
 * Load available AI models from the backend API.
 */
export async function loadModelsFromAPI() {
    try {
        const response = await fetch('/api/generation/gransabio/models');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();

        // Transform API response to frontend format
        state.promptModels = {};
        for (const [provider, models] of Object.entries(data)) {
            state.promptModels[provider] = [
                { value: "", label: "Por defecto" }
            ];
            for (const model of models) {
                state.promptModels[provider].push({
                    value: model.key,
                    label: `${model.name}${model.description ? ' - ' + model.description : ''}`
                });
            }
        }
        state.modelsLoaded = true;
        console.log('AI models loaded from backend:', Object.keys(state.promptModels));

        // Refresh the text AI model selector
        updateTextModelOptions();

        // Restore saved text AI model from localStorage
        const config = loadFromLocalStorage();
        if (config?.text?.model) {
            const modelSelect = document.getElementById('textAiModel');
            const optionExists = Array.from(modelSelect.options).some(opt => opt.value === config.text.model);
            if (optionExists) {
                modelSelect.value = config.text.model;
                updateChipDisplays();
            }
        }

    } catch (error) {
        console.error('Failed to load models from API, using fallback:', error);
        state.promptModels = FALLBACK_PROMPT_MODELS;
        state.modelsLoaded = true;

        updateTextModelOptions();
        const config = loadFromLocalStorage();
        if (config?.text?.model) {
            const modelSelect = document.getElementById('textAiModel');
            const optionExists = Array.from(modelSelect.options).some(opt => opt.value === config.text.model);
            if (optionExists) {
                modelSelect.value = config.text.model;
                updateChipDisplays();
            }
        }
    }
}

/**
 * Check Gran Sabio LLM availability status.
 */
export async function checkGranSabioStatus() {
    const chipStatusEl = document.getElementById('gransabioStatusChip');
    const chipTextEl = chipStatusEl ? chipStatusEl.querySelector('.status-text') : null;

    try {
        const response = await fetch('/api/generation/gransabio/status');
        const data = await response.json();

        if (data.available) {
            if (chipStatusEl) {
                chipStatusEl.classList.remove('unavailable');
                chipStatusEl.classList.add('available');
                chipStatusEl.title = data.message || 'Gran Sabio LLM connected';
            }
            if (chipTextEl) chipTextEl.textContent = 'Online';
        } else {
            if (chipStatusEl) {
                chipStatusEl.classList.remove('available');
                chipStatusEl.classList.add('unavailable');
                chipStatusEl.title = data.message || 'Gran Sabio LLM not available';
            }
            if (chipTextEl) chipTextEl.textContent = 'Offline';
        }
    } catch (e) {
        if (chipStatusEl) {
            chipStatusEl.classList.remove('available');
            chipStatusEl.classList.add('unavailable');
            chipStatusEl.title = 'Could not check Gran Sabio LLM status';
        }
        if (chipTextEl) chipTextEl.textContent = 'Offline';
    }
}

// =========================================================================
// INITIALIZATION
// =========================================================================

/**
 * Initialize AI configuration chips and listeners.
 */
export function init() {
    applyStoredConfig();
    updateTextModelOptions();
    updateChipDisplays();
    updateImageModelOptions();
    setupPersistence();

    // Close popovers on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeAllPopovers();
        }
    });
}
