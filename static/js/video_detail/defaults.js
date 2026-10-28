/**
 * Video Detail Page - Defaults Module
 * ====================================
 * The Generate form starts with the values of the app settings (header ›
 * Settings). Changing them here is for this video only: the folded steps
 * mark what differs from the settings and can go back to them. When the
 * settings change, the fields still on the old value take the new one.
 */

// Form fields that start with a value of the settings' generation defaults
const FIELDS = {
    num_images: { id: 'numImages', step: 'count' },
    expression: { id: 'expression', step: 'look' },
    face_swap: { id: 'applyFaceSwap', step: 'advanced' },
    face_restoration: { id: 'applyFaceRestoration', step: 'advanced' },
    avoid_repeat: { id: 'promptIncludeHistory', step: 'advanced' },
    face_photos: { id: 'facePhotosPerThumbnail', step: 'advanced' },
    provider_fallback: { id: 'imageAiProviderFallback', step: 'advanced' },
};

function element(key) {
    return document.getElementById(FIELDS[key].id);
}

/**
 * Value of a field, as the settings store it.
 * @param {string} key - Key of FIELDS
 * @returns {number|boolean|string|null} Value
 */
function fieldValue(key) {
    const el = element(key);
    if (el.type === 'checkbox') return el.checked;
    if (el.type === 'number') return parseInt(el.value, 10);
    return el.value || null;
}

function setFieldValue(key, value) {
    const el = element(key);
    if (el.type === 'checkbox') el.checked = value;
    else el.value = value ?? '';
}

/**
 * Give the fields the settings' values. With the values the settings had
 * before, only the fields still on them change: the rest were changed here.
 * @param {Object} defaults - Generation defaults of the settings
 * @param {Object} [previous] - Generation defaults before the settings changed
 */
export function apply(defaults, previous) {
    for (const key of Object.keys(FIELDS)) {
        const el = element(key);
        // Face swap without its models stays off
        if (el.disabled) continue;
        if (previous && fieldValue(key) !== previous[key]) continue;
        if (fieldValue(key) === defaults[key]) continue;
        setFieldValue(key, defaults[key]);
        el.dispatchEvent(new Event('change', { bubbles: true }));
    }
}

/**
 * Whether a field differs from the settings (a field that cannot be used never does).
 * @param {string} key - Key of FIELDS
 * @returns {boolean} True if changed for this video
 */
export function isChanged(key) {
    return !element(key).disabled && fieldValue(key) !== AppSettings.get().generation[key];
}

/**
 * Whether any field of a form step differs from the settings.
 * @param {string} step - count, look or advanced
 * @returns {boolean} True if something in the step was changed for this video
 */
export function stepChanged(step) {
    return Object.keys(FIELDS).some(key => FIELDS[key].step === step && isChanged(key));
}

/**
 * Give the fields of a form step their settings' values again.
 * @param {string} step - count, look or advanced
 */
export function reset(step) {
    const defaults = AppSettings.get().generation;
    for (const key of Object.keys(FIELDS)) {
        if (FIELDS[key].step !== step || !isChanged(key)) continue;
        setFieldValue(key, defaults[key]);
        element(key).dispatchEvent(new Event('change', { bubbles: true }));
    }
}
