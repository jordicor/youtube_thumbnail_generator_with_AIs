/**
 * Internationalization (i18n) System for YouTube Thumbnail Generator
 * ===================================================================
 * Client-side translation system with JSON-based locale files.
 *
 * Usage:
 *   // After loading, use the global t() function:
 *   t('common.cancel')           // "Cancelar" or "Cancel"
 *   t('video.count', {count: 5}) // "5 videos" (with pluralization)
 *
 *   // Change language:
 *   await setLanguage('en');
 *
 *   // Get current language:
 *   getLanguage()  // "es" or "en"
 */

// Supported languages
const SUPPORTED_LANGUAGES = ['es', 'en'];
const DEFAULT_LANGUAGE = 'es';

// Cookie name for language preference
const LANG_COOKIE_NAME = 'lang';

// Cache for loaded translations
let translationsCache = {};

// Current language
let currentLanguage = DEFAULT_LANGUAGE;

// Process inline cache immediately (before any async code runs)
if (window.__I18N_CACHE__) {
    const detectedLang = (function() {
        const cookieLang = document.cookie.split('; ').find(c => c.startsWith('lang='));
        if (cookieLang) {
            const lang = cookieLang.split('=')[1];
            if (SUPPORTED_LANGUAGES.includes(lang)) return lang;
        }
        const storedLang = localStorage.getItem('lang');
        if (storedLang && SUPPORTED_LANGUAGES.includes(storedLang)) return storedLang;
        const browserLang = navigator.language.split('-')[0];
        if (SUPPORTED_LANGUAGES.includes(browserLang)) return browserLang;
        return DEFAULT_LANGUAGE;
    })();
    translationsCache[detectedLang] = window.__I18N_CACHE__;
    currentLanguage = detectedLang;
    delete window.__I18N_CACHE__;
}

// Flag to track if i18n is initialized
let isInitialized = false;

// Callbacks to run after language change
const languageChangeCallbacks = [];

/**
 * Get a cookie value by name.
 */
function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) {
        return parts.pop().split(';').shift();
    }
    return null;
}

/**
 * Set a cookie value.
 */
function setCookie(name, value, days = 365) {
    const expires = new Date(Date.now() + days * 864e5).toUTCString();
    document.cookie = `${name}=${value}; expires=${expires}; path=/; SameSite=Lax`;
}

/**
 * Detect initial language from cookie, localStorage, or browser.
 */
function detectLanguage() {
    // 1. Check cookie
    const cookieLang = getCookie(LANG_COOKIE_NAME);
    if (cookieLang && SUPPORTED_LANGUAGES.includes(cookieLang)) {
        return cookieLang;
    }

    // 2. Check localStorage
    const storedLang = localStorage.getItem(LANG_COOKIE_NAME);
    if (storedLang && SUPPORTED_LANGUAGES.includes(storedLang)) {
        return storedLang;
    }

    // 3. Check browser language
    const browserLang = navigator.language.split('-')[0];
    if (SUPPORTED_LANGUAGES.includes(browserLang)) {
        return browserLang;
    }

    // 4. Default
    return DEFAULT_LANGUAGE;
}

/**
 * Load translations for a language.
 */
async function loadTranslations(lang) {
    if (translationsCache[lang]) {
        return translationsCache[lang];
    }

    try {
        const response = await fetch(`/static/locales/${lang}.json`);
        if (!response.ok) {
            console.error(`Failed to load translations for ${lang}: ${response.status}`);
            return {};
        }

        const translations = await response.json();
        translationsCache[lang] = translations;
        return translations;
    } catch (error) {
        console.error(`Error loading translations for ${lang}:`, error);
        return {};
    }
}

/**
 * Get a nested value from an object using dot notation.
 */
function getNestedValue(obj, keyPath) {
    const keys = keyPath.split('.');
    let current = obj;

    for (const key of keys) {
        if (current && typeof current === 'object' && key in current) {
            current = current[key];
        } else {
            return null;
        }
    }

    return typeof current === 'string' ? current : null;
}

/**
 * Format a string with placeholders.
 * Supports {name} style placeholders.
 */
function formatString(template, params = {}) {
    if (!params || Object.keys(params).length === 0) {
        return template;
    }

    return template.replace(/\{(\w+)\}/g, (match, key) => {
        return key in params ? params[key] : match;
    });
}

/**
 * Get the correct plural key based on count.
 * Follows simplified CLDR rules for ES/EN.
 */
function getPluralKey(key, count) {
    if (count === 1) {
        return `${key}_one`;
    }
    return `${key}_other`;
}

/**
 * Get current language.
 */
function getLanguage() {
    return currentLanguage;
}

/**
 * Set current language.
 * Reloads translations and updates UI.
 */
async function setLanguage(lang) {
    if (!SUPPORTED_LANGUAGES.includes(lang)) {
        console.error(`Unsupported language: ${lang}`);
        return false;
    }

    // Load translations for new language
    await loadTranslations(lang);

    // Update current language
    currentLanguage = lang;

    // Persist preference
    setCookie(LANG_COOKIE_NAME, lang);
    localStorage.setItem(LANG_COOKIE_NAME, lang);

    // Notify server (fire and forget)
    fetch('/api/lang', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lang })
    }).catch(() => {});

    // Update HTML lang attribute
    document.documentElement.lang = lang;

    // Update language selector UI
    updateLanguageSelector();

    // Call all registered callbacks
    for (const callback of languageChangeCallbacks) {
        try {
            callback(lang);
        } catch (error) {
            console.error('Error in language change callback:', error);
        }
    }

    return true;
}

/**
 * Register a callback to run when language changes.
 * Useful for re-rendering dynamic content.
 */
function onLanguageChange(callback) {
    if (typeof callback === 'function') {
        languageChangeCallbacks.push(callback);
    }
}

/**
 * Update the language selector UI to reflect current language.
 */
function updateLanguageSelector() {
    const selector = document.querySelector('.language-selector');
    if (!selector) return;

    const buttons = selector.querySelectorAll('[data-lang]');
    buttons.forEach(btn => {
        if (btn.dataset.lang === currentLanguage) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

/**
 * Translate a key to the current language.
 *
 * @param {string} key - Dot-notation key (e.g., 'common.cancel')
 * @param {object} params - Optional parameters for interpolation
 *                          If params.count is provided, uses plural rules
 * @returns {string} Translated string, or [key] if not found
 */
function translate(key, params = {}) {
    const translations = translationsCache[currentLanguage] || {};
    let value = null;

    // Handle pluralization if count is provided
    if ('count' in params) {
        const pluralKey = getPluralKey(key, params.count);
        value = getNestedValue(translations, pluralKey);

        // Fallback to base key if plural not found
        if (value === null) {
            value = getNestedValue(translations, key);
        }
    } else {
        value = getNestedValue(translations, key);
    }

    // Fallback to English if not found
    if (value === null && currentLanguage !== 'en') {
        const enTranslations = translationsCache['en'] || {};
        if ('count' in params) {
            const pluralKey = getPluralKey(key, params.count);
            value = getNestedValue(enTranslations, pluralKey);
            if (value === null) {
                value = getNestedValue(enTranslations, key);
            }
        } else {
            value = getNestedValue(enTranslations, key);
        }
    }

    // Return key as fallback (useful for debugging missing translations)
    if (value === null) {
        console.warn(`Missing translation: ${key}`);
        return `[${key}]`;
    }

    return formatString(value, params);
}

/**
 * Initialize the i18n system.
 * Call this on page load.
 */
async function initI18n() {
    if (isInitialized) {
        return;
    }

    // Detect and set language
    currentLanguage = detectLanguage();

    // Load translations for current language (and English as fallback)
    await Promise.all([
        loadTranslations(currentLanguage),
        currentLanguage !== 'en' ? loadTranslations('en') : Promise.resolve()
    ]);

    // Update HTML lang attribute
    document.documentElement.lang = currentLanguage;

    // Setup language selector if present
    setupLanguageSelector();

    isInitialized = true;

    console.log(`i18n initialized: ${currentLanguage}`);
}

/**
 * Setup click handlers for language selector buttons.
 */
function setupLanguageSelector() {
    const selector = document.querySelector('.language-selector');
    if (!selector) return;

    const buttons = selector.querySelectorAll('[data-lang]');
    buttons.forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.preventDefault();
            const lang = btn.dataset.lang;
            if (lang && lang !== currentLanguage) {
                await setLanguage(lang);
                // Reload page to apply translations to all elements
                window.location.reload();
            }
        });
    });

    // Update initial state
    updateLanguageSelector();
}

// Global shorthand alias
const t = translate;

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initI18n);
} else {
    initI18n();
}
