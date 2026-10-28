/**
 * Settings dialog
 * ===============
 * The header's Settings button opens one dialog for the whole app: who writes
 * the thumbnail ideas, who draws the thumbnails, who writes titles and
 * descriptions, and what the Generate form of every video starts with. A video
 * can change those values for itself; that never touches the settings.
 *
 * window.AppSettings shares them with the pages: `ready` resolves once they are
 * loaded, and saving dispatches "settings:changed" on document with
 * {settings, previous} as its detail. Any element with data-open-settings opens
 * the dialog at that section ("thumbnails" or "texts").
 */

(function () {
    // Names of the text AI providers, for engine summaries without a chosen model
    const TEXT_PROVIDER_NAMES = { anthropic: 'Claude', openai: 'GPT', google: 'Gemini', xai: 'Grok', openrouter: 'OpenRouter', ollama: 'Ollama', llamacpp: 'llama.cpp', openai_compatible: 'Local' };
    const LOCAL_TEXT_PROVIDERS = new Set(['ollama', 'llamacpp', 'openai_compatible']);
    const CATALOG_TEXT_PROVIDERS = new Set([...LOCAL_TEXT_PROVIDERS, 'openrouter']);

    // Image models: short name and, if any, what they are best at (ai_config.<key>)
    const IMAGE_MODEL_LABELS = {
        'gemini-3-pro-image': ['Nano Banana Pro', 'best_quality'],
        'gemini-nano-banana-2.1': ['Nano Banana 2.1', 'fast'],
        'gpt-image-2.5-sunburst': ['GPT Image 2.5 Sunburst', 'best'],
        'gpt-image-2.5-flare': ['GPT Image 2.5 Flare', 'fast'],
        'gpt-image-2': ['GPT Image 2', null],
        'nano-banana-pro': ['Nano Banana Pro', null],
        'nano-banana-2.1': ['Nano Banana 2.1', null],
        'flux2pro': ['FLUX Pro', 'best_quality'],
        'flux2flex': ['FLUX Flex', 'high_res'],
        'fluxkontextpro': ['FLUX Kontext', 'best_prompt'],
        'seedream40': ['Seedream', 'combine_refs'],
        'Ideogram-v3': ['Ideogram', 'text_logos'],
        'flux-1.1-pro': ['FLUX 1.1 Pro', null],
    };

    const TEXT_ENGINES = ['thumbnail_ideas', 'texts'];

    const dialog = document.getElementById('settingsDialog');
    const headerButton = document.getElementById('settingsBtn');
    const saveButton = document.getElementById('settingsSave');
    const imageProvider = document.getElementById('settingsImageProvider');
    const imageModel = document.getElementById('settingsImageModel');
    const numImages = document.getElementById('settingsNumImages');
    const facePhotos = document.getElementById('settingsFacePhotos');

    // GET /api/settings: settings, image_models, default_image_models, face_swap_missing
    let current = null;
    // Gran Sabio's text models by provider, once loaded
    let textModels = null;
    let textModelsRequest = null;
    let opener = null;
    let comfyParameters = {};
    let comfyMetadata = {};

    async function errorText(response) {
        const body = await response.json().catch(() => null);
        const detail = body?.detail;
        if (Array.isArray(detail)) return detail.map(item => item.msg).join('; ');
        return typeof detail === 'string' ? detail : `HTTP ${response.status}`;
    }

    const ready = fetch('/api/settings').then(async (response) => {
        if (!response.ok) throw new Error(await errorText(response));
        current = await response.json();
        return current.settings;
    });
    ready.catch(error => ThumbnailApp.showToast(t('settings.load_failed', { error: error.message }), 'error'));

    // =====================================================================
    // NAMES AND LIMITS
    // =====================================================================

    function imageModelName(model) {
        return current?.comfyui_profiles?.[model]?.name ?? IMAGE_MODEL_LABELS[model]?.[0] ?? model;
    }

    /**
     * Images an image model takes per thumbnail (face photos, face sheets, style image).
     * @param {Object} image - {provider, model}; the saved image engine if omitted
     * @returns {number} Maximum images
     */
    function maxRefs(image = current.settings.image) {
        const max = current.image_models[image.provider]?.[image.model];
        if (!max) throw new Error(`No reference limit for ${image.provider} model ${image.model}`);
        return max;
    }

    /**
     * Name of a text engine for the summaries: "Claude Opus 5.5 with thinking".
     * @param {Object} engine - {provider, model, thinking_enabled}
     * @returns {string} Display name
     */
    function textEngineName(engine) {
        let name = TEXT_PROVIDER_NAMES[engine.provider];
        if (!name) throw new Error(`Unknown text provider: ${engine.provider}`);
        if (engine.model) {
            name = textModels?.[engine.provider]?.find(model => model.key === engine.model)?.name || engine.model;
        }
        return engine.thinking_enabled ? `${name} ${t('ai_config.with_thinking')}` : name;
    }

    /**
     * Load Gran Sabio's text models once (a failed load is tried again next time).
     * @returns {Promise<Object>} Models by provider: [{key, name, description}]
     */
    function loadTextModels() {
        textModelsRequest ??= fetch('/api/generation/gransabio/models')
            .then(async (response) => {
                if (!response.ok) throw new Error(await errorText(response));
                textModels = await response.json();
                return textModels;
            })
            .catch((error) => {
                textModelsRequest = null;
                throw error;
            });
        return textModelsRequest;
    }

    // =====================================================================
    // DIALOG FIELDS
    // =====================================================================

    function engineField(engine, name) {
        return dialog.querySelector(`[data-engine="${engine}"] [data-field="${name}"]`);
    }

    function optionsHtml(options) {
        return options.map(option => `<option value="${escapeHtml(option.value)}"${option.disabled ? ' disabled' : ''}>${escapeHtml(option.label)}</option>`).join('');
    }

    /**
     * Offer the models of the engine's provider and choose one.
     * @param {string} engine - thumbnail_ideas or texts
     * @param {string} selected - Model key ("" is the provider's default)
     */
    function renderTextModels(engine, selected) {
        const select = engineField(engine, 'model');
        const provider = engineField(engine, 'provider').value;
        const explicit = CATALOG_TEXT_PROVIDERS.has(provider);
        const info = engineField(engine, 'provider_info');
        info.classList.toggle('hidden', !explicit);
        info.textContent = t(provider === 'openrouter' ? 'settings.openrouter_info' : 'settings.local_text_info');
        if (!textModels) {
            // Until the list loads, the select keeps the chosen model
            select.innerHTML = optionsHtml([{ value: selected, label: t('ai_config.loading_models') }]);
            return;
        }
        const models = textModels[engineField(engine, 'provider').value] || [];
        // The thumbnail ideas get the face photos: only models that see images
        const offered = engine === 'thumbnail_ideas' ? models.filter(model => model.vision) : models;
        const options = [
            { value: '', label: t(explicit ? 'settings.choose_catalog_model' : 'generation.config.default'), disabled: explicit },
            ...offered.map(model => ({
                value: model.key,
                label: textModelLabel(model),
            })),
        ];
        if (!options.some(option => option.value === selected)) {
            const listed = models.some(model => model.key === selected);
            options.push({ value: selected, label: t(listed ? 'settings.model_no_vision' : 'settings.model_not_listed', { model: selected }) });
        }
        select.innerHTML = optionsHtml(options);
        select.value = selected;
        renderThinking(engine);
    }

    function textModelLabel(model) {
        const price = model.pricing;
        if (Number.isFinite(price?.input_per_million) && Number.isFinite(price?.output_per_million)) {
            return `${model.name} - ${t('settings.token_price', {
                input: price.input_per_million, output: price.output_per_million,
            })}`;
        }
        return model.description ? `${model.name} - ${model.description}` : model.name;
    }

    function renderThinking(engine) {
        const checkbox = engineField(engine, 'thinking_enabled');
        const provider = engineField(engine, 'provider').value;
        const model = textModels?.[provider]?.find(item => item.key === engineField(engine, 'model').value);
        checkbox.disabled = Boolean(textModels && CATALOG_TEXT_PROVIDERS.has(provider) && !model?.reasoning);
        if (checkbox.disabled) checkbox.checked = false;
        const enabled = checkbox.checked;
        engineField(engine, 'thinking_level').closest('.settings-thinking-level').classList.toggle('hidden', !enabled);
    }

    /**
     * Offer the models of the chosen image provider and choose one.
     * @param {string} selected - Model ID of that provider
     */
    function renderImageModels(selected) {
        imageModel.innerHTML = optionsHtml(Object.keys(current.image_models[imageProvider.value]).map(model => {
            const [name, note] = IMAGE_MODEL_LABELS[model] ?? [imageModelName(model), null];
            return { value: model, label: note ? `${name} · ${t(`ai_config.${note}`)}` : name };
        }));
        imageModel.value = selected;
    }

    /**
     * Say what the image model takes and keep the face photos within it.
     * @param {boolean} setToMax - Whether to send as many face photos as it takes (the model changed)
     */
    function renderFacePhotos(setToMax) {
        const max = maxRefs({ provider: imageProvider.value, model: imageModel.value });
        const model = imageModelName(imageModel.value);
        facePhotos.max = max;
        const value = parseInt(facePhotos.value, 10);
        facePhotos.value = setToMax || !Number.isInteger(value) ? max : Math.min(Math.max(value, 1), max);
        document.getElementById('settingsRefsInfo').textContent = t('ai_config.refs_info', { max, model });
        document.getElementById('settingsFacePhotosHint').textContent = t('generate_form.face_photos_hint', { max, model });
    }

    function renderComfyUI(reset = false) {
        const local = imageProvider.value === 'comfyui';
        document.getElementById('settingsComfyUI').classList.toggle('hidden', !local);
        document.getElementById('settingsImageModelLabel').textContent = t(local ? 'comfyui.workflow' : 'ai_config.model');
        const fallback = document.getElementById('settingsFallback');
        fallback.disabled = local;
        if (local) fallback.checked = false;
        if (reset) {
            comfyParameters = {};
            comfyMetadata = {};
            document.getElementById('settingsComfyStatus').textContent = '';
        }
        if (!local) return;
        const parameters = current.comfyui_profiles[imageModel.value].parameters;
        const container = document.getElementById('settingsComfyParameters');
        container.replaceChildren();
        for (const [name, parameter] of Object.entries(parameters)) {
            const label = document.createElement('label');
            label.className = 'field';
            const caption = document.createElement('span');
            caption.className = 'field-label';
            caption.textContent = parameter.label;
            const choices = comfyMetadata[name]?.choices;
            const value = comfyParameters[name] ?? parameter.default;
            const input = document.createElement(choices ? 'select' : 'input');
            input.className = choices ? 'select' : 'input';
            if (choices) {
                // Keep an unavailable saved value visible until the user replaces it.
                const available = choices.includes(value) ? choices : [value, ...choices];
                for (const choice of available) input.add(new Option(choice, choice));
            } else {
                input.type = typeof parameter.default === 'number' ? 'number' : 'text';
                if (input.type === 'number') input.step = 'any';
            }
            input.value = value;
            input.addEventListener('input', () => {
                const chosen = typeof parameter.default === 'number' ? Number(input.value) : input.value;
                if (chosen === parameter.default) delete comfyParameters[name];
                else comfyParameters[name] = chosen;
                document.getElementById('settingsComfyStatus').textContent = '';
                renderSaveState();
            });
            label.append(caption, input);
            container.append(label);
        }
    }

    function readComfyConnection() {
        return {
            url: document.getElementById('settingsComfyURL').value.trim(),
            timeout: Number(document.getElementById('settingsComfyTimeout').value),
            parameters: { ...comfyParameters },
        };
    }

    async function checkComfyUI() {
        const button = document.getElementById('settingsComfyCheck');
        const status = document.getElementById('settingsComfyStatus');
        button.disabled = true;
        status.textContent = t('comfyui.checking');
        // A late response must not change another workflow or URL selected meanwhile.
        const workflow = imageModel.value;
        const connection = readComfyConnection();
        try {
            const response = await fetch('/api/settings/comfyui/check', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ workflow, connection }),
            });
            if (!response.ok) throw new Error(await errorText(response));
            const result = await response.json();
            if (workflow !== imageModel.value || JSON.stringify(connection) !== JSON.stringify(readComfyConnection())) return;
            comfyMetadata = result.parameters;
            renderComfyUI();
            status.textContent = result.validation_error || t('comfyui.connected', { version: result.version });
        } catch (error) {
            if (workflow === imageModel.value && JSON.stringify(connection) === JSON.stringify(readComfyConnection())) {
                status.textContent = error.message;
            }
        } finally {
            button.disabled = false;
        }
    }

    let gpuOperation = false;

    function renderGPUResources(resources) {
        const container = document.getElementById('settingsGPUResources');
        container.replaceChildren();
        for (const resource of resources) {
            const group = document.createElement('div');
            group.className = 'settings-group';
            const title = document.createElement('h5');
            title.textContent = resource.id;
            const state = document.createElement('p');
            state.className = 'form-hint';
            state.textContent = t('gpu.state', { state: resource.state, count: resource.active_dispatches });
            group.append(title, state);
            const errors = [resource.last_error, ...resource.backend_errors.map(error => error.detail || error.code)].filter(Boolean);
            if (errors.length) {
                const error = document.createElement('p');
                error.className = 'form-hint';
                error.textContent = errors.join('; ');
                group.append(error);
            }
            const busy = !['idle', 'unverified'].includes(resource.state) || resource.active_dispatches > 0
                || resource.requires_reconciliation || errors.length > 0 || resource.external_service?.idle === false;
            function addAction(label, request) {
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'btn btn-secondary';
                button.textContent = label;
                button.disabled = busy;
                if (busy) button.title = t('gpu.busy');
                button.addEventListener('click', () => unloadGPUModel(resource.id, request));
                return button;
            }
            for (const model of resource.loaded_models) {
                const row = document.createElement('div');
                const name = document.createElement('p');
                name.textContent = t('gpu.model', {
                    model: model.name, memory: (model.size_vram / (1024 ** 3)).toFixed(1),
                });
                row.append(name, addAction(t('gpu.unload'), {
                    target: 'ollama', endpoint_id: model.endpoint_id, model: model.name,
                }));
                group.append(row);
            }
            if (resource.has_comfyui) {
                const service = resource.external_service;
                const memoryReleased = service?.memory_released === true
                    && Array.isArray(service.loaded_model_classes) && service.loaded_model_classes.length === 0;
                const button = addAction(t(memoryReleased ? 'gpu.images_released' : 'gpu.unload_images'), { target: 'external' });
                if (memoryReleased) button.disabled = true;
                group.append(button);
            }
            container.append(group);
        }
    }

    async function fetchGPUResources() {
        const response = await fetch('/api/settings/local-resources');
        if (!response.ok) throw new Error(await errorText(response));
        const result = await response.json();
        renderGPUResources(result.resources);
        return result.resources;
    }

    async function refreshGPUResources() {
        if (gpuOperation) return;
        gpuOperation = true;
        const button = document.getElementById('settingsGPURefresh');
        const status = document.getElementById('settingsGPUStatus');
        button.disabled = true;
        document.getElementById('settingsGPUResources').replaceChildren();
        status.textContent = t('gpu.loading');
        try {
            const resources = await fetchGPUResources();
            status.textContent = resources.length ? '' : t('gpu.empty');
        } catch (error) {
            status.textContent = error.message;
        } finally {
            gpuOperation = false;
            button.disabled = false;
        }
    }

    async function unloadGPUModel(resourceId, request) {
        if (gpuOperation) return;
        gpuOperation = true;
        const refresh = document.getElementById('settingsGPURefresh');
        const status = document.getElementById('settingsGPUStatus');
        refresh.disabled = true;
        for (const button of document.querySelectorAll('#settingsGPUResources button')) button.disabled = true;
        status.textContent = t('gpu.releasing');
        try {
            const response = await fetch(`/api/settings/local-resources/${encodeURIComponent(resourceId)}/unload`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request),
            });
            if (!response.ok) throw new Error(await errorText(response));
            await fetchGPUResources();
            status.textContent = t('gpu.released');
        } catch (error) {
            // Keep stale action buttons disabled until the owner refreshes the state.
            status.textContent = error.message;
        } finally {
            gpuOperation = false;
            refresh.disabled = false;
        }
    }

    async function importComfyWorkflow(event) {
        const file = event.target.files[0];
        if (!file) return;
        try {
            if (file.size > 1024 * 1024) throw new Error(t('comfyui.file_size'));
            const response = await fetch('/api/settings/comfyui/workflows', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: await file.text(),
            });
            if (!response.ok) throw new Error(await errorText(response));
            const imported = await response.json();
            const catalog = await fetch('/api/settings');
            if (!catalog.ok) throw new Error(await errorText(catalog));
            const data = await catalog.json();
            current.image_models = data.image_models;
            current.comfyui_profiles = data.comfyui_profiles;
            imageProvider.value = 'comfyui';
            renderImageModels(imported.id);
            renderComfyUI(true);
            renderFacePhotos(true);
            renderSaveState();
            document.getElementById('settingsComfyStatus').textContent = t('comfyui.imported');
        } catch (error) {
            showError(error.message);
        } finally {
            event.target.value = '';
        }
    }

    function fill(settings) {
        for (const engine of TEXT_ENGINES) {
            const values = settings[engine];
            engineField(engine, 'provider').value = values.provider;
            engineField(engine, 'thinking_enabled').checked = values.thinking_enabled;
            engineField(engine, 'thinking_level').value = values.thinking_level;
            renderTextModels(engine, values.model);
            renderThinking(engine);
        }

        imageProvider.value = settings.image.provider;
        renderImageModels(settings.image.model);

        const defaults = settings.generation;
        numImages.value = defaults.num_images;
        document.getElementById('settingsExpression').value = defaults.expression ?? '';
        document.getElementById('settingsAvoidRepeat').checked = defaults.avoid_repeat;
        facePhotos.value = defaults.face_photos;
        renderFacePhotos(false);
        document.getElementById('settingsFallback').checked = defaults.provider_fallback;
        document.getElementById('settingsComfyURL').value = settings.comfyui.url;
        document.getElementById('settingsComfyTimeout').value = settings.comfyui.timeout;
        comfyParameters = { ...settings.comfyui.parameters };
        comfyMetadata = {};
        document.getElementById('settingsComfyStatus').textContent = '';
        renderComfyUI();

        // Face swap keeps its value but cannot be changed while its models are missing
        const faceSwap = document.getElementById('settingsFaceSwap');
        const missing = current.face_swap_missing;
        faceSwap.checked = defaults.face_swap;
        faceSwap.disabled = missing.length > 0;
        document.getElementById('settingsFaceSwapRow').classList.toggle('is-unavailable', missing.length > 0);
        const reason = document.getElementById('settingsFaceSwapMissing');
        reason.classList.toggle('hidden', missing.length === 0);
        reason.querySelector('span').textContent = t('generate_form.face_swap_missing', { missing: missing.join(', ') });

        const restoration = document.getElementById('settingsFaceRestoration');
        const restorationMissing = current.face_restoration_missing;
        restoration.checked = defaults.face_restoration;
        restoration.disabled = restorationMissing.length > 0;
        document.getElementById('settingsFaceRestorationRow').classList.toggle('is-unavailable', restoration.disabled);
        const restorationReason = document.getElementById('settingsFaceRestorationMissing');
        restorationReason.classList.toggle('hidden', !restoration.disabled);
        restorationReason.querySelector('span').textContent = t('generate_form.face_restoration_missing', {
            missing: restorationMissing.join(', '),
        });
        renderRestoration();
    }

    function renderRestoration() {
        const faceSwap = document.getElementById('settingsFaceSwap');
        document.getElementById('settingsFaceRestorationRow').classList.toggle('hidden', !faceSwap.checked || faceSwap.disabled);
    }

    /**
     * The settings as the dialog has them, in the order the API returns them.
     * @returns {Object} Settings
     */
    function read() {
        const engine = name => ({
            provider: engineField(name, 'provider').value,
            model: engineField(name, 'model').value,
            thinking_enabled: engineField(name, 'thinking_enabled').checked,
            thinking_level: engineField(name, 'thinking_level').value,
        });
        return {
            thumbnail_ideas: engine('thumbnail_ideas'),
            texts: engine('texts'),
            image: { provider: imageProvider.value, model: imageModel.value },
            generation: {
                num_images: parseInt(numImages.value, 10),
                expression: document.getElementById('settingsExpression').value || null,
                face_swap: document.getElementById('settingsFaceSwap').checked,
                face_restoration: document.getElementById('settingsFaceRestoration').checked,
                avoid_repeat: document.getElementById('settingsAvoidRepeat').checked,
                face_photos: parseInt(facePhotos.value, 10),
                provider_fallback: imageProvider.value !== 'comfyui' && document.getElementById('settingsFallback').checked,
            },
            comfyui: readComfyConnection(),
        };
    }

    function isDirty() {
        return JSON.stringify(read()) !== JSON.stringify(current.settings);
    }

    function showError(message) {
        const error = document.getElementById('settingsError');
        error.textContent = message;
        error.classList.toggle('hidden', !message);
    }

    function renderSaveState() {
        saveButton.disabled = !isDirty();
    }

    // =====================================================================
    // OPEN, CLOSE, SAVE
    // =====================================================================

    function isOpen() {
        return dialog.classList.contains('visible');
    }

    /**
     * Open the dialog with the saved settings, at one of its sections.
     * @param {string} section - thumbnails or texts
     */
    async function open(section) {
        try {
            await ready;
        } catch (error) {
            ThumbnailApp.showToast(t('settings.load_failed', { error: error.message }), 'error');
            return;
        }
        opener = document.activeElement;
        fill(current.settings);
        showError('');
        renderSaveState();
        dialog.classList.add('visible');
        headerButton.setAttribute('aria-expanded', 'true');

        const target = dialog.querySelector(`[data-settings-section="${section}"]`);
        if (!target) throw new Error(`Unknown settings section: ${section}`);
        dialog.querySelector('.settings-body').scrollTop = section === 'thumbnails' ? 0 : target.offsetTop;
        target.querySelector('select, button').focus({ preventScroll: true });

        if (!textModels) {
            loadTextModels().then(() => {
                if (!isOpen()) return;
                // What the owner already chose while the list loaded stays chosen
                for (const engine of TEXT_ENGINES) {
                    renderTextModels(engine, engineField(engine, 'model').value);
                }
                renderSaveState();
            }).catch(error => {
                showError(t('settings.models_failed', { error: error.message }));
                // The chosen models stay, named by their keys instead of "Loading"
                for (const engine of TEXT_ENGINES) {
                    const option = engineField(engine, 'model').options[0];
                    option.textContent = option.value || t('generation.config.default');
                }
            });
        }
    }

    function close() {
        dialog.classList.remove('visible');
        headerButton.setAttribute('aria-expanded', 'false');
        opener?.focus();
    }

    async function save() {
        const settings = read();
        saveButton.disabled = true;
        showError('');
        try {
            const response = await fetch('/api/settings', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings),
            });
            if (!response.ok) throw new Error(await errorText(response));
            const previous = current.settings;
            current = await response.json();
            close();
            ThumbnailApp.showToast(t('settings.saved'), 'success');
            document.dispatchEvent(new CustomEvent('settings:changed', {
                detail: { settings: current.settings, previous },
            }));
        } catch (error) {
            showError(t('settings.save_failed', { error: error.message }));
            renderSaveState();
        }
    }

    // =====================================================================
    // EVENTS
    // =====================================================================

    document.addEventListener('click', (e) => {
        const openButton = e.target.closest('[data-open-settings]');
        if (openButton) {
            e.preventDefault();
            open(openButton.dataset.openSettings);
        }
    });

    dialog.addEventListener('click', (e) => {
        // A click outside the dialog only closes it when nothing would be lost
        if (e.target.closest('[data-settings-cancel]') || (e.target === dialog && !isDirty())) {
            close();
        }
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isOpen()) close();
    });
    saveButton.addEventListener('click', save);
    document.getElementById('settingsComfyCheck').addEventListener('click', checkComfyUI);
    document.getElementById('settingsGPURefresh').addEventListener('click', refreshGPUResources);
    document.getElementById('settingsComfyImport').addEventListener('click', () => document.getElementById('settingsComfyFile').click());
    document.getElementById('settingsComfyFile').addEventListener('change', importComfyWorkflow);

    dialog.addEventListener('change', (e) => {
        const field = e.target.dataset.field;
        const engine = e.target.closest('[data-engine]')?.dataset.engine;
        if (engine && field === 'provider') renderTextModels(engine, '');
        if (engine && field === 'model') renderThinking(engine);
        if (engine && field === 'thinking_enabled') renderThinking(engine);
        if (e.target === imageProvider) {
            renderImageModels(current.default_image_models[imageProvider.value]);
            renderFacePhotos(true);
            renderComfyUI(true);
        }
        if (e.target.id === 'settingsFaceSwap') renderRestoration();
        if (e.target === imageModel) {
            renderFacePhotos(true);
            renderComfyUI(true);
        }
        if (e.target.id === 'settingsComfyURL') {
            comfyMetadata = {};
            document.getElementById('settingsComfyStatus').textContent = '';
            renderComfyUI();
        }
        if (e.target === facePhotos) renderFacePhotos(false);
        if (e.target === numImages) {
            const value = parseInt(numImages.value, 10);
            numImages.value = Number.isInteger(value)
                ? Math.min(Math.max(value, 1), parseInt(numImages.max, 10))
                : current.settings.generation.num_images;
        }
        renderSaveState();
    });
    dialog.addEventListener('input', renderSaveState);

    window.AppSettings = {
        ready,
        get: () => current.settings,
        maxRefs,
        imageModelName,
        textEngineName,
        loadTextModels,
        open,
    };
})();
