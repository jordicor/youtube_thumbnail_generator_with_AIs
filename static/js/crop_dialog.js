/**
 * Crop Dialog
 * ===========
 * Chooses the part of an image a reference photo uses
 * (templates/components/crop_dialog.html). The box is dragged to move it, its
 * handles resize it and dragging outside it draws a new one; arrow keys move it
 * (Shift for bigger steps). Boxes are [left, top, right, bottom] in pixels of
 * the original image.
 */

const MIN_SIDE = 16;
const KEY_STEP = 0.01;
const KEY_STEP_LARGE = 0.05;

const dialog = () => document.getElementById('cropDialog');
const image = () => document.getElementById('cropImage');
const box = () => document.getElementById('cropBox');

let current = null;   // {box, options, resolve}
let drag = null;      // {mode, handle, start, origin}

function size() {
    return [image().naturalWidth, image().naturalHeight];
}

function clampBox([left, top, right, bottom]) {
    const [width, height] = size();
    left = Math.max(0, Math.min(left, width - MIN_SIDE));
    top = Math.max(0, Math.min(top, height - MIN_SIDE));
    right = Math.min(width, Math.max(right, left + MIN_SIDE));
    bottom = Math.min(height, Math.max(bottom, top + MIN_SIDE));
    return [Math.round(left), Math.round(top), Math.round(right), Math.round(bottom)];
}

function drawBox() {
    const [width, height] = size();
    const [left, top, right, bottom] = current.box;
    const position = {
        left: `${(left / width) * 100}%`,
        top: `${(top / height) * 100}%`,
        width: `${((right - left) / width) * 100}%`,
        height: `${((bottom - top) / height) * 100}%`,
    };
    Object.assign(box().style, position);
    Object.assign(document.getElementById('cropShade').style, position);
}

function setBox(value) {
    current.box = clampBox(value);
    drawBox();
    showError('');
}

function showError(message) {
    const error = document.getElementById('cropError');
    error.classList.toggle('hidden', !message);
    error.querySelector('span').textContent = message;
}

/** Point of a pointer event in pixels of the original image. */
function imagePoint(event) {
    const rect = image().getBoundingClientRect();
    const [width, height] = size();
    return [
        Math.max(0, Math.min(width, ((event.clientX - rect.left) / rect.width) * width)),
        Math.max(0, Math.min(height, ((event.clientY - rect.top) / rect.height) * height)),
    ];
}

function onPointerDown(event) {
    if (!current || event.button !== 0) return;
    event.preventDefault();
    const handle = event.target.closest('.crop-handle')?.dataset.handle;
    const point = imagePoint(event);
    if (handle) {
        drag = { mode: 'resize', handle, start: point, origin: [...current.box] };
    } else if (event.target.closest('.crop-box')) {
        drag = { mode: 'move', start: point, origin: [...current.box] };
    } else {
        drag = { mode: 'draw', start: point, origin: [...current.box] };
    }
    document.getElementById('cropStage').setPointerCapture(event.pointerId);
}

function onPointerMove(event) {
    if (!drag) return;
    const [x, y] = imagePoint(event);
    const [dx, dy] = [x - drag.start[0], y - drag.start[1]];
    let [left, top, right, bottom] = drag.origin;
    if (drag.mode === 'move') {
        const [width, height] = size();
        const w = right - left;
        const h = bottom - top;
        left = Math.max(0, Math.min(left + dx, width - w));
        top = Math.max(0, Math.min(top + dy, height - h));
        setBox([left, top, left + w, top + h]);
        return;
    }
    if (drag.mode === 'draw') {
        const [sx, sy] = drag.start;
        // A click without dragging keeps the box
        if (Math.abs(dx) < 3 && Math.abs(dy) < 3) return;
        setBox([Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)]);
        return;
    }
    if (drag.handle.includes('w')) left = Math.min(left + dx, right - MIN_SIDE);
    if (drag.handle.includes('e')) right = Math.max(right + dx, left + MIN_SIDE);
    if (drag.handle.includes('n')) top = Math.min(top + dy, bottom - MIN_SIDE);
    if (drag.handle.includes('s')) bottom = Math.max(bottom + dy, top + MIN_SIDE);
    setBox([left, top, right, bottom]);
}

function onPointerUp() {
    drag = null;
}

function onKeyDown(event) {
    if (!current) return;
    if (event.key === 'Escape') {
        event.preventDefault();
        close(null);
        return;
    }
    const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (!(event.key in moves) || event.target !== box()) return;
    event.preventDefault();
    const [width, height] = size();
    const step = event.shiftKey ? KEY_STEP_LARGE : KEY_STEP;
    const [mx, my] = moves[event.key];
    const [left, top, right, bottom] = current.box;
    const dx = Math.max(-left, Math.min(mx * width * step, width - right));
    const dy = Math.max(-top, Math.min(my * height * step, height - bottom));
    setBox([left + dx, top + dy, right + dx, bottom + dy]);
}

async function apply() {
    const confirmRow = document.getElementById('cropConfirmRow');
    const confirmed = document.getElementById('cropConfirm').checked;
    if (!confirmRow.classList.contains('hidden') && !confirmed) {
        showError(t('crop.confirm_needed'));
        return;
    }
    const button = document.getElementById('cropApply');
    button.disabled = true;
    try {
        if (current.options.validate) await current.options.validate(current.box, confirmed);
        close({ box: current.box, confirmed });
    } catch (error) {
        showError(error.message);
    } finally {
        button.disabled = false;
    }
}

function close(result) {
    if (!current) return;
    const { resolve } = current;
    current = null;
    drag = null;
    dialog().classList.remove('visible');
    resolve(result);
}

let ready = false;

function setup() {
    if (ready) return;
    ready = true;
    const stage = document.getElementById('cropStage');
    stage.addEventListener('pointerdown', onPointerDown);
    stage.addEventListener('pointermove', onPointerMove);
    stage.addEventListener('pointerup', onPointerUp);
    stage.addEventListener('pointercancel', onPointerUp);
    dialog().addEventListener('keydown', onKeyDown);
    document.getElementById('cropApply').addEventListener('click', apply);
    document.getElementById('cropCancel').addEventListener('click', () => close(null));
    document.getElementById('cropFull').addEventListener('click', () => setBox([0, 0, ...size()]));
    document.getElementById('cropAuto').addEventListener('click', () => {
        if (current?.options.autoBox) setBox(current.options.autoBox);
    });
    document.getElementById('cropConfirm').addEventListener('change', () => showError(''));
}

/**
 * Let the user choose a crop of an image.
 * @param {Object} options
 * @param {string} options.title - Dialog title
 * @param {string} options.hint - What the crop is for
 * @param {string} options.src - Image URL
 * @param {number[]} [options.box] - Starting box; the whole image when missing
 * @param {number[]} [options.autoBox] - Box of "Automatic crop", hidden when missing
 * @param {string} [options.confirmText] - Statement the user must tick (uploaded photos)
 * @param {Function} [options.validate] - async (box, confirmed); throws to refuse the crop
 * @returns {Promise<{box: number[], confirmed: boolean}|null>} null when cancelled
 */
export async function openCropDialog(options) {
    setup();
    if (current) close(null);
    const img = image();
    await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = () => reject(new Error(t('crop.image_error')));
        img.src = options.src;
    });
    document.getElementById('cropDialogTitle').textContent = options.title;
    document.getElementById('cropDialogHint').textContent = options.hint;
    document.getElementById('cropAuto').classList.toggle('hidden', !options.autoBox);
    const confirmRow = document.getElementById('cropConfirmRow');
    confirmRow.classList.toggle('hidden', !options.confirmText);
    document.getElementById('cropConfirmText').textContent = options.confirmText || '';
    document.getElementById('cropConfirm').checked = false;

    return new Promise(resolve => {
        current = { box: [0, 0, ...size()], options, resolve };
        setBox(options.box || [0, 0, ...size()]);
        dialog().classList.add('visible');
        box().focus();
    });
}
