"""Exercise production modal behavior with Node.js and an isolated DOM fixture.

Run without application imports: python -B -m unittest discover -s tests -p test_modal.py -v
Node.js must be available on PATH; no browser, npm packages, or server is needed.
"""

from pathlib import Path
import shutil
import subprocess
import unittest


APP_JS = Path(__file__).resolve().parents[1] / "static" / "js" / "app.js"
DOM_FIXTURE = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
    constructor() {
        this.listeners = new Map();
        this.children = new Map();
        this.style = {};
        const classes = new Set();
        this.classList = {
            add: (name) => classes.add(name),
            remove: (name) => classes.delete(name),
            contains: (name) => classes.has(name),
        };
    }
    addEventListener(name, handler) {
        if (!this.listeners.has(name)) this.listeners.set(name, new Set());
        this.listeners.get(name).add(handler);
    }
    removeEventListener(name, handler) {
        this.listeners.get(name)?.delete(handler);
    }
    handlers(name) { return [...(this.listeners.get(name) || [])]; }
    dispatch(name, details = {}) {
        const event = {
            target: this,
            defaultPrevented: false,
            preventDefault() { this.defaultPrevented = true; },
            ...details,
        };
        for (const handler of this.handlers(name)) handler(event);
        return event;
    }
    querySelector(selector) {
        if (!this.children.has(selector)) this.children.set(selector, new Element());
        return this.children.get(selector);
    }
    focus() { document.activeElement = this; }
}

const elementsById = new Map();
const document = new Element();
document.createElement = () => new Element();
document.getElementById = (id) => elementsById.get(id) || null;
document.body = { appendChild(element) { elementsById.set(element.id, element); } };
document.head = { appendChild() {} };
const window = {};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
    document, window, t: (key) => key,
});
const showModal = window.ThumbnailApp.showModal;
const overlay = () => document.getElementById('modalOverlay');
const confirmButton = () => overlay().querySelector('.btn-confirm');
const cancelButton = () => overlay().querySelector('.btn-cancel');
function assertClosed() {
    assert.equal(overlay().classList.contains('visible'), false);
    assert.equal(document.handlers('keydown').length, 0);
    assert.equal(confirmButton().handlers('click').length, 0);
    assert.equal(cancelButton().handlers('click').length, 0);
    assert.equal(overlay().handlers('click').length, 0);
}
"""


class ModalTests(unittest.TestCase):
    def run_scenario(self, scenario):
        node = shutil.which("node")
        self.assertIsNotNone(node, "Node.js is required to run modal regression tests")
        script = (
            DOM_FIXTURE
            + "\n(async () => {\n"
            + scenario
            + "\n})().then(() => console.log('MODAL_SCENARIO_OK')).catch(error => { console.error(error); process.exitCode = 1; });"
        )
        result = subprocess.run(
            [node, "-e", script, str(APP_JS)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), "MODAL_SCENARIO_OK", "Modal scenario did not finish")

    def test_replacement_cancels_original_action_and_escape_cancels_current(self):
        self.run_scenario(r"""
            let destructiveActions = 0;
            const first = showModal({title: 'Delete fixture?'});
            const action = first.then(confirmed => { if (confirmed) destructiveActions++; });
            const second = showModal({title: 'Another confirmation'});
            assert.equal(await first, false);
            await action;
            assert.equal(destructiveActions, 0);
            assert.equal(document.handlers('keydown').length, 1);
            assert.equal(confirmButton().handlers('click').length, 1);
            assert.equal(overlay().handlers('click').length, 1);
            assert.equal(document.activeElement, confirmButton());
            assert.equal(document.dispatch('keydown', {key: 'Escape'}).defaultPrevented, true);
            assert.equal(await second, false);
            assertClosed();
        """)

    def test_stale_events_cannot_close_or_accept_the_replacement(self):
        self.run_scenario(r"""
            const first = showModal({title: 'First'});
            const staleKey = document.handlers('keydown')[0];
            const staleConfirm = confirmButton().handlers('click')[0];
            const staleOverlay = overlay().handlers('click')[0];
            const second = showModal({title: 'Second'});
            assert.equal(await first, false);
            staleConfirm();
            staleKey({key: 'Escape', preventDefault() {}});
            staleOverlay({target: overlay()});
            assert.equal(overlay().classList.contains('visible'), true);
            assert.equal(document.handlers('keydown').length, 1);
            const third = showModal({title: 'Third'});
            assert.equal(await second, false);
            assert.equal(document.dispatch('keydown', {key: 'Enter'}).defaultPrevented, true);
            assert.equal(await third, true);
            assertClosed();
        """)

    def test_click_actions_and_overlay_dismissal_clean_up_handlers(self):
        self.run_scenario(r"""
            const confirmed = showModal({title: 'Confirm'});
            confirmButton().dispatch('click');
            assert.equal(await confirmed, true);
            assertClosed();
            const cancelled = showModal({title: 'Cancel'});
            cancelButton().dispatch('click');
            assert.equal(await cancelled, false);
            assertClosed();
            const dismissed = showModal({title: 'Dismiss'});
            overlay().dispatch('click', {target: confirmButton()});
            assert.equal(overlay().classList.contains('visible'), true);
            overlay().dispatch('click');
            assert.equal(await dismissed, false);
            assertClosed();
        """)

    def test_repeated_replacement_keeps_only_one_active_dialog(self):
        self.run_scenario(r"""
            let current = showModal({title: 'Initial'});
            for (let index = 0; index < 25; index++) {
                const next = showModal({title: `Confirmation ${index}`});
                assert.equal(await current, false);
                current = next;
                assert.equal(document.handlers('keydown').length, 1);
                assert.equal(cancelButton().handlers('click').length, 1);
            }
            document.dispatch('keydown', {key: 'Tab'});
            assert.equal(overlay().classList.contains('visible'), true);
            document.dispatch('keydown', {key: 'Enter'});
            assert.equal(await current, true);
            assertClosed();
        """)


if __name__ == "__main__":
    unittest.main()
