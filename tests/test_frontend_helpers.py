"""Two helpers of static/js/app.js, run with Node.js: the date the server gives
is UTC, and a failed request says why. Node.js must be on PATH; no browser."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "static" / "js" / "app.js"

CHECKS = r"""
const assert = require('node:assert/strict');

// SQLite's timestamps are UTC without saying so: read as local time they
// were shown hours off (5 in Miami)
assert.equal(parseServerDate('2026-01-08 19:45:22').toISOString(), '2026-01-08T19:45:22.000Z');
assert.equal(parseServerDate('2026-01-08T19:45:22').toISOString(), '2026-01-08T19:45:22.000Z');
// A date that carries its zone is read as it is
assert.equal(parseServerDate('2026-01-08T19:45:22.000Z').toISOString(), '2026-01-08T19:45:22.000Z');
assert.equal(parseServerDate('2026-01-08T14:45:22-05:00').toISOString(), '2026-01-08T19:45:22.000Z');
assert.equal(parseServerDate('2026-01-08 19:45:22').getHours(), 14);

const response = (status, body) => ({ status, json: async () => {
    if (body === undefined) throw new SyntaxError('not JSON');
    return body;
} });

(async () => {
    assert.equal(await readApiError(response(409, { detail: 'The video is generating' })), 'The video is generating');
    // A request that does not validate: every field with what is wrong in it
    assert.equal(
        await readApiError(response(422, { detail: [
            { loc: ['body', 'num_images'], msg: 'Input should be less than or equal to 10' },
            { loc: ['body', 'cluster_ids'], msg: 'Field required' },
        ] })),
        'num_images: Input should be less than or equal to 10; cluster_ids: Field required'
    );
    assert.equal(await readApiError(response(500, undefined)), 'HTTP 500');
    assert.equal(await readApiError(response(502, {})), 'HTTP 502');
})().catch((error) => { console.error(error); process.exit(1); });
"""


def function_source(name: str) -> str:
    """A top-level function of app.js, from its declaration to its closing brace."""
    match = re.search(rf"^(?:async )?function {name}\(.*?^}}\n", APP_JS.read_text(encoding="utf-8"), re.M | re.S)
    assert match, f"{name} not found in app.js"
    return match.group(0)


def test_server_dates_and_request_errors():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    script = function_source("parseServerDate") + function_source("readApiError") + CHECKS
    result = subprocess.run(
        [node, "-e", script],
        capture_output=True, text=True, timeout=30,
        env={"TZ": "America/New_York", "SystemRoot": r"C:\Windows", "PATH": str(Path(node).parent)},
    )

    assert result.returncode == 0, result.stderr
