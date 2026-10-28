"""Gran Sabio control of local model memory and shared GPU reservations."""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import quote, urlsplit
from uuid import UUID, uuid4

import requests

import config


class LocalGPUError(RuntimeError):
    def __init__(self, message, status_code=502, code="gpu_request_failed"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class LocalGPUCancelled(LocalGPUError):
    """The application canceled a request while waiting for the GPU."""


def _url_identity(url):
    parsed = urlsplit(url)
    return (parsed.scheme.lower(), parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80),
            parsed.path.rstrip("/"))


def _check_cancelled(check):
    if check and check():
        raise LocalGPUCancelled("Generation canceled while waiting for the shared GPU")


class Client:
    def __init__(self):
        self.url = config.GRANSABIO_LLM_URL.rstrip("/")
        self.session = requests.Session()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.session.close()

    def request(self, method, path, *, body=None, timeout=30):
        try:
            with self.session.request(method, self.url + path, json=body,
                                      timeout=(5, timeout), allow_redirects=False) as response:
                try:
                    data = response.json()
                except ValueError as error:
                    raise LocalGPUError(f"Gran Sabio GPU API returned invalid JSON (HTTP {response.status_code})") from error
                if response.status_code not in (200, 202):
                    detail = data.get("error", data.get("detail", data)) if isinstance(data, dict) else data
                    code = detail.get("code", "gpu_request_failed") if isinstance(detail, dict) else "gpu_request_failed"
                    message = detail.get("detail", str(detail)) if isinstance(detail, dict) else str(detail)
                    raise LocalGPUError(f"Gran Sabio GPU API: {message}", response.status_code, code)
                if not isinstance(data, dict):
                    raise LocalGPUError("Gran Sabio GPU API returned an invalid response")
                return response.status_code, data
        except requests.RequestException as error:
            raise LocalGPUError(f"Gran Sabio GPU API is unavailable: {error}") from error

    def resources(self):
        status, data = self.request("GET", "/local-resources")
        resources = data.get("resources")
        if status != 200 or not isinstance(resources, list) or any(not isinstance(item, dict) for item in resources):
            raise LocalGPUError("Gran Sabio returned an invalid GPU resource list")
        return resources

    def resource_for_comfyui(self, url):
        matches = [item for item in self.resources()
                   if item.get("external_service_url") and _url_identity(item["external_service_url"]) == _url_identity(url)]
        if len(matches) != 1:
            raise LocalGPUError("Configure exactly one Gran Sabio GPU resource for this ComfyUI URL before generating images", 400)
        return matches[0]["id"]

    def operation(self, resource_id, action, body, timeout, *, wait_busy=False, cancellation_check=None, received=None):
        """Advance a pending operation with the same ID; transport uncertainty is fatal."""
        deadline = time.monotonic() + timeout
        path = f"/local-resources/{quote(resource_id, safe='')}/{action}"
        while True:
            _check_cancelled(cancellation_check)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LocalGPUError(f"Timed out waiting for GPU {action} {body['request_id']}; check Gran Sabio", 408)
            try:
                status, result = self.request("POST", path, body=body, timeout=min(30, remaining))
            except LocalGPUError as error:
                if not (wait_busy and error.status_code == 409 and error.code == "resource_busy"):
                    raise
            else:
                if (result.get("resource_id") != resource_id or result.get("request_id") != body["request_id"]
                        or result.get("complete") is not (status == 200)):
                    raise LocalGPUError(f"Gran Sabio returned an invalid GPU {action} receipt")
                if received:
                    received(result)
                if status == 200:
                    return result
            pause_until = min(deadline, time.monotonic() + 1)
            while time.monotonic() < pause_until:
                _check_cancelled(cancellation_check)
                time.sleep(min(0.2, max(0, pause_until - time.monotonic())))

    def unload(self, resource_id, request):
        result = self.operation(resource_id, "unload", request, 120)
        if result.get("state") not in ("idle", "unverified") or result.get("reservation", {}) is not None:
            raise LocalGPUError("Gran Sabio did not confirm that model memory was released")
        return result

    def reserve_images(self, resource_id, request_id, receipt_path, timeout, cancellation_check=None):
        return ImageReservation(self, resource_id, request_id, receipt_path, timeout, cancellation_check)


class ImageReservation:
    """One image batch owns the GPU until Gran Sabio confirms memory release."""

    def __init__(self, client, resource_id, request_id, receipt_path, timeout, cancellation_check):
        self.client = client
        self.resource_id = resource_id
        self.request_id = str(UUID(request_id))
        self.release_id = str(uuid4())
        self.receipt_path = Path(receipt_path)
        self.timeout = timeout
        self.cancellation_check = cancellation_check
        self.owned = False
        self.receipt = {"resource_id": resource_id, "request_id": self.request_id,
                        "reservation_id": self.request_id, "release_request_id": self.release_id, "purpose": "comfyui"}

    def record(self, state, error=None):
        self.receipt.update(state=state)
        if error is not None:
            self.receipt["error"] = str(error)
        self.receipt_path.write_text(json.dumps(self.receipt, indent=2), encoding="utf-8")

    def _received(self, result):
        reservation = result.get("reservation")
        if (not isinstance(reservation, dict) or reservation.get("reservation_id") != self.request_id
                or reservation.get("request_id") != self.request_id
                or result.get("state") not in ("draining", "unloading", "reserved")):
            raise LocalGPUError("Gran Sabio returned an invalid GPU reservation")
        self.owned = True
        self.record(result["state"])
        if result["complete"] and (result["state"] != "reserved" or reservation.get("ready") is not True):
            raise LocalGPUError("Gran Sabio has not made this GPU reservation ready")

    def __enter__(self):
        self.record("requesting")
        try:
            self.client.operation(self.resource_id, "reserve", {"request_id": self.request_id, "purpose": "comfyui"},
                                  self.timeout, wait_busy=True, cancellation_check=self.cancellation_check,
                                  received=self._received)
            _check_cancelled(self.cancellation_check)
            return self
        except BaseException as error:
            self.record("acquire_failed", error)
            if self.owned:
                self.release()
            raise

    def release(self):
        self.record("releasing")
        try:
            result = self.client.operation(self.resource_id, "release", {
                "request_id": self.release_id, "reservation_id": self.request_id,
            }, self.timeout, wait_busy=True)
            if result.get("state") not in ("idle", "unverified") or result.get("reservation", {}) is not None:
                raise LocalGPUError("Gran Sabio did not confirm the GPU reservation release")
        except BaseException as error:
            self.record("release_failed", error)
            raise LocalGPUError(f"GPU reservation {self.request_id} could not be released. Check Gran Sabio before another local job: {error}") from error
        self.owned = False
        self.record("released")

    def __exit__(self, *_args):
        self.release()
