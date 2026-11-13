"""Native ComfyUI transport and portable image workflow profiles."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import time
import uuid
from urllib.parse import urlsplit
from typing import Literal

import requests
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import config
from services.local_gpu_service import Client as GPUClient, LocalGPUError, LocalGPUCancelled


class ComfyUIError(RuntimeError):
    """Local generation failed; never route it to another provider."""


class ComfyUICancelledError(ComfyUIError):
    """The caller canceled this generation."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Connection(StrictModel):
    url: str = Field(default_factory=lambda: config.COMFYUI_URL)
    timeout: int = Field(default=1800, ge=5, le=21600)
    parameters: dict[str, str | int | float] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value):
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Use an HTTP(S) ComfyUI URL without credentials, query, or fragment")
        _ = parsed.port
        return value.rstrip("/")


class Binding(StrictModel):
    node: str
    input: str


class Parameter(Binding):
    label: str


class Profile(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")
    name: str = Field(min_length=1, max_length=120)
    workflow: dict[str, dict]
    prompt: list[Binding] = Field(min_length=1)
    seed: list[Binding] = Field(min_length=1)
    references: list[Binding] = Field(min_length=1, max_length=64)
    output_node: str
    prompt_format: Literal["qwen", "minimax", "json"] = "json"
    parameters: dict[str, Parameter] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_graph(self):
        if len(self.model_dump_json().encode("utf-8")) > 1024 * 1024:
            raise ValueError("Workflow profiles must be smaller than 1 MB")
        if not self.workflow or len(self.workflow) > 200:
            raise ValueError("A profile needs an API workflow with 1-200 nodes")
        for key, node in self.workflow.items():
            if not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), dict):
                raise ValueError(f"Node {key} needs class_type and inputs; export the API format")
        if self.output_node not in self.workflow:
            raise ValueError("The output node does not exist")
        for node in self.workflow.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and (len(value) != 2 or not isinstance(value[0], str) or value[0] not in self.workflow
                                               or not isinstance(value[1], int) or value[1] < 0):
                    raise ValueError("Workflow links must be [existing_node_id, output_index]")
        bindings = self.prompt + self.seed + self.references + list(self.parameters.values())
        targets = [(binding.node, binding.input) for binding in bindings]
        if len(targets) != len(set(targets)):
            raise ValueError("Workflow input bindings must be unique")
        for binding in bindings:
            if binding.node not in self.workflow:
                raise ValueError(f"Binding node {binding.node} does not exist")
        for binding in self.prompt + self.seed + list(self.parameters.values()):
            if binding.input not in self.workflow[binding.node]["inputs"]:
                raise ValueError(f"Bound input {binding.node}.{binding.input} does not exist")
        for binding in self.references:
            if binding.input in self.workflow[binding.node]["inputs"]:
                raise ValueError("Reference bindings must target unconnected inputs")
        return self

    def graph(self, prompt, seed, references, parameters):
        if not 1 <= len(references) <= len(self.references):
            raise ValueError(f"{self.name} needs 1-{len(self.references)} reference images")
        unknown = set(parameters) - self.parameters.keys()
        if unknown:
            raise ValueError(f"Unknown workflow parameters: {', '.join(sorted(unknown))}")
        graph = copy.deepcopy(self.workflow)
        for binding in self.prompt:
            graph[binding.node]["inputs"][binding.input] = prompt
        for binding in self.seed:
            graph[binding.node]["inputs"][binding.input] = seed
        for name, value in parameters.items():
            binding = self.parameters[name]
            graph[binding.node]["inputs"][binding.input] = value
        for index, (filename, binding) in enumerate(zip(references, self.references), 1):
            node_id = f"app_reference_{index}"
            if node_id in graph:
                raise ValueError(f"Reserved workflow node: {node_id}")
            graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": filename}}
            graph[binding.node]["inputs"][binding.input] = [node_id, 0]
        return graph


def profiles() -> dict[str, Profile]:
    result = {}
    for directory in [config.PROJECT_DIR / "workflows" / "comfyui", config.OUTPUT_DIR / "comfyui_workflows"]:
        for path in sorted(directory.glob("*.json")):
            profile = Profile.model_validate_json(path.read_text(encoding="utf-8-sig"))
            if profile.id in result:
                raise ValueError(f"Duplicate ComfyUI profile: {profile.id}")
            result[profile.id] = profile
    return result


def load_profile(profile_id: str | None = None) -> Profile:
    profile_id = profile_id or config.COMFYUI_WORKFLOW
    catalog = profiles()
    if profile_id not in catalog:
        raise ValueError(f"Unknown ComfyUI workflow: {profile_id}")
    return catalog[profile_id]


def import_profile(profile: Profile):
    if profile.id in profiles():
        raise ValueError("This workflow ID already exists. Choose a new ID before importing.")
    directory = config.OUTPUT_DIR / "comfyui_workflows"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{profile.id}.json").open("x", encoding="utf-8") as handle:
        handle.write(profile.model_dump_json(indent=2))


def build_prompt(profile, image, reference_count):
    """Render the existing scene and identity contract without another AI call."""
    from image_generation import build_image_prompt_v3

    mapping = getattr(image, "reference_mapping", None)
    if mapping and [entry["image_index"] for entry in mapping] != list(range(1, reference_count + 1)):
        raise ValueError("Reference mapping does not match the uploaded image order")
    description = build_image_prompt_v3(image, has_reference_images=True)
    if profile.prompt_format == "minimax":
        return _minimax_prompt(image, description, reference_count)
    contract = json.dumps(description, ensure_ascii=False, separators=(",", ":"))
    if profile.prompt_format == "json":
        return contract
    slots = []
    for index in range(1, reference_count + 1):
        entry = mapping[index - 1] if mapping else {}
        role = entry.get("reference_role", "style" if entry.get("is_style_reference") else "face")
        person = entry.get("person_id") or "reference"
        slots.append(f"<image{index}> = image_index {index}: {role} for {person}.")
    return ("Create a new widescreen YouTube thumbnail with the scene and composition below. "
            "Use the attached pictures as references, not as a collage or a layout to copy. "
            "Preserve each person's identity and assigned clothing. Follow the specified "
            "roles for style and environments. Render only the requested visible text.\n"
            + "\n".join(slots) + "\n" + contract)


def _minimax_prompt(image, description, reference_count):
    """Render the existing scene as prose using MiniMax's picture labels."""
    from image_generation import _build_face_source_header

    parts = [
        "Create a new 16:9 YouTube thumbnail. The attached pictures supply the assigned "
        "identity, body, setting, or style references. Compose the target image as described; "
        "do not copy a reference sheet or make a collage of the input pictures.",
        " ".join(f"<Picture {index}> is Image {index}." for index in range(1, reference_count + 1)),
        _build_face_source_header(image.reference_mapping).strip(),
        image.scene or image.image_prompt,
    ]
    for person, character in description.get("characters", {}).items():
        if character.get("outfit"):
            parts.append(f"{person} wears {character['outfit']}.")
        if character.get("outfit_note"):
            parts.append(character["outfit_note"])
    for label, value in (
        ("Pose", image.subject_pose), ("Expression", image.subject_expression),
        ("Action", image.subject_action), ("Mood", image.mood),
    ):
        if value:
            parts.append(f"{label}: {value}.")
    environment = description["scene"]["environment"]
    for label, value in environment.items():
        if value:
            text = "; ".join(value) if isinstance(value, list) else value
            parts.append(f"{label.replace('_', ' ')}: {text}.")
    for label, value in description["camera_and_lighting"].items():
        parts.append(f"{label}: {value}.")
    for label, value in description.get("style_directive", {}).items():
        if value and label != "mandatory":
            text = "; ".join(value) if isinstance(value, list) else value
            parts.append(f"{label.replace('_', ' ')}: {text}.")
    palette = image.color_palette or image.colors
    if palette:
        parts.append(f"Color palette: {', '.join(palette)}.")
    parts.append(f"Quality: {description['output']['quality']}.")
    if image.text_in_image:
        parts.append(f"The only visible lettering is {json.dumps(image.text_in_image, ensure_ascii=False)}. "
                     "Preserve its exact spelling and language, with bold legible typography.")
    else:
        parts.append("The target image contains no lettering, captions, or watermarks.")
    return "\n".join(parts)


class Client:
    def __init__(self, connection: Connection):
        self.connection = connection
        self.session = requests.Session()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.session.close()

    def request(self, method, path, **kwargs):
        try:
            response = self.session.request(method, self.connection.url + path,
                                            timeout=(5, 30), allow_redirects=False, **kwargs)
        except requests.RequestException as error:
            raise ComfyUIError(f"ComfyUI {method} {path} failed: {error}") from error
        if not 200 <= response.status_code < 300:
            detail = response.text[:2500]
            response.close()
            raise ComfyUIError(f"ComfyUI {method} {path}: HTTP {response.status_code}: {detail}")
        return response

    def json(self, method, path, **kwargs):
        with self.request(method, path, **kwargs) as response:
            try:
                return response.json()
            except ValueError as error:
                raise ComfyUIError(f"ComfyUI {path} returned invalid JSON") from error

    def inspect(self, profile):
        system = self.json("GET", "/system_stats")
        classes = {node["class_type"] for node in profile.workflow.values()} | {"LoadImage"}
        nodes = {}
        for name in sorted(classes):
            data = self.json("GET", "/object_info/" + requests.utils.quote(name, safe=""))
            if name not in data:
                raise ComfyUIError(f"Missing ComfyUI node: {name}. Install a compatible ComfyUI version.")
            info = data[name]
            if info.get("api_node"):
                raise ComfyUIError(f"Paid/API node {name} is not allowed in local workflows")
            nodes[name] = info
        version = system["system"]["comfyui_version"]
        import re
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
        if not match or tuple(map(int, match.groups())) < (0, 37, 0):
            raise ComfyUIError("ComfyUI 0.37.0 or newer is required for native job cancellation")
        parameters = {}
        for key, binding in profile.parameters.items():
            node = profile.workflow[binding.node]
            info = nodes[node["class_type"]]["input"]
            spec = {**info.get("required", {}), **info.get("optional", {})}.get(binding.input)
            if spec is None:
                raise ComfyUIError(f"Unknown input: {binding.node}.{binding.input}")
            parameters[key] = {"label": binding.label, "default": node["inputs"][binding.input],
                               "choices": spec[0] if isinstance(spec[0], list) else None}
        return {"version": version, "parameters": parameters, "nodes": nodes}

    def validate(self, profile, info=None):
        info = info if info is not None else self.inspect(profile)
        graph = profile.graph("validation", 1, ["validation.png"], self.connection.parameters)
        for node_id, node in profile.workflow.items():
            schema = info["nodes"][node["class_type"]]["input"]
            inputs = {**schema.get("required", {}), **schema.get("optional", {})}
            for name in node["inputs"]:
                value = graph[node_id]["inputs"][name]
                spec = inputs.get(name)
                if spec is None:
                    raise ComfyUIError(f"Unknown ComfyUI input: {node_id}.{name}")
                if isinstance(value, list):
                    continue
                kind = spec[0]
                if isinstance(kind, list) and value not in kind:
                    raise ComfyUIError(f"Unavailable value for {node_id}.{name}: {value}")
                if kind in ("INT", "FLOAT"):
                    valid_type = type(value) is int if kind == "INT" else type(value) in (int, float)
                    options = spec[1] if len(spec) > 1 else {}
                    if (not valid_type or value < options.get("min", float("-inf"))
                            or value > options.get("max", float("inf"))):
                        raise ComfyUIError(f"Invalid numeric value for {node_id}.{name}: {value}")
                    if kind == "INT" and options.get("step") and (value - options.get("min", 0)) % options["step"]:
                        raise ComfyUIError(f"Invalid numeric step for {node_id}.{name}: {value}")
                if kind == "STRING" and not isinstance(value, str):
                    raise ComfyUIError(f"Expected text for {node_id}.{name}")
        for binding in profile.references:
            schema = info["nodes"][profile.workflow[binding.node]["class_type"]]["input"]
            inputs = {**schema.get("required", {}), **schema.get("optional", {})}
            spec = inputs.get(binding.input)
            if spec and spec[0] == "IMAGE":
                continue
            prefix, separator, slot = binding.input.partition(".")
            dynamic = inputs.get(prefix)
            if not (separator and dynamic and dynamic[0] == "COMFY_AUTOGROW_V3"):
                raise ComfyUIError(f"Unsupported image input: {binding.node}.{binding.input}")
            template = dynamic[1]["template"]
            template_inputs = template["input"].get("required", {})
            if len(template_inputs) != 1 or next(iter(template_inputs.values()))[0] != "IMAGE":
                raise ComfyUIError(f"Unsupported image input: {binding.node}.{binding.input}")
            if "names" in template:
                valid_slot = slot in template["names"]
            else:
                prefix = template.get("prefix", "")
                number = slot.removeprefix(prefix)
                valid_slot = (bool(prefix) and slot.startswith(prefix) and number.isascii() and number.isdigit()
                              and str(int(number)) == number and 1 <= int(number) <= template["max"])
            if not valid_slot:
                raise ComfyUIError(f"Unsupported image input: {binding.node}.{binding.input}")
        if not info["nodes"][profile.workflow[profile.output_node]["class_type"]].get("output_node"):
            raise ComfyUIError("The selected workflow output is not an output node")
        return info

    def upload(self, data, index, batch_id):
        digest = hashlib.sha256(data).hexdigest()[:16]
        response = self.json("POST", "/upload/image",
                             files={"image": (f"ref_{index}_{digest}.png", data, "image/png")},
                             data={"subfolder": f"thumbnail_generator/{batch_id}", "type": "input"})
        if not response.get("name") or response.get("type") != "input":
            raise ComfyUIError("ComfyUI returned an invalid uploaded image reference")
        return "/".join(part for part in [response.get("subfolder"), response["name"]] if part)

    def cancel(self, prompt_id):
        with self.request("POST", f"/api/jobs/{prompt_id}/cancel"):
            pass
        deadline = time.monotonic() + 30
        while True:
            queue = self.json("GET", "/queue")
            if not any(item[1] == prompt_id for key in ("queue_running", "queue_pending") for item in queue[key]):
                return
            if time.monotonic() >= deadline:
                raise ComfyUIError(f"Cancellation requested, but job {prompt_id} is still running. Check ComfyUI.")
            time.sleep(1)

    def generate(self, profile, prompt, seed, references, receipt_path, cancellation_check=None):
        prompt_id = str(uuid.uuid4())
        graph = profile.graph(prompt, seed, references, self.connection.parameters)
        receipt = {"prompt_id": prompt_id, "workflow_id": profile.id, "seed": seed,
                   "server": self.connection.url, "state": "submitting", "workflow": graph}

        def record(state):
            receipt["state"] = state
            receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")

        if cancellation_check and cancellation_check():
            raise ComfyUICancelledError("ComfyUI generation canceled before submission")
        submitted = False
        cancel_attempted = False
        record("submitting")
        try:
            response = self.json("POST", "/prompt", json={"prompt": graph, "prompt_id": prompt_id})
            if response.get("prompt_id") != prompt_id:
                raise ComfyUIError("ComfyUI did not accept the supplied prompt ID; update ComfyUI")
            submitted = True
            record("queued")
            deadline = time.monotonic() + self.connection.timeout
            while True:
                if cancellation_check and cancellation_check():
                    cancel_attempted = True
                    self.cancel(prompt_id)
                    record("cancel_requested")
                    raise ComfyUICancelledError(f"ComfyUI job {prompt_id} was canceled")
                history = self.json("GET", f"/history/{prompt_id}").get(prompt_id)
                if history:
                    status = history.get("status", {})
                    if status.get("status_str") == "error":
                        raise ComfyUIError(f"ComfyUI job {prompt_id} failed: {json.dumps(status.get('messages', []))[:2500]}")
                    if status.get("completed"):
                        images = history.get("outputs", {}).get(profile.output_node, {}).get("images", [])
                        if len(images) != 1:
                            raise ComfyUIError(f"Expected one image from output {profile.output_node}, got {len(images)}")
                        output = images[0]
                        with self.request("GET", "/view", params={key: output[key] for key in ("filename", "subfolder", "type")}) as result:
                            data = result.content
                        with Image.open(io.BytesIO(data)) as decoded:
                            decoded.load()
                        record("completed")
                        return data
                if time.monotonic() >= deadline:
                    cancel_attempted = True
                    self.cancel(prompt_id)
                    record("timeout_cancel_requested")
                    raise ComfyUIError(f"ComfyUI job {prompt_id} timed out; cancellation requested. No resubmission.")
                time.sleep(1)
        except KeyboardInterrupt:
            self.cancel(prompt_id)
            record("cancel_requested")
            raise
        except ComfyUICancelledError:
            raise
        except Exception as error:
            receipt["error"] = str(error)
            if submitted and not cancel_attempted:
                # Fail fast also stops our remote computation when a local callback fails.
                try:
                    self.cancel(prompt_id)
                    record("failed_canceled")
                except ComfyUIError as cancellation_error:
                    receipt["cancellation_error"] = str(cancellation_error)
                    record("failed_or_unknown")
            elif receipt["state"] != "timeout_cancel_requested":
                record("failed_or_unknown")
            raise ComfyUIError(f"{error}. Job ID: {prompt_id}. See {receipt_path}") from error


def prepare(workflow=None, connection=None, snapshot=None):
    """Resolve and validate before the pipeline can call any prompt writer."""
    profile = Profile.model_validate(snapshot) if snapshot else load_profile(workflow)
    if workflow and workflow != profile.id:
        raise ValueError("The workflow ID does not match its snapshot")
    connection = Connection.model_validate(connection or {})
    with Client(connection) as client:
        client.validate(profile)
    return profile.model_dump(), connection.model_dump()


def generate_batch(images, best_frames, output, workflow=None, connection=None, snapshot=None,
                   progress_callback=None, cancellation_check=None):
    """Upload references once, run sequential jobs, and keep each native receipt."""
    import secrets
    from PIL import ImageOps
    from utils import load_reference_images
    from services.generation_service import GenerationCancelledError

    def check_cancelled():
        if cancellation_check and cancellation_check():
            raise GenerationCancelledError("ComfyUI generation canceled")

    check_cancelled()
    profile = Profile.model_validate(snapshot) if snapshot else load_profile(workflow)
    connection = Connection.model_validate(connection or {})
    references = load_reference_images(best_frames, max_count=len(profile.references))
    if not references:
        raise ValueError("This workflow requires at least one reference image")
    # Validate every mapping before uploading or starting the first image.
    prompts = [build_prompt(profile, image, len(references)) for image in images]
    directory = output.output_dir / "thumbnails"
    directory.mkdir(parents=True, exist_ok=True)
    batch_id = uuid.uuid4().hex
    results = []
    with Client(connection) as client:
        client.validate(profile)
        uploaded = []
        for index, data in enumerate(references, 1):
            check_cancelled()
            uploaded.append(client.upload(data, index, batch_id))
        with GPUClient() as gpu:
            try:
                resource_id = gpu.resource_for_comfyui(connection.url)
                reservation = gpu.reserve_images(resource_id, batch_id, directory / f"gpu-{batch_id}.json",
                                                 connection.timeout, cancellation_check)
                with reservation:
                    for completed, (image, prompt) in enumerate(zip(images, prompts), 1):
                        check_cancelled()
                        path = directory / f"comfyui-{profile.id}_{batch_id}_{image.image_index}.png"
                        try:
                            data = client.generate(profile, prompt, secrets.randbits(63), uploaded,
                                                   path.with_suffix(".comfyui.json"), cancellation_check)
                        except ComfyUICancelledError as error:
                            raise GenerationCancelledError(str(error)) from error
                        check_cancelled()
                        if config.KEEP_RAW_THUMBNAILS:
                            raw = directory / "raw"
                            raw.mkdir(exist_ok=True)
                            prefix, suffix = path.name.split("_", 1)
                            (raw / f"{prefix}_raw_{suffix}").write_bytes(data)
                        with Image.open(io.BytesIO(data)) as original:
                            thumbnail = ImageOps.fit(original.convert("RGB"),
                                                     (config.THUMBNAIL_WIDTH, config.THUMBNAIL_HEIGHT),
                                                     method=Image.Resampling.LANCZOS)
                            thumbnail.save(path, "PNG")
                        results.append(path)
                        if progress_callback:
                            progress_callback(completed, len(images), {"path": path, "image_index": image.image_index, "image": image})
            except LocalGPUCancelled as error:
                raise GenerationCancelledError(str(error)) from error
            except LocalGPUError as error:
                raise ComfyUIError(str(error)) from error
    return results
