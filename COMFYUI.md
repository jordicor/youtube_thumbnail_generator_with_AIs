# Local image generation with ComfyUI

ComfyUI is an image provider in this application. It uses the native HTTP API; no bridge, custom node, plugin, or copied Python file is required on the ComfyUI server. The bundled profile runs **Qwen Image 2.1 with visual references**. Other local models can use explicitly configured API workflow profiles.

This changes image generation only. Thumbnail ideas, titles, and descriptions still use their separately configured AI through Gran Sabio. ComfyUI never falls back to a paid image provider, even when the server is unavailable.

## Server setup

1. Install **ComfyUI 0.37.0 or newer** and the Qwen Image 2.1 model files. Follow the [official Qwen Image 2.1 instructions](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1).
2. The bundled profile starts with these official filenames:

   | File | ComfyUI folder |
   | --- | --- |
   | `qwen_image_2.1_int8_convrot.safetensors` | `models/diffusion_models/` |
   | `qwen3vl_8b_int8_convrot.safetensors` | `models/text_encoders/` |
   | `qwen_image_2.1_vae_bf16.safetensors` | `models/vae/` |

   Weights are separate downloads, not included in this repository. Filenames can be selected from the server in Settings. Choose weights compatible with the workflow and available GPU memory.
3. Start ComfyUI normally if both applications run on the same computer. For another computer on a trusted LAN, bind ComfyUI to its LAN address, for example:

   ```sh
   python main.py --listen 192.168.1.50 --port 8188
   ```

   Run that command inside the ComfyUI installation with its Python environment. Use your server's actual address. Restrict firewall access to the intended machines; the native server does not provide application authentication. Do not expose it to the Internet. CORS changes are unnecessary: this application's backend makes the HTTP requests.

## Application settings

Restart the web application and generation worker after updating. Startup adds the connection settings while retaining existing preferences.

In **Settings → Thumbnails → Drawing**, select **ComfyUI (local)**, then:

1. Select **Qwen Image 2.1 (references)** in the model/workflow selector.
2. Enter the ComfyUI URL, such as `http://127.0.0.1:8188`.
3. Click **Check server and models**. The app checks native nodes and shows the installed model filenames. Replace an unavailable filename with the appropriate installed weight.
4. Set the timeout per image, including time waiting in ComfyUI's queue, and save.

Select visual references in the normal reference editor. Their order and roles (face, body, style, environment) remain intact. The app uploads them once per generation batch and uses Qwen's `<image1>`, `<image2>`, etc. bindings. It does not combine different people's references into a batch tensor or silently drop extra images.

The bundled native node exposes up to **16 reference slots**. GPU memory and model behavior still depend on reference count and size. The profile uses a 1024-pixel reference budget inside Qwen's native encoder, a **1536 × 864** canvas, and 25 sampling steps. Local source photos and uploads are not resized first. Final thumbnails are fitted to 1280 × 720; a workflow with another aspect ratio is center-cropped. Originals are retained when `KEEP_RAW_THUMBNAILS=true`.

## Configurable workflows

The bundled profile is [`workflows/comfyui/qwen-image-2.1.json`](workflows/comfyui/qwen-image-2.1.json). Copy it, change its `id` and `name`, adapt the graph and bindings, then use **Import workflow profile**. Imported profiles are stored under `OUTPUT_DIR/comfyui_workflows/`, outside the public export. An existing ID cannot be overwritten through import; use a new ID for a revision.

A profile contains:

- `workflow`: a ComfyUI **API-format** graph keyed by node ID, with `class_type` and `inputs`. A normal UI-layout JSON with a `nodes` array is not this format. Export the API format from ComfyUI when building a graph there.
- `prompt`: node/input bindings receiving the generated scene prompt.
- `seed`: node/input bindings receiving a fresh random seed for each thumbnail.
- `references`: ordered bindings to image inputs. Leave those inputs unconnected in the stored graph; the app inserts native `LoadImage` nodes. Remove old example-image nodes and any unused branches.
- `output_node`: one output node that returns exactly one image through native history and `/view`. Keep batch size at 1; the application runs one job per thumbnail.
- `prompt_format`: `qwen` for numbered Qwen reference instructions plus the scene contract, or `json` for the existing structured scene contract.
- `parameters`: optional named bindings with a `label`, `node`, and `input`. They become editable Settings fields; enum values come from the server's node metadata. The built-in profile exposes the three weight selectors. Other profiles can expose numeric sampling or canvas inputs too.

The native Qwen profile uses `UNETLoader`, `CLIPLoader`, `VAELoader`, `TextEncodeQwenImage21`, `EmptyLatentImage`, `KSampler`, `VAEDecode`, `LoadImage`, and `SaveImage`. It needs no extra installation files in this project. Imported workflows must use available compatible nodes; missing nodes fail validation. ComfyUI nodes marked as paid/API nodes are rejected. Only import workflows and custom server nodes you trust; this client is not a sandbox for arbitrary third-party node code.

Each queued application job stores a snapshot of its graph and connection. Later edits do not change an already queued job. Different image tools need different adapters; an unrelated server URL cannot be used as a ComfyUI endpoint.

## Command line

Existing `process` and `generate` commands accept `--image-provider comfyui`, `--comfyui-url`, `--comfyui-workflow`, and `--comfyui-config`:

```sh
python main.py generate --video 1 --clusters 0 --num-images 1 --image-provider comfyui --comfyui-url http://127.0.0.1:8188 --comfyui-workflow qwen-image-2.1
```

`--comfyui-config` reads a connection JSON file; `--comfyui-url` overrides its URL:

```json
{
  "url": "http://127.0.0.1:8188",
  "timeout": 1800,
  "parameters": {
    "diffusion_model": "qwen_image_2.1_int8_convrot.safetensors",
    "text_encoder": "qwen3vl_8b_int8_convrot.safetensors",
    "vae": "qwen_image_2.1_vae_bf16.safetensors"
  }
}
```

Without CLI overrides, `COMFYUI_URL` and `COMFYUI_WORKFLOW` provide the defaults. Saved web settings apply to web requests. Neither `--provider-fallback` nor `--composite` is accepted for ComfyUI. Configure `JOB_TIMEOUT_GENERATION` to cover the entire batch; the connection timeout applies separately to each image.

## Errors, cancellation, and receipts

Server, node, model, reference, execution, and output errors stop generation. HTTP submissions are never automatically retried. Each submission has a locally chosen native `prompt_id` and a `.comfyui.json` receipt beside its thumbnail, containing the exact graph, seed, server, and state. These receipts are private runtime files and are not exported to the public repository.

Cancellation and timeout call the native `/api/jobs/{prompt_id}/cancel` route and verify that this ID leaves the queue. They never use the global interrupt endpoint or clear another user's jobs. If communication is lost, inspect the receipt and that job in ComfyUI before manually retrying: the server may have accepted a submission whose response was lost.

ComfyUI keeps its uploaded inputs and generated outputs according to its normal storage behavior. The client does not clean a shared server's directories. Review image quality and identity before publishing; a successful request does not guarantee perfect likeness or lettering.
