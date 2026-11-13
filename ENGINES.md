# AI engines and the shared GPU

Thumbnail ideas, titles and descriptions are written by a text model reached
through [Gran Sabio LLM](https://github.com/jordicor/GranSabio_LLM). Choose the
provider and model of each job in **Settings**. Image generation has its own
engine; local image generation with ComfyUI is described in [COMFYUI.md](COMFYUI.md).

## Cloud models

Claude, GPT, Gemini and Grok use the API keys configured in Gran Sabio.
**OpenRouter** offers its whole catalog: the selector shows each model's input
and output price in USD per million tokens when the catalog has it. The chosen
model also handles JSON repair and image preflight; the output budget is capped
at the model's catalog limit and Gran Sabio's model fallback is disabled.

Thumbnail ideas only offer models that see images, because their requests carry
the face photos. Thinking is offered only for models that advertise reasoning.

## Cloud image models

Select **OpenAI** as the image provider in **Settings**, then choose
**GPT Image 2.6 Sunburst** for detailed compositions and precise editing, or
**GPT Image 2.6 Flare** for fast generation and editing. Both use the existing
`OPENAI_API_KEY` and accept up to 16 reference images. The CLI accepts
`--image-provider openai --openai-model gpt-image-2.6-sunburst` or
`--openai-model gpt-image-2.6-flare`. Saved settings and the configured default
remain unchanged until you choose a different model.

The application uses the Images API: `images.edit` with references and
`images.generate` without them, at high quality and `1536x864`, then saves a
`1280x720` thumbnail. GPT Image 2.6 uses its API default for complexity.
See OpenAI's [Sunburst model](https://developers.openai.com/api/docs/models/gpt-image-2.6-sunburst),
[Flare model](https://developers.openai.com/api/docs/models/gpt-image-2.6-flare), and
[image generation guide](https://developers.openai.com/api/docs/guides/image-generation).

## Local models

Choose **Ollama (local)**, **llama.cpp (local)** or **OpenAI-compatible (local)**
for thumbnail ideas or for titles and descriptions, then a model from Gran
Sabio's catalog. Embedding models are not offered.

- Configure the inference server URL and installed models in Gran Sabio. This
  application only needs `GRANSABIO_LLM_URL`, including when Gran Sabio runs on
  another computer.
- Refresh Gran Sabio's local model catalog after installing or removing models.
- To load installed Ollama models on demand, configure its local endpoint with
  `availability: "available"`; `"loaded"` only admits models already in memory.
- When Gran Sabio has several endpoints for one provider, set
  `GRANSABIO_OLLAMA_ENDPOINT_ID`, `GRANSABIO_LLAMACPP_ENDPOINT_ID` or
  `GRANSABIO_OPENAI_COMPATIBLE_ENDPOINT_ID` in this application's `.env`.

Local requests use Gran Sabio's `local_only` policy for generation, JSON repair
and image preflight: no external fallback, and the server must attest local
execution. A missing model, an unsupported capability, an ambiguous endpoint or
an unavailable server gives an explicit error. Text generation with a local
model needs no cloud API key.

## Sharing one GPU between text and images

The **Local GPU** section of Settings reads Gran Sabio's registered resources and
loaded Ollama models. It can unload an idle text model or free ComfyUI's model
memory without stopping either service. Busy resources refuse to unload; this
never cancels a generation. **Check GPU and loaded models** refreshes the status.

ComfyUI batches require exactly one Gran Sabio GPU resource whose ComfyUI URL
matches the one in Settings; there is no unreserved mode. The application takes
an exclusive reservation before the first image and releases it after the batch,
also after a failure or a confirmed cancellation. While the GPU is busy it waits
within the image timeout, checking for cancellation. Local text requests made
during an image batch get Gran Sabio's `resource_busy` error and can be retried
when the batch ends.

A failed or ambiguous reservation never starts image generation, and a failed
release is an error, not permission to reuse the GPU. Each batch writes a
`thumbnails/gpu-*.json` receipt with its resource, reservation ID, request IDs
and state. If a client crashes or a release is not confirmed, stop the job,
inspect that receipt and reconcile the reservation in Gran Sabio; do not force
another client onto the GPU. Requests that bypass the coordinated clients are
outside this guarantee.
