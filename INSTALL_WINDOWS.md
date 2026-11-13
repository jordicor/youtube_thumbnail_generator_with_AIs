# Windows installation: guided or automated

This guide is for a new Windows 10/11 **64-bit x86 PC**. Use a local folder you own,
not Program Files or a network share. Allow several GB of free space and time for downloads.
The installer does not process your videos or make paid AI requests.
Prefer a short path such as `C:\AI\Thumbnails`: deeply nested folders can exceed
Windows' path-length limit when installing PyTorch. Setup does not change that Windows setting.

## The easiest route

1. Download this repository from GitHub (Code > Download ZIP), extract it, and open its folder.
   If you already use Git, cloning is better because the updater can then download changes.
2. Open a PowerShell terminal **in that folder**. On Windows 11, right-click an empty
   part of the folder and choose **Open in Terminal**.
3. Paste this command:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
   ```

   This permits the downloaded script for this invocation only; it does not change
   the computer's permanent PowerShell execution policy.
4. Follow the questions. Before installing, the assistant shows a plain-language summary.
   It stops if something fails; keep the error message and rerun after fixing that issue.
5. Double-click **Start Thumbnail Generator.cmd**, created in the project folder.
   Once it says **Ready**, open the displayed local address. Keep the launcher window open.

**Python**, **Git**, and **FFmpeg** are basic tools: they run the app, download source updates,
and read video/audio. If missing, the assistant offers to install them through Windows
Package Manager. Windows may request administrator approval. If `winget` is missing,
install Microsoft's **App Installer** from the Microsoft Store and run setup again.

## What the questions mean

| Choice | What it does | What to choose if unsure |
|---|---|---|
| CPU or NVIDIA | Chooses the packages used to analyze faces and bodies. NVIDIA is faster but needs a working driver. This does not install a local image AI. | CPU; it works without an NVIDIA card. |
| GranSabio | A separate helper that connects the app to text AIs for ideas, titles, and descriptions. It is free software, not an AI subscription. | Download and set it up. The installer takes the latest published default branch directly from its GitHub repository. |
| Redis | A small background service that keeps the web app's job queue. It is not an AI provider. | Install it in Ubuntu/WSL, unless you already have a Redis service. |
| WSL | Microsoft's way to run Linux tools on Windows. Here it runs Redis. | Allow it if you want setup to manage Redis. Windows may require administrator approval and a restart. |
| Local analysis models | Files that help choose clear face/body photos. | Download the core models. Face replacement is optional and adds about 1 GB. |
| API keys | Secret passwords issued by online AI providers. Their use may cost money. | Enter your own keys, or press Enter and configure engines later. Never share keys in chat, GitHub, screenshots, or command arguments. |

The default online combination is **Claude for ideas** and **Gemini for images**.
Get keys from [Anthropic](https://console.anthropic.com/) and
[Google AI Studio](https://aistudio.google.com/apikey). Keys are entered without
showing them and saved only in local `.env` files. Existing nonempty keys are preserved.
Local Whisper transcription is selected for a new installation; its speech model
and InsightFace's face models download on first use. This can make the first analysis slow.

You can instead choose other text/image engines in the app's **Settings**.
For an existing local AI server, see [ENGINES.md](ENGINES.md) and [COMFYUI.md](COMFYUI.md).
The installer does not install ComfyUI, Ollama, or large image/text models for you.
Model licenses are separate; see the [README license notes](README.md#honest-notes).

## After installation

- `Start Thumbnail Generator.cmd` starts the downloaded GranSabio, the web app, and
  its background worker. It reuses the application's normal entry points and verifies
  health responses plus the worker heartbeat before saying **Ready**.
- Existing GranSabio/Redis services are used without replacing their configuration.
  Downloaded GranSabio and the web app listen on `127.0.0.1`, for this PC only.
- Finish or cancel jobs in the app before pressing **Ctrl+C** in the launcher.
  It stops only the processes it started. Redis and externally managed servers keep running.
- The app's packages live in `.venv`; downloaded GranSabio, its separate environment,
  installation choices, and service logs live in `.install`. Those folders and `.env`
  are excluded from Git. No API keys are stored in the installation-choice JSON.
- A successful package installation is different from having usable AI accounts.
  Checks report missing configuration, but never spend money to test a key.

Check the installation or its running services:

```powershell
.\.venv\Scripts\python.exe install.py check
.\.venv\Scripts\python.exe install.py check --services --json
```

To update, finish jobs and stop the launcher, then run:

```powershell
.\.venv\Scripts\python.exe install.py update
```

Updates require a clean Git clone of this public repository. They use `git pull --ff-only`;
local edits or diverged branches stop the update instead of being reset or merged.
Downloaded GranSabio is updated from its own repository. An existing external GranSabio
is maintained by its owner. Updates are explicit, never performed at every launch.
ZIP installations work for first setup; use a Git clone for later automatic updates.
Rerunning the wizard keeps your previous choices as the defaults. Keep the same CPU/NVIDIA
profile for an existing environment; to change it, use a separate project folder.

## Installation by an AI agent or script

Read this section before installing. Do not run the interactive wizard through a pipe.
All choices have CLI arguments, and `plan` and `check` support JSON output.
The assistant's explanations above are the basis for questions to the user: explain
the purpose and tradeoff, rather than asking whether they want a component by name.

### 1. Inspect, explain, and choose

```powershell
git clone https://github.com/jordicor/youtube_thumbnail_generator_with_AIs.git
cd youtube_thumbnail_generator_with_AIs
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -NonInteractive plan --json
```

If Python is missing, install the prerequisites with the owner's authorization; the
PowerShell bootstrap can do that with `-InstallPrerequisites`. `-PythonPath` selects
an existing 64-bit Python 3.11 executable. No Conda installation is required.

Before installing, establish:

- **A new folder or an existing installation?** Inspect Git status and existing configuration.
  Preserve local edits, `.env`, videos, database, and models. Never reset or overwrite them.
- **Hardware and disk space:** check the NVIDIA driver if acceleration was selected;
  do not silently change a failed GPU installation to CPU. Allow several GB for downloads.
  Useful commands: `nvidia-smi`, `Get-PSDrive -PSProvider FileSystem`, and `wsl --list --verbose`.
- **Who runs the helper services?** For existing GranSabio, obtain its address and local
  repository folder (the client package is installed from that folder). For existing Redis,
  obtain its connection address. Prefer dedicated services/queues for this installation.
- **WSL/system-tool installation:** explain the possible Windows approval/restart. Never
  approve a restart or install system components without the owner's authorization.
- **AI engines and costs:** explain the default Claude/Gemini combination or follow the
  user's existing local-engine setup. Leave keys for the user to enter locally or read
  them from an authorized private dotenv file. Never ask for keys in chat.

### 2. Install with explicit choices

New CPU installation, latest GranSabio, Redis managed in WSL:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -InstallPrerequisites -NonInteractive install --hardware cpu --gransabio download --redis wsl --models core --json
```

Existing Redis, downloaded GranSabio, NVIDIA analysis:

```powershell
py -3.11 install.py install --non-interactive --hardware cuda --gransabio download --redis existing --redis-url redis://127.0.0.1:6379/0 --models core --json
```

Existing GranSabio server and client checkout:

```powershell
py -3.11 install.py install --non-interactive --hardware cpu --gransabio existing --gransabio-dir C:\AI\GranSabio_LLM --gransabio-url http://127.0.0.1:8000 --redis existing --models core --json
```

Additional options: `--keys-file C:\private\provider-keys.env`, `--models all` for
face replacement, `--models none` to defer downloads, `--web-port 8057`,
`--wsl-distribution Ubuntu`, and `--gransabio-dir` to choose the download folder.
Use `install.py --help` for the complete interface. Private Redis credentials belong
in `.env`, not in a URL passed as a command argument.

### 3. Verify and report the exact result

```powershell
.\.venv\Scripts\python.exe install.py check --json
.\.venv\Scripts\python.exe install.py start
# While the launcher is running, use another terminal:
.\.venv\Scripts\python.exe install.py check --services --json
```

Exit codes: **0** = requested checks passed; **1** = failure; **2** = invalid arguments;
**3** = packages checked, but models/default AI configuration still need attention.
Read `installation_ok`, `configuration_pending`, and every check instead of treating
any completed process as a successful installation. JSON goes to stdout; installation
progress goes to stderr. A plan is read-only and does not test network credentials.

Report separately: packages installed, model checksums verified, service health checked,
worker heartbeat checked, and whether an actual AI generation was tested. Paid generation
requires separate authorization and is never part of setup checks. If the user chose
custom engines in Settings, review those selections rather than assuming the default-key
reminders describe their setup.

## If something fails

Do not delete the environment, remove the database, or retry with random package versions.
Keep the first error. The script stops and can be rerun with the same choices.

- **WSL requests a restart/admin rights:** follow the Windows message, complete the restart
  when the owner permits it, and rerun the command. The installer never restarts Windows.
- **Port occupied or worker already running:** use the existing application, or stop it
  deliberately after its jobs finish. The launcher does not kill processes by port.
- **Existing `.env` conflicts with chosen addresses:** review the named settings locally.
  Match the existing settings or edit them deliberately; secrets are not printed.
- **Package import/CUDA failure:** inspect the failing package/driver. NVIDIA uses the
  official PyTorch CUDA 12.8 index; ONNX needs matching CUDA/cuDNN libraries. CPU uses the
  official CPU index. See [PyTorch](https://pytorch.org/get-started/locally/) and
  [ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
  The NVIDIA requirements keep ONNX below 1.27 because newer default PyPI builds use CUDA 13.
  Setup follows [InsightFace's NVIDIA instructions](https://pypi.org/project/insightface/#nvidia-cuda)
  by replacing its CPU runtime with the GPU distribution after installing dependencies.
  A later `pip check` therefore reports InsightFace's CPU-package requirement as missing;
  this is an upstream packaging limitation. Do not install both runtimes to suppress it.
  Use `install.py check` to verify the actual GPU operation and application imports.
- **A service exits:** read `.install/web.log`, `.install/worker.log`, or
  `.install/gransabio.log`. Review logs locally before sharing: upstream services may
  include private paths or request information.

Shared packages remain in `requirements.txt`; the two hardware requirement files only
select ONNX Runtime. GranSabio supplies its own requirements and client package. Keep
installation fixes in these sources rather than duplicating dependency lists in scripts.
