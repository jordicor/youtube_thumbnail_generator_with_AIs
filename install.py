"""Windows setup: a guided wizard and the same operations as explicit CLI commands."""

import argparse
import getpass
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".install" / "settings.json"
GRANSABIO_REPO = "https://github.com/jordicor/GranSabio_LLM.git"
APP_REPO = "https://github.com/jordicor/youtube_thumbnail_generator_with_AIs.git"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
KEY_NAMES = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
             "POE_API_KEY", "REPLICATE_API_TOKEN", "ELEVENLABS_API_KEY")


def run(command, *, cwd=ROOT, capture=False, input_text=None, timeout=None):
    """Run argument lists directly; never interpret user paths as shell commands."""
    result = subprocess.run([str(value) for value in command], cwd=cwd, input=input_text,
                            stdout=subprocess.PIPE if capture else sys.stderr,
                            stderr=subprocess.PIPE if capture else sys.stderr,
                            text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode:
        # Captured configuration can contain credentials. Never include it in errors.
        raise RuntimeError(f"{Path(str(command[0])).name} failed (exit {result.returncode}). "
                           "See the command output above, or INSTALL_WINDOWS.md for recovery steps.")
    return result.stdout.strip() if capture else None


def git(folder, *arguments):
    return run(["git", "-c", f"safe.directory={Path(folder).resolve().as_posix()}",
                "-C", folder, *arguments], capture=True)


def check_checkout(folder, expected_url):
    if not (folder / ".git").is_dir():
        raise ValueError(f"{folder} must be a Git clone, not a ZIP or an unrelated folder.")
    if git(folder, "remote", "get-url", "origin").removesuffix(".git") != expected_url.removesuffix(".git"):
        raise ValueError(f"Unexpected repository in {folder}. Nothing was updated.")
    if git(folder, "status", "--porcelain"):
        raise ValueError(f"{folder} has local changes. Save or review them before updating; nothing will be discarded.")
    if not git(folder, "branch", "--show-current"):
        raise ValueError(f"{folder} is not on a branch. Select its published branch before updating.")


def python_info():
    return {"executable": sys.executable, "version": platform.python_version(),
            "supported": sys.version_info[:2] == (3, 11) and sys.maxsize > 2**32}


def plan(options):
    gpu = shutil.which("nvidia-smi")
    return {
        "platform": platform.system(), "python": python_info(), "project": str(ROOT),
        "tools": {name: shutil.which(name) for name in ("git", "ffmpeg", "ffprobe", "wsl")},
        "nvidia_driver_tool": gpu, "options": options,
        "steps": ["Install Redis in Ubuntu/WSL; stop with instructions if Windows needs a restart."
                  if options["redis"] == "wsl" else "Use your existing Redis queue server.",
                  "Create or reuse this project's .venv (isolated Python packages).",
                  "Install the selected CPU/NVIDIA packages and the project's requirements.",
                  "Download the latest published GranSabio and install it in a separate environment."
                  if options["gransabio"] == "download" else "Use your existing GranSabio server; install its client only.",
                  "Keep existing secrets; create missing configuration and download the selected models.",
                  "Check packages and models without calling paid AI services. Create a Start shortcut."],
        "notes": ["Plan does not install, download, start services, or change files.",
                  "Downloads can take several GB. Windows may request administrator approval for system tools.",
                  "Cloud AI accounts and any usage charges are separate. No paid request is made during setup."],
    }


def choose(question, choices, default=1):
    print(f"\n{question}")
    for index, label in enumerate(choices, 1):
        print(f"  {index}. {label}")
    while True:
        value = input(f"Choose 1-{len(choices)} [Enter = {default}]: ").strip() or str(default)
        if value.isdigit() and 1 <= int(value) <= len(choices):
            return int(value)
        print("Please enter one of the numbers shown above.")


def show_plan(options):
    print(f"\nInstallation folder: {ROOT}")
    print("Video analysis: " + ("NVIDIA graphics card" if options["hardware"] == "cuda" else "CPU (no NVIDIA card required)"))
    print("Ideas helper: " + ("download the latest GranSabio" if options["gransabio"] == "download" else "use your existing GranSabio"))
    print("Background job queue: " + ("install Redis in Ubuntu/WSL" if options["redis"] == "wsl" else "use your existing Redis"))
    print("Models: " + {"core": "download the face/body analysis models", "all": "analysis and optional face replacement", "none": "download later"}[options["models"]])
    print("\nSetup can download several GB and take a while. It keeps each app's packages\n"
          "in a separate folder and does not reset existing files or secrets.\n"
          "Windows may request approval to install system tools. No paid AI requests are made.")


def option_arguments(options):
    return [value for key, setting in options.items() for value in ("--" + key.replace("_", "-"), str(setting))]


def require_stopped(options):
    from urllib.parse import urlsplit
    ports = [options["web_port"]]
    if options["gransabio"] == "download":
        ports.append(urlsplit(options["gransabio_url"]).port or 8000)
    for port in ports:
        with socket.socket() as probe:
            probe.settimeout(1)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError(f"Port {port} is in use. Finish jobs and stop the app/helper before installing or updating, or choose a different port.")


def wizard(options):
    print("\nYouTube Thumbnail Generator - guided Windows setup\n")
    print("This installs the tools that analyze your videos and create thumbnail ideas and images.\n"
          "Your videos stay on your computer. Online AI services receive the transcript/photos\n"
          "needed for their task and may charge for use. Installation itself makes no paid requests.\n"
          "A 'virtual environment' is just a private folder of Python packages for this app;\n"
          "you will not need to activate it or type Python commands to start the app.")
    options["hardware"] = "cuda" if choose(
        "How should this computer analyze videos? An NVIDIA graphics card can make this much faster.",
        ["CPU: works without an NVIDIA card; slower, smaller download. Choose this if unsure.",
         "NVIDIA GPU: faster; requires a working NVIDIA driver. Setup will verify acceleration."],
        default=2 if options["hardware"] == "cuda" else 1) == 2 else "cpu"
    options["gransabio"] = "existing" if choose(
        "The app needs a helper called GranSabio to ask text AIs for ideas, titles, and descriptions.\n"
        "GranSabio is free software, not an AI subscription. It connects to your online AI account\n"
        "or to an AI model you already run locally. Most new users should choose option 1.",
        ["Download and set up the latest GranSabio for me (recommended).",
         "I already run GranSabio and know its address and local repository folder."],
        default=2 if options["gransabio"] == "existing" else 1) == 2 else "download"
    if options["gransabio"] == "existing":
        options["gransabio_dir"] = input(f"GranSabio repository folder [{options['gransabio_dir']}]: ").strip().strip('"') or options["gransabio_dir"]
        options["gransabio_url"] = input(f"GranSabio server address [{options['gransabio_url']}]: ").strip() or options["gransabio_url"]
    options["redis"] = "existing" if choose(
        "The web app needs a small queue service called Redis to keep track of background jobs.\n"
        "It is not an AI service and does not charge per job. On Windows we can install it in\n"
        "Ubuntu through WSL (Microsoft's Linux support). This may need administrator approval\n"
        "and a Windows restart. No Docker is needed.",
        ["Set up Redis in Ubuntu/WSL for me (recommended for a new installation).",
         "I already have Redis running and know its connection address."],
        default=2 if options["redis"] == "existing" else 1) == 2 else "wsl"
    if options["redis"] == "existing":
        options["redis_url"] = input(f"Redis address [{options['redis_url']}]: ").strip() or options["redis_url"]
    options["models"] = ("core", "all", "none")[choose(
        "Local models help choose clear photos of faces and bodies. They are downloaded once.\n"
        "Face replacement is optional and adds roughly 1 GB. Model licenses are separate from\n"
        "this app's license; read README's model-license note before downloading.",
        ["Download the models needed for video analysis (recommended).",
         "Also download the optional face replacement models.",
         "Download later; video analysis will not be ready yet."],
        default=("core", "all", "none").index(options["models"]) + 1) - 1]
    return options


def validate(options):
    from urllib.parse import urlsplit
    for name, schemes in (("gransabio_url", ("http", "https")), ("redis_url", ("redis", "rediss"))):
        url = urlsplit(options[name])
        if url.scheme not in schemes or not url.hostname or url.query or url.fragment or url.username or url.password:
            raise ValueError(f"{name} must be a server address without credentials, query, or fragment. "
                             "Put private Redis credentials in .env after installation.")
        try:
            if url.port == 0:
                raise ValueError()
            _ = url.port
        except ValueError:
            raise ValueError(f"{name} has an invalid port; use a number from 1 to 65535.") from None
    folder = Path(options["gransabio_dir"]).expanduser().resolve()
    options["gransabio_dir"] = str(folder)
    if folder == ROOT or folder == PYTHON.parent.parent or ROOT.is_relative_to(folder):
        raise ValueError("GranSabio needs its own folder, separate from this application and its environment.")
    if options["gransabio"] == "existing" and not (folder / "client" / "pyproject.toml").is_file():
        raise ValueError("Choose the GranSabio repository folder containing client/pyproject.toml.")
    if options["redis"] == "wsl" and options["redis_url"] != "redis://127.0.0.1:6379/0":
        raise ValueError("WSL setup uses redis://127.0.0.1:6379/0. Choose existing Redis for a custom address.")
    address = urlsplit(options["gransabio_url"])
    if options["gransabio"] == "download" and (address.hostname != "127.0.0.1" or address.scheme != "http" or address.path not in ("", "/")):
        raise ValueError("Downloaded GranSabio runs locally. Use http://127.0.0.1:PORT or select an existing server.")


def require_tools():
    if platform.system() != "Windows" or not python_info()["supported"]:
        raise ValueError("Use 64-bit Python 3.11 on Windows. install.ps1 can install the prerequisites.")
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            long_paths = winreg.QueryValueEx(key, "LongPathsEnabled")[0]
    except FileNotFoundError:
        long_paths = 0  # Windows requires this opt-in value to enable long paths.
    if not long_paths and len(str(ROOT)) > 80:
        raise ValueError("This folder is too deeply nested for Windows package installation with long paths disabled. "
                         "Extract the project into a short folder such as C:\\AI\\Thumbnails and run setup there. "
                         "Windows settings were not changed.")
    missing = [name for name in ("git", "ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise ValueError(f"Missing tools: {', '.join(missing)}. Run install.ps1 -InstallPrerequisites.")


def make_venv(folder):
    executable = folder / "Scripts" / "python.exe"
    if not executable.exists():
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f"{folder} is not an empty folder or a usable environment. Nothing was removed.")
        run([sys.executable, "-m", "venv", folder])
    if run([executable, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], capture=True) != "3.11":
        raise ValueError(f"{folder} must use Python 3.11. Existing environments are never replaced automatically.")
    run([executable, "-m", "pip", "install", "--upgrade", "pip"])
    return executable


def install_pinned_sources(python):
    """Honor Git pins even when upstream reuses the same package version."""
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if " @ git+" not in line or line.lstrip().startswith("#"):
            continue
        requirement = line.split("#", 1)[0].strip()
        name, source = requirement.split(" @ ", 1)
        revision = source.rsplit("@", 1)[1]
        current = json.loads(run([python, "-c", "import json,sys; from importlib.metadata import distribution; "
                                  "print(distribution(sys.argv[1]).read_text('direct_url.json') or '{}')", name], capture=True))
        if current.get("vcs_info", {}).get("commit_id") != revision:
            run([python, "-m", "pip", "install", "--force-reinstall", "--no-deps", requirement])


def setup_redis(options):
    if options["redis"] != "wsl":
        return
    if not shutil.which("wsl"):
        raise ValueError("WSL is missing. Run 'wsl --install -d Ubuntu' in an Administrator terminal, restart if requested, then rerun setup.")
    available = subprocess.run(["wsl", "--list", "--quiet"], capture_output=True, timeout=20)
    names = available.stdout.decode("utf-16-le", errors="replace").replace("\x00", "")
    if options["wsl_distribution"] not in names.split():
        try:
            run(["wsl", "--install", "--distribution", options["wsl_distribution"], "--no-launch"])
        except RuntimeError:
            raise RuntimeError("WSL could not finish installation. Open a Windows Terminal as Administrator, "
                               f"run 'wsl --install -d {options['wsl_distribution']} --no-launch', "
                               "restart Windows only if requested, then rerun this installer in a normal terminal.") from None
    print("Installing the Redis job queue in Ubuntu. Windows may need a restart before WSL can run.", file=sys.stderr)
    run(["wsl", "-d", options["wsl_distribution"], "-u", "root", "--", "sh", "-c",
         "set -eu; apt-get update; DEBIAN_FRONTEND=noninteractive apt-get install -y redis-server; service redis-server start"])


def read_env(path):
    return json.loads(run([PYTHON, "-c", "import json,sys; from dotenv import dotenv_values; "
                          "print(json.dumps(dict(dotenv_values(sys.argv[1], interpolate=False))))", path], capture=True))


def configure_env(path, template, values, secrets):
    """Use python-dotenv itself; preserve existing secrets and reject conflicting setup choices."""
    existing = read_env(path) if path.exists() else {}
    values = values.copy()
    if existing.get("REDIS_URL") and "REDIS_URL" in values:
        from urllib.parse import urlsplit
        address = urlsplit(existing["REDIS_URL"])
        public_address = address._replace(netloc=address.netloc.rsplit("@", 1)[-1]).geturl()
        if public_address == values["REDIS_URL"]:
            del values["REDIS_URL"]  # Keep credentials entered locally, outside command arguments/state.
    conflicts = [key for key, value in values.items() if key in existing and existing[key] != value]
    conflicts += [key for key, value in secrets.items() if value and existing.get(key) and existing[key] != value]
    if conflicts:
        raise ValueError(f"Existing {path.name} differs for {', '.join(conflicts)}. "
                         "Use its existing settings or edit it deliberately; setup will not overwrite it.")
    if not path.exists():
        path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
        # Templates may contain example credentials. They must never count as configured.
        values = {**{key: "" for key in read_env(path) if key.endswith(("_KEY", "_TOKEN", "_PASSWORD"))}, **values}
    values.update({key: value for key, value in secrets.items() if value and not existing.get(key)})
    program = ("import json,sys; from dotenv import set_key; data=json.load(sys.stdin); "
               "[set_key(sys.argv[1], key, value, quote_mode='always') for key,value in data.items()]")
    run([PYTHON, "-c", program, path], input_text=json.dumps(values), capture=True)


def install(options, args):
    require_tools()
    validate(options)
    require_stopped(options)
    if args.keys_file and not args.keys_file.is_file():
        raise ValueError("The private keys file does not exist. Check its path; no credentials were read.")
    if options["hardware"] == "cuda" and not shutil.which("nvidia-smi"):
        raise ValueError("The NVIDIA driver tool was not found. Install the NVIDIA driver first, or select CPU analysis.")
    folder = Path(options["gransabio_dir"])
    if options["gransabio"] == "download" and folder.exists():
        check_checkout(folder, GRANSABIO_REPO)
    if PYTHON.exists() and STATE.exists():
        previous = json.loads(STATE.read_text(encoding="utf-8"))
        if previous["hardware"] != options["hardware"]:
            raise ValueError("This environment uses a different CPU/NVIDIA profile. Keep its existing choice, "
                             "or install the other profile in a new project folder; packages were not changed.")
    # Persist non-secret choices so an interrupted installation can be resumed.
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(options, indent=2) + "\n", encoding="utf-8")
    setup_redis(options)
    app_python = make_venv(PYTHON.parent.parent)
    index = "cu128" if options["hardware"] == "cuda" else "cpu"
    run([app_python, "-m", "pip", "install", "torch", "torchvision", "--index-url", f"https://download.pytorch.org/whl/{index}"])
    run([app_python, "-m", "pip", "install", "-r", ROOT / f"requirements-{options['hardware']}.txt"])
    install_pinned_sources(app_python)
    if options["gransabio"] == "download":
        if not folder.exists():
            run(["git", "clone", "--depth", "1", GRANSABIO_REPO, folder])
        else:
            git(folder, "pull", "--ff-only")
        gran_python = make_venv(STATE.parent / "gransabio-venv")
        run([gran_python, "-m", "pip", "install", "-r", folder / "requirements.txt"], cwd=folder)
        print(f"GranSabio revision: {git(folder, 'rev-parse', 'HEAD')}", file=sys.stderr)
    run([app_python, "-m", "pip", "install", folder / "client"])
    run([app_python, "-m", "pip", "check"])
    if options["hardware"] == "cuda":
        # InsightFace explicitly requires replacing its CPU runtime after installation.
        # Uninstall removes shared files, so reinstall the already resolved GPU version.
        version = run([app_python, "-c", "from importlib.metadata import version; print(version('onnxruntime-gpu'))"], capture=True)
        run([app_python, "-m", "pip", "uninstall", "-y", "onnxruntime"])
        run([app_python, "-m", "pip", "install", "--force-reinstall", "--no-deps", f"onnxruntime-gpu=={version}"])
    secrets = read_env(args.keys_file) if args.keys_file else {}
    secrets = {key: value for key, value in secrets.items() if key in KEY_NAMES}
    if not args.non_interactive:
        print("\nThe default setup uses Claude for ideas and Gemini for images.\n"
              "An API key is a secret password issued by that service. Never post it in GitHub or chat.\n"
              "You can press Enter to configure keys or local AI engines later in Settings and .env.\n"
              "Claude keys: https://console.anthropic.com/\nGemini keys: https://aistudio.google.com/apikey")
        for key, label in (("ANTHROPIC_API_KEY", "Claude API key"), ("GEMINI_API_KEY", "Gemini API key")):
            value = getpass.getpass(f"{label} (hidden; Enter to keep/skip): ").strip()
            if value:
                secrets[key] = value
    app_values = {
        "GRANSABIO_CLIENT_PATH": (folder / "client").as_posix(),
        "GRANSABIO_LLM_URL": options["gransabio_url"], "REDIS_URL": options["redis_url"],
    }
    if not (ROOT / ".env").exists():
        app_values.update(USE_LOCAL_WHISPER="true", SERVER_HOST="127.0.0.1", DEBUG_MODE="false")
    configure_env(ROOT / ".env", ROOT / ".env.example", app_values, secrets)
    if options["gransabio"] == "download":
        from urllib.parse import urlsplit
        address = urlsplit(options["gransabio_url"])
        configure_env(folder / ".env", folder / ".env.template", {
            "APP_HOST": "127.0.0.1", "APP_PORT": str(address.port or 8000), "APP_RELOAD": "false",
        }, {**secrets, "GOOGLE_API_KEY": secrets.get("GOOGLE_API_KEY") or secrets.get("GEMINI_API_KEY", "")})
    if options["models"] != "none":
        for script in ("download_portrait_model.py", "download_body_model.py"):
            run([app_python, ROOT / "scripts" / script])
    if options["models"] == "all":
        run([app_python, ROOT / "scripts" / "download_face_swap_models.py"])
    (ROOT / "Start Thumbnail Generator.cmd").write_text(
        '@echo off\r\ncd /d "%~dp0"\r\n"%~dp0.venv\\Scripts\\python.exe" "%~dp0install.py" start\r\npause\r\n', encoding="utf-8")
    return check(options, services=False)


def check(options, services):
    if not PYTHON.exists():
        raise ValueError("This application's environment is missing. Run install.ps1 first.")
    command = [PYTHON, ROOT / "setup_check.py", "--json", "--hardware", options["hardware"],
               "--models", options["models"], "--gransabio-dir", options["gransabio_dir"],
               "--web-port", str(options["web_port"])]
    if services:
        command.append("--services")
    result = subprocess.run([str(value) for value in command], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", cwd=ROOT, timeout=180)
    if result.returncode not in (0, 1, 3):
        raise RuntimeError("The installation checker could not run. Run .venv\\Scripts\\python.exe setup_check.py for diagnostics.")
    return json.loads(result.stdout)


def start(options):
    if not STATE.is_file() or not (ROOT / ".env").is_file() or not PYTHON.is_file():
        raise ValueError("Installation is incomplete. Run install.ps1 before starting the app.")
    # Run the supervisor in the app environment so it reuses dotenv and Redis.
    if Path(sys.executable).resolve() != PYTHON.resolve():
        return subprocess.call([str(PYTHON), str(ROOT / "install.py"), "start", *option_arguments(options)], cwd=ROOT)
    from dotenv import dotenv_values
    import redis
    signal.signal(signal.SIGBREAK, signal.default_int_handler)
    configuration = {**dotenv_values(ROOT / ".env"), **os.environ}
    client = redis.Redis.from_url(configuration["REDIS_URL"], socket_connect_timeout=3, socket_timeout=3)
    if options["redis"] == "wsl":
        run(["wsl", "-d", options["wsl_distribution"], "-u", "root", "--", "service", "redis-server", "start"])
    client.ping()
    from services.queue_health import WORKER_HEALTH_KEY
    if client.exists(WORKER_HEALTH_KEY):
        raise ValueError("A worker already uses this Redis queue. Use the running app; do not start a second worker.")
    children = []
    logs = []

    def launch(name, command, cwd, environment):
        path = STATE.parent / f"{name}.log"
        log = path.open("a", encoding="utf-8")
        logs.append(log)
        child = subprocess.Popen([str(value) for value in command], cwd=cwd, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                                 startupinfo=subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESHOWWINDOW, wShowWindow=0))
        children.append(child)
        return child

    def wait_http(url, expected):
        for _ in range(120):
            if any(child.poll() is not None for child in children):
                raise RuntimeError(f"A service exited. See logs in {STATE.parent}.")
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if json.load(response).get("status") == expected:
                        return
            except (OSError, ValueError):
                pass  # Bounded startup wait, not a provider fallback.
            time.sleep(0.5)
        raise RuntimeError(f"Service did not become ready at {url}. See logs in {STATE.parent}.")

    def require_free(port):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", port))

    try:
        require_free(options["web_port"])
        if options["gransabio"] == "download":
            from urllib.parse import urlsplit
            port = urlsplit(configuration["GRANSABIO_LLM_URL"]).port or 8000
            require_free(port)
            launch("gransabio", [STATE.parent / "gransabio-venv/Scripts/python.exe", "main.py"],
                   Path(options["gransabio_dir"]), {**os.environ, "APP_HOST": "127.0.0.1", "APP_PORT": str(port), "APP_RELOAD": "false"})
        wait_http(configuration["GRANSABIO_LLM_URL"].rstrip("/") + "/health", "healthy")
        launch("web", [PYTHON, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", options["web_port"]], ROOT, os.environ.copy())
        url = f"http://127.0.0.1:{options['web_port']}"
        wait_http(url + "/health", "ok")
        launch("worker", [PYTHON.parent / "arq.exe", "workers.main.WorkerSettings"], ROOT, os.environ.copy())
        for _ in range(120):
            if any(child.poll() is not None for child in children):
                raise RuntimeError(f"A service exited. See logs in {STATE.parent}.")
            if client.exists(WORKER_HEALTH_KEY):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("The background worker did not become ready. See .install/worker.log.")
        print(f"\nReady: open {url}\nKeep this window open. Finish or cancel jobs in the app before pressing Ctrl+C to stop.", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(1)
        raise RuntimeError(f"A service stopped unexpectedly. See logs in {STATE.parent}.")
    except KeyboardInterrupt:
        print("\nStopping the processes started by this launcher. Redis and existing servers are left running.")
        return 0
    finally:
        still_running = []
        for child in reversed(children):
            if child.poll() is None:
                child.send_signal(signal.CTRL_BREAK_EVENT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    still_running.append(child.pid)
        for log in logs:
            log.close()
        client.close()
        if still_running:
            raise RuntimeError(f"Processes {still_running} are still stopping. They were not force-killed; inspect the logs before restarting.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("action", nargs="?", default="wizard", choices=("wizard", "plan", "install", "check", "start", "update"))
    parser.add_argument("--non-interactive", action="store_true", help="Never ask questions; required for automated install/update.")
    parser.add_argument("--json", action="store_true", help="Print the plan/check result as JSON; progress goes to stderr.")
    parser.add_argument("--hardware", choices=("cpu", "cuda"))
    parser.add_argument("--gransabio", choices=("download", "existing"))
    parser.add_argument("--gransabio-dir", type=Path)
    parser.add_argument("--gransabio-url")
    parser.add_argument("--redis", choices=("wsl", "existing"))
    parser.add_argument("--redis-url")
    parser.add_argument("--wsl-distribution")
    parser.add_argument("--models", choices=("core", "all", "none"))
    parser.add_argument("--web-port", type=int)
    parser.add_argument("--keys-file", type=Path, help="Read selected API keys from a private dotenv file; never pass keys in arguments.")
    parser.add_argument("--services", action="store_true", help="With check, also verify running GranSabio, Redis, web, and worker.")
    args = parser.parse_args(argv)
    options = {"hardware": "cpu", "gransabio": "download", "gransabio_dir": str(STATE.parent / "GranSabio_LLM"),
               "gransabio_url": "http://127.0.0.1:8000", "redis": "wsl", "redis_url": "redis://127.0.0.1:6379/0",
               "wsl_distribution": "Ubuntu", "models": "core", "web_port": 8057}
    try:
        if STATE.exists():
            saved = json.loads(STATE.read_text(encoding="utf-8"))
            if set(saved) != set(options):
                raise ValueError("The saved installation options are incomplete or unknown. Inspect .install/settings.json before continuing.")
            options.update(saved)
        options.update({key: str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)
                        for key in options if getattr(args, key) is not None})
        if not 1 <= options["web_port"] <= 65535:
            raise ValueError("The web port must be between 1 and 65535.")
        if args.action == "wizard":
            if args.non_interactive or not sys.stdin.isatty():
                raise ValueError("Automation must choose an action. Start with: install.py plan --json")
            options = wizard(options)
        validate(options)
        if args.action == "plan":
            result = plan(options)
        elif args.action == "check":
            result = check(options, args.services)
        elif args.action == "start":
            return start(options)
        else:
            if args.json and not args.non_interactive:
                raise ValueError("JSON installation/update requires --non-interactive; use the wizard for guided setup.")
            if not args.non_interactive and not sys.stdin.isatty():
                raise ValueError("Automated installation requires --non-interactive. Inspect install.py plan --json first.")
            if not args.non_interactive:
                show_plan(options)
                if input("\nInstall these components now? [y/N]: ").strip().lower() != "y":
                    print("Canceled. Nothing was installed.")
                    return 0
            if args.action == "update":
                require_tools()
                check_checkout(ROOT, APP_REPO)
                if options["gransabio"] == "download":
                    check_checkout(Path(options["gransabio_dir"]), GRANSABIO_REPO)
                # Updating a running installation could mix old and new code.
                require_stopped(options)
                git(ROOT, "pull", "--ff-only")
                command = [sys.executable, ROOT / "install.py", "install", "--non-interactive", *option_arguments(options)]
                if args.keys_file:
                    command.extend(["--keys-file", args.keys_file])
                if args.json:
                    command.append("--json")
                return subprocess.call([str(value) for value in command], cwd=ROOT)
            result = install(options, args)
        if args.json:
            print(json.dumps(result, indent=2))
        elif args.action == "plan":
            show_plan(options)
        else:
            for item in result["checks"]:
                print(f"[{item['status'].upper()}] {item['name']}: {item['detail']}")
            print("\nNext: double-click 'Start Thumbnail Generator.cmd'. Open Settings to choose your AI engines.\n"
                  "Missing keys/models are listed above. Installation checks do not prove a paid provider account works.")
        return result.get("exit_code", 0)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc), "exit_code": 1}))
        else:
            print(f"\nSetup stopped: {exc}\nYour files were not reset. Fix the reported issue and run the same command again.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
