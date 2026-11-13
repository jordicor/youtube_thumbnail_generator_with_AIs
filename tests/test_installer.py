"""Installer boundaries: temporary files, no package downloads or host changes."""
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

import install


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(install, "ROOT", tmp_path)
    monkeypatch.setattr(install, "STATE", tmp_path / ".install" / "settings.json")
    monkeypatch.setattr(install, "PYTHON", Path(sys.executable))
    options = {"hardware": "cpu", "gransabio": "download", "gransabio_dir": str(tmp_path / ".install/GranSabio_LLM"),
               "gransabio_url": "http://127.0.0.1:18007", "redis": "existing", "redis_url": "redis://127.0.0.1:16389/0",
               "wsl_distribution": "Ubuntu", "models": "core", "web_port": 18057}
    return SimpleNamespace(root=tmp_path, options=options)


def test_plan_is_read_only_and_reports_the_selected_operations(setup, capsys):
    assert install.main(["plan", "--json", "--redis", "existing", "--hardware", "cuda"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["options"]["hardware"] == "cuda"
    assert any("latest published GranSabio" in step for step in result["steps"])
    assert not list(setup.root.iterdir())


@pytest.mark.parametrize("arguments", [[], ["install"], ["wizard", "--non-interactive"]])
def test_piped_invocation_never_waits_for_an_answer(setup, monkeypatch, capsys, arguments):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert install.main(arguments) == 1
    assert not install.STATE.exists()
    assert "install.py plan --json" in capsys.readouterr().err


@pytest.mark.parametrize("name,value", [
    # Deliberately fake credentials exercise rejection without exposing real secrets.
    ("redis_url", "redis://user:private-password@localhost:6379/0"),
    ("gransabio_url", "https://example.test/?token=private-token"),
    ("gransabio_url", "http://remote.test:8000"),
    ("gransabio_dir", "."),
])
def test_invalid_or_secret_addresses_are_rejected_without_echoing_the_value(setup, name, value):
    options = {**setup.options, name: str(setup.root) if name == "gransabio_dir" else value}
    with pytest.raises(ValueError) as error:
        install.validate(options)
    assert "private-password" not in str(error.value)
    assert "private-token" not in str(error.value)
    assert not install.STATE.exists()


def test_existing_gransabio_requires_the_packaged_client(setup):
    setup.options["gransabio"] = "existing"
    with pytest.raises(ValueError, match="client/pyproject.toml"):
        install.validate(setup.options)


def test_environment_creation_clears_examples_and_quotes_keys_without_shell_execution(setup, capsys):
    template = setup.root / "example"
    template.write_text("ANTHROPIC_API_KEY=your-key\nREDIS_URL=redis://localhost:6379\n", encoding="utf-8")
    path = setup.root / ".env"
    secret = "dummy-secret-'quoted'-$()"
    install.configure_env(path, template, {"REDIS_URL": setup.options["redis_url"]}, {"ANTHROPIC_API_KEY": secret})
    parsed = install.read_env(path)
    assert parsed["ANTHROPIC_API_KEY"] == secret
    assert parsed["REDIS_URL"] == setup.options["redis_url"]
    assert secret not in capsys.readouterr().out


def test_existing_keys_are_preserved_and_conflicting_settings_do_not_partially_write(setup):
    path = setup.root / ".env"
    original = "ANTHROPIC_API_KEY=existing-secret\nREDIS_URL=redis://localhost:6379\n"
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY") as error:
        install.configure_env(path, setup.root / "unused", {"REDIS_URL": "redis://localhost:6379"},
                              {"ANTHROPIC_API_KEY": "replacement-secret"})
    assert "existing-secret" not in str(error.value)
    assert "replacement-secret" not in str(error.value)
    assert install.read_env(path)["ANTHROPIC_API_KEY"] == "existing-secret"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="REDIS_URL"):
        install.configure_env(path, setup.root / "unused", {"REDIS_URL": setup.options["redis_url"]}, {})
    assert path.read_bytes() == before


def test_templates_without_real_keys_do_not_become_configured(setup):
    template = setup.root / "example"
    template.write_text("OPENAI_API_KEY=sk-your-placeholder\nSOME_API_TOKEN=example\n", encoding="utf-8")
    install.configure_env(setup.root / ".env", template, {}, {})
    assert install.read_env(setup.root / ".env") == {"OPENAI_API_KEY": "", "SOME_API_TOKEN": ""}


def test_rerunning_setup_preserves_redis_credentials_entered_locally(setup):
    path = setup.root / ".env"
    # Synthetic local fixture, never a real Redis password.
    original = "REDIS_URL=redis://user:private-password@127.0.0.1:16389/0\n"
    path.write_text(original, encoding="utf-8")
    install.configure_env(path, setup.root / "unused", {"REDIS_URL": setup.options["redis_url"]}, {})
    assert path.read_text(encoding="utf-8") == original


def test_wizard_explains_components_and_costs_before_asking(setup, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda question: "1")
    options = install.wizard(setup.options.copy())
    output = capsys.readouterr().out
    assert "not an AI subscription" in output
    assert "keep track of background jobs" in output
    assert "may charge" in output
    assert "Microsoft" in output
    assert options["gransabio"] == "download"
    assert options["hardware"] == "cpu"


def test_existing_redis_does_not_run_wsl_commands(setup, monkeypatch):
    monkeypatch.setattr(install, "run", lambda *args, **kwargs: pytest.fail("Must not change an existing Redis service"))
    install.setup_redis(setup.options)


@pytest.mark.parametrize("installed", [False, True])
def test_wsl_provisions_only_a_missing_distribution_then_installs_redis(setup, monkeypatch, installed):
    setup.options["redis"] = "wsl"
    monkeypatch.setattr(install.shutil, "which", lambda name: "wsl.exe")
    monkeypatch.setattr(install.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=("Ubuntu\r\n" if installed else "").encode("utf-16-le"), returncode=0))
    commands = []
    monkeypatch.setattr(install, "run", lambda command, **kwargs: commands.append(command))
    install.setup_redis(setup.options)
    assert any("--install" in command for command in commands) is not installed
    assert commands[-1][:7] == ["wsl", "-d", "Ubuntu", "-u", "root", "--", "sh"]
    assert "apt-get install -y redis-server" in commands[-1][-1]


def test_wsl_install_failure_stops_with_resume_instructions(setup, monkeypatch):
    setup.options["redis"] = "wsl"
    monkeypatch.setattr(install.shutil, "which", lambda name: "wsl.exe")
    monkeypatch.setattr(install.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=b"", returncode=1))
    def failed(command, **kwargs):
        assert "--install" in command
        raise RuntimeError("Windows needs administrator approval")
    monkeypatch.setattr(install, "run", failed)
    with pytest.raises(RuntimeError, match="rerun this installer in a normal terminal"):
        install.setup_redis(setup.options)


def test_wizard_defaults_preserve_saved_choices_and_addresses(setup, monkeypatch):
    options = {**setup.options, "hardware": "cuda", "models": "all", "gransabio": "existing"}
    monkeypatch.setattr("builtins.input", lambda question: "")
    assert install.wizard(options.copy()) == options


def test_profile_change_stops_before_rewriting_choices_or_packages(setup, monkeypatch):
    install.STATE.parent.mkdir()
    install.STATE.write_text(json.dumps(setup.options), encoding="utf-8")
    monkeypatch.setattr(install, "require_tools", lambda: None)
    monkeypatch.setattr(install, "require_stopped", lambda options: None)
    monkeypatch.setattr(install.shutil, "which", lambda name: "nvidia-smi.exe")
    monkeypatch.setattr(install, "run", lambda *args, **kwargs: pytest.fail("No package changes allowed"))
    with pytest.raises(ValueError, match="different CPU/NVIDIA profile"):
        install.install({**setup.options, "hardware": "cuda"}, SimpleNamespace(keys_file=None))
    assert json.loads(install.STATE.read_text(encoding="utf-8")) == setup.options


def test_missing_private_keys_file_fails_before_creating_environments(setup, monkeypatch):
    monkeypatch.setattr(install, "require_tools", lambda: None)
    monkeypatch.setattr(install, "require_stopped", lambda options: None)
    args = SimpleNamespace(keys_file=setup.root / "missing.env", non_interactive=True)
    with pytest.raises(ValueError, match="keys file"):
        install.install(setup.options, args)
    assert not install.STATE.exists()


@pytest.mark.parametrize("running_service", ["web", "gransabio"])
def test_update_refuses_running_services_before_pulling(setup, monkeypatch, capsys, running_service):
    monkeypatch.setattr(install, "require_tools", lambda: None)
    monkeypatch.setattr(install, "check_checkout", lambda *args: None)
    monkeypatch.setattr(install, "git", lambda *args: pytest.fail("Must not pull a running installation"))
    with socket.socket() as listener, socket.socket() as unused:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        unused.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        unused_port = unused.getsockname()[1]
        arguments = ["update", "--non-interactive", "--redis", "existing"]
        if running_service == "web":
            arguments += ["--web-port", str(port), "--gransabio-url", f"http://127.0.0.1:{unused_port}"]
        else:
            arguments += ["--gransabio-url", f"http://127.0.0.1:{port}", "--web-port", str(unused_port)]
        assert install.main(arguments) == 1
        assert f"Port {port}" in capsys.readouterr().err


def test_dirty_checkout_is_not_updated(setup, monkeypatch):
    (setup.root / ".git").mkdir()
    def git(folder, *arguments):
        return install.GRANSABIO_REPO if arguments[0] == "remote" else " M requirements.txt"
    monkeypatch.setattr(install, "git", git)
    with pytest.raises(ValueError, match="local changes"):
        install.check_checkout(setup.root, install.GRANSABIO_REPO)


def test_update_fast_forwards_a_clean_clone_and_runs_updated_installer(setup, monkeypatch):
    source = setup.root / "source"
    checkout = setup.root / "checkout"
    client = setup.root / "gransabio" / "client"
    source.mkdir()
    client.mkdir(parents=True)
    (client / "pyproject.toml").write_text("", encoding="utf-8")
    install.git(source, "init", "-b", "main")
    install.git(source, "config", "user.name", "Installer Test")
    install.git(source, "config", "user.email", "test@example.invalid")
    (source / "install.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
    install.git(source, "add", "install.py")
    install.git(source, "commit", "-m", "Initial installer")
    install.git(setup.root, "clone", str(source), str(checkout))
    (source / "install.py").write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "Path('invocation.json').write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n", encoding="utf-8")
    install.git(source, "commit", "-am", "Update installer")
    monkeypatch.setattr(install, "ROOT", checkout)
    monkeypatch.setattr(install, "APP_REPO", str(source))
    monkeypatch.setattr(install, "require_tools", lambda: None)
    monkeypatch.setattr(install, "require_stopped", lambda options: None)
    assert install.main(["update", "--non-interactive", "--json", "--gransabio", "existing",
                         "--gransabio-dir", str(client.parent), "--redis", "existing", "--models", "none"]) == 0
    assert install.git(checkout, "rev-parse", "HEAD") == install.git(source, "rev-parse", "HEAD")
    arguments = json.loads((checkout / "invocation.json").read_text(encoding="utf-8"))
    assert arguments[:2] == ["install", "--non-interactive"]
    assert arguments[arguments.index("--models") + 1] == "none"
    assert arguments[-1] == "--json"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows bootstrap")
def test_powershell_bootstrap_forwards_automation_arguments_without_installing_tools():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root / "install.ps1"),
                             "-PythonPath", sys.executable, "-NonInteractive", "plan", "--json", "--redis", "existing"],
                            capture_output=True, text=True, encoding="utf-8", cwd=root, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["options"]["redis"] == "existing"


def test_checker_preserves_failure_and_incomplete_configuration_status(setup, monkeypatch):
    monkeypatch.setattr(install.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=3, stdout=json.dumps({"installation_ok": True, "configuration_pending": True, "exit_code": 3})))
    result = install.check(setup.options, services=False)
    assert result["exit_code"] == 3
    assert result["configuration_pending"]


@pytest.mark.parametrize("installed_revision", [None, "correct"])
def test_git_dependency_pin_replaces_a_different_source_but_keeps_the_correct_one(setup, monkeypatch, installed_revision):
    (setup.root / "requirements.txt").write_text("example @ git+https://example.test/source@correct\n", encoding="utf-8")
    installs = []
    def run(command, **kwargs):
        if kwargs.get("capture"):
            return json.dumps({"vcs_info": {"commit_id": installed_revision}})
        installs.append(command)
    monkeypatch.setattr(install, "run", run)
    install.install_pinned_sources(Path(sys.executable))
    assert bool(installs) == (installed_revision is None)
    if installs:
        assert "--no-deps" in installs[0]
        assert installs[0][-1].endswith("@correct")
