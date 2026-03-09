import importlib.util
import json
import sys
import threading
import time
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_sidecar_client_module():
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv_stub)

    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = app_pkg

    services_pkg = types.ModuleType("app.services")
    services_pkg.__path__ = [str(ROOT / "app" / "services")]
    sys.modules["app.services"] = services_pkg

    module_path = ROOT / "app" / "services" / "cognee_sidecar_client.py"
    spec = importlib.util.spec_from_file_location("app.services.cognee_sidecar_client", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["app.services.cognee_sidecar_client"] = module
    spec.loader.exec_module(module)
    return module


class CompletedProcessStub:
    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_sidecar_client_parses_sentinel_response(monkeypatch, tmp_path):
    module = load_sidecar_client_module()
    client = module.CogneeSidecarClient()
    client.sidecar_root = tmp_path / "cognee_sidecar"
    client.sidecar_python = tmp_path / "python"
    client.sidecar_script = tmp_path / "bridge.py"
    client.workspace_root = client.sidecar_root / "workspaces"
    client.sidecar_root.mkdir(parents=True)
    client.sidecar_python.write_text("", encoding="utf-8")
    client.sidecar_script.write_text("", encoding="utf-8")

    def fake_run(*args, **kwargs):
        return CompletedProcessStub(stdout=f"noise\n{module.SENTINEL}{{\"success\": true, \"data\": {{\"graph_id\": \"g1\"}}}}")

    monkeypatch.setattr("subprocess.run", fake_run)
    data = client.get_graph_data("g1")
    assert data["graph_id"] == "g1"


def test_sidecar_client_delete_graph_removes_workspace(tmp_path):
    module = load_sidecar_client_module()
    client = module.CogneeSidecarClient()
    client.sidecar_root = tmp_path / "cognee_sidecar"
    client.workspace_root = client.sidecar_root / "workspaces"
    workspace = client.ensure_workspace("graph_x")
    (workspace / "marker.txt").write_text("ok", encoding="utf-8")

    client.delete_graph("graph_x")
    assert not workspace.exists()


def test_sidecar_client_serializes_subprocess_access_per_graph(monkeypatch, tmp_path):
    module = load_sidecar_client_module()
    client = module.CogneeSidecarClient()
    client.sidecar_root = tmp_path / "cognee_sidecar"
    client.sidecar_python = tmp_path / "python"
    client.sidecar_script = tmp_path / "bridge.py"
    client.workspace_root = client.sidecar_root / "workspaces"
    client.sidecar_root.mkdir(parents=True)
    client.sidecar_python.write_text("", encoding="utf-8")
    client.sidecar_script.write_text("", encoding="utf-8")

    state_lock = threading.Lock()
    active_calls = 0
    max_active_calls = 0

    def fake_run(*args, **kwargs):
        nonlocal active_calls, max_active_calls
        with state_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        time.sleep(0.05)
        with state_lock:
            active_calls -= 1
        return CompletedProcessStub(stdout=f"{module.SENTINEL}{{\"success\": true, \"data\": {{\"graph_id\": \"g1\"}}}}")

    monkeypatch.setattr("subprocess.run", fake_run)

    first = threading.Thread(target=client.get_graph_data, args=("g1",))
    second = threading.Thread(target=client.get_graph_data, args=("g1",))
    first.start()
    second.start()
    first.join()
    second.join()

    assert max_active_calls == 1


def test_sidecar_client_search_graph_forwards_runtime_llm_config(monkeypatch, tmp_path):
    module = load_sidecar_client_module()
    client = module.CogneeSidecarClient()
    client.sidecar_root = tmp_path / "cognee_sidecar"
    client.sidecar_python = tmp_path / "python"
    client.sidecar_script = tmp_path / "bridge.py"
    client.workspace_root = client.sidecar_root / "workspaces"
    client.sidecar_root.mkdir(parents=True)
    client.sidecar_python.write_text("", encoding="utf-8")
    client.sidecar_script.write_text("", encoding="utf-8")

    captured_payload = {}

    def fake_run(*args, **kwargs):
        nonlocal captured_payload
        captured_payload = json.loads(kwargs["input"])
        return CompletedProcessStub(stdout=f"{module.SENTINEL}{{\"success\": true, \"data\": {{\"graph_id\": \"g1\", \"results\": []}}}}")

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("REPORT_LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr("subprocess.run", fake_run)

    client.search_graph("g1", "harbor rumor", limit=3)

    assert captured_payload["graph_id"] == "g1"
    assert captured_payload["dataset_name"] == "g1"
    assert captured_payload["query"] == "harbor rumor"
    assert captured_payload["limit"] == 3
    assert captured_payload["llm_api_key"] == "test-key"
    assert captured_payload["llm_endpoint"] == "https://api.groq.com/openai/v1"
    assert captured_payload["llm_model"] == "openai/gpt-oss-20b"
    assert captured_payload["llm_provider"] == "openai"