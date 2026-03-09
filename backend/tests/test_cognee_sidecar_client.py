import importlib.util
import sys
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