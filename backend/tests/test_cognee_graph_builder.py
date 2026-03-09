import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_builder_module():
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv_stub)

    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = app_pkg

    services_pkg = types.ModuleType("app.services")
    services_pkg.__path__ = [str(ROOT / "app" / "services")]
    sys.modules["app.services"] = services_pkg

    config_spec = importlib.util.spec_from_file_location("app.config", ROOT / "app" / "config.py")
    config_module = importlib.util.module_from_spec(config_spec)
    assert config_spec.loader is not None
    sys.modules["app.config"] = config_module
    config_spec.loader.exec_module(config_module)

    sidecar_module = types.ModuleType("app.services.cognee_sidecar_client")

    class CogneeSidecarClient:
        pass

    sidecar_module.CogneeSidecarClient = CogneeSidecarClient
    sys.modules["app.services.cognee_sidecar_client"] = sidecar_module

    builder_spec = importlib.util.spec_from_file_location(
        "app.services.cognee_graph_builder",
        ROOT / "app" / "services" / "cognee_graph_builder.py",
    )
    builder_module = importlib.util.module_from_spec(builder_spec)
    assert builder_spec.loader is not None
    sys.modules["app.services.cognee_graph_builder"] = builder_module
    builder_spec.loader.exec_module(builder_module)
    return builder_module


def test_cognee_builder_wait_for_episodes_is_compatible_noop():
    module = load_builder_module()

    class SidecarStub:
        pass

    builder = module.CogneeGraphBuilderService(sidecar_client=SidecarStub())
    progress_updates = []

    builder._wait_for_episodes(["graph_1"], lambda msg, progress: progress_updates.append((msg, progress)))

    assert progress_updates == [("Cognee data is already synced; no extra wait required", 1.0)]


def test_cognee_builder_applies_projection_semantic_types(tmp_path):
    module = load_builder_module()

    class SidecarStub:
        def __init__(self, root):
            self.root = root

        def ensure_workspace(self, graph_id):
            workspace = self.root / graph_id
            workspace.mkdir(parents=True, exist_ok=True)
            return workspace

    builder = module.CogneeGraphBuilderService(sidecar_client=SidecarStub(tmp_path))
    builder._persist_projection_semantic_types(
        "graph_1",
        [
            "## Actors\nActor: Aria the Captain\n",
            "## Organizations\nOrganization: Harbor Guild\n",
        ],
    )

    graph_data = builder._format_graph_data(
        "graph_1",
        {
            "graph_id": "graph_1",
            "nodes": [
                {"uuid": "n1", "name": "aria the captain", "labels": ["Entity", "Entity"], "summary": "Captain", "attributes": {"type": "Entity"}},
                {"uuid": "n2", "name": "harbor guild", "labels": ["Entity", "Entity"], "summary": "Guild", "attributes": {"type": "Entity"}},
            ],
            "edges": [],
        },
    )

    labels_by_name = {node["name"]: node["labels"] for node in graph_data["nodes"]}
    attrs_by_name = {node["name"]: node["attributes"] for node in graph_data["nodes"]}

    assert "Actor" in labels_by_name["aria the captain"]
    assert "Organization" in labels_by_name["harbor guild"]
    assert attrs_by_name["aria the captain"]["semantic_type"] == "Actor"
    assert attrs_by_name["harbor guild"]["type"] == "Organization"