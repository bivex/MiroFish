import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_cognee_tools_module():
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = app_pkg

    services_pkg = types.ModuleType("app.services")
    services_pkg.__path__ = [str(ROOT / "app" / "services")]
    sys.modules["app.services"] = services_pkg

    entity_reader_module = types.ModuleType("app.services.cognee_entity_reader")
    entity_reader_module.CogneeEntityReader = object
    sys.modules["app.services.cognee_entity_reader"] = entity_reader_module

    sidecar_client_module = types.ModuleType("app.services.cognee_sidecar_client")
    sidecar_client_module.CogneeSidecarClient = object
    sys.modules["app.services.cognee_sidecar_client"] = sidecar_client_module

    module_path = ROOT / "app" / "services" / "cognee_tools.py"
    spec = importlib.util.spec_from_file_location("app.services.cognee_tools", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["app.services.cognee_tools"] = module
    spec.loader.exec_module(module)
    return module


class SidecarStub:
    def get_graph_data(self, graph_id):
        return {
            "graph_id": graph_id,
            "nodes": [
                {"uuid": "n1", "name": "Harbor Guild", "labels": ["Entity", "Organization"], "summary": "Guild controls the harbor", "attributes": {"type": "Organization"}},
                {"uuid": "n2", "name": "Aria", "labels": ["Entity", "Actor"], "summary": "Captain aligned with the guild", "attributes": {"type": "Actor"}},
            ],
            "edges": [
                {"uuid": "e1", "name": "SUPPORTS", "fact": "Aria supports Harbor Guild", "source_node_uuid": "n2", "target_node_uuid": "n1"},
            ],
        }

    def search_graph(self, graph_id, query, limit):
        raise RuntimeError("search unavailable")


def test_cognee_tools_search_graph_falls_back_to_local_matches():
    module = load_cognee_tools_module()
    service = module.CogneeToolsService(sidecar_client=SidecarStub(), entity_reader=object())

    result = service.search_graph("graph_1", "Harbor Guild", limit=5)

    assert result.total_count == 1
    assert result.facts == ["Aria supports Harbor Guild"]
    assert result.nodes[0]["name"] == "Harbor Guild"


def test_cognee_tools_get_simulation_context_matches_zep_shape():
    module = load_cognee_tools_module()
    service = module.CogneeToolsService(sidecar_client=SidecarStub(), entity_reader=object())

    result = service.get_simulation_context("graph_1", "Run a harbor crisis simulation", limit=10)

    assert result["graph_statistics"]["total_nodes"] == 2
    assert result["graph_statistics"]["entity_types"]["Organization"] == 1
    assert result["total_entities"] == 2
    assert result["entities"][0]["type"] in {"Organization", "Actor"}