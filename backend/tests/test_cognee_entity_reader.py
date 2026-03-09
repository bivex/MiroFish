import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_entity_reader_module():
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

    module_path = ROOT / "app" / "services" / "cognee_entity_reader.py"
    spec = importlib.util.spec_from_file_location("app.services.cognee_entity_reader", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["app.services.cognee_entity_reader"] = module
    spec.loader.exec_module(module)
    return module


class BuilderStub:
    def get_graph_data(self, graph_id):
        return {
            "graph_id": graph_id,
            "nodes": [
                {"uuid": "n1", "name": "Alice", "labels": ["Entity", "person"], "summary": "Hero", "attributes": {"type": "person"}},
                {"uuid": "n2", "name": "Guild", "labels": ["Entity", "organization"], "summary": "Faction", "attributes": {"type": "organization"}},
            ],
            "edges": [
                {"uuid": "e1", "name": "MEMBER_OF", "fact": "Alice belongs to Guild", "source_node_uuid": "n1", "target_node_uuid": "n2", "attributes": {}},
            ],
        }


def test_cognee_entity_reader_filters_and_enriches_entities():
    module = load_entity_reader_module()
    reader = module.CogneeEntityReader(builder=BuilderStub())
    result = reader.filter_defined_entities("graph_1", defined_entity_types=["person"], enrich_with_edges=True)

    assert result.filtered_count == 1
    assert result.entities[0].name == "Alice"
    assert result.entities[0].related_edges[0]["edge_name"] == "MEMBER_OF"
    assert result.entities[0].related_nodes[0]["name"] == "Guild"