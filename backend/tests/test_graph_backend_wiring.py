from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_backend_neutral_entities_are_extracted_from_zep_reader():
    entities_path = ROOT / "app" / "services" / "graph_entities.py"
    reader_path = ROOT / "app" / "services" / "zep_entity_reader.py"

    assert "class EntityNode" in entities_path.read_text(encoding="utf-8")
    assert "from .graph_entities import EntityNode, FilteredEntities" in reader_path.read_text(encoding="utf-8")


def test_graph_backend_factories_are_used_in_wiring_points():
    expectations = {
        ROOT / "app" / "services" / "simulation_manager.py": "get_entity_reader_service",
        ROOT / "app" / "services" / "simulation_runner.py": "get_graph_memory_manager_class",
        ROOT / "app" / "services" / "report_agent.py": "get_report_tools_service",
        ROOT / "app" / "api" / "graph.py": "get_graph_builder_service",
    }

    for path, needle in expectations.items():
        assert needle in path.read_text(encoding="utf-8")


def test_graph_backend_is_persisted_in_project_and_simulation_state_files():
    project_text = (ROOT / "app" / "models" / "project.py").read_text(encoding="utf-8")
    simulation_text = (ROOT / "app" / "services" / "simulation_manager.py").read_text(encoding="utf-8")

    assert '"graph_backend": self.graph_backend' in project_text
    assert 'graph_backend=data.get("graph_backend", "zep")' in simulation_text