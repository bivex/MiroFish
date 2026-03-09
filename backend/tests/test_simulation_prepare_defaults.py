import importlib.util
import sys
import threading as real_threading
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_simulation_api_module(sim_root: Path):
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = app_pkg

    api_pkg = types.ModuleType("app.api")

    class BlueprintStub:
        def route(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    api_pkg.simulation_bp = BlueprintStub()
    sys.modules["app.api"] = api_pkg

    flask_module = types.ModuleType("flask")
    flask_module.request = types.SimpleNamespace(args={}, get_json=lambda: {})
    flask_module.jsonify = lambda payload: payload
    flask_module.send_file = lambda *args, **kwargs: None
    sys.modules["flask"] = flask_module

    config_module = types.ModuleType("app.config")
    config_module.Config = type("Config", (), {"OASIS_SIMULATION_DATA_DIR": str(sim_root), "get_graph_backend": classmethod(lambda cls: "cognee")})
    sys.modules["app.config"] = config_module

    factory_module = types.ModuleType("app.services.graph_backend_factory")
    factory_module.get_entity_reader_service = lambda graph_backend=None: object()
    factory_module.validate_graph_backend_requirements = lambda graph_backend=None: []
    sys.modules["app.services.graph_backend_factory"] = factory_module

    sys.modules["app.services.oasis_profile_generator"] = types.SimpleNamespace(OasisProfileGenerator=object)
    sys.modules["app.services.simulation_manager"] = types.SimpleNamespace(SimulationManager=object, SimulationStatus=types.SimpleNamespace(PREPARING="preparing"))
    sys.modules["app.services.simulation_runner"] = types.SimpleNamespace(SimulationRunner=object, RunnerStatus=types.SimpleNamespace())
    sys.modules["app.models.project"] = types.SimpleNamespace(ProjectManager=object)
    sys.modules["app.models.task"] = types.SimpleNamespace(TaskManager=object, TaskStatus=types.SimpleNamespace(PROCESSING="processing"))
    sys.modules["app.utils.logger"] = types.SimpleNamespace(get_logger=lambda name: types.SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None))

    module_path = ROOT / "app" / "api" / "simulation.py"
    spec = importlib.util.spec_from_file_location("app.api.simulation", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["app.api.simulation"] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_uses_project_recommended_entity_types_when_request_omits_them(tmp_path, monkeypatch):
    module = load_simulation_api_module(tmp_path)
    captured = {"preview": None, "prepare": None}

    state = types.SimpleNamespace(
        project_id="proj_1",
        graph_id="graph_1",
        graph_backend="cognee",
        status="created",
        entities_count=0,
        entity_types=[],
    )

    class ManagerStub:
        def get_simulation(self, simulation_id):
            assert simulation_id == "sim_1"
            return state

        def _save_simulation_state(self, saved_state):
            return None

        def prepare_simulation(self, **kwargs):
            captured["prepare"] = kwargs["defined_entity_types"]
            return types.SimpleNamespace(to_simple_dict=lambda: {
                "simulation_id": "sim_1",
                "status": "ready",
                "entities_count": 16,
                "profiles_count": 16,
                "entity_types": ["Actor", "Organization"],
            })

    class ReaderStub:
        def filter_defined_entities(self, graph_id, defined_entity_types, enrich_with_edges=False):
            captured["preview"] = defined_entity_types
            return types.SimpleNamespace(filtered_count=16, entity_types=["Actor", "Organization"])

    class TaskManagerStub:
        def create_task(self, **kwargs):
            return "task_1"

        def update_task(self, *args, **kwargs):
            return None

        def complete_task(self, *args, **kwargs):
            return None

        def fail_task(self, *args, **kwargs):
            return None

    class ThreadStub:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self):
            if self.target:
                self.target()

    module.request.get_json = lambda: {
        "simulation_id": "sim_1",
        "force_regenerate": True,
        "use_llm_for_profiles": False,
        "parallel_profile_count": 1,
    }
    module.SimulationManager = lambda: ManagerStub()
    module.get_entity_reader_service = lambda graph_backend=None: ReaderStub()
    module.ProjectManager = types.SimpleNamespace(
        get_project=lambda project_id: types.SimpleNamespace(
            project_id=project_id,
            simulation_requirement="Simulate a court rumor.",
            ontology={"entity_types": [{"name": "DocumentChunk"}]},
            recommended_prepare_entity_types=["Actor", "Organization"],
        ),
        get_extracted_text=lambda project_id: "Projection text",
    )
    sys.modules["app.models.task"].TaskManager = TaskManagerStub
    monkeypatch.setattr(real_threading, "Thread", ThreadStub)

    response = module.prepare_simulation()

    assert response["success"] is True
    assert response["data"]["expected_entities_count"] == 16
    assert response["data"]["entity_types"] == ["Actor", "Organization"]
    assert captured["preview"] == ["Actor", "Organization"]
    assert captured["prepare"] == ["Actor", "Organization"]