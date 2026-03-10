import importlib.util
import json
import sys
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
    sys.modules["app.services.simulation_manager"] = types.SimpleNamespace(SimulationManager=object, SimulationStatus=types.SimpleNamespace())
    sys.modules["app.services.simulation_runner"] = types.SimpleNamespace(SimulationRunner=object, RunnerStatus=types.SimpleNamespace())
    sys.modules["app.models.project"] = types.SimpleNamespace(ProjectManager=object)
    sys.modules["app.utils.logger"] = types.SimpleNamespace(get_logger=lambda name: types.SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None))

    module_path = ROOT / "app" / "api" / "simulation.py"
    spec = importlib.util.spec_from_file_location("app.api.simulation", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["app.api.simulation"] = module
    spec.loader.exec_module(module)
    return module


def test_check_simulation_prepared_allows_twitter_only_simulations(tmp_path):
    sim_dir = tmp_path / "sim_twitter_only"
    sim_dir.mkdir()
    (sim_dir / "state.json").write_text(json.dumps({
        "simulation_id": "sim_twitter_only",
        "status": "stopped",
        "config_generated": True,
        "enable_twitter": True,
        "enable_reddit": False,
    }), encoding="utf-8")
    (sim_dir / "simulation_config.json").write_text("{}", encoding="utf-8")
    (sim_dir / "twitter_profiles.csv").write_text("username\nagent_1\n", encoding="utf-8")

    module = load_simulation_api_module(tmp_path)

    is_prepared, info = module._check_simulation_prepared("sim_twitter_only")

    assert is_prepared is True
    assert info["status"] == "stopped"
    assert info["profiles_count"] == 1


def test_get_run_status_uses_public_runner_state_payload(tmp_path):
    module = load_simulation_api_module(tmp_path)

    raw_payload = {
        "simulation_id": "sim_waiting_completed",
        "runner_status": "running",
        "current_round": 5,
    }
    public_payload = {
        "simulation_id": "sim_waiting_completed",
        "runner_status": "completed",
        "current_round": 5,
        "completed_at": "2026-03-10T13:23:00",
    }

    run_state = types.SimpleNamespace(to_dict=lambda: raw_payload)
    module.SimulationRunner = types.SimpleNamespace(
        get_run_state=lambda simulation_id: run_state,
        get_public_run_state_dict=lambda state: public_payload,
    )

    payload = module.get_run_status("sim_waiting_completed")

    assert payload["success"] is True
    assert payload["data"] == public_payload