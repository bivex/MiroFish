import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_simulation_manager_module():
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = app_pkg

    services_pkg = types.ModuleType("app.services")
    services_pkg.__path__ = [str(ROOT / "app" / "services")]
    sys.modules["app.services"] = services_pkg

    config_module = types.ModuleType("app.config")
    config_module.Config = type("Config", (), {})
    sys.modules["app.config"] = config_module

    factory_module = types.ModuleType("app.services.graph_backend_factory")
    factory_module.get_entity_reader_service = lambda graph_backend=None: object()
    sys.modules["app.services.graph_backend_factory"] = factory_module

    sys.modules["app.services.oasis_profile_generator"] = types.SimpleNamespace(OasisProfileGenerator=object)
    sys.modules["app.services.simulation_config_generator"] = types.SimpleNamespace(SimulationConfigGenerator=object)
    sys.modules["app.utils.logger"] = types.SimpleNamespace(
        get_logger=lambda name: types.SimpleNamespace(
            debug=lambda *a, **k: None,
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
    )

    module_path = ROOT / "app" / "services" / "simulation_manager.py"
    spec = importlib.util.spec_from_file_location("app.services.simulation_manager", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["app.services.simulation_manager"] = module
    spec.loader.exec_module(module)
    return module


def test_get_profiles_reads_twitter_csv(tmp_path):
    module = load_simulation_manager_module()
    module.SimulationManager.SIMULATION_DATA_DIR = str(tmp_path)
    manager = module.SimulationManager()

    sim_dir = tmp_path / "sim_twitter"
    sim_dir.mkdir()
    (sim_dir / "state.json").write_text(json.dumps({
        "simulation_id": "sim_twitter",
        "project_id": "proj_1",
        "graph_id": "graph_1",
        "status": "ready",
    }), encoding="utf-8")
    (sim_dir / "twitter_profiles.csv").write_text(
        "user_id,name,username,user_char,description\n"
        "0,Princess Ilyra,princess_ilyra,Heir under pressure,Public heir\n",
        encoding="utf-8",
    )

    profiles = manager.get_profiles("sim_twitter", platform="twitter")

    assert profiles == [{
        "user_id": "0",
        "name": "Princess Ilyra",
        "username": "princess_ilyra",
        "user_char": "Heir under pressure",
        "description": "Public heir",
    }]


def test_get_profiles_normalizes_legacy_twitter_csv(tmp_path):
    module = load_simulation_manager_module()
    module.SimulationManager.SIMULATION_DATA_DIR = str(tmp_path)
    manager = module.SimulationManager()

    sim_dir = tmp_path / "sim_twitter_legacy"
    sim_dir.mkdir()
    (sim_dir / "state.json").write_text(json.dumps({
        "simulation_id": "sim_twitter_legacy",
        "project_id": "proj_1",
        "graph_id": "graph_1",
        "status": "ready",
    }), encoding="utf-8")
    (sim_dir / "twitter_profiles.csv").write_text(
        "user_id,name,username,bio,persona\n"
        "0,Archivist Maelin,archivist_maelin,Keeper of seals,Warns about forged decrees\n",
        encoding="utf-8",
    )

    profiles = manager.get_profiles("sim_twitter_legacy", platform="twitter")

    assert profiles == [{
        "user_id": "0",
        "name": "Archivist Maelin",
        "username": "archivist_maelin",
        "bio": "Keeper of seals",
        "persona": "Warns about forged decrees",
        "user_char": "Warns about forged decrees",
        "description": "Keeper of seals",
    }]


def test_get_profiles_reads_reddit_json(tmp_path):
    module = load_simulation_manager_module()
    module.SimulationManager.SIMULATION_DATA_DIR = str(tmp_path)
    manager = module.SimulationManager()

    sim_dir = tmp_path / "sim_reddit"
    sim_dir.mkdir()
    (sim_dir / "state.json").write_text(json.dumps({
        "simulation_id": "sim_reddit",
        "project_id": "proj_1",
        "graph_id": "graph_1",
        "status": "ready",
    }), encoding="utf-8")
    (sim_dir / "reddit_profiles.json").write_text(json.dumps([
        {"user_id": 0, "username": "scribe_1", "profile": "Court scribe"}
    ]), encoding="utf-8")

    profiles = manager.get_profiles("sim_reddit", platform="reddit")

    assert profiles == [{"user_id": 0, "username": "scribe_1", "profile": "Court scribe"}]