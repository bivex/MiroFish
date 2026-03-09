import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_config_module():
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv_stub)

    config_path = ROOT / "app" / "config.py"
    spec = importlib.util.spec_from_file_location("mirofish_test_config", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_stage_model_returns_stage_specific_values(monkeypatch):
    config_module = load_config_module()
    Config = config_module.Config

    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "base-model", raising=False)
    monkeypatch.setattr(Config, "ONTOLOGY_LLM_MODEL", "ontology-model", raising=False)
    monkeypatch.setattr(Config, "PROFILE_LLM_MODEL", "profile-model", raising=False)
    monkeypatch.setattr(Config, "SIM_CONFIG_LLM_MODEL", "sim-model", raising=False)
    monkeypatch.setattr(Config, "REPORT_LLM_MODEL", "report-model", raising=False)

    assert Config.get_stage_model("ontology") == "ontology-model"
    assert Config.get_stage_model("profile") == "profile-model"
    assert Config.get_stage_model("sim_config") == "sim-model"
    assert Config.get_stage_model("report") == "report-model"


def test_get_stage_model_falls_back_to_default(monkeypatch):
    config_module = load_config_module()
    Config = config_module.Config

    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "base-model", raising=False)

    assert Config.get_stage_model("unknown-stage") == "base-model"


def test_services_use_stage_specific_model_routing():
    files_and_needles = {
        ROOT / "app" / "services" / "ontology_generator.py": "Config.get_stage_model('ontology')",
        ROOT / "app" / "services" / "oasis_profile_generator.py": "Config.get_stage_model('profile')",
        ROOT / "app" / "services" / "simulation_config_generator.py": "Config.get_stage_model('sim_config')",
        ROOT / "app" / "services" / "report_agent.py": "Config.get_stage_model('report')",
    }

    for path, needle in files_and_needles.items():
        assert needle in path.read_text(encoding="utf-8")

