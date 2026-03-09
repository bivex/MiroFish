import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_config_and_factory_modules():
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

    factory_spec = importlib.util.spec_from_file_location(
        "app.services.graph_backend_factory",
        ROOT / "app" / "services" / "graph_backend_factory.py",
    )
    factory_module = importlib.util.module_from_spec(factory_spec)
    assert factory_spec.loader is not None
    sys.modules["app.services.graph_backend_factory"] = factory_module
    factory_spec.loader.exec_module(factory_module)

    return config_module, factory_module


def test_validate_graph_backend_only_requires_zep_key(monkeypatch):
    config_module, factory_module = load_config_and_factory_modules()
    Config = config_module.Config

    monkeypatch.setattr(Config, "LLM_API_KEY", "dummy-llm", raising=False)
    monkeypatch.setattr(Config, "ZEP_API_KEY", None, raising=False)
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "zep", raising=False)

    assert factory_module.validate_graph_backend_requirements("zep") == ["ZEP_API_KEY 未配置"]
    assert factory_module.validate_graph_backend_requirements("cognee") == []


def test_resolve_graph_backend_normalizes_and_rejects_unknown(monkeypatch):
    config_module, factory_module = load_config_and_factory_modules()
    Config = config_module.Config

    monkeypatch.setattr(Config, "GRAPH_BACKEND", "CoGnEe", raising=False)
    assert factory_module.resolve_graph_backend() == "cognee"

    try:
        factory_module.resolve_graph_backend("unknown")
    except ValueError as exc:
        assert "GRAPH_BACKEND" in str(exc)
    else:
        raise AssertionError("resolve_graph_backend should reject unsupported values")


def test_cognee_factories_are_lazy_and_not_implemented():
    _, factory_module = load_config_and_factory_modules()
    reader = factory_module.get_entity_reader_service("cognee")
    assert reader.__class__.__name__ == "CogneeEntityReader"