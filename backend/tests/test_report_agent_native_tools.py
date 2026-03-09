import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_report_agent_module():
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = app_pkg

    services_pkg = types.ModuleType("app.services")
    services_pkg.__path__ = [str(ROOT / "app" / "services")]
    sys.modules["app.services"] = services_pkg

    utils_pkg = types.ModuleType("app.utils")
    utils_pkg.__path__ = [str(ROOT / "app" / "utils")]
    sys.modules["app.utils"] = utils_pkg

    config_module = types.ModuleType("app.config")

    class Config:
        UPLOAD_FOLDER = "/tmp"

        @classmethod
        def get_graph_backend(cls):
            return "cognee"

        @classmethod
        def get_stage_model(cls, stage):
            return "dummy-model"

    config_module.Config = Config
    sys.modules["app.config"] = config_module

    llm_client_module = types.ModuleType("app.utils.llm_client")
    llm_client_module.LLMClient = object
    sys.modules["app.utils.llm_client"] = llm_client_module

    logger_module = types.ModuleType("app.utils.logger")
    logger_module.get_logger = lambda name: types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    sys.modules["app.utils.logger"] = logger_module

    factory_module = types.ModuleType("app.services.graph_backend_factory")
    factory_module.get_report_tools_service = lambda graph_backend=None: object()
    sys.modules["app.services.graph_backend_factory"] = factory_module

    module_path = ROOT / "app" / "services" / "report_agent.py"
    spec = importlib.util.spec_from_file_location("app.services.report_agent", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["app.services.report_agent"] = module
    spec.loader.exec_module(module)
    return module


def test_report_agent_builds_native_tool_specs_for_groq_compatible_llm():
    module = load_report_agent_module()

    class LLMStub:
        def prefers_native_tools(self):
            return True

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="s1",
        simulation_requirement="Run a harbor crisis simulation",
        llm_client=LLMStub(),
        zep_tools=object(),
        graph_backend="cognee",
    )

    specs = agent._get_native_llm_tools()
    quick_search = next(item for item in specs if item["function"]["name"] == "quick_search")
    tool_names = {item["function"]["name"] for item in specs}

    assert agent._should_use_native_tools() is True
    assert len(specs) == 3
    assert tool_names == {"insight_forge", "panorama_search", "quick_search"}
    assert "interview_agents" not in agent.tools
    assert quick_search["function"]["parameters"]["properties"]["limit"]["type"] == "integer"
    assert quick_search["function"]["parameters"]["required"] == ["query"]


def test_report_agent_requires_runtime_evidence_for_cognee_reports():
    module = load_report_agent_module()

    class RuntimeEvidenceStub:
        def get_runtime_evidence(self, simulation_id, limit=10):
            return {
                "simulation_id": simulation_id,
                "has_runtime_evidence": False,
                "message": "runtime evidence missing",
            }

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="s1",
        simulation_requirement="Run a harbor crisis simulation",
        llm_client=object(),
        zep_tools=RuntimeEvidenceStub(),
        graph_backend="cognee",
    )

    try:
        agent._ensure_generation_readiness()
    except ValueError as exc:
        assert "runtime evidence missing" in str(exc)
    else:
        raise AssertionError("Expected _ensure_generation_readiness to reject empty Cognee evidence")