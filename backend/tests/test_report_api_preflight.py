import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_report_api_module():
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = app_pkg

    api_pkg = types.ModuleType("app.api")

    class BlueprintStub:
        def route(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    api_pkg.report_bp = BlueprintStub()
    sys.modules["app.api"] = api_pkg

    services_pkg = types.ModuleType("app.services")
    services_pkg.__path__ = [str(ROOT / "app" / "services")]
    sys.modules["app.services"] = services_pkg

    flask_module = types.ModuleType("flask")
    flask_module.request = types.SimpleNamespace(get_json=lambda: {})
    flask_module.jsonify = lambda payload: payload
    flask_module.send_file = lambda *args, **kwargs: None
    sys.modules["flask"] = flask_module

    config_module = types.ModuleType("app.config")

    class Config:
        @classmethod
        def get_graph_backend(cls):
            return "cognee"

    config_module.Config = Config
    sys.modules["app.config"] = config_module

    factory_module = types.ModuleType("app.services.graph_backend_factory")
    factory_module.get_report_tools_service = lambda graph_backend=None: object()
    sys.modules["app.services.graph_backend_factory"] = factory_module

    sys.modules["app.services.mirofish_writeback_bridge"] = types.SimpleNamespace(
        try_post_report_writeback=lambda report, graph_backend=None: {"enabled": False, "skipped": True}
    )

    report_agent_module = types.ModuleType("app.services.report_agent")
    report_agent_module.ReportAgent = object
    report_agent_module.ReportManager = types.SimpleNamespace(get_report_by_simulation=lambda simulation_id: None)
    report_agent_module.ReportStatus = types.SimpleNamespace(COMPLETED="completed")
    sys.modules["app.services.report_agent"] = report_agent_module

    simulation_manager_module = types.ModuleType("app.services.simulation_manager")
    simulation_manager_module.SimulationManager = object
    sys.modules["app.services.simulation_manager"] = simulation_manager_module

    simulation_runner_module = types.ModuleType("app.services.simulation_runner")

    class SimulationRunner:
        run_state = None

        @classmethod
        def get_run_state(cls, simulation_id):
            return cls.run_state

        @classmethod
        def get_public_runner_status(cls, run_state):
            return getattr(getattr(run_state, "runner_status", None), "value", "idle")

    simulation_runner_module.SimulationRunner = SimulationRunner
    sys.modules["app.services.simulation_runner"] = simulation_runner_module

    models_project_module = types.ModuleType("app.models.project")
    models_project_module.ProjectManager = object
    sys.modules["app.models.project"] = models_project_module

    models_task_module = types.ModuleType("app.models.task")
    models_task_module.TaskManager = object
    models_task_module.TaskStatus = object
    sys.modules["app.models.task"] = models_task_module

    logger_module = types.ModuleType("app.utils.logger")
    logger_module.get_logger = lambda name: types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    sys.modules["app.utils.logger"] = logger_module

    module_path = ROOT / "app" / "api" / "report.py"
    spec = importlib.util.spec_from_file_location("app.api.report", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["app.api.report"] = module
    spec.loader.exec_module(module)
    return module


def test_report_api_preflight_rejects_cognee_without_runtime_evidence():
    module = load_report_api_module()
    module.SimulationRunner.run_state = None
    state = types.SimpleNamespace(simulation_id="sim_1", status=types.SimpleNamespace(value="ready"))

    error = module._get_report_generation_preflight_error(state, "cognee")

    assert error["code"] == "insufficient_runtime_evidence"
    assert error["status"] == 409
    assert error["details"]["simulation_status"] == "ready"
    assert error["details"]["has_runtime_evidence"] is False


def test_report_api_preflight_accepts_runtime_evidence():
    module = load_report_api_module()
    module.SimulationRunner.run_state = types.SimpleNamespace(
        runner_status=types.SimpleNamespace(value="running"),
        current_round=1,
        twitter_actions_count=2,
        reddit_actions_count=0,
        recent_actions=[],
    )
    state = types.SimpleNamespace(simulation_id="sim_1", status=types.SimpleNamespace(value="running"))

    runtime = module._collect_runtime_evidence("sim_1")
    error = module._get_report_generation_preflight_error(state, "cognee")

    assert runtime["has_runtime_evidence"] is True
    assert runtime["total_actions"] == 2
    assert runtime["runner_status"] == "running"
    assert error is None


def test_generate_status_prefers_task_id_over_completed_simulation_report():
    module = load_report_api_module()
    module.request.get_json = lambda: {
        "task_id": "task_new",
        "simulation_id": "sim_1",
    }
    module.ReportManager = types.SimpleNamespace(
        get_report_by_simulation=lambda simulation_id: types.SimpleNamespace(
            report_id="report_old",
            status=module.ReportStatus.COMPLETED,
        )
    )

    class TaskStub:
        def to_dict(self):
            return {
                "task_id": "task_new",
                "status": "processing",
                "metadata": {"report_id": "report_new"},
            }

    class TaskManagerStub:
        def get_task(self, task_id):
            assert task_id == "task_new"
            return TaskStub()

    module.TaskManager = TaskManagerStub

    response = module.get_generate_status()

    assert response["success"] is True
    assert response["data"]["task_id"] == "task_new"
    assert response["data"]["status"] == "processing"
    assert response["data"]["metadata"]["report_id"] == "report_new"


def test_generate_status_uses_completed_report_when_only_simulation_id_is_provided():
    module = load_report_api_module()
    module.request.get_json = lambda: {"simulation_id": "sim_1"}
    module.ReportManager = types.SimpleNamespace(
        get_report_by_simulation=lambda simulation_id: types.SimpleNamespace(
            report_id="report_old",
            status=module.ReportStatus.COMPLETED,
        )
    )

    response = module.get_generate_status()

    assert response["success"] is True
    assert response["data"]["simulation_id"] == "sim_1"
    assert response["data"]["report_id"] == "report_old"
    assert response["data"]["already_completed"] is True