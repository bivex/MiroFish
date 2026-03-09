import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FakeBlueprint:
    def __init__(self, name, import_name):
        self.name = name
        self.import_name = import_name
        self.routes = {}

    def route(self, rule, methods=None):
        methods = tuple((methods or ["GET"]))

        def decorator(func):
            for method in methods:
                self.routes[(method.upper(), rule)] = func
            return func

        return decorator


class FakeRequest:
    def __init__(self):
        self._json = {}

    def get_json(self):
        return self._json


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def get_json(self):
        return self._payload


class FakeClient:
    def __init__(self, routes, request_obj):
        self.routes = routes
        self.request_obj = request_obj

    def post(self, path, json=None):
        self.request_obj._json = json or {}
        handler = self.routes[("POST", path)]
        result = handler()
        if isinstance(result, tuple):
            payload, status_code = result
        else:
            payload, status_code = result, 200
        return FakeResponse(payload, status_code)


class FakeFlaskApp:
    def __init__(self, request_obj):
        self.routes = {}
        self.request_obj = request_obj

    def register_blueprint(self, blueprint, url_prefix=""):
        for (method, rule), handler in blueprint.routes.items():
            self.routes[(method, f"{url_prefix}{rule}")] = handler

    def test_client(self):
        return FakeClient(self.routes, self.request_obj)


def load_report_api_blueprint_module():
    request_obj = FakeRequest()
    flask_module = types.ModuleType("flask")
    flask_module.request = request_obj
    flask_module.jsonify = lambda payload: payload
    flask_module.send_file = lambda *args, **kwargs: None
    sys.modules["flask"] = flask_module

    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = app_pkg

    api_pkg = types.ModuleType("app.api")
    api_pkg.report_bp = FakeBlueprint("report", __name__)
    sys.modules["app.api"] = api_pkg

    config_module = types.ModuleType("app.config")
    config_module.Config = type("Config", (), {"get_graph_backend": classmethod(lambda cls: "cognee")})
    sys.modules["app.config"] = config_module

    factory_module = types.ModuleType("app.services.graph_backend_factory")
    factory_module.get_report_tools_service = lambda graph_backend=None: object()
    sys.modules["app.services.graph_backend_factory"] = factory_module

    report_agent_module = types.ModuleType("app.services.report_agent")
    report_agent_module.ReportAgent = object
    report_agent_module.ReportManager = types.SimpleNamespace(get_report_by_simulation=lambda simulation_id: None)
    report_agent_module.ReportStatus = types.SimpleNamespace(COMPLETED="completed")
    sys.modules["app.services.report_agent"] = report_agent_module

    sys.modules["app.services.simulation_manager"] = types.SimpleNamespace(SimulationManager=object)
    sys.modules["app.services.simulation_runner"] = types.SimpleNamespace(SimulationRunner=types.SimpleNamespace(get_run_state=lambda simulation_id: None))
    sys.modules["app.models.project"] = types.SimpleNamespace(ProjectManager=object)
    sys.modules["app.models.task"] = types.SimpleNamespace(TaskManager=object, TaskStatus=types.SimpleNamespace())
    sys.modules["app.utils.logger"] = types.SimpleNamespace(
        get_logger=lambda name: types.SimpleNamespace(
            debug=lambda *a, **k: None,
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
    )

    module_path = ROOT / "app" / "api" / "report.py"
    spec = importlib.util.spec_from_file_location("app.api.report", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["app.api.report"] = module
    spec.loader.exec_module(module)
    return module, api_pkg.report_bp, request_obj


def test_generate_status_http_prefers_task_id_over_completed_simulation_report():
    module, report_bp, request_obj = load_report_api_blueprint_module()
    calls = {"task_ids": [], "simulation_ids": []}

    class TaskStub:
        def to_dict(self):
            return {
                "task_id": "task_new",
                "status": "processing",
                "metadata": {"report_id": "report_new"},
            }

    class TaskManagerStub:
        def get_task(self, task_id):
            calls["task_ids"].append(task_id)
            return TaskStub()

    def get_report_by_simulation(simulation_id):
        calls["simulation_ids"].append(simulation_id)
        return types.SimpleNamespace(report_id="report_old", status=module.ReportStatus.COMPLETED)

    module.TaskManager = TaskManagerStub
    module.ReportManager = types.SimpleNamespace(get_report_by_simulation=get_report_by_simulation)

    app = FakeFlaskApp(request_obj)
    app.register_blueprint(report_bp, url_prefix="/api/report")
    client = app.test_client()

    response = client.post(
        "/api/report/generate/status",
        json={"task_id": "task_new", "simulation_id": "sim_1"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["data"]["task_id"] == "task_new"
    assert payload["data"]["status"] == "processing"
    assert payload["data"]["metadata"]["report_id"] == "report_new"
    assert calls["task_ids"] == ["task_new"]
    assert calls["simulation_ids"] == []