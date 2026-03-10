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

    sys.modules["app.services.mirofish_writeback_bridge"] = types.SimpleNamespace(
        try_post_report_writeback=lambda report, graph_backend=None: {"enabled": False, "skipped": True}
    )

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


def test_get_report_by_simulation_http_returns_manager_selected_report():
    module, _, _ = load_report_api_blueprint_module()

    report = types.SimpleNamespace(
        report_id="report_latest_completed",
        status="completed",
        to_dict=lambda: {
            "report_id": "report_latest_completed",
            "simulation_id": "sim_1",
            "status": "completed",
        },
    )
    module.ReportManager = types.SimpleNamespace(get_report_by_simulation=lambda simulation_id: report)

    response = module.get_report_by_simulation("sim_1")

    assert response["success"] is True
    assert response["has_report"] is True
    assert response["data"]["report_id"] == "report_latest_completed"


def test_get_report_http_returns_persisted_writeback_result():
    module, _, _ = load_report_api_blueprint_module()

    report = types.SimpleNamespace(
        to_dict=lambda: {
            "report_id": "report_with_writeback",
            "simulation_id": "sim_1",
            "status": "completed",
            "writeback_result": {
                "enabled": True,
                "ok": True,
                "candidate_deltas_count": 2,
            },
        }
    )
    module.ReportManager = types.SimpleNamespace(get_report=lambda report_id: report)

    response = module.get_report("report_with_writeback")

    assert response["success"] is True
    assert response["data"]["writeback_result"]["candidate_deltas_count"] == 2


def test_collect_runtime_evidence_uses_public_runner_status():
    module, _, _ = load_report_api_blueprint_module()

    run_state = types.SimpleNamespace(
        runner_status=types.SimpleNamespace(value="running"),
        current_round=12,
        twitter_actions_count=18,
        reddit_actions_count=0,
        recent_actions=[],
    )
    module.SimulationRunner = types.SimpleNamespace(
        get_run_state=lambda simulation_id: run_state,
        get_public_runner_status=lambda state: types.SimpleNamespace(value="completed"),
    )

    payload = module._collect_runtime_evidence("sim_1")

    assert payload["simulation_id"] == "sim_1"
    assert payload["runner_status"] == "completed"
    assert payload["current_round"] == 12
    assert payload["total_actions"] == 18
    assert payload["has_runtime_evidence"] is True


def test_generate_report_http_completed_task_includes_writeback_result():
    module, report_bp, request_obj = load_report_api_blueprint_module()
    completed = {}
    saved_reports = []

    class TaskManagerStub:
        def create_task(self, task_type, metadata=None):
            return "task_1"

        def update_task(self, *args, **kwargs):
            return None

        def complete_task(self, task_id, result):
            completed["task_id"] = task_id
            completed["result"] = result

        def fail_task(self, task_id, error):
            raise AssertionError(f"unexpected fail_task: {task_id} {error}")

    class ReportAgentStub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate_report(self, progress_callback=None, report_id=None):
            return types.SimpleNamespace(
                report_id=report_id,
                simulation_id="sim_1",
                graph_id="graph_1",
                simulation_requirement="Track rumor spread.",
                status=module.ReportStatus.COMPLETED,
                error=None,
            )

    class ThreadStub:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self):
            self.target()

    module.TaskManager = TaskManagerStub
    module.TaskStatus = types.SimpleNamespace(PROCESSING="processing")
    module.ReportAgent = ReportAgentStub
    module.threading.Thread = ThreadStub
    module.ReportManager = types.SimpleNamespace(
        get_report_by_simulation=lambda simulation_id: None,
        save_report=lambda report: saved_reports.append(report),
    )
    module.SimulationManager = lambda: types.SimpleNamespace(
        get_simulation=lambda simulation_id: types.SimpleNamespace(
            simulation_id=simulation_id,
            project_id="proj_1",
            graph_id="graph_1",
            graph_backend="zep",
        )
    )
    module.ProjectManager = types.SimpleNamespace(
        get_project=lambda project_id: types.SimpleNamespace(
            project_id=project_id,
            graph_id="graph_1",
            graph_backend="zep",
            simulation_requirement="Track rumor spread.",
        )
    )
    module.try_post_report_writeback = lambda report, graph_backend=None: {
        "enabled": True,
        "ok": True,
        "run_id": report.report_id,
    }

    app = FakeFlaskApp(request_obj)
    app.register_blueprint(report_bp, url_prefix="/api/report")
    client = app.test_client()

    response = client.post("/api/report/generate", json={"simulation_id": "sim_1", "graph_backend": "zep"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert completed["task_id"] == "task_1"
    assert completed["result"]["status"] == "completed"
    assert completed["result"]["writeback"] == {
        "enabled": True,
        "ok": True,
        "run_id": payload["data"]["report_id"],
    }
    assert len(saved_reports) == 2
    assert saved_reports[-1].writeback_result == completed["result"]["writeback"]


def test_generate_report_http_bridge_exception_does_not_fail_completed_task():
    module, report_bp, request_obj = load_report_api_blueprint_module()
    completed = {}
    saved_reports = []

    class TaskManagerStub:
        def create_task(self, task_type, metadata=None):
            return "task_2"

        def update_task(self, *args, **kwargs):
            return None

        def complete_task(self, task_id, result):
            completed["task_id"] = task_id
            completed["result"] = result

        def fail_task(self, task_id, error):
            raise AssertionError(f"unexpected fail_task: {task_id} {error}")

    class ReportAgentStub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate_report(self, progress_callback=None, report_id=None):
            return types.SimpleNamespace(
                report_id=report_id,
                simulation_id="sim_2",
                graph_id="graph_2",
                simulation_requirement="Track rumor spread.",
                status=module.ReportStatus.COMPLETED,
                error=None,
            )

    class ThreadStub:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self):
            self.target()

    module.TaskManager = TaskManagerStub
    module.TaskStatus = types.SimpleNamespace(PROCESSING="processing")
    module.ReportAgent = ReportAgentStub
    module.threading.Thread = ThreadStub
    module.ReportManager = types.SimpleNamespace(
        get_report_by_simulation=lambda simulation_id: None,
        save_report=lambda report: saved_reports.append(report),
    )
    module.SimulationManager = lambda: types.SimpleNamespace(
        get_simulation=lambda simulation_id: types.SimpleNamespace(
            simulation_id=simulation_id,
            project_id="proj_2",
            graph_id="graph_2",
            graph_backend="zep",
        )
    )
    module.ProjectManager = types.SimpleNamespace(
        get_project=lambda project_id: types.SimpleNamespace(
            project_id=project_id,
            graph_id="graph_2",
            graph_backend="zep",
            simulation_requirement="Track rumor spread.",
        )
    )

    def explode(report, graph_backend=None):
        raise RuntimeError("bridge boom")

    module.try_post_report_writeback = explode

    app = FakeFlaskApp(request_obj)
    app.register_blueprint(report_bp, url_prefix="/api/report")
    client = app.test_client()

    response = client.post("/api/report/generate", json={"simulation_id": "sim_2", "graph_backend": "zep"})

    assert response.status_code == 200
    assert response.get_json()["success"] is True
    assert completed["task_id"] == "task_2"
    assert completed["result"]["status"] == "completed"
    assert completed["result"]["writeback"]["ok"] is False
    assert completed["result"]["writeback"]["error"] == "bridge boom"
    assert len(saved_reports) == 2
    assert saved_reports[-1].writeback_result == completed["result"]["writeback"]


def test_panorama_tool_http_returns_serialized_result():
    module, report_bp, request_obj = load_report_api_blueprint_module()

    class ToolStub:
        def __init__(self):
            self.calls = []

        def panorama_search(self, **kwargs):
            self.calls.append(kwargs)
            return types.SimpleNamespace(
                to_dict=lambda: {
                    "query": kwargs["query"],
                    "active_facts": ["Royal Court monitors the rumor"],
                    "diagnostics": {"graph_id": kwargs["graph_id"], "fallback_used": False},
                }
            )

    tools = ToolStub()
    module.get_report_tools_service = lambda graph_backend=None: tools

    app = FakeFlaskApp(request_obj)
    app.register_blueprint(report_bp, url_prefix="/api/report")
    client = app.test_client()

    response = client.post(
        "/api/report/tools/panorama",
        json={
            "graph_id": "g1",
            "query": "royal court rumor",
            "limit": 7,
            "graph_backend": "cognee",
            "simulation_id": "sim_1",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["data"]["query"] == "royal court rumor"
    assert payload["data"]["diagnostics"]["graph_id"] == "g1"
    assert tools.calls == [{
        "graph_id": "g1",
        "query": "royal court rumor",
        "include_expired": True,
        "limit": 7,
        "simulation_id": "sim_1",
    }]


def test_insight_forge_tool_http_defaults_simulation_requirement_to_query():
    module, report_bp, request_obj = load_report_api_blueprint_module()

    class ToolStub:
        def __init__(self):
            self.calls = []

        def insight_forge(self, **kwargs):
            self.calls.append(kwargs)
            return types.SimpleNamespace(
                to_dict=lambda: {
                    "query": kwargs["query"],
                    "simulation_requirement": kwargs["simulation_requirement"],
                    "semantic_facts": ["Forgery claim is spreading through the court"],
                    "diagnostics": {"quick_search": {"sidecar_search_ok": True}},
                }
            )

    tools = ToolStub()
    module.get_report_tools_service = lambda graph_backend=None: tools

    app = FakeFlaskApp(request_obj)
    app.register_blueprint(report_bp, url_prefix="/api/report")
    client = app.test_client()

    response = client.post(
        "/api/report/tools/insight-forge",
        json={
            "graph_id": "g1",
            "query": "royal succession rumor",
            "max_sub_queries": 4,
            "graph_backend": "cognee",
            "simulation_id": "sim_2",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["data"]["simulation_requirement"] == "royal succession rumor"
    assert payload["data"]["diagnostics"]["quick_search"]["sidecar_search_ok"] is True
    assert tools.calls == [{
        "graph_id": "g1",
        "query": "royal succession rumor",
        "simulation_requirement": "royal succession rumor",
        "report_context": "",
        "max_sub_queries": 4,
        "simulation_id": "sim_2",
    }]