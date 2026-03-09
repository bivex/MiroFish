import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_project_module():
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv_stub)

    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = app_pkg

    models_pkg = types.ModuleType("app.models")
    models_pkg.__path__ = [str(ROOT / "app" / "models")]
    sys.modules["app.models"] = models_pkg

    config_spec = importlib.util.spec_from_file_location("app.config", ROOT / "app" / "config.py")
    config_module = importlib.util.module_from_spec(config_spec)
    assert config_spec.loader is not None
    sys.modules["app.config"] = config_module
    config_spec.loader.exec_module(config_module)

    project_spec = importlib.util.spec_from_file_location(
        "app.models.project",
        ROOT / "app" / "models" / "project.py",
    )
    project_module = importlib.util.module_from_spec(project_spec)
    assert project_spec.loader is not None
    sys.modules["app.models.project"] = project_module
    project_spec.loader.exec_module(project_module)

    return project_module


def test_project_serializes_graph_backend():
    project_module = load_project_module()

    project = project_module.Project(
        project_id="proj_test",
        name="Test",
        status=project_module.ProjectStatus.CREATED,
        created_at="2026-03-09T00:00:00",
        updated_at="2026-03-09T00:00:00",
        graph_backend="cognee",
    )

    assert project.to_dict()["graph_backend"] == "cognee"


def test_project_from_dict_defaults_legacy_projects_to_zep():
    project_module = load_project_module()

    project = project_module.Project.from_dict(
        {
            "project_id": "proj_legacy",
            "name": "Legacy",
            "status": "created",
            "created_at": "2026-03-09T00:00:00",
            "updated_at": "2026-03-09T00:00:00",
        }
    )

    assert project.graph_backend == "zep"