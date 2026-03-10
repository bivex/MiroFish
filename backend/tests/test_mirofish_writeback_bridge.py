import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_bridge_module():
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = app_pkg

    services_pkg = types.ModuleType("app.services")
    services_pkg.__path__ = [str(ROOT / "app" / "services")]
    sys.modules["app.services"] = services_pkg

    config_module = types.ModuleType("app.config")
    config_module.Config = type(
        "Config",
        (),
        {
            "MIROFISH_WRITEBACK_ENABLED": True,
            "MIROFISH_WRITEBACK_BASE_URL": "http://127.0.0.1:8080",
            "MIROFISH_WRITEBACK_TIMEOUT_SECONDS": 3,
            "MIROFISH_WRITEBACK_AUTO_PROMOTE_ENABLED": False,
            "MIROFISH_WRITEBACK_AUTO_PROMOTE_TENANT_ID": None,
            "MIROFISH_WRITEBACK_AUTO_PROMOTE_WORLD_ID": None,
        },
    )
    sys.modules["app.config"] = config_module
    sys.modules["app.models.project"] = types.SimpleNamespace(ProjectManager=object)
    sys.modules["app.services.simulation_manager"] = types.SimpleNamespace(SimulationManager=object)
    sys.modules["app.services.simulation_runner"] = types.SimpleNamespace(SimulationRunner=object)
    sys.modules["app.utils.logger"] = types.SimpleNamespace(
        get_logger=lambda name: types.SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
    )

    module_path = ROOT / "app" / "services" / "mirofish_writeback_bridge.py"
    spec = importlib.util.spec_from_file_location("app.services.mirofish_writeback_bridge", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["app.services.mirofish_writeback_bridge"] = module
    spec.loader.exec_module(module)
    return module


def test_build_result_bundle_prefers_projection_metadata_and_subject_sections():
    module = load_bridge_module()

    class Action:
        def __init__(self):
            self.round_num = 3
            self.timestamp = "2026-03-10T12:05:00"
            self.platform = "twitter"
            self.agent_id = 7
            self.agent_name = "Captain Mira"
            self.action_type = "CREATE_POST"
            self.action_args = {"post": "Council denies the dock strike rumor."}
            self.result = "Council denial published"
            self.success = True

        def to_dict(self):
            return {
                "round_num": self.round_num,
                "timestamp": self.timestamp,
                "platform": self.platform,
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "action_type": self.action_type,
                "action_args": self.action_args,
                "result": self.result,
                "success": self.success,
            }

    module.ProjectManager = types.SimpleNamespace(
        get_project=lambda project_id: types.SimpleNamespace(
            project_id=project_id,
            project_name=project_id,
            name="Harbor Crisis",
            graph_id="graph_1",
            graph_backend="cognee",
            analysis_summary="Harbor rumor scenario",
            recommended_prepare_entity_types=["Actor", "Organization"],
            updated_at="2026-03-10T12:10:00Z",
        ),
        get_extracted_text=lambda project_id: "\n".join([
            "# Projection Bundle Import",
            "schema_version: 1.0",
            "world_id: world-1",
            "world_version: 7",
            "scenario_id: succession-crisis",
            "",
            "## Actors",
            "Actor: Captain Mira",
            "- id: actor:mira",
            "- canonical_id: char-1",
            "- canonical_type: Character",
            "- speaker_mode: individual",
            "",
            "## Organizations",
            "Organization: Harbor Council",
            "- id: org:council",
            "- canonical_id: faction-1",
            "- canonical_type: Faction",
            "- speaker_mode: official_account",
            "",
        ]),
    )
    module.SimulationManager = lambda: types.SimpleNamespace(
        get_simulation=lambda simulation_id: types.SimpleNamespace(
            simulation_id=simulation_id,
            project_id="proj_1",
            graph_id="graph_1",
            graph_backend="cognee",
            to_dict=lambda: {"simulation_id": simulation_id, "project_id": "proj_1"},
        ),
        get_profiles=lambda simulation_id, platform="twitter": [],
    )
    module.SimulationRunner = types.SimpleNamespace(get_all_actions=lambda simulation_id: [Action()])

    report = types.SimpleNamespace(
        report_id="report_1",
        simulation_id="sim_1",
        graph_id="graph_1",
        simulation_requirement="Track rumor spread.",
        status=types.SimpleNamespace(value="completed"),
        outline=types.SimpleNamespace(title="Harbor Report", summary="Rumor reached the council"),
        completed_at="2026-03-10T12:15:00",
        to_dict=lambda: {"report_id": "report_1", "status": "completed"},
    )

    bundle = module.build_result_bundle(report, graph_backend="cognee")

    assert bundle["world_id"] == "world-1"
    assert bundle["scenario_id"] == "succession-crisis"
    assert bundle["run_id"] == "report_1"
    assert bundle["projection_version"] == "1.0"
    assert bundle["actors"][0]["id"] == "actor:mira"
    assert bundle["organizations"][0]["id"] == "org:council"
    assert bundle["runtime_evidence"][0]["actor_refs"] == ["actor:mira"]
    assert len(bundle["emergent_events"]) == 1
    assert len(bundle["rumor_candidates"]) == 1
    assert len(bundle["candidate_deltas"]) == 2
    assert bundle["candidate_deltas"][0]["candidate_type"] == "scenario_event"
    assert bundle["candidate_deltas"][0]["confidence"] >= 0.9
    assert len(bundle["candidate_deltas"][0]["evidence_ids"]) >= 2
    assert bundle["candidate_deltas"][0]["proposed_change"]["timestamp"].endswith("Z")
    assert bundle["candidate_deltas"][1]["candidate_type"] == "rumor_candidate"
    assert bundle["candidate_deltas"][1]["confidence"] >= 0.9
    assert len(bundle["candidate_deltas"][1]["evidence_ids"]) >= 2
    assert bundle["runtime_evidence"][0]["timestamp"].endswith("Z")
    assert bundle["prediction_summary"]["rumors"][0]["source_name"] == "Captain Mira"
    assert any(
        any(ref.get("collection") == "prediction_summary" for ref in item["source_refs"])
        for item in bundle["runtime_evidence"]
    )
    assert any(
        any(ref.get("collection") == "emergent_event" for ref in item["source_refs"])
        for item in bundle["runtime_evidence"]
    )


def test_try_post_report_writeback_skips_when_disabled():
    module = load_bridge_module()
    module.Config.MIROFISH_WRITEBACK_ENABLED = False

    result = module.try_post_report_writeback(types.SimpleNamespace(report_id="report_skip"))

    assert result == {"enabled": False, "skipped": True, "reason": "bridge_disabled"}


def test_try_post_report_writeback_posts_built_bundle(monkeypatch):
    module = load_bridge_module()
    captured = {}
    module.build_result_bundle = lambda report, graph_backend=None: {
        "world_id": "world-1",
        "scenario_id": "scenario-1",
        "run_id": "report_2",
        "runtime_evidence": [{"evidence_id": "ev-1"}],
        "candidate_deltas": [],
        "actors": [{"id": "actor:mira"}],
        "organizations": [],
    }
    def fake_post_bundle(bundle):
        captured["bundle"] = bundle
        return {
            "url": "http://127.0.0.1:8080/api/mirofish/writeback/ingest",
            "http_status": 200,
            "body": {"success": True},
        }

    module._post_bundle = fake_post_bundle

    result = module.try_post_report_writeback(types.SimpleNamespace(report_id="report_2"), graph_backend="cognee")

    assert captured["bundle"]["scenario_id"] == "scenario-1"
    assert result["enabled"] is True
    assert result["ok"] is True
    assert result["run_id"] == "report_2"
    assert result["candidate_deltas_count"] == 0


def test_try_post_report_writeback_runs_auto_promote_when_enabled():
    module = load_bridge_module()
    module.Config.MIROFISH_WRITEBACK_AUTO_PROMOTE_ENABLED = True
    module.Config.MIROFISH_WRITEBACK_AUTO_PROMOTE_TENANT_ID = 1
    module.Config.MIROFISH_WRITEBACK_AUTO_PROMOTE_WORLD_ID = 101
    module.build_result_bundle = lambda report, graph_backend=None: {
        "world_id": "world-1",
        "scenario_id": "scenario-1",
        "run_id": "report_3",
        "runtime_evidence": [{"evidence_id": "ev-1"}, {"evidence_id": "ev-2"}],
        "candidate_deltas": [{"candidate_id": "cand-1"}],
        "actors": [{"id": "actor:mira", "canonical_id": "char-1", "canonical_type": "Character"}],
        "organizations": [],
    }
    module._post_bundle = lambda bundle: {
        "url": "http://127.0.0.1:8080/api/mirofish/writeback/ingest",
        "http_status": 200,
        "body": {"success": True},
    }
    module._post_auto_promote = lambda bundle: {
        "enabled": True,
        "ok": True,
        "policy_count": 1,
        "attempted_candidate_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "responses": [],
    }

    result = module.try_post_report_writeback(types.SimpleNamespace(report_id="report_3"), graph_backend="cognee")

    assert result["enabled"] is True
    assert result["ok"] is True
    assert result["candidate_deltas_count"] == 1
    assert result["auto_promote"]["ok"] is True