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


def test_build_result_bundle_extracts_new_entity_candidates_from_report_markdown():
    module = load_bridge_module()

    module.ProjectManager = types.SimpleNamespace(
        get_project=lambda project_id: types.SimpleNamespace(
            project_id=project_id,
            project_name=project_id,
            name="Succession Crisis",
            graph_id="graph_2",
            graph_backend="cognee",
            analysis_summary="Palace rumor scenario",
            recommended_prepare_entity_types=["Actor", "Organization"],
            updated_at="2026-03-10T12:20:00Z",
        ),
        get_extracted_text=lambda project_id: "\n".join([
            "# Projection Bundle Import",
            "schema_version: 1.0",
            "world_id: world-2",
            "world_version: 9",
            "scenario_id: palace-crisis",
            "",
            "## Actors",
            "Actor: Princess Ilyra",
            "- id: actor:ilyra",
            "- canonical_id: char-2",
            "- canonical_type: Character",
            "",
            "## Organizations",
            "Organization: Royal Court",
            "- id: org:royal-court",
            "- canonical_id: faction-2",
            "- canonical_type: Faction",
            "",
        ]),
    )
    module.SimulationManager = lambda: types.SimpleNamespace(
        get_simulation=lambda simulation_id: types.SimpleNamespace(
            simulation_id=simulation_id,
            project_id="proj_2",
            graph_id="graph_2",
            graph_backend="cognee",
            to_dict=lambda: {"simulation_id": simulation_id, "project_id": "proj_2"},
        ),
        get_profiles=lambda simulation_id, platform="twitter": [],
    )
    module.SimulationRunner = types.SimpleNamespace(get_all_actions=lambda simulation_id: [])

    report = types.SimpleNamespace(
        report_id="report_2",
        simulation_id="sim_2",
        graph_id="graph_2",
        simulation_requirement="Track decree rumors.",
        status=types.SimpleNamespace(value="completed"),
        outline=types.SimpleNamespace(title="Succession Report", summary="Rumor pressure reaches the palace."),
        markdown_content="\n".join([
            "# Succession Report",
            "> The response shifted toward Sunspire Palace as tension spread.",
            "## Geographic escalation",
            "The rumor spread from Sunspire Palace into the Market of Bells as merchants panicked.",
            "Dockworkers Union organized messengers near Sunspire Palace to steady trade routes.",
            "Later, Dockworkers Union relayed updates from the Market of Bells to nearby stalls.",
        ]),
        completed_at="2026-03-10T12:25:00",
        to_dict=lambda: {"report_id": "report_2", "status": "completed"},
    )

    bundle = module.build_result_bundle(report, graph_backend="cognee")

    new_candidates = [item for item in bundle["candidate_deltas"] if item["candidate_type"] == "new_entity_candidate"]
    assert len(new_candidates) >= 3
    assert bundle["new_entity_candidates"]
    assert any(item["source_type"] == "report_markdown_entity" for item in bundle["runtime_evidence"])

    location_names = {item["name"] for item in new_candidates if item["target_canonical_type"] == "Location"}
    faction_names = {item["name"] for item in new_candidates if item["target_canonical_type"] == "Faction"}
    assert "Sunspire Palace" in location_names
    assert "Market of Bells" in location_names
    assert "Dockworkers Union" in faction_names
    assert "Royal Court" not in faction_names

    sunspire = next(item for item in new_candidates if item["name"] == "Sunspire Palace")
    dockworkers = next(item for item in new_candidates if item["name"] == "Dockworkers Union")
    assert sunspire["proposed_change"]["location_type"] == "castle"
    assert dockworkers["proposed_change"]["faction_type"] == "merchant"
    assert dockworkers["proposed_change"]["alignment"] == "neutral"


def test_build_result_bundle_extracts_character_candidates_from_report_markdown_without_known_actor_duplicates():
    module = load_bridge_module()

    module.ProjectManager = types.SimpleNamespace(
        get_project=lambda project_id: types.SimpleNamespace(
            project_id=project_id,
            project_name=project_id,
            name="Palace Escalation",
            graph_id="graph_3",
            graph_backend="cognee",
            analysis_summary="Character-heavy report",
            recommended_prepare_entity_types=["Actor", "Organization"],
            updated_at="2026-03-10T13:00:00Z",
        ),
        get_extracted_text=lambda project_id: "\n".join([
            "# Projection Bundle Import",
            "schema_version: 1.0",
            "world_id: world-3",
            "world_version: 10",
            "scenario_id: palace-escalation",
            "",
            "## Actors",
            "Actor: Princess Ilyra",
            "- id: actor:ilyra",
            "- canonical_id: char-3",
            "- canonical_type: Character",
            "",
            "Actor: Captain Serik",
            "- id: actor:serik",
            "- canonical_id: char-4",
            "- canonical_type: Character",
            "",
            "Actor: Nessa",
            "- id: actor:nessa",
            "- canonical_id: char-5",
            "- canonical_type: Character",
            "",
            "## Organizations",
            "Organization: Royal Court",
            "- id: org:royal-court",
            "- canonical_id: faction-3",
            "- canonical_type: Faction",
            "",
        ]),
    )
    module.SimulationManager = lambda: types.SimpleNamespace(
        get_simulation=lambda simulation_id: types.SimpleNamespace(
            simulation_id=simulation_id,
            project_id="proj_3",
            graph_id="graph_3",
            graph_backend="cognee",
            to_dict=lambda: {"simulation_id": simulation_id, "project_id": "proj_3"},
        ),
        get_profiles=lambda simulation_id, platform="twitter": [],
    )
    module.SimulationRunner = types.SimpleNamespace(get_all_actions=lambda simulation_id: [])

    report = types.SimpleNamespace(
        report_id="report_3",
        simulation_id="sim_3",
        graph_id="graph_3",
        simulation_requirement="Track palace responses.",
        status=types.SimpleNamespace(value="completed"),
        outline=types.SimpleNamespace(title="Palace Escalation", summary="New palace figures emerged during the response."),
        markdown_content="\n".join([
            "# Palace Escalation",
            "Princess Ilyra denied the forged decree while archivist Maelin verified seals.",
            "Town Crier Nessa repeated the claim before Archivist Maelin corrected the record.",
            "Later, Chancellor Varos pressured Captain Serik to contain unrest around the palace.",
        ]),
        completed_at="2026-03-10T13:05:00",
        to_dict=lambda: {"report_id": "report_3", "status": "completed"},
    )

    bundle = module.build_result_bundle(report, graph_backend="cognee")

    character_candidates = [
        item for item in bundle["candidate_deltas"]
        if item["candidate_type"] == "new_entity_candidate" and item["target_canonical_type"] == "Character"
    ]
    character_names = {item["name"] for item in character_candidates}

    assert "Archivist Maelin" in character_names
    assert "Chancellor Varos" in character_names
    assert "Princess Ilyra" not in character_names
    assert "Captain Serik" not in character_names
    assert "Town Crier Nessa" not in character_names

    maelin = next(item for item in character_candidates if item["name"] == "Archivist Maelin")
    varos = next(item for item in character_candidates if item["name"] == "Chancellor Varos")
    assert maelin["proposed_change"]["status"] == "active"
    assert maelin["proposed_change"]["character_title"] == "Archivist"
    assert maelin["proposed_change"]["mention_count"] == 2
    assert varos["proposed_change"]["status"] == "active"
    assert varos["proposed_change"]["character_title"] == "Chancellor"


def test_build_result_bundle_extracts_new_entities_from_structured_interview_answers():
    module = load_bridge_module()

    module.ProjectManager = types.SimpleNamespace(
        get_project=lambda project_id: types.SimpleNamespace(
            project_id=project_id,
            project_name=project_id,
            name="Interview World",
            graph_id="graph_4",
            graph_backend="zep",
            analysis_summary="Interview-heavy scenario",
            recommended_prepare_entity_types=["Actor", "Organization"],
            updated_at="2026-03-10T13:20:00Z",
        ),
        get_extracted_text=lambda project_id: "\n".join([
            "# Projection Bundle Import",
            "schema_version: 1.0",
            "world_id: world-4",
            "world_version: 11",
            "scenario_id: interview-world",
            "",
            "## Actors",
            "Actor: Captain Serik",
            "- id: actor:serik",
            "- canonical_id: char-6",
            "- canonical_type: Character",
            "",
            "## Organizations",
            "Organization: Royal Court",
            "- id: org:royal-court",
            "- canonical_id: faction-4",
            "- canonical_type: Faction",
            "",
        ]),
    )
    module.SimulationManager = lambda: types.SimpleNamespace(
        get_simulation=lambda simulation_id: types.SimpleNamespace(
            simulation_id=simulation_id,
            project_id="proj_4",
            graph_id="graph_4",
            graph_backend="zep",
            to_dict=lambda: {"simulation_id": simulation_id, "project_id": "proj_4"},
        ),
        get_profiles=lambda simulation_id, platform="twitter": [],
    )
    module.SimulationRunner = types.SimpleNamespace(get_all_actions=lambda simulation_id: [])

    report = types.SimpleNamespace(
        report_id="report_4",
        simulation_id="sim_4",
        graph_id="graph_4",
        simulation_requirement="Collect interviews.",
        status=types.SimpleNamespace(value="completed"),
        outline=types.SimpleNamespace(title="Interview Report", summary="Interview answers reveal new entities."),
        markdown_content="# Interview Report\n\nGeneral report text only.",
        completed_at="2026-03-10T13:25:00",
        to_dict=lambda: {
            "report_id": "report_4",
            "status": "completed",
            "interview_result": {
                "interviews": [
                    {
                        "agent_name": "Captain Serik",
                        "response": "I met Archivist Sel beside Harbor of Glass after the Lantern Guild moved supplies through the plaza.",
                        "key_quotes": [
                            "Archivist Sel warned that Harbor of Glass would close before dawn.",
                        ],
                    }
                ]
            },
        },
    )

    bundle = module.build_result_bundle(report, graph_backend="zep")

    new_candidates = [item for item in bundle["candidate_deltas"] if item["candidate_type"] == "new_entity_candidate"]
    candidate_names = {item["name"] for item in new_candidates}

    assert "Archivist Sel" in candidate_names
    assert "Harbor of Glass" in candidate_names
    assert "Lantern Guild" in candidate_names
    assert "Captain Serik" not in candidate_names

    raw_archivist = next(item for item in bundle["new_entity_candidates"] if item["name"] == "Archivist Sel")
    archivist = next(item for item in new_candidates if item["name"] == "Archivist Sel")
    harbor = next(item for item in new_candidates if item["name"] == "Harbor of Glass")
    guild = next(item for item in new_candidates if item["name"] == "Lantern Guild")
    assert "interview_response_entity" in raw_archivist["source_types"]
    assert archivist["proposed_change"]["character_title"] == "Archivist"
    assert harbor["proposed_change"]["location_type"] == "landmark"
    assert guild["proposed_change"]["faction_type"] == "merchant"
    assert any(item["source_type"] == "interview_response_entity" for item in bundle["runtime_evidence"])
    assert any(item["source_type"] == "interview_quote_entity" for item in bundle["runtime_evidence"])


def test_build_result_bundle_extracts_rumor_and_relationship_candidates_from_structured_interviews():
    module = load_bridge_module()

    module.ProjectManager = types.SimpleNamespace(
        get_project=lambda project_id: types.SimpleNamespace(
            project_id=project_id,
            project_name=project_id,
            name="Interview Signals World",
            graph_id="graph_6",
            graph_backend="zep",
            analysis_summary="Structured interview rumor/relationship scenario",
            recommended_prepare_entity_types=["Actor", "Organization"],
            updated_at="2026-03-10T14:00:00Z",
        ),
        get_extracted_text=lambda project_id: "\n".join([
            "# Projection Bundle Import",
            "schema_version: 1.0",
            "world_id: world-6",
            "world_version: 13",
            "scenario_id: interview-signal-world",
            "",
            "## Actors",
            "Actor: Captain Serik",
            "- id: actor:serik",
            "- canonical_id: char-8",
            "- canonical_type: Character",
            "",
            "Actor: Nessa",
            "- id: actor:nessa",
            "- canonical_id: char-9",
            "- canonical_type: Character",
            "",
            "## Organizations",
            "Organization: Royal Court",
            "- id: org:royal-court",
            "- canonical_id: faction-6",
            "- canonical_type: Faction",
            "",
        ]),
    )
    module.SimulationManager = lambda: types.SimpleNamespace(
        get_simulation=lambda simulation_id: types.SimpleNamespace(
            simulation_id=simulation_id,
            project_id="proj_6",
            graph_id="graph_6",
            graph_backend="zep",
            to_dict=lambda: {"simulation_id": simulation_id, "project_id": "proj_6"},
        ),
        get_profiles=lambda simulation_id, platform="twitter": [],
    )
    module.SimulationRunner = types.SimpleNamespace(get_all_actions=lambda simulation_id: [])

    report = types.SimpleNamespace(
        report_id="report_6",
        simulation_id="sim_6",
        graph_id="graph_6",
        simulation_requirement="Collect structured interview signals.",
        status=types.SimpleNamespace(value="completed"),
        outline=types.SimpleNamespace(title="Structured Interview Signals", summary="Interviews surface trust shifts and rumors."),
        markdown_content="# Structured Interview Signals\n",
        completed_at="2026-03-10T14:05:00",
        to_dict=lambda: {
            "report_id": "report_6",
            "status": "completed",
            "interview_result": {
                "interviews": [
                    {
                        "agent_name": "Captain Serik",
                        "response": "I distrust Nessa after her latest rumor about a forged succession decree reached the barracks.",
                        "key_quotes": [
                            "Nessa keeps spreading the forged decree rumor to every patrol post.",
                        ],
                    }
                ]
            },
        },
    )

    bundle = module.build_result_bundle(report, graph_backend="zep")

    rumor_candidates = [item for item in bundle["candidate_deltas"] if item["candidate_type"] == "rumor_candidate"]
    relationship_candidates = [item for item in bundle["candidate_deltas"] if item["candidate_type"] == "relationship_change"]

    assert rumor_candidates
    assert relationship_candidates
    assert any("forged succession decree" in item["summary"].lower() for item in rumor_candidates)

    relationship = relationship_candidates[0]
    assert relationship["proposed_change"]["actor_refs"] == ["actor:serik", "actor:nessa"]
    assert relationship["proposed_change"]["relationship_level"] < 0
    assert any(
        item["evidence_type"] == "rumor_signal" and item["source_type"] == "interview_response_entity"
        for item in bundle["runtime_evidence"]
    )
    assert any(
        item["evidence_type"] == "relationship_change" and item["source_type"] == "interview_response_entity"
        for item in bundle["runtime_evidence"]
    )


def test_build_result_bundle_extracts_new_entities_from_markdown_interview_blocks():
    module = load_bridge_module()

    module.ProjectManager = types.SimpleNamespace(
        get_project=lambda project_id: types.SimpleNamespace(
            project_id=project_id,
            project_name=project_id,
            name="Interview Markdown World",
            graph_id="graph_5",
            graph_backend="zep",
            analysis_summary="Markdown interview scenario",
            recommended_prepare_entity_types=["Actor", "Organization"],
            updated_at="2026-03-10T13:40:00Z",
        ),
        get_extracted_text=lambda project_id: "\n".join([
            "# Projection Bundle Import",
            "schema_version: 1.0",
            "world_id: world-5",
            "world_version: 12",
            "scenario_id: interview-markdown-world",
            "",
            "## Actors",
            "Actor: Nessa",
            "- id: actor:nessa",
            "- canonical_id: char-7",
            "- canonical_type: Character",
            "",
            "## Organizations",
            "Organization: Harbor Council",
            "- id: org:harbor-council",
            "- canonical_id: faction-5",
            "- canonical_type: Faction",
            "",
        ]),
    )
    module.SimulationManager = lambda: types.SimpleNamespace(
        get_simulation=lambda simulation_id: types.SimpleNamespace(
            simulation_id=simulation_id,
            project_id="proj_5",
            graph_id="graph_5",
            graph_backend="zep",
            to_dict=lambda: {"simulation_id": simulation_id, "project_id": "proj_5"},
        ),
        get_profiles=lambda simulation_id, platform="twitter": [],
    )
    module.SimulationRunner = types.SimpleNamespace(get_all_actions=lambda simulation_id: [])

    report = types.SimpleNamespace(
        report_id="report_5",
        simulation_id="sim_5",
        graph_id="graph_5",
        simulation_requirement="Collect markdown interviews.",
        status=types.SimpleNamespace(value="completed"),
        outline=types.SimpleNamespace(title="Interview Markdown Report", summary="Markdown interviews expose new names."),
        markdown_content="\n".join([
            "# Interview Markdown Report",
            "## Interview excerpts",
            "**Captain Rowan** (harbor guard)",
            "**Q:** What changed overnight?",
            "**A:** Dockmaster Elra redirected supplies to Moonlit Port before sunrise while the Iron Union secured the pier.",
            "**Key Quotes:**",
            "> Dockmaster Elra said Moonlit Port would stay open as the Iron Union secured the pier.",
        ]),
        completed_at="2026-03-10T13:45:00",
        to_dict=lambda: {"report_id": "report_5", "status": "completed"},
    )

    bundle = module.build_result_bundle(report, graph_backend="zep")

    new_candidates = [item for item in bundle["candidate_deltas"] if item["candidate_type"] == "new_entity_candidate"]
    candidate_names = {item["name"] for item in new_candidates}
    assert "Dockmaster Elra" in candidate_names
    assert "Moonlit Port" in candidate_names
    assert "Iron Union" in candidate_names
    assert any(item["source_type"] == "interview_response_entity" for item in bundle["runtime_evidence"])
    assert any(item["source_type"] == "interview_quote_entity" for item in bundle["runtime_evidence"])


def test_build_result_bundle_extracts_rumor_and_relationship_candidates_from_markdown_interviews():
    module = load_bridge_module()

    module.ProjectManager = types.SimpleNamespace(
        get_project=lambda project_id: types.SimpleNamespace(
            project_id=project_id,
            project_name=project_id,
            name="Interview Markdown Signals",
            graph_id="graph_7",
            graph_backend="zep",
            analysis_summary="Markdown interview rumor/relationship scenario",
            recommended_prepare_entity_types=["Actor", "Organization"],
            updated_at="2026-03-10T14:15:00Z",
        ),
        get_extracted_text=lambda project_id: "\n".join([
            "# Projection Bundle Import",
            "schema_version: 1.0",
            "world_id: world-7",
            "world_version: 14",
            "scenario_id: interview-markdown-signals",
            "",
            "## Actors",
            "Actor: Captain Rowan",
            "- id: actor:rowan",
            "- canonical_id: char-10",
            "- canonical_type: Character",
            "",
            "Actor: Nessa",
            "- id: actor:nessa",
            "- canonical_id: char-11",
            "- canonical_type: Character",
            "",
            "## Organizations",
            "Organization: Harbor Council",
            "- id: org:harbor-council",
            "- canonical_id: faction-7",
            "- canonical_type: Faction",
            "",
        ]),
    )
    module.SimulationManager = lambda: types.SimpleNamespace(
        get_simulation=lambda simulation_id: types.SimpleNamespace(
            simulation_id=simulation_id,
            project_id="proj_7",
            graph_id="graph_7",
            graph_backend="zep",
            to_dict=lambda: {"simulation_id": simulation_id, "project_id": "proj_7"},
        ),
        get_profiles=lambda simulation_id, platform="twitter": [],
    )
    module.SimulationRunner = types.SimpleNamespace(get_all_actions=lambda simulation_id: [])

    report = types.SimpleNamespace(
        report_id="report_7",
        simulation_id="sim_7",
        graph_id="graph_7",
        simulation_requirement="Collect markdown interview signals.",
        status=types.SimpleNamespace(value="completed"),
        outline=types.SimpleNamespace(title="Markdown Interview Signals", summary="Markdown interviews expose distrust and rumor spread."),
        markdown_content="\n".join([
            "# Markdown Interview Signals",
            "## Interview excerpts",
            "**Captain Rowan** (harbor guard)",
            "**Q:** What changed overnight?",
            "**A:** I distrust Nessa because her rumor about a forged decree keeps spreading through the harbor.",
            "**Key Quotes:**",
            "> Nessa keeps pushing the forged decree rumor faster than the Harbor Council can answer.",
        ]),
        completed_at="2026-03-10T14:20:00",
        to_dict=lambda: {"report_id": "report_7", "status": "completed"},
    )

    bundle = module.build_result_bundle(report, graph_backend="zep")

    rumor_candidates = [item for item in bundle["candidate_deltas"] if item["candidate_type"] == "rumor_candidate"]
    relationship_candidates = [item for item in bundle["candidate_deltas"] if item["candidate_type"] == "relationship_change"]

    assert rumor_candidates
    assert relationship_candidates
    assert any("forged decree" in item["summary"].lower() for item in rumor_candidates)

    relationship = relationship_candidates[0]
    assert relationship["proposed_change"]["actor_refs"] == ["actor:rowan", "actor:nessa"]
    assert relationship["proposed_change"]["relationship_type"] == "enemy"
    assert any(
        item["evidence_type"] == "rumor_signal" and item["source_type"] == "interview_quote_entity"
        for item in bundle["runtime_evidence"]
    )
    assert any(
        item["evidence_type"] == "relationship_change" and item["source_type"] == "interview_response_entity"
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