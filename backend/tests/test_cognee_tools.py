import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_cognee_tools_module():
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = app_pkg

    services_pkg = types.ModuleType("app.services")
    services_pkg.__path__ = [str(ROOT / "app" / "services")]
    sys.modules["app.services"] = services_pkg

    entity_reader_module = types.ModuleType("app.services.cognee_entity_reader")
    entity_reader_module.CogneeEntityReader = object
    sys.modules["app.services.cognee_entity_reader"] = entity_reader_module

    sidecar_client_module = types.ModuleType("app.services.cognee_sidecar_client")
    sidecar_client_module.CogneeSidecarClient = object
    sys.modules["app.services.cognee_sidecar_client"] = sidecar_client_module

    simulation_runner_module = types.ModuleType("app.services.simulation_runner")

    class SimulationRunner:
        run_state = None
        actions = []

        @classmethod
        def get_run_state(cls, simulation_id):
            return cls.run_state

        @classmethod
        def get_all_actions(cls, simulation_id):
            return list(cls.actions)

    simulation_runner_module.SimulationRunner = SimulationRunner
    sys.modules["app.services.simulation_runner"] = simulation_runner_module

    module_path = ROOT / "app" / "services" / "cognee_tools.py"
    spec = importlib.util.spec_from_file_location("app.services.cognee_tools", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["app.services.cognee_tools"] = module
    spec.loader.exec_module(module)
    return module


class SidecarStub:
    def __init__(self, *, search_results=None, search_error="search unavailable"):
        self.get_graph_data_calls = 0
        self.search_results = list(search_results or [])
        self.search_error = search_error

    def get_graph_data(self, graph_id):
        self.get_graph_data_calls += 1
        return {
            "graph_id": graph_id,
            "nodes": [
                {"uuid": "n1", "name": "Harbor Guild", "labels": ["Entity", "Organization"], "summary": "Guild controls the harbor", "attributes": {"type": "Organization"}},
                {"uuid": "n2", "name": "Aria", "labels": ["Entity", "Actor"], "summary": "Captain aligned with the guild", "attributes": {"type": "Actor"}},
            ],
            "edges": [
                {"uuid": "e1", "name": "SUPPORTS", "fact": "Aria supports Harbor Guild", "source_node_uuid": "n2", "target_node_uuid": "n1"},
            ],
        }

    def search_graph(self, graph_id, query, limit):
        if self.search_error:
            raise RuntimeError(self.search_error)
        return {"graph_id": graph_id, "results": self.search_results[:limit]}


@dataclass
class RunStateStub:
    runner_status: object = "running"
    current_round: int = 2
    twitter_actions_count: int = 1
    reddit_actions_count: int = 1
    recent_actions: list = None


class ActionStub:
    def __init__(self, *, round_num, platform, agent_name, action_type, result, agent_id=1):
        self.round_num = round_num
        self.platform = platform
        self.agent_name = agent_name
        self.action_type = action_type
        self.result = result
        self.agent_id = agent_id
        self.timestamp = "2026-03-09T00:00:00"
        self.action_args = {}
        self.success = True


def test_cognee_tools_search_graph_falls_back_to_local_matches():
    module = load_cognee_tools_module()
    runner = sys.modules["app.services.simulation_runner"].SimulationRunner
    runner.run_state = None
    runner.actions = []
    sidecar = SidecarStub()
    service = module.CogneeToolsService(sidecar_client=sidecar, entity_reader=object())

    result = service.search_graph("graph_1", "Harbor Guild", limit=5)

    assert result.total_count >= 1
    assert result.facts[0] == "Aria supports Harbor Guild"
    assert "Guild controls the harbor" in result.facts
    assert result.nodes[0]["name"] == "Harbor Guild"
    assert result.diagnostics["fallback_used"] is True
    assert result.diagnostics["fallback_mode"] == "local_graph_edges"
    assert result.diagnostics["sidecar_search_ok"] is False
    assert "search unavailable" in result.diagnostics["sidecar_search_error"]
    assert sidecar.get_graph_data_calls == 1


def test_cognee_tools_get_simulation_context_matches_zep_shape():
    module = load_cognee_tools_module()
    runner = sys.modules["app.services.simulation_runner"].SimulationRunner
    runner.run_state = None
    runner.actions = []
    sidecar = SidecarStub()
    service = module.CogneeToolsService(sidecar_client=sidecar, entity_reader=object())

    result = service.get_simulation_context("graph_1", "Run a harbor crisis simulation", limit=10)

    assert result["graph_statistics"]["total_nodes"] == 2
    assert result["graph_statistics"]["entity_types"]["Organization"] == 1
    assert result["total_entities"] == 2
    assert result["entities"][0]["type"] in {"Organization", "Actor"}
    assert result["diagnostics"]["search"]["graph_data_source"] == "cache"
    assert result["graph_statistics"]["diagnostics"]["graph_data_source"] == "cache"
    assert sidecar.get_graph_data_calls == 1


def test_cognee_tools_runtime_evidence_is_exposed_in_context_and_search():
    module = load_cognee_tools_module()
    runner = sys.modules["app.services.simulation_runner"].SimulationRunner
    runner.run_state = RunStateStub(recent_actions=[])
    runner.actions = [
        ActionStub(
            round_num=2,
            platform="twitter",
            agent_name="Aria",
            action_type="post_message",
            result={"content": "Harbor Guild boycott is spreading across the docks."},
        )
    ]
    service = module.CogneeToolsService(sidecar_client=SidecarStub(), entity_reader=object())

    evidence = service.get_runtime_evidence("sim_1", limit=5)
    result = service.get_simulation_context("graph_1", "boycott in the harbor", limit=10, simulation_id="sim_1")
    search = service.quick_search("graph_1", "boycott harbor", limit=5, simulation_id="sim_1")

    assert evidence["has_runtime_evidence"] is True
    assert evidence["total_actions"] == 2
    assert evidence["action_facts"][0].startswith("[round 2] [twitter] Aria post_message")
    assert result["runtime_evidence"]["has_runtime_evidence"] is True
    assert result["related_facts"][0] == evidence["action_facts"][0]
    assert search.facts[0] == evidence["action_facts"][0]
    assert search.diagnostics["runtime_fact_count"] == 1
    assert search.diagnostics["fallback_mode"] == "runtime_actions"
    assert "runtime_actions" in search.diagnostics["evidence_sources"]


def test_cognee_tools_insight_forge_exposes_nested_diagnostics_and_uses_cache():
    module = load_cognee_tools_module()
    runner = sys.modules["app.services.simulation_runner"].SimulationRunner
    runner.run_state = None
    runner.actions = []
    sidecar = SidecarStub(search_results=[{"search_result": "Harbor Guild tightened tariff policy."}], search_error=None)
    service = module.CogneeToolsService(sidecar_client=sidecar, entity_reader=object())

    result = service.insight_forge("graph_1", "Harbor Guild", "Analyze the harbor crisis", simulation_id="sim_1")

    assert result.semantic_facts[0] == "Harbor Guild tightened tariff policy."
    assert result.diagnostics["quick_search"]["sidecar_search_ok"] is True
    assert "sidecar_search" in result.diagnostics["evidence_sources"]
    assert result.diagnostics["panorama_search"]["graph_data_source"] == "cache"
    assert sidecar.get_graph_data_calls == 1


def test_cognee_tools_filters_opaque_ids_from_fallback_fact_candidates():
    module = load_cognee_tools_module()
    service = module.CogneeToolsService(sidecar_client=SidecarStub(), entity_reader=object())

    assert service._edge_fact_candidates([{"name": "967c8fef-d2a0-5a49-84b1-cb331289ed7c"}]) == []
    assert service._node_fact_candidates([{"name": "Aria"}, {"summary": "Harbor unrest is escalating"}]) == [
        "Aria",
        "Harbor unrest is escalating",
    ]


def test_cognee_tools_panorama_search_humanizes_edge_facts_and_names():
    module = load_cognee_tools_module()

    class SidecarHumanizeStub(SidecarStub):
        def get_graph_data(self, graph_id):
            self.get_graph_data_calls += 1
            return {
                "graph_id": graph_id,
                "nodes": [
                    {"uuid": "n1", "name": "Royal Court", "labels": ["Entity", "Organization"], "summary": "Court manages succession disputes", "attributes": {"type": "Organization"}},
                    {"uuid": "n2", "name": "Harbor Guard", "labels": ["Entity", "Faction"], "summary": "Guard controls the docks", "attributes": {"type": "Faction"}},
                ],
                "edges": [
                    {"uuid": "e1", "name": "CONTAINS", "fact": "967c8fef-d2a0-5a49-84b1-cb331289ed7c", "source_node_uuid": "n1", "target_node_uuid": "n2"},
                ],
            }

    runner = sys.modules["app.services.simulation_runner"].SimulationRunner
    runner.run_state = None
    runner.actions = []
    service = module.CogneeToolsService(sidecar_client=SidecarHumanizeStub(), entity_reader=object())

    result = service.panorama_search("graph_1", "contains", limit=5)

    assert result.all_edges[0].name == "contains"
    assert result.all_edges[0].fact == "Royal Court contains Harbor Guard"
    assert result.all_edges[0].source_node_name == "Royal Court"
    assert result.all_edges[0].target_node_name == "Harbor Guard"
    assert result.active_facts == ["Royal Court contains Harbor Guard"]


def test_cognee_tools_insight_forge_compacts_noisy_semantic_facts():
    module = load_cognee_tools_module()

    class SidecarInsightStub(SidecarStub):
        def get_graph_data(self, graph_id):
            self.get_graph_data_calls += 1
            return {
                "graph_id": graph_id,
                "nodes": [
                    {"uuid": "n1", "name": "Royal Court", "labels": ["Entity", "Organization"], "summary": "Court manages succession disputes", "attributes": {"type": "Organization"}},
                    {"uuid": "n2", "name": "Court Guard", "labels": ["Entity", "Faction"], "summary": "Guard circulates news through the city", "attributes": {"type": "Faction"}},
                ],
                "edges": [
                    {"uuid": "e1", "name": "SPREADS_RUMOR_TO", "fact": "SPREADS_RUMOR_TO", "source_node_uuid": "n2", "target_node_uuid": "n1"},
                ],
            }

    runner = sys.modules["app.services.simulation_runner"].SimulationRunner
    runner.run_state = None
    runner.actions = []
    sidecar = SidecarInsightStub(
        search_results=[
            {
                "search_result": "canonical_id: 967c8fef-d2a0-5a49-84b1-cb331289ed7c\nThe forged decree reached the Royal Court.\nCourt guards spread the rumor through the harbor.",
            }
        ],
        search_error=None,
    )
    service = module.CogneeToolsService(sidecar_client=sidecar, entity_reader=object())

    result = service.insight_forge("graph_1", "spreads rumor", "Analyze rumor escalation", simulation_id="sim_1")

    assert result.semantic_facts == [
        "The forged decree reached the Royal Court. Court guards spread the rumor through the harbor."
    ]
    assert result.relationship_chains[0] == "Court Guard spreads rumor to Royal Court"