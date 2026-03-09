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
    def get_graph_data(self, graph_id):
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
        raise RuntimeError("search unavailable")


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
    service = module.CogneeToolsService(sidecar_client=SidecarStub(), entity_reader=object())

    result = service.search_graph("graph_1", "Harbor Guild", limit=5)

    assert result.total_count == 1
    assert result.facts == ["Aria supports Harbor Guild"]
    assert result.nodes[0]["name"] == "Harbor Guild"


def test_cognee_tools_get_simulation_context_matches_zep_shape():
    module = load_cognee_tools_module()
    runner = sys.modules["app.services.simulation_runner"].SimulationRunner
    runner.run_state = None
    runner.actions = []
    service = module.CogneeToolsService(sidecar_client=SidecarStub(), entity_reader=object())

    result = service.get_simulation_context("graph_1", "Run a harbor crisis simulation", limit=10)

    assert result["graph_statistics"]["total_nodes"] == 2
    assert result["graph_statistics"]["entity_types"]["Organization"] == 1
    assert result["total_entities"] == 2
    assert result["entities"][0]["type"] in {"Organization", "Actor"}


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