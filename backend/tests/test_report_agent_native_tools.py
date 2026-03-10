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


def test_report_manager_prefers_latest_completed_report(tmp_path: Path):
    module = load_report_agent_module()
    module.ReportManager.REPORTS_DIR = str(tmp_path / "reports")

    module.ReportManager.save_report(module.Report(
        report_id="report_pending_newer",
        simulation_id="sim_1",
        graph_id="g1",
        simulation_requirement="req",
        status=module.ReportStatus.GENERATING,
        created_at="2026-03-09T23:00:00",
    ))
    module.ReportManager.save_report(module.Report(
        report_id="report_completed_old",
        simulation_id="sim_1",
        graph_id="g1",
        simulation_requirement="req",
        status=module.ReportStatus.COMPLETED,
        created_at="2026-03-09T22:00:00",
        completed_at="2026-03-09T22:10:00",
    ))
    module.ReportManager.save_report(module.Report(
        report_id="report_completed_latest",
        simulation_id="sim_1",
        graph_id="g1",
        simulation_requirement="req",
        status=module.ReportStatus.COMPLETED,
        created_at="2026-03-09T22:30:00",
        completed_at="2026-03-09T22:40:00",
    ))

    report = module.ReportManager.get_report_by_simulation("sim_1")

    assert report is not None
    assert report.report_id == "report_completed_latest"


def test_report_manager_falls_back_to_latest_created_when_no_completed_reports(tmp_path: Path):
    module = load_report_agent_module()
    module.ReportManager.REPORTS_DIR = str(tmp_path / "reports")

    module.ReportManager.save_report(module.Report(
        report_id="report_old",
        simulation_id="sim_1",
        graph_id="g1",
        simulation_requirement="req",
        status=module.ReportStatus.PENDING,
        created_at="2026-03-09T21:00:00",
    ))
    module.ReportManager.save_report(module.Report(
        report_id="report_new",
        simulation_id="sim_1",
        graph_id="g1",
        simulation_requirement="req",
        status=module.ReportStatus.GENERATING,
        created_at="2026-03-09T22:00:00",
    ))

    report = module.ReportManager.get_report_by_simulation("sim_1")

    assert report is not None
    assert report.report_id == "report_new"


def test_report_manager_roundtrip_preserves_writeback_result(tmp_path: Path):
    module = load_report_agent_module()
    module.ReportManager.REPORTS_DIR = str(tmp_path / "reports")

    writeback_result = {
        "enabled": True,
        "ok": True,
        "candidate_deltas_count": 2,
        "auto_promote": {
            "success_count": 2,
            "failure_count": 0,
        },
    }
    module.ReportManager.save_report(module.Report(
        report_id="report_with_writeback",
        simulation_id="sim_1",
        graph_id="g1",
        simulation_requirement="req",
        status=module.ReportStatus.COMPLETED,
        created_at="2026-03-10T12:00:00",
        completed_at="2026-03-10T12:05:00",
        writeback_result=writeback_result,
    ))

    loaded = module.ReportManager.get_report("report_with_writeback")

    assert loaded is not None
    assert loaded.writeback_result == writeback_result
    assert loaded.to_dict()["writeback_result"]["auto_promote"]["success_count"] == 2


def test_generate_section_react_uses_runtime_fallback_when_all_tool_results_are_empty():
    module = load_report_agent_module()

    class LLMStub:
        def __init__(self):
            self.responses = iter([
                '<tool_call>{"name": "insight_forge", "parameters": {"query": "rumor diffusion"}}</tool_call>',
                '<tool_call>{"name": "panorama_search", "parameters": {"query": "court reaction"}}</tool_call>',
                '<tool_call>{"name": "quick_search", "parameters": {"query": "town crier nessa", "limit": 5}}</tool_call>',
                'This confident narrative should not be accepted as-is.',
            ])

        def chat(self, **kwargs):
            return next(self.responses)

    class ToolsStub:
        def insight_forge(self, **kwargs):
            return types.SimpleNamespace(
                semantic_facts=[],
                entity_insights=[],
                relationship_chains=[],
                diagnostics={"evidence_sources": ["no_evidence"]},
                to_text=lambda: "Current Key Memory (0)\nCore Entities (0)\nRelationship Chains (0)",
            )

        def panorama_search(self, **kwargs):
            return types.SimpleNamespace(
                active_facts=[],
                all_nodes=[],
                all_edges=[],
                diagnostics={"evidence_sources": ["no_evidence"], "fallback_used": True},
                to_text=lambda: "Active Memory (0)\nReferenced Entities (0)",
            )

        def quick_search(self, **kwargs):
            return types.SimpleNamespace(
                facts=[],
                nodes=[],
                edges=[],
                total_count=0,
                diagnostics={"evidence_sources": ["no_evidence"]},
                to_text=lambda: "Search Results\n0 facts\nNo related results found",
            )

        def get_runtime_evidence(self, simulation_id, limit=10):
            return {
                "simulation_id": simulation_id,
                "has_runtime_evidence": True,
                "current_round": 10,
                "total_actions": 68,
                "action_facts": [
                    "[round 10] [twitter] captain_serik_488 quote_post: {\"trace_created_at\": 4}",
                    "[round 0] [twitter] Town Crier Nessa posted the initial forged-decree rumor.",
                    "[round 10] [twitter] Royal Court posted that the decree remains valid and the succession is unchanged.",
                    "[round 10] [twitter] Archivist Maelin posted that no evidence of forgery was found in the seals and records.",
                ],
            }

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_1",
        simulation_requirement="Analyze how the forged-decree rumor spreads and how institutions react.",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    outline = module.ReportOutline(
        title="Rumor Report",
        summary="Grounded analysis only",
        sections=[module.ReportSection(title="Rumor Diffusion Pathways")],
    )

    content = agent._generate_section_react(
        section=outline.sections[0],
        outline=outline,
        previous_sections=[],
        section_index=0,
    )

    assert "graph retrieval returned sparse results" in content
    assert "Town Crier Nessa posted the initial forged-decree rumor" in content
    assert "Royal Court posted that the decree remains valid" in content
    assert "trace_created_at" not in content
    assert "This confident narrative should not be accepted as-is." not in content


def test_generate_section_react_keeps_llm_content_when_tool_evidence_exists():
    module = load_report_agent_module()

    class LLMStub:
        def __init__(self):
            self.responses = iter([
                '<tool_call>{"name": "insight_forge", "parameters": {"query": "institutional response"}}</tool_call>',
                '<tool_call>{"name": "panorama_search", "parameters": {"query": "guard statements"}}</tool_call>',
                '<tool_call>{"name": "quick_search", "parameters": {"query": "royal court decree", "limit": 5}}</tool_call>',
                'Final Answer: Verified synthesis based on retrieved evidence.',
            ])

        def chat(self, **kwargs):
            return next(self.responses)

    class ToolsStub:
        def insight_forge(self, **kwargs):
            return types.SimpleNamespace(
                semantic_facts=[],
                entity_insights=[],
                relationship_chains=[],
                diagnostics={"evidence_sources": ["no_evidence"]},
                to_text=lambda: "Current Key Memory (0)",
            )

        def panorama_search(self, **kwargs):
            return types.SimpleNamespace(
                active_facts=[],
                all_nodes=[],
                all_edges=[],
                diagnostics={"evidence_sources": ["no_evidence"]},
                to_text=lambda: "Active Memory (0)",
            )

        def quick_search(self, **kwargs):
            return types.SimpleNamespace(
                facts=["Royal Court posted that the decree remains valid."],
                nodes=[],
                edges=[],
                total_count=1,
                diagnostics={"evidence_sources": ["runtime_actions"]},
                to_text=lambda: "Search Results\n1 facts\nRoyal Court posted that the decree remains valid.",
            )

        def get_runtime_evidence(self, simulation_id, limit=10):
            return {"simulation_id": simulation_id, "has_runtime_evidence": True}

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_2",
        simulation_requirement="Analyze official messaging.",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    outline = module.ReportOutline(
        title="Institutional Report",
        summary="Use retrieved evidence",
        sections=[module.ReportSection(title="Institutional Reactions")],
    )

    content = agent._generate_section_react(
        section=outline.sections[0],
        outline=outline,
        previous_sections=[],
        section_index=0,
    )

    assert content == "Verified synthesis based on retrieved evidence."


def test_generate_section_react_uses_strict_grounded_fallback_when_final_answer_prefix_is_missing_and_content_is_not_substantive():
    module = load_report_agent_module()

    class LLMStub:
        def __init__(self):
            self.responses = iter([
                '<tool_call>{"name": "insight_forge", "parameters": {"query": "institutional response"}}</tool_call>',
                '<tool_call>{"name": "panorama_search", "parameters": {"query": "guard statements"}}</tool_call>',
                '<tool_call>{"name": "quick_search", "parameters": {"query": "royal court decree", "limit": 5}}</tool_call>',
                'Verified synthesis based on retrieved evidence.',
            ])

        def chat(self, **kwargs):
            return next(self.responses)

    class ToolsStub:
        def insight_forge(self, **kwargs):
            return types.SimpleNamespace(
                semantic_facts=[],
                entity_insights=[],
                relationship_chains=[],
                diagnostics={"evidence_sources": ["no_evidence"]},
                to_text=lambda: "Current Key Memory (0)",
            )

        def panorama_search(self, **kwargs):
            return types.SimpleNamespace(
                active_facts=[],
                all_nodes=[],
                all_edges=[],
                diagnostics={"evidence_sources": ["no_evidence"]},
                to_text=lambda: "Active Memory (0)",
            )

        def quick_search(self, **kwargs):
            return types.SimpleNamespace(
                facts=["[round 10] [twitter] Royal Court posted that the decree remains valid."],
                nodes=[],
                edges=[],
                total_count=1,
                diagnostics={"evidence_sources": ["runtime_actions"]},
                to_text=lambda: "Search Results\n1 facts\n[round 10] [twitter] Royal Court posted that the decree remains valid.",
            )

        def get_runtime_evidence(self, simulation_id, limit=10):
            return {"simulation_id": simulation_id, "has_runtime_evidence": True}

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_strict_prefix",
        simulation_requirement="Analyze official messaging.",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    outline = module.ReportOutline(
        title="Institutional Report",
        summary="Use retrieved evidence",
        sections=[module.ReportSection(title="Institutional Reactions")],
    )

    content = agent._generate_section_react(
        section=outline.sections[0],
        outline=outline,
        previous_sections=[],
        section_index=0,
    )

    assert "auto-finalized in strict grounding mode" in content
    assert "Royal Court posted that the decree remains valid." in content
    assert "Verified synthesis based on retrieved evidence." not in content


def test_generate_section_react_keeps_substantive_content_when_final_answer_prefix_is_missing():
    module = load_report_agent_module()

    substantive_response = (
        "The Royal Court reacted quickly to the forged-decree rumor, but the response stayed fragmented across platforms. "
        "Boros kept the controversy visible by repeating and reframing the claim, which prevented a clean institutional reset.\n\n"
        "That pattern suggests the crisis is being sustained by repeated amplification and role conflict rather than by a single viral spike."
    )

    class LLMStub:
        def __init__(self):
            self.responses = iter([
                '<tool_call>{"name": "insight_forge", "parameters": {"query": "institutional response"}}</tool_call>',
                '<tool_call>{"name": "panorama_search", "parameters": {"query": "guard statements"}}</tool_call>',
                '<tool_call>{"name": "quick_search", "parameters": {"query": "royal court decree", "limit": 5}}</tool_call>',
                substantive_response,
            ])

        def chat(self, **kwargs):
            return next(self.responses)

    class ToolsStub:
        def insight_forge(self, **kwargs):
            return types.SimpleNamespace(
                semantic_facts=[],
                entity_insights=[],
                relationship_chains=[],
                diagnostics={"evidence_sources": ["runtime_actions"]},
                to_text=lambda: "Current Key Memory (0)",
            )

        def panorama_search(self, **kwargs):
            return types.SimpleNamespace(
                active_facts=[],
                all_nodes=[],
                all_edges=[],
                diagnostics={"evidence_sources": ["runtime_actions"]},
                to_text=lambda: "Active Memory (0)",
            )

        def quick_search(self, **kwargs):
            return types.SimpleNamespace(
                facts=[
                    "[round 10] [twitter] Royal Court posted that the decree remains valid.",
                    "[round 9] [twitter] Boros the Dockmaster repeated the forged-decree claim.",
                ],
                nodes=[],
                edges=[],
                total_count=2,
                diagnostics={"evidence_sources": ["runtime_actions"]},
                to_text=lambda: "Search Results\n2 facts\n[round 10] [twitter] Royal Court posted that the decree remains valid.",
            )

        def get_runtime_evidence(self, simulation_id, limit=10):
            return {"simulation_id": simulation_id, "has_runtime_evidence": True}

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_substantive_prefixless",
        simulation_requirement="Analyze official messaging.",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    outline = module.ReportOutline(
        title="Institutional Report",
        summary="Use retrieved evidence",
        sections=[module.ReportSection(title="Institutional Reactions")],
    )

    content = agent._generate_section_react(
        section=outline.sections[0],
        outline=outline,
        previous_sections=[],
        section_index=0,
    )

    assert content == substantive_response
    assert "auto-finalized in strict grounding mode" not in content


def test_generate_section_react_uses_runtime_fallback_when_tool_results_are_graph_noise_only():
    module = load_report_agent_module()

    class LLMStub:
        def __init__(self):
            self.responses = iter([
                '<tool_call>{"name": "insight_forge", "parameters": {"query": "rumor polarization"}}</tool_call>',
                '<tool_call>{"name": "panorama_search", "parameters": {"query": "court rumor graph"}}</tool_call>',
                '<tool_call>{"name": "quick_search", "parameters": {"query": "guard influence", "limit": 5}}</tool_call>',
                'Confident synthesis based on graph lore only.',
            ])

        def chat(self, **kwargs):
            return next(self.responses)

    class ToolsStub:
        def insight_forge(self, **kwargs):
            return types.SimpleNamespace(
                semantic_facts=["WorldRule: Unverified rumors spread quickly online"],
                entity_insights=[{"name": "Sunspire Palace"}],
                relationship_chains=["SocialEdge: Chancellor Varos -> Captain Serik (pressures)"],
                diagnostics={"evidence_sources": ["local_graph_nodes"]},
                to_text=lambda: "Current Key Memory\nWorldRule: Unverified rumors spread quickly online",
            )

        def panorama_search(self, **kwargs):
            return types.SimpleNamespace(
                active_facts=["EventSeed: Forged succession decree rumor"],
                all_nodes=[{"name": "Market of Bells"}],
                all_edges=[],
                diagnostics={"evidence_sources": ["local_graph_nodes"]},
                to_text=lambda: "Active Memory\nEventSeed: Forged succession decree rumor",
            )

        def quick_search(self, **kwargs):
            return types.SimpleNamespace(
                facts=["ContextLocation: Sunspire Palace"],
                nodes=[],
                edges=[],
                total_count=1,
                diagnostics={"evidence_sources": ["local_graph_nodes"]},
                to_text=lambda: "Search Results\nContextLocation: Sunspire Palace",
            )

        def get_runtime_evidence(self, simulation_id, limit=10):
            return {
                "simulation_id": simulation_id,
                "has_runtime_evidence": True,
                "current_round": 10,
                "total_actions": 68,
                "action_facts": [
                    "[round 9] [twitter] Town Crier Nessa posted a forged-decree claim.",
                    "[round 10] [twitter] Royal Court posted that the decree remains valid.",
                ],
            }

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_noise",
        simulation_requirement="Analyze the rumor dynamics.",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    outline = module.ReportOutline(
        title="Noise Report",
        summary="Runtime facts only",
        sections=[module.ReportSection(title="Narrative Risk")],
    )

    content = agent._generate_section_react(
        section=outline.sections[0],
        outline=outline,
        previous_sections=[],
        section_index=0,
    )

    assert "graph retrieval returned sparse results" in content
    assert "Town Crier Nessa posted a forged-decree claim" in content
    assert "Confident synthesis based on graph lore only." not in content


def test_execute_tool_payload_prefers_runtime_facts_for_cognee_tool_text():
    module = load_report_agent_module()

    class ToolsStub:
        def quick_search(self, **kwargs):
            return types.SimpleNamespace(
                facts=[
                    "[round 10] DocumentChunk rumor-brief --[contains]--> TextSummary court-response",
                    "[round 10] [twitter] captain_serik_488 quote_post: {\"trace_created_at\": 4}",
                    "[round 10] [twitter] Royal Court posted that the decree remains valid.",
                    "WorldRule: Merchants panic quickly when succession looks unstable",
                ],
                nodes=[],
                edges=[],
                total_count=4,
                diagnostics={"evidence_sources": ["runtime_actions", "local_graph_nodes"]},
                to_text=lambda: "Search Results\n[round 10] DocumentChunk rumor-brief --[contains]--> TextSummary court-response\n[round 10] [twitter] captain_serik_488 quote_post: {\"trace_created_at\": 4}\n[round 10] [twitter] Royal Court posted that the decree remains valid.\nWorldRule: Merchants panic quickly when succession looks unstable",
            )

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_runtime",
        simulation_requirement="Analyze official messaging.",
        llm_client=object(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    payload = agent._execute_tool_payload("quick_search", {"query": "royal court decree", "limit": 5})

    assert "Verified runtime evidence from quick_search:" in payload["text"]
    assert "[round 10] [twitter] Royal Court posted that the decree remains valid." in payload["text"]
    assert "DocumentChunk" not in payload["text"]
    assert "trace_created_at" not in payload["text"]
    assert "WorldRule:" not in payload["text"]
    assert payload["evidence"]["meaningful"] is True
    assert payload["evidence"]["grounded_line_count"] == 1


def test_chat_returns_only_retrieved_facts_when_tool_results_exist():
    module = load_report_agent_module()

    class LLMStub:
        def __init__(self):
            self.responses = iter([
                '<tool_call>{"name": "quick_search", "parameters": {"query": "Aria Harbor Guild", "limit": 5}}</tool_call>',
                'Aria is the unquestioned ruler of the port city.',
            ])

        def chat(self, **kwargs):
            return next(self.responses)

    class ToolsStub:
        def quick_search(self, **kwargs):
            return types.SimpleNamespace(
                facts=[
                    'Actor: Aria the Captain\n- role: captain\nSocialEdge: Aria the Captain -> Harbor Guild (MEMBER_OF)',
                ],
                nodes=[],
                edges=[],
                total_count=1,
                diagnostics={"evidence_sources": ["sidecar_search"]},
                to_text=lambda: 'Search Results\nActor: Aria the Captain\n- role: captain\nSocialEdge: Aria the Captain -> Harbor Guild (MEMBER_OF)',
            )

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_chat_1",
        simulation_requirement="Answer actor questions from retrieved memory.",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    result = agent.chat("Who is Aria?")

    assert "Based only on retrieved facts:" in result["response"]
    assert "Actor: Aria the Captain" in result["response"]
    assert "Aria the Captain — role: captain" in result["response"]
    assert "Relation: Aria the Captain -> Harbor Guild (MEMBER_OF)" in result["response"]
    assert "unquestioned ruler" not in result["response"]


def test_chat_returns_limited_fallback_when_retrieved_facts_are_sparse():
    module = load_report_agent_module()

    class LLMStub:
        def __init__(self):
            self.responses = iter([
                '<tool_call>{"name": "quick_search", "parameters": {"query": "Aria Harbor Guild", "limit": 5}}</tool_call>',
                'Aria definitely controls the guild.',
            ])

        def chat(self, **kwargs):
            return next(self.responses)

    class ToolsStub:
        def quick_search(self, **kwargs):
            return types.SimpleNamespace(
                facts=["schema_version: 1.0", "world_id: mythweave"],
                nodes=[],
                edges=[],
                total_count=2,
                diagnostics={"evidence_sources": ["sidecar_search"]},
                to_text=lambda: 'Search Results\nschema_version: 1.0\nworld_id: mythweave',
            )

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_chat_2",
        simulation_requirement="Answer actor questions from retrieved memory.",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    result = agent.chat("Who is Aria?")

    assert "I could not find enough verified facts" in result["response"]
    assert "definitely controls the guild" not in result["response"]


def test_chat_prefers_query_matching_facts_over_unrelated_runtime_lines():
    module = load_report_agent_module()

    class LLMStub:
        def __init__(self):
            self.responses = iter([
                '<tool_call>{"name": "quick_search", "parameters": {"query": "Aria the Captain", "limit": 5}}</tool_call>',
                'Aria clearly dominates the city.',
            ])

        def chat(self, **kwargs):
            return next(self.responses)

    class ToolsStub:
        def quick_search(self, **kwargs):
            return types.SimpleNamespace(
                facts=[
                    '[round 10] [twitter] city guard CREATE_POST',
                    'Actor: Aria the Captain\n- role: captain',
                ],
                nodes=[],
                edges=[],
                total_count=2,
                diagnostics={"evidence_sources": ["sidecar_search"]},
                to_text=lambda: 'Search Results\n[round 10] [twitter] city guard CREATE_POST\nActor: Aria the Captain\n- role: captain',
            )

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_chat_4",
        simulation_requirement="Answer actor questions from retrieved memory.",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    result = agent.chat("Who is Aria?")

    assert "Actor: Aria the Captain" in result["response"]
    assert "Aria the Captain — role: captain" in result["response"]
    assert "city guard CREATE_POST" not in result["response"]


def test_chat_auto_uses_quick_search_when_llm_skips_tools_for_cognee():
    module = load_report_agent_module()

    class LLMStub:
        def chat(self, **kwargs):
            return "Aria is probably the central authority here."

    class ToolsStub:
        def quick_search(self, **kwargs):
            return types.SimpleNamespace(
                facts=['Actor: Aria the Captain\n- role: captain'],
                nodes=[],
                edges=[],
                total_count=1,
                diagnostics={"evidence_sources": ["sidecar_search"]},
                to_text=lambda: 'Search Results\nActor: Aria the Captain\n- role: captain',
            )

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_chat_3",
        simulation_requirement="Answer actor questions from retrieved memory.",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    result = agent.chat("Who is Aria?")

    assert "Actor: Aria the Captain" in result["response"]
    assert "Aria the Captain — role: captain" in result["response"]
    assert "probably the central authority" not in result["response"]
    assert result["tool_calls"]
    assert result["tool_calls"][0]["name"] == "quick_search"


def test_chat_auto_extracts_entity_query_from_russian_prompt():
    module = load_report_agent_module()

    class LLMStub:
        def chat(self, **kwargs):
            return "I don't know."

    class ToolsStub:
        def __init__(self):
            self.last_query = None

        def quick_search(self, **kwargs):
            self.last_query = kwargs.get("query")
            return types.SimpleNamespace(
                facts=['Organization: Harbor Guild'],
                nodes=[],
                edges=[],
                total_count=1,
                diagnostics={"evidence_sources": ["sidecar_search"]},
                to_text=lambda: 'Search Results\nOrganization: Harbor Guild',
            )

    tools = ToolsStub()
    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_chat_5",
        simulation_requirement="Answer actor questions from retrieved memory.",
        llm_client=LLMStub(),
        zep_tools=tools,
        graph_backend="cognee",
    )

    result = agent.chat("Что такое Harbor Guild? Ответь одной строкой.")

    assert tools.last_query == "Harbor Guild"
    assert "Organization: Harbor Guild" in result["response"]


def test_chat_returns_russian_actor_list_for_actor_overview_question():
    module = load_report_agent_module()

    class LLMStub:
        def chat(self, **kwargs):
            return "Не знаю."

    class ToolsStub:
        def quick_search(self, **kwargs):
            return types.SimpleNamespace(
                facts=[
                    "[round 10] [twitter] town crier nessa CREATEPOST",
                    "[round 10] [twitter] captain serik QUOTEPOST",
                    "[round 10] [twitter] royal court QUOTEPOST",
                ],
                nodes=[],
                edges=[],
                total_count=3,
                diagnostics={"evidence_sources": ["runtime_actions"]},
                to_text=lambda: "Search Results",
            )

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_chat_ru_actors",
        simulation_requirement="Answer actor questions from retrieved memory.",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    result = agent.chat("Кто главные актеры?")

    assert result["response"].startswith("По найденным фактам наиболее заметные акторы:")
    assert "Town Crier Nessa" in result["response"]
    assert "Captain Serik" in result["response"]
    assert "Royal Court" in result["response"]


def test_chat_returns_russian_no_fact_message_for_unmatched_backend_question():
    module = load_report_agent_module()

    class LLMStub:
        def chat(self, **kwargs):
            return "Не знаю."

    class ToolsStub:
        def quick_search(self, **kwargs):
            return types.SimpleNamespace(
                facts=["[round 10] [twitter] town crier nessa CREATEPOST"],
                nodes=[],
                edges=[],
                total_count=1,
                diagnostics={"evidence_sources": ["runtime_actions"]},
                to_text=lambda: "Search Results",
            )

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_chat_ru_backend",
        simulation_requirement="Answer actor questions from retrieved memory.",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    result = agent.chat("Какой бекенд ты юзаешь?")

    assert "не могу определить" in result["response"].lower()
    assert "town crier nessa" not in result["response"].lower()


def test_chat_filters_generic_placeholder_summary_nodes():
    module = load_report_agent_module()

    class LLMStub:
        def chat(self, **kwargs):
            return "I don't know."

    class ToolsStub:
        def quick_search(self, **kwargs):
            return types.SimpleNamespace(
                facts=[],
                nodes=[{"name": "artifact", "type": "artifact", "summary": "artifact"}],
                edges=[],
                total_count=1,
                diagnostics={"evidence_sources": ["local_graph_nodes"]},
                to_text=lambda: "Search Results",
            )

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_chat_placeholder",
        simulation_requirement="Answer actor questions from retrieved memory.",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    result = agent.chat("artifact какой")

    assert "Вопрос слишком общий" in result["response"]
    assert result["tool_calls"] == []
    assert "artifact — summary: artifact" not in result["response"]


def test_chat_restates_previous_answer_in_russian_without_new_tools():
    module = load_report_agent_module()

    class LLMStub:
        def chat(self, **kwargs):
            raise AssertionError("LLM should not be used for language-only follow-up")

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_chat_followup_ru",
        simulation_requirement="Answer actor questions from retrieved memory.",
        llm_client=LLMStub(),
        zep_tools=types.SimpleNamespace(),
        graph_backend="cognee",
    )

    history = [{"role": "assistant", "content": "Based on retrieved facts, the most visible actors are: Town Crier Nessa, Captain Serik."}]
    result = agent.chat("ответь по-русски", chat_history=history)

    assert result["response"].startswith("По найденным фактам наиболее заметные акторы:")
    assert result["tool_calls"] == []


def test_chat_summarizes_main_conflict_from_direct_search():
    module = load_report_agent_module()

    class LLMStub:
        def chat(self, **kwargs):
            raise AssertionError("LLM should not be used for direct main conflict intent")

    class ToolsStub:
        def insight_forge(self, **kwargs):
            return types.SimpleNamespace(
                semantic_facts=["Forged succession decree rumor — summary: Rumor about a forged succession decree destabilizing the royal court."],
                relationship_chains=[],
                entity_insights=[],
                diagnostics={"evidence_sources": ["sidecar_search"]},
                to_text=lambda: "Search Results",
            )

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_chat_conflict",
        simulation_requirement="forged royal succession decree rumor",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    result = agent.chat("какой главный конфликт сейчас? ответь коротко")

    assert result["response"].startswith("По найденным фактам главный конфликт сейчас —")
    assert "forged succession decree" in result["response"].lower()
    assert result["tool_calls"][0]["name"] == "insight_forge"


def test_chat_main_conflict_question_with_short_instruction_is_not_treated_as_followup():
    module = load_report_agent_module()

    class LLMStub:
        def chat(self, **kwargs):
            raise AssertionError("LLM should not be used for direct main conflict intent")

    class ToolsStub:
        def insight_forge(self, **kwargs):
            return types.SimpleNamespace(
                semantic_facts=["Forged succession decree rumor — summary: Rumor about a forged succession decree destabilizing the royal court."],
                relationship_chains=[],
                entity_insights=[],
                diagnostics={"evidence_sources": ["sidecar_search"]},
                to_text=lambda: "Search Results",
            )

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_chat_conflict_history",
        simulation_requirement="forged royal succession decree rumor",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    history = [{"role": "assistant", "content": "Вопрос слишком общий: уточни сущность."}]
    result = agent.chat("какой главный конфликт сейчас? ответь коротко", chat_history=history)

    assert result["response"].startswith("По найденным фактам главный конфликт сейчас —")
    assert result["tool_calls"][0]["name"] == "insight_forge"


def test_chat_main_conflict_refuses_when_only_runtime_lines_exist():
    module = load_report_agent_module()

    class LLMStub:
        def chat(self, **kwargs):
            raise AssertionError("LLM should not be used for direct main conflict intent")

    class ToolsStub:
        def insight_forge(self, **kwargs):
            return types.SimpleNamespace(
                semantic_facts=["[round 12] [twitter] captain_serik_488 quote_post: {\"trace_created_at\": 4}"],
                relationship_chains=[],
                entity_insights=[],
                diagnostics={"evidence_sources": ["runtime_actions"]},
                to_text=lambda: "Search Results",
            )

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_chat_conflict_runtime_only",
        simulation_requirement="forged royal succession decree rumor",
        llm_client=LLMStub(),
        zep_tools=ToolsStub(),
        graph_backend="cognee",
    )

    result = agent.chat("какой главный конфликт сейчас? ответь коротко")

    assert "Не нашёл достаточно верифицированных фактов" in result["response"]


def test_chat_shortens_previous_fact_dump_into_one_line():
    module = load_report_agent_module()

    class LLMStub:
        def chat(self, **kwargs):
            raise AssertionError("LLM should not be used for brevity-only follow-up")

    agent = module.ReportAgent(
        graph_id="g1",
        simulation_id="sim_chat_followup_short",
        simulation_requirement="Answer actor questions from retrieved memory.",
        llm_client=LLMStub(),
        zep_tools=types.SimpleNamespace(),
        graph_backend="cognee",
    )

    history = [{
        "role": "assistant",
        "content": "Только по найденным фактам:\n- Fact one\n- Fact two\n- Fact three",
    }]
    result = agent.chat("короче, одной строкой", chat_history=history)

    assert result["response"] == "Только по найденным фактам: Fact one; Fact two"
    assert result["tool_calls"] == []