import asyncio
import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_bridge_module():
    module_path = ROOT / "cognee_sidecar" / "bridge.py"
    spec = importlib.util.spec_from_file_location("mirofish_cognee_bridge", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fetch_graph_data_awaits_async_graph_engine():
    module = load_bridge_module()

    class Engine:
        async def get_graph_data(self):
            return ([{"id": "n1"}], [{"id": "e1"}])

    async def get_graph_engine():
        return Engine()

    nodes, edges = asyncio.run(module.fetch_graph_data(get_graph_engine))

    assert nodes == [{"id": "n1"}]
    assert edges == [{"id": "e1"}]


def test_resolve_groq_provider_from_endpoint_and_normalize_model():
    module = load_bridge_module()

    payload = {
        "llm_provider": "openai",
        "llm_endpoint": "https://api.groq.com/openai/v1",
        "llm_model": "openai/gpt-oss-20b",
    }

    provider = module.resolve_llm_provider(payload)

    assert module.is_groq_routed(payload) is True
    assert provider == "openai"
    assert module.normalize_llm_model("openai/gpt-oss-20b", True) == "groq/openai/gpt-oss-20b"


def test_resolve_embedding_settings_defaults_to_fastembed_for_groq():
    module = load_bridge_module()

    settings = module.resolve_embedding_settings({}, True)

    assert settings["embedding_provider"] == "fastembed"
    assert settings["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert settings["embedding_dimensions"] == 384


def test_is_groq_routed_uses_env_fallback(monkeypatch):
    module = load_bridge_module()

    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.delenv("REPORT_LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)

    assert module.is_groq_routed({}) is True


def test_apply_non_null_settings_preserves_existing_defaults():
    module = load_bridge_module()

    target = types.SimpleNamespace(
        embedding_provider="openai",
        embedding_model="openai/text-embedding-3-large",
        embedding_dimensions=3072,
    )

    module.apply_non_null_settings(
        target,
        {
            "embedding_provider": None,
            "embedding_model": None,
            "embedding_dimensions": 384,
        },
    )

    assert target.embedding_provider == "openai"
    assert target.embedding_model == "openai/text-embedding-3-large"
    assert target.embedding_dimensions == 384


def test_resolve_llm_instructor_mode_defaults_to_json_schema_for_groq():
    module = load_bridge_module()

    assert module.resolve_llm_instructor_mode({}, True) == "json_schema_mode"


def test_normalize_knowledge_graph_payload_maps_groq_style_fields():
    module = load_bridge_module()

    payload = {
        "nodes": [
            {"id": "Aria the Captain", "type": "Person"},
            {"id": "Harbor Guild", "label": "Faction", "properties": {"summary": "Trade guild"}},
        ],
        "edges": [
            {
                "source_id": "Aria the Captain",
                "target_id": "Harbor Guild",
                "relation_type": "member_of",
            }
        ],
    }

    normalized = module.normalize_knowledge_graph_payload(payload)

    assert normalized["nodes"] == [
        {
            "id": "Aria the Captain",
            "name": "Aria the Captain",
            "type": "Person",
            "description": "Aria the Captain",
        },
        {
            "id": "Harbor Guild",
            "name": "Harbor Guild",
            "type": "Faction",
            "description": "Trade guild",
        },
    ]
    assert normalized["edges"] == [
        {
            "source_node_id": "Aria the Captain",
            "target_node_id": "Harbor Guild",
            "relationship_name": "member_of",
        }
    ]


def test_extract_completion_payload_reads_json_from_tool_arguments():
    module = load_bridge_module()

    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"nodes":[{"id":"Aria","type":"Person"}],"edges":[]}'
                            }
                        }
                    ]
                }
            }
        ]
    }

    extracted = module.extract_completion_payload(response)

    assert extracted == {"nodes": [{"id": "Aria", "type": "Person"}], "edges": []}


def test_normalize_search_result_handles_none_payloads():
    module = load_bridge_module()

    assert module.normalize_search_result(None) == {"dataset_name": None, "search_result": ""}
    assert module.normalize_search_result({"dataset_name": "g1", "search_result": None, "text": "fallback chunk"}) == {
        "dataset_name": "g1",
        "search_result": "fallback chunk",
    }


def test_search_graph_skips_empty_results_from_cognee():
    module = load_bridge_module()

    class FakeCognee:
        async def search(self, *args, **kwargs):
            return [None, {"dataset_name": "g1", "search_result": None, "content": "useful chunk"}]

    module.configure_cognee = lambda payload: FakeCognee()

    search_type_module = types.ModuleType("cognee.modules.search.types.SearchType")
    search_type_module.SearchType = type("SearchType", (), {"CHUNKS": "chunks"})
    sys.modules["cognee.modules.search.types.SearchType"] = search_type_module

    result = asyncio.run(module.search_graph({"graph_id": "g1", "query": "harbor", "limit": 5}))

    assert result == {
        "graph_id": "g1",
        "results": [{"dataset_name": "g1", "search_result": "useful chunk"}],
    }


def test_should_bypass_knowledge_graph_structured_output_for_groq_route():
    module = load_bridge_module()

    class FakeKnowledgeGraph:
        __name__ = "TotallyNotNamedKnowledgeGraph"
        model_fields = {"nodes": object(), "edges": object()}

    class FakeAdapter:
        endpoint = "https://api.groq.com/openai/v1"
        model = "groq/openai/gpt-oss-20b"

    assert module.is_knowledge_graph_response_model(FakeKnowledgeGraph) is True
    assert (
        module.should_bypass_knowledge_graph_structured_output(FakeAdapter(), FakeKnowledgeGraph)
        is True
    )