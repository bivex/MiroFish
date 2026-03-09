import asyncio
import functools
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


SENTINEL = "__MIROFISH_COGNEE_JSON__"
GROQ_KG_PATCH_FLAG = "_mirofish_groq_kg_patch"
GROQ_KG_GATEWAY_PATCH_FLAG = "_mirofish_groq_kg_gateway_patch"
KNOWLEDGE_GRAPH_JSON_INSTRUCTIONS = """
Return only one valid JSON object matching this schema:
{
  "nodes": [
    {"id": "string", "name": "string", "type": "string", "description": "string"}
  ],
  "edges": [
    {
      "source_node_id": "string",
      "target_node_id": "string",
      "relationship_name": "string"
    }
  ]
}
Use empty arrays when unsure. Do not include markdown fences or extra text.
""".strip()


def emit(payload: dict[str, Any]) -> None:
    print(SENTINEL + json.dumps(payload, ensure_ascii=False))


def load_payload() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    return json.loads(raw) if raw else {}


def normalize_node(raw: Any) -> dict[str, Any]:
    node_id = ""
    props: dict[str, Any] = {}

    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        node_id = str(raw[0])
        props = raw[1] or {}
    elif isinstance(raw, dict):
        node_id = str(raw.get("id") or raw.get("uuid") or raw.get("name") or "")
        props = raw.get("properties") or raw

    node_type = props.get("type") or props.get("node_type") or "Entity"
    name = props.get("name") or props.get("id") or node_id or node_type
    summary = props.get("summary") or props.get("description") or ""
    labels = props.get("labels") or ["Entity", node_type]

    return {
        "uuid": node_id or name,
        "name": name,
        "labels": labels,
        "summary": summary,
        "attributes": props,
    }


def normalize_edge(raw: Any) -> dict[str, Any]:
    source = target = relation = ""
    props: dict[str, Any] = {}

    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        props = raw[3] or {}

        source = str(raw[0])
        second = str(raw[1])
        third = str(raw[2])

        relation_hint = str(
            props.get("relationship_name")
            or props.get("relationship_type")
            or props.get("relation_type")
            or props.get("relationship")
            or ""
        )
        target_hint = str(
            props.get("target_node_uuid")
            or props.get("target_node_id")
            or props.get("target_id")
            or ""
        )

        if relation_hint and relation_hint == second and relation_hint != third:
            relation = second
            target = target_hint or third
        elif relation_hint and relation_hint == third:
            target = target_hint or second
            relation = third
        elif target_hint and target_hint == third and third != second:
            relation = second
            target = third
        else:
            # Cognee graph tuples arrive as: (source_node_id, target_node_id, relationship_name, props)
            target = second
            relation = third
    elif isinstance(raw, dict):
        source = str(
            raw.get("source")
            or raw.get("source_node_uuid")
            or raw.get("source_node_id")
            or raw.get("source_id")
            or ""
        )
        relation = str(
            raw.get("relationship_name")
            or raw.get("name")
            or raw.get("relationship_type")
            or raw.get("relation_type")
            or raw.get("relationship")
            or "RELATED_TO"
        )
        target = str(
            raw.get("target")
            or raw.get("target_node_uuid")
            or raw.get("target_node_id")
            or raw.get("target_id")
            or ""
        )
        props = raw.get("properties") or raw.get("attributes") or raw

    fact = props.get("fact") or props.get("relationship_description") or relation
    edge_uuid = str(props.get("id") or props.get("uuid") or f"{source}:{relation}:{target}")

    return {
        "uuid": edge_uuid,
        "name": relation,
        "fact": fact,
        "source_node_uuid": source,
        "target_node_uuid": target,
        "attributes": props,
    }


def normalize_search_result(raw: Any) -> dict[str, Any]:
    raw = _model_dump(raw)
    if raw is None:
        return {"dataset_name": None, "search_result": ""}

    result = raw.get("search_result") if isinstance(raw, dict) else raw
    result = _model_dump(result)
    if result is None and isinstance(raw, dict):
        result = raw.get("text") or raw.get("content") or raw.get("chunk_text") or ""
    if result is None:
        result = ""
    elif not isinstance(result, (str, dict, list, int, float, bool)):
        result = str(result)

    return {
        "dataset_name": raw.get("dataset_name") if isinstance(raw, dict) else None,
        "search_result": result,
    }


def _coerce_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip() or default
    return str(value).strip() or default


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
            continue
        return value
    return None


def _model_dump(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def is_knowledge_graph_response_model(response_model: Any) -> bool:
    fields = getattr(response_model, "model_fields", {}) or {}
    return {"nodes", "edges"}.issubset(fields.keys())


def is_groq_adapter(adapter: Any) -> bool:
    endpoint = _coerce_string(getattr(adapter, "endpoint", "")).lower()
    model = _coerce_string(getattr(adapter, "model", "")).lower()
    return "api.groq.com" in endpoint or model.startswith("groq/")


def should_bypass_knowledge_graph_structured_output(adapter: Any, response_model: Any) -> bool:
    return is_groq_adapter(adapter) and is_knowledge_graph_response_model(response_model)


def normalize_knowledge_graph_node(raw: Any) -> dict[str, str]:
    raw = _model_dump(raw)
    props = raw.get("properties") if isinstance(raw, dict) else {}
    props = props if isinstance(props, dict) else {}

    node_id = _coerce_string(
        (raw.get("id") if isinstance(raw, dict) else None)
        or (raw.get("uuid") if isinstance(raw, dict) else None)
        or (raw.get("name") if isinstance(raw, dict) else None)
        or (raw.get("label") if isinstance(raw, dict) else None)
        or props.get("id")
        or props.get("name")
    )
    name = _coerce_string(
        (raw.get("name") if isinstance(raw, dict) else None)
        or (raw.get("id") if isinstance(raw, dict) else None)
        or props.get("name")
        or node_id
    )
    node_type = _coerce_string(
        (raw.get("type") if isinstance(raw, dict) else None)
        or (raw.get("label") if isinstance(raw, dict) else None)
        or (raw.get("category") if isinstance(raw, dict) else None)
        or props.get("type")
        or props.get("label")
        or "Entity"
    )
    description = _coerce_string(
        (raw.get("description") if isinstance(raw, dict) else None)
        or (raw.get("summary") if isinstance(raw, dict) else None)
        or props.get("description")
        or props.get("summary")
        or name
        or node_id
        or node_type
    )

    return {
        "id": node_id or name,
        "name": name or node_id,
        "type": node_type,
        "description": description,
    }


def normalize_knowledge_graph_edge(raw: Any) -> dict[str, str] | None:
    raw = _model_dump(raw)
    props = raw.get("properties") if isinstance(raw, dict) else {}
    props = props if isinstance(props, dict) else {}

    source = _coerce_string(
        (raw.get("source_node_id") if isinstance(raw, dict) else None)
        or (raw.get("source_id") if isinstance(raw, dict) else None)
        or (raw.get("source") if isinstance(raw, dict) else None)
        or props.get("source_node_id")
        or props.get("source_id")
        or props.get("source")
    )
    target = _coerce_string(
        (raw.get("target_node_id") if isinstance(raw, dict) else None)
        or (raw.get("target_id") if isinstance(raw, dict) else None)
        or (raw.get("target") if isinstance(raw, dict) else None)
        or props.get("target_node_id")
        or props.get("target_id")
        or props.get("target")
    )
    relationship = _coerce_string(
        (raw.get("relationship_name") if isinstance(raw, dict) else None)
        or (raw.get("relation_type") if isinstance(raw, dict) else None)
        or (raw.get("relationship") if isinstance(raw, dict) else None)
        or (raw.get("relationship_type") if isinstance(raw, dict) else None)
        or (raw.get("name") if isinstance(raw, dict) else None)
        or props.get("relationship_name")
        or props.get("relation_type")
        or props.get("relationship")
        or props.get("name")
    )

    if not (source and target and relationship):
        return None

    return {
        "source_node_id": source,
        "target_node_id": target,
        "relationship_name": relationship,
    }


def normalize_knowledge_graph_payload(raw: Any) -> dict[str, Any]:
    raw = _model_dump(raw)
    if not isinstance(raw, dict):
        raw = {}

    payload = raw.get("knowledge_graph") or raw.get("graph") or raw
    payload = _model_dump(payload)
    if not isinstance(payload, dict):
        payload = {}

    raw_nodes = payload.get("nodes") or payload.get("entities") or payload.get("vertices") or []
    raw_edges = (
        payload.get("edges")
        or payload.get("relationships")
        or payload.get("relations")
        or payload.get("triplets")
        or []
    )

    nodes = [normalize_knowledge_graph_node(node) for node in raw_nodes if node is not None]
    node_ids = {node["id"] for node in nodes if node["id"]}

    edges = []
    for edge in raw_edges:
        normalized_edge = normalize_knowledge_graph_edge(edge)
        if not normalized_edge:
            continue
        if normalized_edge["source_node_id"] not in node_ids:
            continue
        if normalized_edge["target_node_id"] not in node_ids:
            continue
        edges.append(normalized_edge)

    return {
        "summary": _coerce_string(payload.get("summary")),
        "description": _coerce_string(payload.get("description")),
        "nodes": nodes,
        "edges": edges,
    }


def extract_json_payload(text: str) -> Any:
    decoder = json.JSONDecoder()
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    candidates.extend(match.strip() for match in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL))

    if "{" in stripped and "}" in stripped:
        candidates.append(stripped[stripped.find("{") : stripped.rfind("}") + 1])
    if "[" in stripped and "]" in stripped:
        candidates.append(stripped[stripped.find("[") : stripped.rfind("]") + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        for index, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                payload, _ = decoder.raw_decode(candidate[index:])
                return payload
            except json.JSONDecodeError:
                continue

    raise ValueError("Unable to extract JSON payload from LLM response")


def extract_completion_payload(response: Any) -> Any:
    response = _model_dump(response)
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    choices = choices or []
    if not choices:
        raise ValueError("LLM response did not contain choices")

    choice = _model_dump(choices[0])
    if not isinstance(choice, dict):
        raise ValueError("Unsupported choice payload")

    message = _model_dump(choice.get("message") or {})
    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        tool_call = _model_dump(tool_call)
        function = _model_dump(tool_call.get("function") or {})
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments.strip():
            return extract_json_payload(arguments)
        if isinstance(arguments, dict):
            return arguments

    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(
            _coerce_string(item.get("text") if isinstance(item, dict) else item)
            for item in content
            if _coerce_string(item.get("text") if isinstance(item, dict) else item)
        )
    if isinstance(content, str) and content.strip():
        return extract_json_payload(content)

    text = choice.get("text")
    if isinstance(text, str) and text.strip():
        return extract_json_payload(text)

    raise ValueError("LLM response did not contain JSON content")


def build_knowledge_graph_messages(text_input: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": f"{system_prompt}\n\n{KNOWLEDGE_GRAPH_JSON_INSTRUCTIONS}",
        },
        {"role": "user", "content": text_input},
    ]


async def request_raw_knowledge_graph(adapter: Any, text_input: str, system_prompt: str) -> Any:
    import litellm

    kwargs = {
        "model": getattr(adapter, "model", None),
        "messages": build_knowledge_graph_messages(text_input, system_prompt),
        "api_key": getattr(adapter, "api_key", None),
        "api_base": getattr(adapter, "endpoint", None),
        "api_version": getattr(adapter, "api_version", None),
        "max_completion_tokens": getattr(adapter, "max_completion_tokens", None),
        "max_retries": 1,
        "temperature": 0,
    }
    kwargs = {key: value for key, value in kwargs.items() if value not in (None, "")}

    try:
        response = await litellm.acompletion(response_format={"type": "json_object"}, **kwargs)
    except Exception:
        response = await litellm.acompletion(**kwargs)

    return extract_completion_payload(response)


async def fallback_knowledge_graph_response(
    adapter: Any,
    text_input: str,
    system_prompt: str,
    response_model: Any,
) -> Any:
    raw_payload = await request_raw_knowledge_graph(adapter, text_input, system_prompt)
    normalized_payload = normalize_knowledge_graph_payload(raw_payload)
    return response_model.model_validate(normalized_payload)


def install_groq_knowledge_graph_patch(groq_routed: bool) -> None:
    if not groq_routed:
        return

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
        GenericAPIAdapter,
    )
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
        get_llm_client,
    )
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
        OpenAIAdapter,
    )

    def patch_class(adapter_class: type[Any]) -> None:
        if getattr(adapter_class, GROQ_KG_PATCH_FLAG, False):
            return

        original = adapter_class.acreate_structured_output

        @functools.wraps(original)
        async def patched(self, text_input: str, system_prompt: str, response_model: Any, **kwargs):
            if should_bypass_knowledge_graph_structured_output(self, response_model):
                return await fallback_knowledge_graph_response(
                    self,
                    text_input=text_input,
                    system_prompt=system_prompt,
                    response_model=response_model,
                )
            try:
                return await original(self, text_input, system_prompt, response_model, **kwargs)
            except Exception as exc:
                if not should_bypass_knowledge_graph_structured_output(self, response_model):
                    raise
                try:
                    return await fallback_knowledge_graph_response(
                        self,
                        text_input=text_input,
                        system_prompt=system_prompt,
                        response_model=response_model,
                    )
                except Exception as fallback_exc:
                    raise fallback_exc from exc

        adapter_class.acreate_structured_output = patched
        setattr(adapter_class, GROQ_KG_PATCH_FLAG, True)

    patch_class(GenericAPIAdapter)
    patch_class(OpenAIAdapter)

    if not getattr(LLMGateway, GROQ_KG_GATEWAY_PATCH_FLAG, False):
        original_gateway = LLMGateway.acreate_structured_output

        async def patched_gateway(
            text_input: str,
            system_prompt: str,
            response_model: Any,
            **kwargs,
        ):
            llm_client = get_llm_client()
            if should_bypass_knowledge_graph_structured_output(llm_client, response_model):
                return await fallback_knowledge_graph_response(
                    llm_client,
                    text_input=text_input,
                    system_prompt=system_prompt,
                    response_model=response_model,
                )
            return await original_gateway(
                text_input=text_input,
                system_prompt=system_prompt,
                response_model=response_model,
                **kwargs,
            )

        LLMGateway.acreate_structured_output = staticmethod(patched_gateway)
        setattr(LLMGateway, GROQ_KG_GATEWAY_PATCH_FLAG, True)


def resolve_llm_provider(payload: dict[str, Any]) -> str:
    provider = _coerce_string(_first_present(payload.get("llm_provider"), os.environ.get("LLM_PROVIDER"))).lower()

    if provider == "groq":
        return "openai"
    if provider:
        return provider
    return "openai"


def is_groq_routed(payload: dict[str, Any]) -> bool:
    endpoint = _coerce_string(_first_present(payload.get("llm_endpoint"), os.environ.get("LLM_BASE_URL"))).lower()
    model = _coerce_string(
        _first_present(
            payload.get("llm_model"),
            os.environ.get("REPORT_LLM_MODEL"),
            os.environ.get("LLM_MODEL_NAME"),
        )
    ).lower()
    provider = _coerce_string(_first_present(payload.get("llm_provider"), os.environ.get("LLM_PROVIDER"))).lower()
    return "api.groq.com" in endpoint or model.startswith("groq/") or provider == "groq"


def normalize_llm_model(model: str | None, groq_routed: bool) -> str | None:
    if not model:
        return model
    if groq_routed and not model.startswith("groq/"):
        return f"groq/{model}"
    return model


def resolve_embedding_settings(payload: dict[str, Any], groq_routed: bool) -> dict[str, Any]:
    provider = str(payload.get("embedding_provider") or os.environ.get("EMBEDDING_PROVIDER") or "").lower()
    model = payload.get("embedding_model") or os.environ.get("EMBEDDING_MODEL")
    dimensions = payload.get("embedding_dimensions") or os.environ.get("EMBEDDING_DIMENSIONS")
    endpoint = payload.get("embedding_endpoint") or os.environ.get("EMBEDDING_ENDPOINT")
    api_key = payload.get("embedding_api_key") or os.environ.get("EMBEDDING_API_KEY")
    api_version = payload.get("embedding_api_version") or os.environ.get("EMBEDDING_API_VERSION")

    if not provider and groq_routed:
        provider = "fastembed"
    if provider == "fastembed":
        model = model or "BAAI/bge-small-en-v1.5"
        dimensions = int(dimensions or 384)
    elif dimensions is not None:
        dimensions = int(dimensions)

    return {
        "embedding_provider": provider or None,
        "embedding_model": model,
        "embedding_dimensions": dimensions,
        "embedding_endpoint": endpoint,
        "embedding_api_key": api_key,
        "embedding_api_version": api_version,
    }


def resolve_llm_instructor_mode(payload: dict[str, Any], groq_routed: bool) -> str:
    mode = str(payload.get("llm_instructor_mode") or os.environ.get("LLM_INSTRUCTOR_MODE") or "").strip()
    if mode:
        return mode
    if groq_routed:
        return "json_schema_mode"
    return ""


def apply_non_null_settings(target: Any, settings: dict[str, Any]) -> None:
    for key, value in settings.items():
        if value is None:
            continue
        object.__setattr__(target, key, value)


def configure_cognee(payload: dict[str, Any]):
    os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    import cognee
    from cognee.infrastructure.llm import get_llm_config
    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config

    llm_api_key = _first_present(payload.get("llm_api_key"), os.environ.get("LLM_API_KEY"))
    llm_endpoint = _first_present(payload.get("llm_endpoint"), os.environ.get("LLM_BASE_URL"))
    llm_provider = resolve_llm_provider(payload)
    groq_routed = is_groq_routed(payload)
    llm_model = normalize_llm_model(
        _first_present(
            payload.get("llm_model"),
            os.environ.get("REPORT_LLM_MODEL"),
            os.environ.get("LLM_MODEL_NAME"),
        ),
        groq_routed,
    )
    embedding_settings = resolve_embedding_settings(payload, groq_routed)
    llm_instructor_mode = resolve_llm_instructor_mode(payload, groq_routed)
    install_groq_knowledge_graph_patch(groq_routed)
    llm_config = get_llm_config()
    embedding_config = get_embedding_config()

    cognee.config.system_root_directory = str(Path.cwd() / ".cognee_system")
    cognee.config.data_root_directory = str(Path.cwd() / ".data_storage")
    cognee.config.set_llm_provider(llm_provider)

    if llm_api_key:
        cognee.config.set_llm_api_key(llm_api_key)
    if llm_endpoint:
        cognee.config.set_llm_endpoint(llm_endpoint)
    if llm_model:
        cognee.config.set_llm_model(llm_model)
    object.__setattr__(llm_config, "llm_instructor_mode", llm_instructor_mode)

    apply_non_null_settings(embedding_config, embedding_settings)

    return cognee


async def fetch_graph_data(get_graph_engine: Any) -> tuple[Any, Any]:
    graph_engine = get_graph_engine()
    if inspect.isawaitable(graph_engine):
        graph_engine = await graph_engine

    graph_data = graph_engine.get_graph_data()
    if inspect.isawaitable(graph_data):
        graph_data = await graph_data

    return graph_data


async def build_graph(payload: dict[str, Any]) -> dict[str, Any]:
    cognee = configure_cognee(payload)
    from cognee.infrastructure.databases.graph import get_graph_engine

    dataset_name = payload.get("dataset_name") or payload.get("graph_id") or "main_dataset"
    texts = payload.get("texts") or []
    await cognee.add(texts, dataset_name=dataset_name)
    await cognee.cognify(datasets=dataset_name)

    nodes, edges = await fetch_graph_data(get_graph_engine)
    return {
        "graph_id": payload.get("graph_id") or dataset_name,
        "nodes": [normalize_node(node) for node in nodes],
        "edges": [normalize_edge(edge) for edge in edges],
    }


async def get_graph_data(payload: dict[str, Any]) -> dict[str, Any]:
    configure_cognee(payload)
    from cognee.infrastructure.databases.graph import get_graph_engine

    nodes, edges = await fetch_graph_data(get_graph_engine)
    return {
        "graph_id": payload.get("graph_id"),
        "nodes": [normalize_node(node) for node in nodes],
        "edges": [normalize_edge(edge) for edge in edges],
    }


async def search_graph(payload: dict[str, Any]) -> dict[str, Any]:
    cognee = configure_cognee(payload)
    from cognee.modules.search.types.SearchType import SearchType

    results = await cognee.search(
        payload["query"],
        query_type=SearchType.CHUNKS,
        datasets=payload.get("dataset_name") or payload.get("graph_id"),
        top_k=payload.get("limit", 10),
        only_context=False,
    ) or []

    normalized_results = []
    for item in results:
        normalized = normalize_search_result(item)
        if normalized.get("search_result") in {None, ""}:
            continue
        normalized_results.append(normalized)

    return {
        "graph_id": payload.get("graph_id"),
        "results": normalized_results,
    }


async def main() -> int:
    operation = sys.argv[1] if len(sys.argv) > 1 else ""
    payload = load_payload()

    try:
        if operation == "build_graph":
            result = await build_graph(payload)
        elif operation == "get_graph_data":
            result = await get_graph_data(payload)
        elif operation == "search_graph":
            result = await search_graph(payload)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        emit({"success": True, "data": result})
        return 0
    except Exception as exc:  # pragma: no cover - surfaced to caller
        emit({"success": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))