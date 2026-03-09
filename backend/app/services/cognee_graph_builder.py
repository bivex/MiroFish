"""Cognee-backed graph builder using an isolated sidecar runtime."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config import Config
from .cognee_sidecar_client import CogneeSidecarClient


class CogneeGraphBuilderService:
    def __init__(self, sidecar_client: Optional[CogneeSidecarClient] = None, **_: Any):
        self.sidecar = sidecar_client or CogneeSidecarClient()
        self._graph_cache: Dict[str, Dict[str, Any]] = {}

    def create_graph(self, name: str) -> str:
        return f"mirofish_cognee_{uuid.uuid4().hex[:16]}"

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        workspace = self.sidecar.ensure_workspace(graph_id)
        (workspace / "ontology.json").write_text(json.dumps(ontology, ensure_ascii=False, indent=2), encoding="utf-8")

    def _semantic_types_path(self, graph_id: str) -> Path:
        return self.sidecar.ensure_workspace(graph_id) / "semantic_types.json"

    def _normalize_entity_name(self, value: str) -> str:
        return " ".join((value or "").strip().lower().split())

    def _extract_projection_semantic_types(self, chunks: List[str]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        pattern = re.compile(r"^\s*(Actor|Organization):\s*(.+?)\s*$", re.IGNORECASE)

        for chunk in chunks:
            for line in str(chunk).splitlines():
                match = pattern.match(line)
                if not match:
                    continue
                entity_type = match.group(1).capitalize()
                name = self._normalize_entity_name(match.group(2))
                if name:
                    mapping[name] = entity_type

        return mapping

    def _persist_projection_semantic_types(self, graph_id: str, chunks: List[str]) -> None:
        mapping = self._extract_projection_semantic_types(chunks)
        if not mapping:
            return
        self._semantic_types_path(graph_id).write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_projection_semantic_types(self, graph_id: str) -> Dict[str, str]:
        path = self._semantic_types_path(graph_id)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _apply_projection_semantic_types(self, graph_id: str, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        mapping = self._load_projection_semantic_types(graph_id)
        if not mapping:
            return graph_data

        nodes = []
        for node in graph_data.get("nodes", []):
            node_copy = dict(node)
            normalized_name = self._normalize_entity_name(node_copy.get("name", ""))
            semantic_type = mapping.get(normalized_name)
            if semantic_type:
                labels = list(node_copy.get("labels", []))
                if semantic_type not in labels:
                    labels.append(semantic_type)
                node_copy["labels"] = labels

                attributes = dict(node_copy.get("attributes", {}))
                if attributes.get("type") in (None, "", "Entity"):
                    attributes["type"] = semantic_type
                attributes["semantic_type"] = semantic_type
                node_copy["attributes"] = attributes

            nodes.append(node_copy)

        enriched = dict(graph_data)
        enriched["nodes"] = nodes
        return enriched

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        total_chunks = len(chunks)
        total_batches = max((total_chunks + batch_size - 1) // batch_size, 1)

        for i in range(0, total_chunks, batch_size):
            if progress_callback:
                batch_num = i // batch_size + 1
                progress_callback(f"Preparing Cognee data batch {batch_num}/{total_batches}...", min((batch_num / total_batches) * 0.4, 0.4))

        self._persist_projection_semantic_types(graph_id, chunks)

        graph_data = self.sidecar.build_graph(
            graph_id=graph_id,
            graph_name=graph_id,
            texts=chunks,
            llm_api_key=Config.LLM_API_KEY,
            llm_endpoint=Config.LLM_BASE_URL,
            llm_model=Config.get_stage_model("ontology"),
        )
        self._graph_cache[graph_id] = self._format_graph_data(graph_id, graph_data)

        if progress_callback:
            progress_callback("Cognee graph build complete", 1.0)

        return [graph_id]

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        if graph_id in self._graph_cache:
            return self._graph_cache[graph_id]
        return self._format_graph_data(graph_id, self.sidecar.get_graph_data(graph_id))

    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600,
    ):
        del episode_uuids, timeout
        if progress_callback:
            progress_callback("Cognee data is already synced; no extra wait required", 1.0)

    def delete_graph(self, graph_id: str):
        self.sidecar.delete_graph(graph_id)
        self._graph_cache.pop(graph_id, None)

    def _format_graph_data(self, graph_id: str, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        graph_data = self._apply_projection_semantic_types(graph_id, graph_data)
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        return {
            "graph_id": graph_data.get("graph_id"),
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }