"""Cognee-backed graph builder using an isolated sidecar runtime."""

from __future__ import annotations

import json
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
                progress_callback(f"准备 Cognee 数据批次 {batch_num}/{total_batches}...", min((batch_num / total_batches) * 0.4, 0.4))

        graph_data = self.sidecar.build_graph(
            graph_id=graph_id,
            graph_name=graph_id,
            texts=chunks,
            llm_api_key=Config.LLM_API_KEY,
            llm_endpoint=Config.LLM_BASE_URL,
            llm_model=Config.get_stage_model("ontology"),
        )
        self._graph_cache[graph_id] = self._format_graph_data(graph_data)

        if progress_callback:
            progress_callback("Cognee 图谱构建完成", 1.0)

        return [graph_id]

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        if graph_id in self._graph_cache:
            return self._graph_cache[graph_id]
        return self._format_graph_data(self.sidecar.get_graph_data(graph_id))

    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600,
    ):
        del episode_uuids, timeout
        if progress_callback:
            progress_callback("Cognee 数据已同步，无需额外等待", 1.0)

    def delete_graph(self, graph_id: str):
        self.sidecar.delete_graph(graph_id)
        self._graph_cache.pop(graph_id, None)

    def _format_graph_data(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        return {
            "graph_id": graph_data.get("graph_id"),
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }