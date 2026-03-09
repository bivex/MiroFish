"""Simplified report/search tools for Cognee backend."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .cognee_entity_reader import CogneeEntityReader
from .cognee_sidecar_client import CogneeSidecarClient


@dataclass
class SearchResult:
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {"facts": self.facts, "edges": self.edges, "nodes": self.nodes, "query": self.query, "total_count": self.total_count}

    def to_text(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class NodeInfo:
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"uuid": self.uuid, "name": self.name, "labels": self.labels, "summary": self.summary, "attributes": self.attributes}


@dataclass
class EdgeInfo:
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class InsightForgeResult:
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2)


@dataclass
class PanoramaResult:
    query: str
    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        return json.dumps({
            "query": self.query,
            "all_nodes": [node.to_dict() for node in self.all_nodes],
            "all_edges": [edge.to_dict() for edge in self.all_edges],
            "active_facts": self.active_facts,
        }, ensure_ascii=False, indent=2)


@dataclass
class InterviewResult:
    interview_topic: str
    interview_questions: List[str]
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    interviews: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = "Cognee backend 暂不支持 interview_agents，返回空结果。"

    def to_text(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2)


class CogneeToolsService:
    def __init__(self, sidecar_client: Optional[CogneeSidecarClient] = None, entity_reader: Optional[CogneeEntityReader] = None, **_: Any):
        self.sidecar = sidecar_client or CogneeSidecarClient()
        self.entity_reader = entity_reader or CogneeEntityReader()

    def search_graph(self, graph_id: str, query: str, limit: int = 10, scope: str = "edges") -> SearchResult:
        search_data = self.sidecar.search_graph(graph_id, query, limit)
        graph_data = self.sidecar.get_graph_data(graph_id)
        lowered = query.lower()
        nodes = [node for node in graph_data.get("nodes", []) if lowered in json.dumps(node, ensure_ascii=False).lower()][:limit]
        edges = [edge for edge in graph_data.get("edges", []) if lowered in json.dumps(edge, ensure_ascii=False).lower()][:limit]
        facts = [str(item.get("search_result")) for item in search_data.get("results", [])[:limit]]
        return SearchResult(facts=facts, edges=edges, nodes=nodes, query=query, total_count=len(facts))

    def quick_search(self, graph_id: str, query: str, limit: int = 10) -> SearchResult:
        return self.search_graph(graph_id, query, limit=limit)

    def get_entities_by_type(self, graph_id: str, entity_type: str) -> List[NodeInfo]:
        entities = self.entity_reader.get_entities_by_type(graph_id, entity_type, enrich_with_edges=False)
        return [NodeInfo(uuid=e.uuid, name=e.name, labels=e.labels, summary=e.summary, attributes=e.attributes) for e in entities]

    def get_entity_summary(self, graph_id: str, entity_name: str) -> Dict[str, Any]:
        graph_data = self.sidecar.get_graph_data(graph_id)
        for node in graph_data.get("nodes", []):
            if node.get("name") == entity_name:
                related_edges = [
                    edge for edge in graph_data.get("edges", [])
                    if edge.get("source_node_uuid") == node["uuid"] or edge.get("target_node_uuid") == node["uuid"]
                ]
                return {"entity": node, "related_edges": related_edges, "related_fact_count": len(related_edges)}
        return {"entity": None, "related_edges": [], "related_fact_count": 0}

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        graph_data = self.sidecar.get_graph_data(graph_id)
        entity_type_counts: Dict[str, int] = {}
        for node in graph_data.get("nodes", []):
            for label in node.get("labels", []):
                if label not in {"Entity", "Node"}:
                    entity_type_counts[label] = entity_type_counts.get(label, 0) + 1
        return {
            "graph_id": graph_id,
            "node_count": len(graph_data.get("nodes", [])),
            "edge_count": len(graph_data.get("edges", [])),
            "entity_type_counts": entity_type_counts,
        }

    def panorama_search(self, graph_id: str, query: str, include_expired: bool = True, limit: int = 50) -> PanoramaResult:
        graph_data = self.sidecar.get_graph_data(graph_id)
        lowered = query.lower()
        nodes = [NodeInfo(**{k: node[k] for k in ["uuid", "name", "labels", "summary", "attributes"]}) for node in graph_data.get("nodes", []) if lowered in json.dumps(node, ensure_ascii=False).lower()][:limit]
        edges = [EdgeInfo(**{k: edge[k] for k in ["uuid", "name", "fact", "source_node_uuid", "target_node_uuid"]}) for edge in graph_data.get("edges", []) if lowered in json.dumps(edge, ensure_ascii=False).lower()][:limit]
        return PanoramaResult(query=query, all_nodes=nodes, all_edges=edges, active_facts=[edge.fact for edge in edges])

    def insight_forge(self, graph_id: str, query: str, simulation_requirement: str, report_context: str = "", max_sub_queries: int = 5) -> InsightForgeResult:
        quick = self.search_graph(graph_id, query, limit=max_sub_queries)
        pano = self.panorama_search(graph_id, query, limit=max_sub_queries)
        return InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[query],
            semantic_facts=quick.facts,
            entity_insights=[node.to_dict() for node in pano.all_nodes],
            relationship_chains=[edge.fact for edge in pano.all_edges],
        )

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None,
    ) -> InterviewResult:
        questions = custom_questions or [interview_requirement]
        return InterviewResult(interview_topic=interview_requirement, interview_questions=questions)