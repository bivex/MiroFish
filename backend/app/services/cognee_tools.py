"""Simplified report/search tools for Cognee backend."""

from __future__ import annotations

import json
import re
from collections import Counter
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

    @staticmethod
    def _dedupe_keep_order(values: List[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        return [term for term in re.findall(r"[a-z0-9_]+", text.lower()) if len(term) >= 3]

    def _score_payload(self, payload: Any, lowered_query: str, query_terms: List[str]) -> int:
        haystack = json.dumps(payload, ensure_ascii=False).lower()
        score = 0
        if lowered_query and lowered_query in haystack:
            score += len(query_terms) + 3
        score += sum(1 for term in query_terms if term in haystack)
        return score

    @staticmethod
    def _action_to_dict(action: Any) -> Dict[str, Any]:
        if hasattr(action, "to_dict"):
            return action.to_dict()
        return {
            "round_num": getattr(action, "round_num", 0),
            "timestamp": getattr(action, "timestamp", ""),
            "platform": getattr(action, "platform", ""),
            "agent_id": getattr(action, "agent_id", 0),
            "agent_name": getattr(action, "agent_name", ""),
            "action_type": getattr(action, "action_type", ""),
            "action_args": getattr(action, "action_args", {}),
            "result": getattr(action, "result", None),
            "success": getattr(action, "success", True),
        }

    def _format_action_fact(self, action: Any) -> str:
        data = self._action_to_dict(action)
        agent_name = data.get("agent_name") or f"Agent {data.get('agent_id', 0)}"
        platform = data.get("platform") or "unknown"
        action_type = data.get("action_type") or "action"
        round_num = data.get("round_num") or 0
        result = data.get("result")
        if isinstance(result, dict):
            result_text = json.dumps(result, ensure_ascii=False, sort_keys=True)
        else:
            result_text = str(result or "")
        result_text = " ".join(result_text.split())[:240]
        suffix = f": {result_text}" if result_text else ""
        return f"[round {round_num}] [{platform}] {agent_name} {action_type}{suffix}"

    def get_runtime_evidence(self, simulation_id: str, limit: int = 10) -> Dict[str, Any]:
        from .simulation_runner import SimulationRunner

        run_state = SimulationRunner.get_run_state(simulation_id)
        actions = SimulationRunner.get_all_actions(simulation_id)[:limit]
        if not actions and run_state:
            actions = list(getattr(run_state, "recent_actions", [])[:limit])

        total_actions = len(actions)
        current_round = 0
        runner_status = "idle"
        if run_state:
            total_actions = max(
                total_actions,
                int(getattr(run_state, "twitter_actions_count", 0)) + int(getattr(run_state, "reddit_actions_count", 0)),
            )
            current_round = int(getattr(run_state, "current_round", 0))
            runner_status_value = getattr(run_state, "runner_status", "idle")
            runner_status = getattr(runner_status_value, "value", runner_status_value)

        action_facts = [self._format_action_fact(action) for action in actions[:limit]]
        action_dicts = [self._action_to_dict(action) for action in actions[:limit]]
        top_agents = Counter(data.get("agent_name") or f"Agent {data.get('agent_id', 0)}" for data in action_dicts)
        top_action_types = Counter(data.get("action_type") or "action" for data in action_dicts)

        has_runtime_evidence = bool(total_actions or current_round or action_facts)
        message = ""
        if not has_runtime_evidence:
            message = (
                "Simulation has not produced runtime evidence yet. "
                "Start the simulation and wait for actions before generating a report."
            )

        return {
            "simulation_id": simulation_id,
            "runner_status": runner_status,
            "current_round": current_round,
            "total_actions": total_actions,
            "has_runtime_evidence": has_runtime_evidence,
            "recent_actions": action_dicts,
            "action_facts": action_facts,
            "top_agents": [{"name": name, "count": count} for name, count in top_agents.most_common(5)],
            "top_action_types": [{"name": name, "count": count} for name, count in top_action_types.most_common(5)],
            "message": message,
        }

    def _get_runtime_action_facts(self, simulation_id: Optional[str], query: str, limit: int) -> List[str]:
        if not simulation_id:
            return []

        runtime = self.get_runtime_evidence(simulation_id, limit=max(limit, 5))
        action_facts = runtime.get("action_facts", [])
        if not action_facts:
            return []

        lowered_query = query.lower().strip()
        query_terms = self._extract_terms(query)
        scored: List[tuple[int, str]] = []
        for fact in action_facts:
            score = self._score_payload(fact, lowered_query, query_terms)
            if score > 0:
                scored.append((score, fact))

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            return [fact for _, fact in scored[:limit]]

        return action_facts[:limit]

    def search_graph(self, graph_id: str, query: str, limit: int = 10, scope: str = "edges", simulation_id: Optional[str] = None) -> SearchResult:
        graph_data = self.sidecar.get_graph_data(graph_id)
        lowered = query.lower().strip()
        query_terms = self._extract_terms(query)
        node_matches = []
        edge_matches = []
        for node in graph_data.get("nodes", []):
            score = self._score_payload(node, lowered, query_terms)
            if score > 0:
                node_matches.append((score, node))
        for edge in graph_data.get("edges", []):
            score = self._score_payload(edge, lowered, query_terms)
            if score > 0:
                edge_matches.append((score, edge))
        node_matches.sort(key=lambda item: item[0], reverse=True)
        edge_matches.sort(key=lambda item: item[0], reverse=True)
        nodes = [node for _, node in node_matches[:limit]]
        edges = [edge for _, edge in edge_matches[:limit]]
        try:
            search_data = self.sidecar.search_graph(graph_id, query, limit)
            raw_results = search_data.get("results") or []
        except Exception:
            raw_results = []

        facts = [
            str(item.get("search_result"))
            for item in raw_results[:limit]
            if isinstance(item, dict) and item.get("search_result") is not None
        ]
        runtime_facts = self._get_runtime_action_facts(simulation_id=simulation_id, query=query, limit=limit)
        if runtime_facts:
            facts = self._dedupe_keep_order(runtime_facts + facts)
        if not facts:
            facts = [
                edge.get("fact") or edge.get("name") or edge.get("uuid")
                for edge in edges
                if edge.get("fact") or edge.get("name") or edge.get("uuid")
            ][:limit]
        if not facts:
            facts = [
                node.get("summary") or node.get("name") or node.get("uuid")
                for node in nodes
                if node.get("summary") or node.get("name") or node.get("uuid")
            ][:limit]
        facts = self._dedupe_keep_order(facts)[:limit]
        return SearchResult(facts=facts, edges=edges, nodes=nodes, query=query, total_count=len(facts))

    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        simulation_id: Optional[str] = None,
    ) -> SearchResult:
        return self.search_graph(graph_id, query, limit=limit, simulation_id=simulation_id)

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
        relation_type_counts: Dict[str, int] = {}
        for node in graph_data.get("nodes", []):
            for label in node.get("labels", []):
                if label not in {"Entity", "Node"}:
                    entity_type_counts[label] = entity_type_counts.get(label, 0) + 1
        for edge in graph_data.get("edges", []):
            relation = edge.get("name") or edge.get("edge_name") or "related_to"
            relation_type_counts[relation] = relation_type_counts.get(relation, 0) + 1
        return {
            "graph_id": graph_id,
            "node_count": len(graph_data.get("nodes", [])),
            "edge_count": len(graph_data.get("edges", [])),
            "entity_type_counts": entity_type_counts,
            "relation_type_counts": relation_type_counts,
            "total_nodes": len(graph_data.get("nodes", [])),
            "total_edges": len(graph_data.get("edges", [])),
            "entity_types": entity_type_counts,
            "relation_types": relation_type_counts,
        }

    def get_simulation_context(
        self,
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30,
        simulation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        graph_data = self.sidecar.get_graph_data(graph_id)
        stats = self.get_graph_statistics(graph_id)
        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit,
            simulation_id=simulation_id,
        )
        runtime_evidence = self.get_runtime_evidence(simulation_id, limit=min(limit, 10)) if simulation_id else {
            "has_runtime_evidence": False,
            "recent_actions": [],
            "action_facts": [],
            "message": "",
        }

        entities = []
        for node in graph_data.get("nodes", []):
            labels = [label for label in node.get("labels", []) if label not in {"Entity", "Node"}]
            entity_type = labels[0] if labels else (node.get("attributes") or {}).get("type")
            if not entity_type:
                continue
            entities.append(
                {
                    "name": node.get("name", ""),
                    "type": entity_type,
                    "summary": node.get("summary", ""),
                }
            )

        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": self._dedupe_keep_order(runtime_evidence.get("action_facts", []) + search_result.facts)[:limit],
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities),
            "runtime_evidence": runtime_evidence,
        }

    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50,
        simulation_id: Optional[str] = None,
    ) -> PanoramaResult:
        graph_data = self.sidecar.get_graph_data(graph_id)
        lowered = query.lower().strip()
        query_terms = self._extract_terms(query)
        node_matches = []
        edge_matches = []
        for node in graph_data.get("nodes", []):
            score = self._score_payload(node, lowered, query_terms)
            if score > 0:
                node_matches.append((score, node))
        for edge in graph_data.get("edges", []):
            score = self._score_payload(edge, lowered, query_terms)
            if score > 0:
                edge_matches.append((score, edge))
        node_matches.sort(key=lambda item: item[0], reverse=True)
        edge_matches.sort(key=lambda item: item[0], reverse=True)

        nodes = [NodeInfo(**{k: node[k] for k in ["uuid", "name", "labels", "summary", "attributes"]}) for _, node in node_matches[:limit]]
        edges = [EdgeInfo(**{k: edge[k] for k in ["uuid", "name", "fact", "source_node_uuid", "target_node_uuid"]}) for _, edge in edge_matches[:limit]]
        runtime_facts = self._get_runtime_action_facts(simulation_id=simulation_id, query=query, limit=limit)
        active_facts = self._dedupe_keep_order([edge.fact for edge in edges] + runtime_facts)[:limit]
        return PanoramaResult(query=query, all_nodes=nodes, all_edges=edges, active_facts=active_facts)

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5,
        simulation_id: Optional[str] = None,
    ) -> InsightForgeResult:
        quick = self.search_graph(graph_id, query, limit=max_sub_queries, simulation_id=simulation_id)
        pano = self.panorama_search(graph_id, query, limit=max_sub_queries, simulation_id=simulation_id)
        return InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[query],
            semantic_facts=quick.facts,
            entity_insights=[node.to_dict() for node in pano.all_nodes],
            relationship_chains=pano.active_facts,
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