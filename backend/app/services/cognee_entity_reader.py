"""Entity reader for Cognee graph data."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .cognee_graph_builder import CogneeGraphBuilderService
from .graph_entities import EntityNode, FilteredEntities


class CogneeEntityReader:
    def __init__(self, builder: Optional[CogneeGraphBuilderService] = None, **_: Any):
        self.builder = builder or CogneeGraphBuilderService()

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        return self.builder.get_graph_data(graph_id).get("nodes", [])

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        return self.builder.get_graph_data(graph_id).get("edges", [])

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        all_nodes = self.get_all_nodes(graph_id)
        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []
        node_map = {node["uuid"]: node for node in all_nodes}
        total_count = len(all_nodes)

        filtered_entities: List[EntityNode] = []
        entity_types_found = set()

        for node in all_nodes:
            labels = node.get("labels", [])
            custom_labels = [label for label in labels if label not in ["Entity", "Node"]]
            if not custom_labels:
                continue

            if defined_entity_types:
                matching_labels = [label for label in custom_labels if label in defined_entity_types]
                if not matching_labels:
                    continue

            entity_type = custom_labels[0]
            entity_types_found.add(entity_type)
            entity = EntityNode(
                uuid=node["uuid"],
                name=node.get("name", ""),
                labels=labels,
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
            )

            if enrich_with_edges:
                related_edges = []
                related_node_uuids = set()
                for edge in all_edges:
                    if edge["source_node_uuid"] == entity.uuid:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge.get("fact", ""),
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == entity.uuid:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge.get("fact", ""),
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])

                entity.related_edges = related_edges
                entity.related_nodes = [
                    {
                        "uuid": related["uuid"],
                        "name": related.get("name", ""),
                        "labels": related.get("labels", []),
                        "summary": related.get("summary", ""),
                    }
                    for related_uuid in related_node_uuids
                    if (related := node_map.get(related_uuid)) is not None
                ]

            filtered_entities.append(entity)

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    def get_entity_with_context(self, graph_id: str, entity_uuid: str) -> Optional[EntityNode]:
        result = self.filter_defined_entities(graph_id, enrich_with_edges=True)
        for entity in result.entities:
            if entity.uuid == entity_uuid:
                return entity
        return None

    def get_entities_by_type(self, graph_id: str, entity_type: str, enrich_with_edges: bool = True) -> List[EntityNode]:
        return self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges,
        ).entities