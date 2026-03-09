"""Projection bundle adapter for MiroFish integrated mode."""

from typing import Any, Dict, List


class ProjectionBundleAdapter:
    """Converts a social projection bundle into ontology + synthetic text."""

    REQUIRED_COLLECTIONS = (
        "actors",
        "organizations",
        "social_edges",
        "context_locations",
        "event_seeds",
        "world_rules",
    )
    SPEAKER_ENTITY_TYPES = ["Actor", "Organization"]

    def adapt(self, bundle: Dict[str, Any], additional_context: str | None = None) -> Dict[str, Any]:
        normalized = self._normalize_bundle(bundle)
        return {
            "bundle": normalized,
            "ontology": self._build_ontology(normalized),
            "extracted_text": self._build_text(normalized, additional_context),
            "analysis_summary": self._build_summary(normalized),
            "recommended_prepare_entity_types": list(self.SPEAKER_ENTITY_TYPES),
        }

    def _normalize_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(bundle, dict):
            raise ValueError("projection_bundle must be a JSON object")

        normalized = dict(bundle)
        for key in self.REQUIRED_COLLECTIONS:
            value = normalized.get(key) or []
            if not isinstance(value, list):
                raise ValueError(f"projection_bundle.{key} must be a list")
            normalized[key] = value

        if not normalized["actors"] and not normalized["organizations"]:
            raise ValueError("projection_bundle must contain at least one actor or organization")

        self._validate_entities(normalized["actors"], "actors")
        self._validate_entities(normalized["organizations"], "organizations")
        self._validate_edges(normalized["social_edges"])
        return normalized

    def _validate_entities(self, entities: List[Dict[str, Any]], field_name: str) -> None:
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise ValueError(f"projection_bundle.{field_name}[{index}] must be an object")
            missing = [key for key in ("id", "name") if not entity.get(key)]
            if missing:
                raise ValueError(f"projection_bundle.{field_name}[{index}] missing fields: {', '.join(missing)}")

    def _validate_edges(self, edges: List[Dict[str, Any]]) -> None:
        for index, edge in enumerate(edges):
            if not isinstance(edge, dict):
                raise ValueError(f"projection_bundle.social_edges[{index}] must be an object")
            missing = [key for key in ("source_id", "target_id") if not edge.get(key)]
            if missing:
                raise ValueError(f"projection_bundle.social_edges[{index}] missing fields: {', '.join(missing)}")

    def _build_ontology(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        actor_examples = [item["name"] for item in bundle["actors"][:3]]
        org_examples = [item["name"] for item in bundle["organizations"][:3]]
        return {
            "entity_types": [
                {
                    "name": "Actor",
                    "description": "A speaker-capable individual or representative actor.",
                    "attributes": self._attributes("canonical_id", "canonical_type", "speaker_mode", "represented_entity_id", "role", "stance"),
                    "examples": actor_examples,
                },
                {
                    "name": "Organization",
                    "description": "A collective actor, institution, or official channel.",
                    "attributes": self._attributes("canonical_id", "canonical_type", "speaker_mode", "represented_entity_id", "domain", "public_position"),
                    "examples": org_examples,
                },
            ],
            "edge_types": [
                {
                    "name": "SOCIAL_LINK",
                    "description": "A social or political relationship between projected entities.",
                    "source_targets": [
                        {"source": "Actor", "target": "Actor"},
                        {"source": "Actor", "target": "Organization"},
                        {"source": "Organization", "target": "Actor"},
                        {"source": "Organization", "target": "Organization"},
                    ],
                    "attributes": self._attributes("relation_type", "status", "strength"),
                }
            ],
            "analysis_summary": self._build_summary(bundle),
        }

    def _build_summary(self, bundle: Dict[str, Any]) -> str:
        return (
            "Projection bundle import: "
            f"{len(bundle['actors'])} actors, "
            f"{len(bundle['organizations'])} organizations, "
            f"{len(bundle['social_edges'])} social edges, "
            f"{len(bundle['context_locations'])} context locations, "
            f"{len(bundle['event_seeds'])} event seeds, "
            f"{len(bundle['world_rules'])} world rules."
        )

    def _build_text(self, bundle: Dict[str, Any], additional_context: str | None) -> str:
        entity_index = {
            item["id"]: item.get("name", item["id"])
            for item in bundle["actors"] + bundle["organizations"]
            if isinstance(item, dict) and item.get("id")
        }
        lines = [
            "# Projection Bundle Import",
            f"schema_version: {bundle.get('schema_version', '1.0')}",
            f"world_id: {bundle.get('world_id', 'unspecified')}",
            f"world_version: {bundle.get('world_version', 'unspecified')}",
            f"scenario_id: {bundle.get('scenario_id', 'unspecified')}",
            "",
        ]
        lines += self._render_section("Actors", "Actor", bundle["actors"])
        lines += self._render_section("Organizations", "Organization", bundle["organizations"])
        lines += self._render_section("Context Locations", "ContextLocation", bundle["context_locations"])
        lines += self._render_section("Event Seeds", "EventSeed", bundle["event_seeds"])
        lines += self._render_section("World Rules", "WorldRule", bundle["world_rules"])
        lines += self._render_edges(bundle["social_edges"], entity_index)
        if additional_context:
            lines += ["## Additional Context", additional_context.strip(), ""]
        return "\n".join(lines).strip()

    def _render_section(self, title: str, item_type: str, items: List[Dict[str, Any]]) -> List[str]:
        lines = [f"## {title}"]
        if not items:
            return lines + ["None", ""]
        for item in items:
            lines.append(f"{item_type}: {item.get('name', item.get('id', 'unnamed'))}")
            for key in sorted(item.keys()):
                if key == "name" or item.get(key) in (None, "", [], {}):
                    continue
                lines.append(f"- {key}: {self._stringify(item[key])}")
            lines.append("")
        return lines

    def _render_edges(self, edges: List[Dict[str, Any]], entity_index: Dict[str, str]) -> List[str]:
        lines = ["## Social Edges"]
        if not edges:
            return lines + ["None", ""]
        for edge in edges:
            relation = edge.get("relation_type") or edge.get("type") or "RELATED_TO"
            source_id = edge["source_id"]
            target_id = edge["target_id"]
            source_name = entity_index.get(source_id, source_id)
            target_name = entity_index.get(target_id, target_id)
            lines.append(f"SocialEdge: {source_name} -> {target_name} ({relation})")
            lines.append(f"- source_id: {source_id}")
            lines.append(f"- target_id: {target_id}")
            for key in sorted(edge.keys()):
                if key in {"source_id", "target_id"} or edge.get(key) in (None, "", [], {}):
                    continue
                lines.append(f"- {key}: {self._stringify(edge[key])}")
            lines.append("")
        return lines

    def _attributes(self, *names: str) -> List[Dict[str, str]]:
        return [{"name": name, "type": "text", "description": name.replace("_", " ")} for name in names]

    def _stringify(self, value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        if isinstance(value, dict):
            return "; ".join(f"{key}={self._stringify(val)}" for key, val in value.items())
        return str(value)