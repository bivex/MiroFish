import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_adapter_module():
    path = ROOT / "app" / "services" / "projection_bundle_adapter.py"
    spec = importlib.util.spec_from_file_location("mirofish_projection_bundle_adapter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_bundle():
    return {
        "schema_version": "1.0",
        "world_id": "world-1",
        "scenario_id": "scenario-1",
        "actors": [
            {
                "id": "actor-1",
                "name": "Captain Mira",
                "canonical_id": "char-1",
                "canonical_type": "Character",
                "speaker_mode": "individual",
                "role": "harbor captain",
            }
        ],
        "organizations": [
            {
                "id": "org-1",
                "name": "Harbor Council",
                "canonical_id": "faction-1",
                "canonical_type": "Faction",
                "speaker_mode": "official_account",
            }
        ],
        "social_edges": [
            {
                "source_id": "actor-1",
                "target_id": "org-1",
                "relation_type": "REPRESENTS",
                "status": "active",
            }
        ],
        "context_locations": [{"id": "loc-1", "name": "North Harbor"}],
        "event_seeds": [{"id": "event-1", "name": "Dock strike rumors"}],
        "world_rules": [{"id": "rule-1", "name": "Speech against the council is monitored"}],
    }


def test_adapter_builds_speaker_safe_projection_payload():
    module = load_adapter_module()
    adapter = module.ProjectionBundleAdapter()

    adapted = adapter.adapt(sample_bundle(), additional_context="Storm season is near.")

    entity_type_names = [item["name"] for item in adapted["ontology"]["entity_types"]]
    assert entity_type_names == ["Actor", "Organization"]
    assert adapted["recommended_prepare_entity_types"] == ["Actor", "Organization"]
    assert adapted["ontology"]["edge_types"][0]["name"] == "SOCIAL_LINK"
    assert "Captain Mira" in adapted["extracted_text"]
    assert "Harbor Council" in adapted["extracted_text"]
    assert "North Harbor" in adapted["extracted_text"]
    assert "Dock strike rumors" in adapted["extracted_text"]
    assert "Storm season is near." in adapted["extracted_text"]


def test_adapter_summary_reports_bundle_counts():
    module = load_adapter_module()
    adapter = module.ProjectionBundleAdapter()

    adapted = adapter.adapt(sample_bundle())

    assert "1 actors" in adapted["analysis_summary"]
    assert "1 organizations" in adapted["analysis_summary"]
    assert "1 social edges" in adapted["analysis_summary"]


def test_adapter_rejects_invalid_social_edge():
    module = load_adapter_module()
    adapter = module.ProjectionBundleAdapter()
    broken_bundle = sample_bundle()
    broken_bundle["social_edges"] = [{"source_id": "actor-1"}]

    try:
        adapter.adapt(broken_bundle)
    except ValueError as exc:
        assert "social_edges[0]" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid social edge")