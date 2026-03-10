import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_ontology_generator_module():
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = app_pkg

    services_pkg = types.ModuleType("app.services")
    services_pkg.__path__ = [str(ROOT / "app" / "services")]
    sys.modules["app.services"] = services_pkg

    utils_pkg = types.ModuleType("app.utils")
    utils_pkg.__path__ = [str(ROOT / "app" / "utils")]
    sys.modules["app.utils"] = utils_pkg

    llm_client_module = types.ModuleType("app.utils.llm_client")

    class LLMClient:
        def __init__(self, model=None):
            self.model = model

    llm_client_module.LLMClient = LLMClient
    sys.modules["app.utils.llm_client"] = llm_client_module

    config_module = types.ModuleType("app.config")

    class Config:
        @classmethod
        def get_stage_model(cls, stage):
            return "dummy-model"

    config_module.Config = Config
    sys.modules["app.config"] = config_module

    spec = importlib.util.spec_from_file_location(
        "app.services.ontology_generator",
        ROOT / "app" / "services" / "ontology_generator.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["app.services.ontology_generator"] = module
    spec.loader.exec_module(module)
    return module


def test_build_user_message_adds_narrative_guidance_for_fantasy_context():
    module = load_ontology_generator_module()
    generator = module.OntologyGenerator()

    message = generator._build_user_message(
        document_texts=[
            "Queen Mira warned the Royal Court that the forged succession decree rumor was spreading across the kingdom.",
        ],
        simulation_requirement="Generate ontology for a fantasy court-intrigue narrative.",
        additional_context="Focus on lore-stable ontology classes rather than one-off titles.",
    )

    assert "This source appears narrative/lore-heavy" in message
    assert "Prefer `Character` over types like `Queen`, `Prince`, or `Alchemist`" in message
    assert "Only use metaphysical types like `Void`" in message


def test_build_user_message_keeps_base_rules_for_non_narrative_context():
    module = load_ontology_generator_module()
    generator = module.OntologyGenerator()

    message = generator._build_user_message(
        document_texts=[
            "The company announced a hiring plan for analysts, managers, and suppliers across three offices.",
        ],
        simulation_requirement="Generate ontology for a corporate operations simulation.",
        additional_context=None,
    )

    assert "Return exactly 10 entity types" in message
    assert "This source appears narrative/lore-heavy" not in message


def test_validate_and_process_preserves_fallback_types_with_narrative_specific_types():
    module = load_ontology_generator_module()
    generator = module.OntologyGenerator()

    result = generator._validate_and_process({
        "entity_types": [
            {"name": "Character", "description": "Named acting character in the story."},
            {"name": "Faction", "description": "Political or social faction."},
            {"name": "Location", "description": "Important place in the world."},
        ],
        "edge_types": [],
    })

    entity_names = [entity["name"] for entity in result["entity_types"]]
    assert entity_names[-2:] == ["Person", "Organization"]
    assert "Character" in entity_names