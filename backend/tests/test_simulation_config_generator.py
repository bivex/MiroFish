import importlib.util
import sys
import types
from pathlib import Path


def _load_module():
    module_name = 'app.services.simulation_config_generator'
    path = Path(__file__).resolve().parents[1] / 'app/services/simulation_config_generator.py'

    for name in ['app', 'app.services', 'app.utils']:
        sys.modules.setdefault(name, types.ModuleType(name))

    config_mod = types.ModuleType('app.config')
    config_mod.Config = type('Config', (), {
        'LLM_API_KEY': 'test-key',
        'LLM_BASE_URL': 'http://example.test',
        'get_stage_model': staticmethod(lambda stage: 'test-model'),
    })
    sys.modules['app.config'] = config_mod

    logger_mod = types.ModuleType('app.utils.logger')
    logger_mod.get_logger = lambda name: type('Logger', (), {
        'info': staticmethod(lambda *args, **kwargs: None),
        'warning': staticmethod(lambda *args, **kwargs: None),
        'error': staticmethod(lambda *args, **kwargs: None),
        'debug': staticmethod(lambda *args, **kwargs: None),
    })()
    sys.modules['app.utils.logger'] = logger_mod

    graph_mod = types.ModuleType('app.services.graph_entities')
    graph_mod.EntityNode = object
    sys.modules['app.services.graph_entities'] = graph_mod

    openai_mod = types.ModuleType('openai')
    openai_mod.OpenAI = object
    sys.modules['openai'] = openai_mod

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_assign_initial_posts_prefers_named_agents_found_in_content():
    mod = _load_module()
    generator = object.__new__(mod.SimulationConfigGenerator)

    event_config = mod.EventConfig(initial_posts=[
        {'content': 'Aria the Captain warns the harbor is no longer safe.', 'poster_type': 'person'},
        {'content': 'Boros the Dockmaster says patrols have been doubled.', 'poster_type': 'person'},
    ])
    agent_configs = [
        mod.AgentActivityConfig(agent_id=2, entity_uuid='org-1', entity_name='harbor guild', entity_type='Organization', influence_weight=8),
        mod.AgentActivityConfig(agent_id=10, entity_uuid='actor-1', entity_name='aria the captain', entity_type='Actor', influence_weight=7),
        mod.AgentActivityConfig(agent_id=11, entity_uuid='actor-2', entity_name='boros the dockmaster', entity_type='Actor', influence_weight=6),
    ]

    updated = generator._assign_initial_post_agents(event_config, agent_configs)

    assert updated.initial_posts[0]['poster_agent_id'] == 10
    assert updated.initial_posts[1]['poster_agent_id'] == 11


def test_assign_initial_posts_matches_reordered_titles_in_content():
    mod = _load_module()
    generator = object.__new__(mod.SimulationConfigGenerator)

    event_config = mod.EventConfig(initial_posts=[
        {'content': 'Captain Aria reports worsening shortages near the docks.', 'poster_type': 'person'},
        {'content': 'Dockmaster Boros says patrols are increasing tonight.', 'poster_type': 'person'},
    ])
    agent_configs = [
        mod.AgentActivityConfig(agent_id=2, entity_uuid='org-1', entity_name='harbor guild', entity_type='Organization', influence_weight=8),
        mod.AgentActivityConfig(agent_id=10, entity_uuid='actor-1', entity_name='aria the captain', entity_type='Actor', influence_weight=7),
        mod.AgentActivityConfig(agent_id=11, entity_uuid='actor-2', entity_name='boros the dockmaster', entity_type='Actor', influence_weight=6),
    ]

    updated = generator._assign_initial_post_agents(event_config, agent_configs)

    assert updated.initial_posts[0]['poster_agent_id'] == 10
    assert updated.initial_posts[1]['poster_agent_id'] == 11


def test_assign_initial_posts_fallback_ignores_meta_agents():
    mod = _load_module()
    generator = object.__new__(mod.SimulationConfigGenerator)

    event_config = mod.EventConfig(initial_posts=[
        {'content': 'A mysterious rumor starts spreading.', 'poster_type': 'unknown'},
    ])
    agent_configs = [
        mod.AgentActivityConfig(agent_id=0, entity_uuid='chunk-1', entity_name='chunk', entity_type='DocumentChunk', influence_weight=99),
        mod.AgentActivityConfig(agent_id=1, entity_uuid='schema-1', entity_name='schema', entity_type='EntityType', influence_weight=88),
        mod.AgentActivityConfig(agent_id=2, entity_uuid='org-1', entity_name='harbor guild', entity_type='Organization', influence_weight=8),
    ]

    updated = generator._assign_initial_post_agents(event_config, agent_configs)

    assert updated.initial_posts[0]['poster_agent_id'] == 2


def test_parse_time_config_normalizes_compact_hour_sequences():
    mod = _load_module()
    generator = object.__new__(mod.SimulationConfigGenerator)

    parsed = generator._parse_time_config({
        'total_simulation_hours': '24',
        'minutes_per_round': '60',
        'agents_per_hour_min': 1,
        'agents_per_hour_max': 3,
        'peak_hours': [10111213141516],
        'off_peak_hours': ['012345'],
        'morning_hours': [678],
        'work_hours': [91011121314151617],
    }, num_entities=4)

    assert parsed.peak_hours == [10, 11, 12, 13, 14, 15, 16]
    assert parsed.off_peak_hours == [0, 1, 2, 3, 4, 5]
    assert parsed.morning_hours == [6, 7, 8]
    assert parsed.work_hours == [9, 10, 11, 12, 13, 14, 15, 16, 17]


def test_parse_event_config_normalizes_string_initial_posts():
    mod = _load_module()
    generator = object.__new__(mod.SimulationConfigGenerator)

    parsed = generator._parse_event_config({
        'initial_posts': [
            'Harbor rumors are spreading fast.',
            {'message': 'The council will address the public tonight.', 'author_type': 'Organization'},
        ],
        'hot_topics': ['rumors'],
        'narrative_direction': 'Escalating concern.',
    })

    assert parsed.initial_posts == [
        {'content': 'Harbor rumors are spreading fast.', 'poster_type': 'Unknown'},
        {'message': 'The council will address the public tonight.', 'author_type': 'Organization', 'content': 'The council will address the public tonight.', 'poster_type': 'Organization'},
    ]


def test_generate_agent_configs_batch_normalizes_active_hours():
    mod = _load_module()
    generator = object.__new__(mod.SimulationConfigGenerator)
    generator._call_llm_with_retry = lambda prompt, system_prompt: {
        'agent_configs': [{
            'agent_id': 0,
            'activity_level': 0.7,
            'posts_per_hour': 0.5,
            'comments_per_hour': 1.0,
            'active_hours': [891011121314151617181920212223],
            'response_delay_min': 5,
            'response_delay_max': 10,
            'sentiment_bias': 0.0,
            'stance': 'neutral',
            'influence_weight': 1.0,
        }],
    }

    class FakeEntity:
        uuid = 'actor-1'
        name = 'Alice'
        summary = 'Test summary for active hours normalization.'

        def get_entity_type(self):
            return 'Actor'

    configs = generator._generate_agent_configs_batch(
        context='ctx',
        entities=[FakeEntity()],
        start_idx=0,
        simulation_requirement='req',
    )

    assert configs[0].active_hours == [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]