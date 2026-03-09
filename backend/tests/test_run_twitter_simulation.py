import importlib.util
import json
import sqlite3
import sys
import types
from pathlib import Path


def _load_module():
    module_name = 'scripts.run_twitter_simulation'
    path = Path(__file__).resolve().parents[1] / 'scripts/run_twitter_simulation.py'

    dotenv_mod = types.ModuleType('dotenv')
    dotenv_mod.load_dotenv = lambda *args, **kwargs: None
    sys.modules['dotenv'] = dotenv_mod

    camel_models = types.ModuleType('camel.models')
    camel_models.ModelFactory = type('ModelFactory', (), {'create': staticmethod(lambda **kwargs: object())})
    sys.modules['camel.models'] = camel_models

    camel_types = types.ModuleType('camel.types')
    camel_types.ModelPlatformType = type('ModelPlatformType', (), {'OPENAI': 'OPENAI'})
    sys.modules['camel.types'] = camel_types

    oasis_mod = types.ModuleType('oasis')
    oasis_mod.ActionType = type('ActionType', (), {
        'CREATE_POST': 'create_post',
        'LIKE_POST': 'like_post',
        'REPOST': 'repost',
        'FOLLOW': 'follow',
        'DO_NOTHING': 'do_nothing',
        'QUOTE_POST': 'quote_post',
    })
    oasis_mod.LLMAction = object
    oasis_mod.ManualAction = object
    oasis_mod.generate_twitter_agent_graph = lambda **kwargs: None
    sys.modules['oasis'] = oasis_mod

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeTool:
    def __init__(self, name, schema):
        self.func = type('Func', (), {'__name__': name})()
        self._schema = schema

    def get_openai_tool_schema(self):
        return self._schema


class _FakeAgent:
    def __init__(self, tool):
        self.action_tools = [tool]
        self._internal_tools = {'do_nothing': tool}


class _FakeGraph:
    def __init__(self, tool):
        self.tool = tool

    def get_agents(self):
        return [(1, _FakeAgent(self.tool))]


def test_sanitize_agent_tool_schemas_removes_empty_required_for_do_nothing():
    mod = _load_module()
    runner = object.__new__(mod.TwitterSimulationRunner)
    tool = _FakeTool('do_nothing', {
        'type': 'function',
        'function': {'name': 'do_nothing', 'parameters': {'type': 'object', 'properties': {}, 'required': []}},
    })
    runner.agent_graph = _FakeGraph(tool)

    patched = runner._sanitize_agent_tool_schemas()
    schema = tool.get_openai_tool_schema()

    assert patched == 1
    assert schema['function']['parameters']['properties'] == {}
    assert 'required' not in schema['function']['parameters']


def test_sanitize_tool_schema_ignores_non_do_nothing_tools():
    mod = _load_module()
    tool = _FakeTool('create_post', {
        'type': 'function',
        'function': {'name': 'create_post', 'parameters': {'type': 'object', 'properties': {'content': {'type': 'string'}}, 'required': ['content']}},
    })

    assert mod.TwitterSimulationRunner._sanitize_tool_schema(tool) is False
    assert tool.get_openai_tool_schema()['function']['parameters']['required'] == ['content']


def test_export_new_trace_actions_writes_simulationrunner_compatible_jsonl(tmp_path):
    mod = _load_module()
    runner = object.__new__(mod.TwitterSimulationRunner)
    runner.simulation_dir = str(tmp_path)

    db_path = tmp_path / 'twitter_simulation.db'
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('CREATE TABLE user (user_id INTEGER PRIMARY KEY, agent_id INTEGER, user_name TEXT, name TEXT)')
    cur.execute('CREATE TABLE trace (user_id INTEGER, created_at DATETIME, action TEXT, info TEXT)')
    cur.execute("INSERT INTO user (user_id, agent_id, user_name, name) VALUES (1, 1, NULL, 'captain_mira_1')")
    cur.execute("INSERT INTO trace (user_id, created_at, action, info) VALUES (1, 0, 'sign_up', '{}')")
    cur.execute("INSERT INTO trace (user_id, created_at, action, info) VALUES (1, 1, 'refresh', '{\"posts\": []}')")
    cur.execute(
        "INSERT INTO trace (user_id, created_at, action, info) VALUES (1, 1, 'quote_post', ?)",
        (json.dumps({'quoted_id': 4, 'new_post_id': 8}),),
    )
    conn.commit()
    conn.close()

    agent_names = runner._load_agent_names()
    last_rowid, exported = runner._export_new_trace_actions(0, 2, agent_names)

    actions_log = tmp_path / 'twitter' / 'actions.jsonl'
    lines = [json.loads(line) for line in actions_log.read_text(encoding='utf-8').splitlines()]

    assert last_rowid == 3
    assert exported == 1
    assert len(lines) == 1
    assert lines[0]['platform'] == 'twitter'
    assert lines[0]['round'] == 2
    assert lines[0]['agent_id'] == 1
    assert lines[0]['agent_name'] == 'captain_mira_1'
    assert lines[0]['action_type'] == 'quote_post'
    assert lines[0]['action_args']['new_post_id'] == 8