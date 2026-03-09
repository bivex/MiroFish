import importlib.util
import sys
import types
from pathlib import Path


def _load_module():
    module_name = 'scripts.run_reddit_simulation'
    path = Path(__file__).resolve().parents[1] / 'scripts/run_reddit_simulation.py'

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
        'LIKE_POST': 'like_post',
        'DISLIKE_POST': 'dislike_post',
        'CREATE_POST': 'create_post',
        'CREATE_COMMENT': 'create_comment',
        'LIKE_COMMENT': 'like_comment',
        'DISLIKE_COMMENT': 'dislike_comment',
        'SEARCH_POSTS': 'search_posts',
        'SEARCH_USER': 'search_user',
        'TREND': 'trend',
        'REFRESH': 'refresh',
        'DO_NOTHING': 'do_nothing',
        'FOLLOW': 'follow',
        'MUTE': 'mute',
    })
    oasis_mod.LLMAction = object
    oasis_mod.ManualAction = object
    oasis_mod.generate_reddit_agent_graph = lambda **kwargs: None
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
        self._internal_tools = {'tool': tool}


class _FakeGraph:
    def __init__(self, tool):
        self.tool = tool

    def get_agents(self):
        return [(1, _FakeAgent(self.tool))]


def test_sanitize_agent_tool_schemas_removes_empty_required_for_zero_arg_tool():
    mod = _load_module()
    runner = object.__new__(mod.RedditSimulationRunner)
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


def test_sanitize_tool_schema_ignores_tools_with_real_parameters():
    mod = _load_module()
    tool = _FakeTool('create_post', {
        'type': 'function',
        'function': {'name': 'create_post', 'parameters': {'type': 'object', 'properties': {'content': {'type': 'string'}}, 'required': ['content']}},
    })

    assert mod.RedditSimulationRunner._sanitize_tool_schema(tool) is False
    assert tool.get_openai_tool_schema()['function']['parameters']['required'] == ['content']