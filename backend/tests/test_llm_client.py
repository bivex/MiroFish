import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_llm_client_module():
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = app_pkg

    utils_pkg = types.ModuleType("app.utils")
    utils_pkg.__path__ = [str(ROOT / "app" / "utils")]
    sys.modules["app.utils"] = utils_pkg

    config_module = types.ModuleType("app.config")

    class Config:
        LLM_API_KEY = "dummy-key"
        LLM_BASE_URL = "https://api.openai.com/v1"
        LLM_MODEL_NAME = "dummy-model"

    config_module.Config = Config
    sys.modules["app.config"] = config_module

    openai_module = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **_: None))

    openai_module.OpenAI = OpenAI
    sys.modules["openai"] = openai_module

    module_path = ROOT / "app" / "utils" / "llm_client.py"
    spec = importlib.util.spec_from_file_location("app.utils.llm_client", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["app.utils.llm_client"] = module
    spec.loader.exec_module(module)
    return module


def test_llm_client_formats_native_tool_calls_as_tool_call_xml():
    module = load_llm_client_module()
    client = module.LLMClient(
        api_key="dummy-key",
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-20b",
    )

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        message = types.SimpleNamespace(
            content=None,
            tool_calls=[
                types.SimpleNamespace(
                    function=types.SimpleNamespace(
                        name="quick_search",
                        arguments='{"query":"Harbor Guild","limit":3}',
                    )
                )
            ],
        )
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    client.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))
    )

    response = client.chat(
        messages=[{"role": "user", "content": "Search"}],
        tools=[{"type": "function", "function": {"name": "quick_search"}}],
        tool_choice="auto",
    )

    assert captured["tool_choice"] == "auto"
    assert captured["tools"][0]["function"]["name"] == "quick_search"
    assert '"name": "quick_search"' in response
    assert '"query": "Harbor Guild"' in response
    assert response.startswith("<tool_call>")


def test_llm_client_prefers_native_tools_for_groq_routes():
    module = load_llm_client_module()

    groq_client = module.LLMClient(api_key="k", base_url="https://api.groq.com/openai/v1", model="m")
    openai_client = module.LLMClient(api_key="k", base_url="https://api.openai.com/v1", model="m")

    assert groq_client.prefers_native_tools() is True
    assert openai_client.prefers_native_tools() is False