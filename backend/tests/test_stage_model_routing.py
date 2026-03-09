import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_config_module():
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv_stub)

    config_path = ROOT / "app" / "config.py"
    spec = importlib.util.spec_from_file_location("mirofish_test_config", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_stage_model_returns_stage_specific_values(monkeypatch):
    config_module = load_config_module()
    Config = config_module.Config

    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "base-model", raising=False)
    monkeypatch.setattr(Config, "ONTOLOGY_LLM_MODEL", "ontology-model", raising=False)
    monkeypatch.setattr(Config, "PROFILE_LLM_MODEL", "profile-model", raising=False)
    monkeypatch.setattr(Config, "SIM_CONFIG_LLM_MODEL", "sim-model", raising=False)
    monkeypatch.setattr(Config, "REPORT_LLM_MODEL", "report-model", raising=False)

    assert Config.get_stage_model("ontology") == "ontology-model"
    assert Config.get_stage_model("profile") == "profile-model"
    assert Config.get_stage_model("sim_config") == "sim-model"
    assert Config.get_stage_model("report") == "report-model"


def test_get_stage_model_falls_back_to_default(monkeypatch):
    config_module = load_config_module()
    Config = config_module.Config

    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "base-model", raising=False)

    assert Config.get_stage_model("unknown-stage") == "base-model"


def test_services_use_stage_specific_model_routing():
    files_and_needles = {
        ROOT / "app" / "services" / "ontology_generator.py": "Config.get_stage_model('ontology')",
        ROOT / "app" / "services" / "oasis_profile_generator.py": "Config.get_stage_model('profile')",
        ROOT / "app" / "services" / "simulation_config_generator.py": "Config.get_stage_model('sim_config')",
        ROOT / "app" / "services" / "report_agent.py": "Config.get_stage_model('report')",
    }

    for path, needle in files_and_needles.items():
        assert needle in path.read_text(encoding="utf-8")


def test_active_prompt_layers_use_english_llm_contracts():
    checks = {
        ROOT / "app" / "services" / "ontology_generator.py": {
            "must_have": [
                "You are an expert knowledge-graph ontology designer.",
                "## Simulation Requirement",
                '"analysis_summary": "Brief English analysis summary of the source text"',
            ],
            "must_not_have": [
                "请根据以上内容，设计适合情景推演与社会投影模拟的实体类型和关系类型。",
                '"analysis_summary": "对文本内容的简要分析说明（中文）"',
            ],
        },
        ROOT / "app" / "services" / "oasis_profile_generator.py": {
            "must_have": [
                "You are an expert in generating simulation-ready character and organization profiles.",
                "Generate a detailed individual profile for this entity",
                "Generate a detailed representative public profile for this organization or group",
            ],
            "must_not_have": [
                "默认使用中文",
                "请生成JSON，包含以下字段",
                "无额外上下文",
            ],
        },
        ROOT / "app" / "services" / "simulation_config_generator.py": {
            "must_have": [
                "## Simulation Requirement",
                "Generate the time simulation configuration based on the scenario below.",
                "Generate the event configuration for the following scenario.",
                "Generate public activity configurations for each entity using the scenario information below.",
            ],
            "must_not_have": [
                "## 模拟需求",
                "基于以下模拟需求，生成时间模拟配置。",
                "基于以下模拟需求，生成事件配置。",
                "基于以下信息，为每个实体生成公开行为活动配置。",
            ],
        },
        ROOT / "app" / "services" / "report_agent.py": {
            "must_have": [
                "You are an expert writer of scenario simulation analysis reports.",
                "Available tools:",
                "Answer the user's question concisely.",
                "(This is the first section.)",
            ],
            "must_not_have": [
                "（这是第一个章节）",
                "（暂无报告）",
                "未知工具:",
                "请简洁回答问题。",
            ],
        },
    }

    for path, expectations in checks.items():
        content = path.read_text(encoding="utf-8")
        for needle in expectations["must_have"]:
            assert needle in content, f"Missing expected English marker in {path.name}: {needle}"
        for needle in expectations["must_not_have"]:
            assert needle not in content, f"Found legacy non-English marker in {path.name}: {needle}"


def test_api_user_facing_responses_use_english_markers():
    checks = {
        ROOT / "app" / "api" / "report.py": {
            "must_have": [
                "Please provide simulation_id",
                "Report already exists",
                "Report generation task started. Check progress via /api/report/generate/status.",
                "Please provide graph_id and query",
            ],
            "must_not_have": [
                "请提供 simulation_id",
                "报告已存在",
                "报告生成任务已启动，请通过 /api/report/generate/status 查询进度",
                "请提供 graph_id 和 query",
            ],
        },
        ROOT / "app" / "api" / "simulation.py": {
            "must_have": [
                "Please provide simulation_id",
                "Preparation task started. Check progress via /api/simulation/prepare/status.",
                "Environment is running and ready to receive interview commands",
                "Simulation environment is not running or has been closed. Make sure the simulation finished and is waiting for commands.",
            ],
            "must_not_have": [
                "请提供 simulation_id",
                "准备任务已启动，请通过 /api/simulation/prepare/status 查询进度",
                "环境正在运行，可以接收Interview命令",
                "模拟环境未运行或已关闭。请确保模拟已完成并进入等待命令模式。",
            ],
        },
    }

    for path, expectations in checks.items():
        content = path.read_text(encoding="utf-8")
        for needle in expectations["must_have"]:
            assert needle in content, f"Missing expected English API marker in {path.name}: {needle}"
        for needle in expectations["must_not_have"]:
            assert needle not in content, f"Found legacy non-English API marker in {path.name}: {needle}"

