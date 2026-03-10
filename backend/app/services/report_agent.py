"""
Report Agent服务
使用LangChain + Zep实现ReACT模式的模拟报告生成

功能：
1. 根据模拟需求和Zep图谱信息生成报告
2. 先规划目录结构，然后分段生成
3. 每段采用ReACT多轮思考与反思模式
4. 支持与用户对话，在对话中自主调用检索工具
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .graph_backend_factory import get_report_tools_service

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Report Agent 详细日志记录器
    
    在报告文件夹中生成 agent_log.jsonl 文件，记录每一步详细动作。
    每行是一个完整的 JSON 对象，包含时间戳、动作类型、详细内容等。
    """
    
    def __init__(self, report_id: str):
        """
        初始化日志记录器
        
        Args:
            report_id: 报告ID，用于确定日志文件路径
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """确保日志文件所在目录存在"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """获取从开始到现在的耗时（秒）"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self, 
        action: str, 
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        记录一条日志
        
        Args:
            action: 动作类型，如 'start', 'tool_call', 'llm_response', 'section_complete' 等
            stage: 当前阶段，如 'planning', 'generating', 'completed'
            details: 详细内容字典，不截断
            section_title: 当前章节标题（可选）
            section_index: 当前章节索引（可选）
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # 追加写入 JSONL 文件
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """记录报告生成开始"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "报告生成任务开始"
            }
        )
    
    def log_planning_start(self):
        """记录大纲规划开始"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "开始规划报告大纲"}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """记录规划时获取的上下文信息"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "获取模拟上下文信息",
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """记录大纲规划完成"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "大纲规划完成",
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """记录章节生成开始"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"开始生成章节: {section_title}"}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """记录 ReACT 思考过程"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT 第{iteration}轮思考"
            }
        )
    
    def log_tool_call(
        self, 
        section_title: str, 
        section_index: int,
        tool_name: str, 
        parameters: Dict[str, Any],
        iteration: int
    ):
        """记录工具调用"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"调用工具: {tool_name}"
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """记录工具调用结果（完整内容，不截断）"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # 完整结果，不截断
                "result_length": len(result),
                "message": f"工具 {tool_name} 返回结果"
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """记录 LLM 响应（完整内容，不截断）"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # 完整响应，不截断
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"LLM 响应 (工具调用: {has_tool_calls}, 最终答案: {has_final_answer})"
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """记录章节内容生成完成（仅记录内容，不代表整个章节完成）"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # 完整内容，不截断
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"章节 {section_title} 内容生成完成"
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        记录章节生成完成

        前端应监听此日志来判断一个章节是否真正完成，并获取完整内容
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"章节 {section_title} 生成完成"
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """记录报告生成完成"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "报告生成完成"
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """记录错误"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"发生错误: {error_message}"
            }
        )


class ReportConsoleLogger:
    """
    Report Agent 控制台日志记录器
    
    将控制台风格的日志（INFO、WARNING等）写入报告文件夹中的 console_log.txt 文件。
    这些日志与 agent_log.jsonl 不同，是纯文本格式的控制台输出。
    """
    
    def __init__(self, report_id: str):
        """
        初始化控制台日志记录器
        
        Args:
            report_id: 报告ID，用于确定日志文件路径
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """确保日志文件所在目录存在"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """设置文件处理器，将日志同时写入文件"""
        import logging
        
        # 创建文件处理器
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)
        
        # 使用与控制台相同的简洁格式
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)
        
        # 添加到 report_agent 相关的 logger
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.zep_tools',
        ]
        
        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # 避免重复添加
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """关闭文件处理器并从 logger 中移除"""
        import logging
        
        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.zep_tools',
            ]
            
            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)
            
            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """析构时确保关闭文件处理器"""
        self.close()


class ReportStatus(str, Enum):
    """报告状态"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """转换为Markdown格式"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """报告大纲"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """转换为Markdown格式"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """完整报告"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Prompt 模板常量
# ═══════════════════════════════════════════════════════════════

# ── 工具描述 ──

TOOL_DESC_INSIGHT_FORGE = """\
[Deep Insight Retrieval - advanced analysis tool]
Use this tool when you need the richest evidence package for a report section. It will:
1. Break your question into sub-questions automatically
2. Search the simulation graph from multiple angles
3. Combine semantic facts, entity insights, and relationship chains
4. Return the deepest and most comprehensive retrieval result

[Best for]
- deep analysis of a topic
- understanding multiple facets of an event
- collecting strong evidence for a report section

[Returns]
- relevant fact snippets you can cite directly
- key entity insights
- relationship chain analysis
- diagnostics about evidence sources and fallback mode"""

TOOL_DESC_PANORAMA_SEARCH = """\
[Panorama Search - broad situational view]
Use this tool to understand the full picture of a simulation outcome. It will:
1. Collect all relevant nodes and relationships
2. Distinguish active facts from historical or expired ones
3. Help you trace how the situation or public narrative evolved

[Best for]
- understanding the full development path of an event
- comparing different stages of the situation or narrative
- gathering comprehensive entity and relationship coverage

[Returns]
- active facts from the latest simulation state
- historical or expired facts from earlier stages
- all involved entities
- diagnostics about evidence sources and fallback mode"""

TOOL_DESC_QUICK_SEARCH = """\
[Quick Search - lightweight fact lookup]
A lightweight retrieval tool for simple and direct information checks.

[Best for]
- finding a specific detail quickly
- verifying a concrete fact
- lightweight evidence lookup

[Returns]
- a list of facts most relevant to the query
- diagnostics about evidence sources and fallback mode"""

TOOL_DESC_INTERVIEW_AGENTS = """\
[Agent Interviews - real first-person responses across platforms]
Call the OASIS interview API to interview running simulation agents directly.
This is not a synthetic LLM summary. It fetches real responses from the active simulation environment.
By default it interviews agents on both Twitter and Reddit for broader coverage.

[Workflow]
1. Read the profile files for the simulation agents
2. Select the agents most relevant to the interview topic
3. Generate interview questions automatically
4. Call /api/simulation/interview/batch to run real interviews
5. Merge the responses into a multi-perspective evidence set

[Best for]
- collecting first-person viewpoints from different roles
- comparing positions across groups
- making the report more vivid with interview-style evidence

[Returns]
- identity information for interviewed agents
- interview responses across platforms
- key quotations you can cite directly
- a summary comparing viewpoints

[Important]
This tool requires the OASIS simulation environment to be running."""

# ── 大纲规划 prompt ──

PLAN_SYSTEM_PROMPT = """\
You are an expert writer of scenario simulation analysis reports. You have a god's-eye view of the simulation world and can examine the behavior, statements, and interactions of every agent.

[Core framing]
We built a simulation world and injected a specific simulation requirement as the scenario variable. The resulting world state is not abstract experiment data. It is a scenario exercise showing how a world, conflict, or public situation may evolve under those conditions.

[Your task]
Design a report outline that answers:
1. How does the situation develop under the configured conditions?
2. How do different agents or groups react and act?
3. What trends, risks, opportunities, or power shifts does the simulation reveal?

[What this report is]
- A simulation-based scenario analysis report about what happens under the stated conditions
- Focused on outcomes: event trajectory, group reactions, emergent patterns, risks, and opportunities
- Grounded in agent behavior and simulation evidence rather than free invention

[What this report is not]
- Not an analysis of the real-world status quo
- Not a generic overview detached from the scenario setup

[Section limits]
- Minimum 2 sections, maximum 5 sections
- No subsections are needed
- Each section should be concise and focused on core findings
- You should choose the most appropriate structure based on the simulation outcomes

Return the report outline as JSON in this format:
{
    "title": "Report title",
    "summary": "One-sentence summary of the core scenario finding",
    "sections": [
        {
            "title": "Section title",
            "description": "What this section should cover"
        }
    ]
}

Important: the sections array must contain between 2 and 5 items."""

PLAN_USER_PROMPT_TEMPLATE = """\
[Scenario setup]
Injected simulation requirement: {simulation_requirement}

[Simulation world size]
- Total participating entities: {total_nodes}
- Total relationships produced: {total_edges}
- Entity type distribution: {entity_types}
- Active agent count: {total_entities}

[Sample facts observed in the simulation]
{related_facts_json}

Review this scenario exercise from a god's-eye view:
1. What overall state does the situation reach under these conditions?
2. How do different groups or agents react and act?
3. What notable trends, risks, opportunities, or structural changes emerge?

Based on the simulation outcomes, design the most appropriate report section structure.

Reminder: the report must contain 2 to 5 sections and stay focused on the core scenario findings."""

# ── 章节生成 prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
You are writing one section of a scenario simulation analysis report.

Report title: {report_title}
Report summary: {report_summary}
Scenario condition (simulation requirement): {simulation_requirement}

Current section: {section_title}

═══════════════════════════════════════════════════════════════
[Core framing]
═══════════════════════════════════════════════════════════════

The simulation world is a scenario exercise constrained by worldbuilding, role logic, and agent behavior. The injected condition is the simulation requirement, and the observed agent behavior is the evidence of how the situation evolves under that condition.

Your job is to:
- explain how the situation unfolds under the configured condition
- describe how agents react, act, align, clash, or spread information
- identify notable trends, risks, opportunities, and structural changes

Do not write a report about the real-world status quo.
Focus on what develops inside this configured scenario. The simulation results are the evidence.

═══════════════════════════════════════════════════════════════
[Most important rules - must follow]
═══════════════════════════════════════════════════════════════

1. [You must use tools to observe the simulation world]
   - You are observing a scenario exercise from a god's-eye view
   - Every claim must come from events, behavior, or statements that exist in the simulation output
   - Do not use outside knowledge to write the section
   - Each section must call tools at least 3 times and at most 5 times

2. [You must use original agent evidence]
   - Agent behavior and statements are key evidence
   - Use quote formatting to present this evidence when helpful
   - These quotations are central scenario evidence, not decoration

3. [Language consistency - the report must be fully in English]
   - Tool results may contain Chinese, English, or mixed-language text
   - The final report must be written entirely in fluent English
   - When quoting tool output that is not already in English, translate it into natural English while preserving the original meaning
   - This rule applies to both body text and block quotes

4. [Stay faithful to the simulation]
   - Reflect the scenario outcome faithfully
   - Do not invent evidence that is not present in the simulation
   - If evidence is insufficient, say so clearly

═══════════════════════════════════════════════════════════════
[Formatting rules - critical]
═══════════════════════════════════════════════════════════════

[One section = one minimal content unit]
- Each section is a single self-contained report block
- Do not use Markdown headings inside the section (#, ##, ###, ####, etc.)
- Do not start by repeating the section title
- The system adds the title automatically; you should write body content only
- You may use **bold text**, paragraphs, quotes, and lists, but not headings

[Correct example]
```
This section examines how the event spread and intensified under the simulated scenario. Based on the retrieved evidence, we observe...

**Early ignition phase**

Key public channels carried the first wave of amplification:

> "This channel accounted for 68% of the initial spread..."

**Escalation phase**

The next wave amplified both visibility and emotional intensity:

- high visual impact
- strong emotional resonance
```

[Incorrect example]
```
## Executive Summary      ← Wrong: do not add headings
### Phase One             ← Wrong: do not use subsection headings
#### Detailed Analysis    ← Wrong: do not subdivide with heading syntax

This section examines...
```

═══════════════════════════════════════════════════════════════
[Available retrieval tools] (use 3-5 times per section)
═══════════════════════════════════════════════════════════════

{tools_description}

[Tool usage strategy]
- Mix tools when useful; do not rely on a single tool only
- Use `insight_forge` for deep, multi-angle evidence gathering
- Use `panorama_search` for broad development, timeline, and overall-picture retrieval
- Use `quick_search` for targeted fact checks
- Use `interview_agents` only if it appears in the available tools list above

═══════════════════════════════════════════════════════════════
[Workflow]
═══════════════════════════════════════════════════════════════

Each reply may do exactly one of the following, not both:

Option A - Call one tool:
Write your reasoning, then call a tool in this format:
<tool_call>
{{"name": "tool_name", "parameters": {{"parameter_name": "value"}}}}
</tool_call>
The system will execute the tool and inject the result. You must not invent tool results yourself.

Option B - Produce the final section content:
Once you have enough evidence, output the section using the prefix "Final Answer:".

Strictly forbidden:
- including both a tool call and Final Answer in the same reply
- inventing Observation content yourself
- calling more than one tool in a single reply

═══════════════════════════════════════════════════════════════
[Section content requirements]
═══════════════════════════════════════════════════════════════

1. Every section must be grounded in retrieved simulation evidence
2. Use direct evidence and quotations generously when they strengthen the analysis
3. Use Markdown without headings:
   - use **bold text** for emphasis or mini-labels
   - use bullet lists or numbered lists for structured points
   - separate paragraphs with blank lines
   - do not use heading syntax such as #, ##, ###, ####
4. [Quote formatting rule - block quotes must stand alone]
   Quotes must appear as their own paragraph with blank lines before and after them

   Correct format:
   ```
   The institution's response was widely seen as too limited.

   > "The institution's response pattern appeared rigid and slow in a fast-moving media environment."

   This judgment captures a broader wave of dissatisfaction.
   ```

   Incorrect format:
   ```
   The institution's response was too limited. > "The response pattern..." This reflects dissatisfaction.
   ```
5. Maintain logical continuity with the other sections
6. Read previously completed sections carefully and avoid repeating the same points
7. Do not add any headings. Use **bold text** instead of subsection titles."""

SECTION_USER_PROMPT_TEMPLATE = """\
Previously completed sections (read carefully and avoid repetition):
{previous_content}

═══════════════════════════════════════════════════════════════
[Current task] Write section: {section_title}
═══════════════════════════════════════════════════════════════

[Important reminders]
1. Read the completed sections above and avoid repeating the same points
2. You must call tools before writing the final section
3. Mix different tools when helpful instead of using only one
4. The report content must come from retrieved simulation evidence, not from your own knowledge

[Format warning - must follow]
- Do not write any headings (#, ##, ###, ####, etc.)
- Do not start with "{section_title}"
- The section title is added automatically by the system
- Write body text directly and use **bold text** instead of subsection headings

Begin now:
1. Think about what evidence this section needs
2. Call tools to collect simulation data
3. Once the evidence is sufficient, output Final Answer with body text only and no headings"""

# ── ReACT 循环内消息模板 ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (retrieved evidence):

═══ Result from tool `{tool_name}` ═══
{result}

═══════════════════════════════════════════════════════════════
Tools used: {tool_calls_count}/{max_tool_calls} (used so far: {used_tools_str}){unused_hint}
- If the evidence is sufficient, output the section with the prefix "Final Answer:"
- If you still need evidence, call one more tool
- Read any `diagnostics` fields carefully. If fallback was used or evidence is sparse, say that explicitly and do not overclaim.
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "You have only used {tool_calls_count} tool calls, but at least {min_tool_calls} are required. "
    "Call more tools to gather simulation evidence before outputting Final Answer.{unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "You have used only {tool_calls_count} tool calls so far, but at least {min_tool_calls} are required. "
    "Call tools to gather simulation evidence.{unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "You have reached the tool-call limit ({tool_calls_count}/{max_tool_calls}) and cannot call more tools. "
    'Now output the section immediately using the prefix "Final Answer:" and only the evidence already collected.'
)

REACT_UNUSED_TOOLS_HINT = "\nTip: you have not used these tools yet: {unused_list}. Consider mixing tools for broader evidence."

REACT_FORCE_FINAL_MSG = "The tool-call phase is over. Output Final Answer now and write the section strictly from the evidence already collected."

# ── Chat prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
You are a concise and efficient scenario simulation assistant.

[Context]
Scenario condition: {simulation_requirement}

[Generated analysis report]
{report_content}

[Rules]
1. Answer from the report content first whenever possible
2. Respond directly and avoid long chains of meta-reasoning
3. Only call tools when the report content is not enough to answer the question
4. Keep answers concise, clear, well-structured, and faithful to the scenario outcome
5. If you used tool results, answer only from explicit retrieved facts; do not infer beyond them
6. If the retrieved facts are insufficient, say that clearly instead of guessing

[Available tools] (use only if needed, at most 1-2 calls)
{tools_description}

[Tool-call format]
<tool_call>
{{"name": "tool_name", "parameters": {{"parameter_name": "value"}}}}
</tool_call>

[Answer style]
- concise and direct
- use > block quotes for key evidence when helpful
- lead with the conclusion, then explain the reason"""

CHAT_OBSERVATION_SUFFIX = "\n\nAnswer the user's question concisely."


# ═══════════════════════════════════════════════════════════════
# ReportAgent 主类
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - 模拟报告生成Agent

    采用ReACT（Reasoning + Acting）模式：
    1. 规划阶段：分析模拟需求，规划报告目录结构
    2. 生成阶段：逐章节生成内容，每章节可多次调用工具获取信息
    3. 反思阶段：检查内容完整性和准确性
    """
    
    # 最大工具调用次数（每个章节）
    MAX_TOOL_CALLS_PER_SECTION = 5
    
    # 最大反思轮数
    MAX_REFLECTION_ROUNDS = 3
    
    # 对话中的最大工具调用次数
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[Any] = None,
        graph_backend: Optional[str] = None,
    ):
        """
        初始化Report Agent
        
        Args:
            graph_id: 图谱ID
            simulation_id: 模拟ID
            simulation_requirement: 模拟需求描述
            llm_client: LLM客户端（可选）
            zep_tools: Zep工具服务（可选）
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        self.graph_backend = graph_backend or Config.get_graph_backend()
        
        self.llm = llm_client or LLMClient(model=Config.get_stage_model('report'))
        self.zep_tools = zep_tools or get_report_tools_service(graph_backend=self.graph_backend)
        
        # 工具定义
        self.tools = self._define_tools()
        
        # 日志记录器（在 generate_report 中初始化）
        self.report_logger: Optional[ReportLogger] = None
        # 控制台日志记录器（在 generate_report 中初始化）
        self.console_logger: Optional[ReportConsoleLogger] = None
        
        logger.info(
            f"ReportAgent 初始化完成: graph_id={graph_id}, simulation_id={simulation_id}, backend={self.graph_backend}"
        )

    def _supports_interview_agents(self) -> bool:
        return self.graph_backend != "cognee"

    def _available_tool_names(self) -> set[str]:
        return set(self.tools.keys())

    def _get_runtime_evidence(self) -> Dict[str, Any]:
        provider = getattr(self.zep_tools, "get_runtime_evidence", None)
        if callable(provider):
            return provider(self.simulation_id, limit=10)
        return {"has_runtime_evidence": True, "message": ""}

    def _ensure_generation_readiness(self) -> None:
        if self.graph_backend != "cognee":
            return
        runtime_evidence = self._get_runtime_evidence()
        if runtime_evidence.get("has_runtime_evidence"):
            return
        message = runtime_evidence.get("message") or (
            "Cognee report generation requires runtime evidence. "
            "Start the simulation and wait for actions before generating a report."
        )
        raise ValueError(message)
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """定义可用工具"""
        tools = {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "Question or topic to analyze in depth",
                    "report_context": "Current report-section context (optional, helps generate better sub-queries)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Search query used for relevance ranking",
                    "include_expired": "Whether to include expired or historical content (default: True)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Search query string",
                    "limit": "Number of results to return (optional, default: 10)"
                }
            },
        }
        if self._supports_interview_agents():
            tools["interview_agents"] = {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Interview topic or evidence need (for example: collect student views on the crisis)",
                    "max_agents": "Maximum number of agents to interview (optional, default: 5, max: 10)"
                }
            }
        return tools
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """执行工具调用并返回文本格式结果。"""
        return self._execute_tool_payload(tool_name, parameters, report_context).get("text", "")

    def _serialize_tool_result(self, result: Any) -> str:
        """将工具结果稳定序列化为字符串。"""
        if result is None:
            return ""
        to_text = getattr(result, "to_text", None)
        if callable(to_text):
            return to_text()
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    def _normalize_tool_fact(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def _is_runtime_fact_like(self, fact: str) -> bool:
        text = self._normalize_tool_fact(fact)
        if not text:
            return False

        lowered = text.lower()
        structural_markers = (
            "source_id:",
            "target_id:",
            "canonical_id:",
            "relation_type:",
            "speaker_mode:",
            "schema_version:",
            "scenario_id:",
            "world_id:",
        )
        if any(marker in lowered for marker in structural_markers):
            return False

        structural_prefixes = (
            "socialedge:",
            "worldrule:",
            "contextlocation:",
            "eventseed:",
            "organization:",
            "actor:",
            "## ",
        )
        if lowered.startswith(structural_prefixes):
            return False

        if "[round" in lowered or "[twitter]" in lowered or "[reddit]" in lowered:
            return True

        runtime_tokens = (
            " create_post",
            " create_comment",
            " quote_post",
            " like_post",
            " do_nothing",
            " posted ",
            " commented ",
            " replied ",
            " quoted ",
            " reposted ",
            " retweeted ",
            " shared ",
        )
        return any(token in lowered for token in runtime_tokens)

    def _extract_grounded_tool_lines(self, tool_name: str, result: Any) -> List[str]:
        if tool_name == "insight_forge":
            candidates = list(getattr(result, "semantic_facts", []) or []) + list(
                getattr(result, "relationship_chains", []) or []
            )
        elif tool_name == "panorama_search":
            candidates = list(getattr(result, "active_facts", []) or [])
        elif tool_name == "quick_search":
            candidates = list(getattr(result, "facts", []) or [])
        else:
            return []

        grounded: List[str] = []
        seen = set()
        for item in candidates:
            text = self._normalize_tool_fact(item)
            if not text or text in seen or not self._is_runtime_fact_like(text):
                continue
            grounded.append(text)
            seen.add(text)
        return grounded

    def _iter_chat_tool_candidates(self, tool_name: str, result: Any) -> List[Any]:
        if tool_name == "insight_forge":
            return list(getattr(result, "semantic_facts", []) or []) + list(
                getattr(result, "relationship_chains", []) or []
            ) + list(getattr(result, "entity_insights", []) or [])
        if tool_name == "panorama_search":
            return list(getattr(result, "active_facts", []) or []) + list(
                getattr(result, "all_nodes", []) or []
            ) + list(getattr(result, "all_edges", []) or [])
        if tool_name == "quick_search":
            return list(getattr(result, "facts", []) or []) + list(
                getattr(result, "nodes", []) or []
            ) + list(getattr(result, "edges", []) or [])
        return []

    def _extract_chat_subject(self, line: str) -> Optional[str]:
        text = self._normalize_tool_fact(line)
        if ":" not in text:
            return None
        prefix, value = text.split(":", 1)
        prefix = prefix.strip().lower()
        value = value.strip()
        if prefix in {"actor", "organization", "contextlocation", "eventseed", "worldrule", "location", "event"} and value:
            return value
        return None

    def _normalize_chat_fact_line(self, line: str, current_subject: Optional[str] = None) -> Optional[str]:
        text = str(line or "").strip()
        if not text:
            return None

        text = re.sub(r"^[\-•*]\s*", "", text)
        text = self._normalize_tool_fact(text)
        if not text:
            return None

        lowered = text.lower()
        structural_markers = (
            "source_id:",
            "target_id:",
            "canonical_id:",
            "schema_version:",
            "scenario_id:",
            "world_id:",
            "world_version:",
            "speaker_mode:",
            "uuid:",
        )
        if any(marker in lowered for marker in structural_markers):
            return None

        if lowered.startswith((
            "search results",
            "current key memory",
            "active memory",
            "referenced entities",
            "core entities",
            "relationship chains",
            "grounded facts:",
            "evidence sources:",
            "verified runtime evidence from",
        )):
            return None

        if re.match(r"^[0-9]+\s+(facts?|nodes?|edges?|results?)$", lowered):
            return None

        prefix_labels = {
            "actor:": "Actor",
            "organization:": "Organization",
            "contextlocation:": "Location",
            "eventseed:": "Event",
            "worldrule:": "World rule",
        }
        for prefix, label in prefix_labels.items():
            if lowered.startswith(prefix):
                value = text.split(":", 1)[1].strip()
                return f"{label}: {value}" if value else None

        relation_match = re.match(r"socialedge:\s*(.+?)\s*->\s*(.+?)\s*\((.+?)\)\s*$", text, re.IGNORECASE)
        if relation_match:
            return (
                f"Relation: {relation_match.group(1).strip()} -> "
                f"{relation_match.group(2).strip()} ({relation_match.group(3).strip()})"
            )

        if current_subject:
            field_match = re.match(r"([A-Za-z_ ]+):\s*(.+)$", text)
            if field_match:
                key = field_match.group(1).strip().lower().replace(" ", "_")
                value = field_match.group(2).strip()
                if key in {"role", "domain", "public_position", "stance", "kind", "summary", "status"}:
                    label = key.replace("_", " ")
                    return f"{current_subject} — {label}: {value}"

        return text

    def _extract_chat_tool_lines(self, tool_name: str, result: Any) -> List[str]:
        facts: List[str] = []
        seen = set()

        for item in self._iter_chat_tool_candidates(tool_name, result):
            current_subject: Optional[str] = None

            if isinstance(item, dict):
                structured_lines = []
                name = self._normalize_tool_fact(item.get("name") or item.get("label") or item.get("title") or "")
                item_type = self._normalize_tool_fact(item.get("type") or item.get("kind") or "")
                if name:
                    if item_type:
                        structured_lines.append(f"{item_type}: {name}")
                    else:
                        structured_lines.append(name)
                    current_subject = name
                for key in ("role", "domain", "public_position", "stance", "kind", "summary", "status"):
                    value = item.get(key)
                    if value:
                        structured_lines.append(f"{key}: {value}")
                lines = structured_lines or [json.dumps(item, ensure_ascii=False)]
            else:
                lines = str(item or "").splitlines()

            for raw_line in lines:
                subject = self._extract_chat_subject(raw_line)
                if subject:
                    current_subject = subject
                normalized = self._normalize_chat_fact_line(raw_line, current_subject=current_subject)
                if not normalized or normalized in seen:
                    continue
                facts.append(normalized)
                seen.add(normalized)

        return facts

    def _clean_chat_response(self, response: Optional[str]) -> str:
        text = response or ""
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
        text = re.sub(r'\[TOOL_CALL\].*?\)', '', text)
        return text.strip()

    def _build_chat_search_query(self, message: str) -> str:
        text = self._normalize_tool_fact(message)
        latin_matches = re.findall(
            r"[A-Z][A-Za-z0-9_'-]*(?:\s+(?:the\s+)?[A-Z][A-Za-z0-9_'-]*)*",
            text,
        )
        if latin_matches:
            return max(latin_matches, key=len).strip(" ?!.,")

        text = re.sub(r"^(кто\s+(такая|такой|это)|что\s+такое|расскажи\s+про)\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*ответь.*$", "", text, flags=re.IGNORECASE)
        return text.strip(" ?!.,") or self._normalize_tool_fact(message)

    def _tokenize_chat_query(self, query: str) -> List[str]:
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "кто",
            "такая",
            "такой",
            "это",
            "что",
            "такое",
            "ответь",
            "одной",
            "строкой",
        }
        tokens = re.findall(r"[A-Za-zА-Яа-я0-9_'-]+", query.lower())
        return [token for token in tokens if len(token) >= 3 and token not in stop_words]

    def _build_fact_bound_chat_response(self, tool_fact_records: List[Dict[str, Any]]) -> str:
        matched_facts: List[str] = []
        unmatched_facts: List[str] = []
        seen_matched = set()
        seen_unmatched = set()
        checked_queries: List[str] = []

        for record in tool_fact_records:
            query = self._normalize_tool_fact(record.get("query", ""))
            query_terms = self._tokenize_chat_query(query)
            if query and query not in checked_queries:
                checked_queries.append(query)
            for fact in record.get("facts") or []:
                normalized = self._normalize_tool_fact(fact)
                if not normalized:
                    continue
                if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", normalized, re.IGNORECASE):
                    continue

                lowered_fact = normalized.lower()
                is_match = bool(query_terms) and any(term in lowered_fact for term in query_terms)
                if is_match:
                    if normalized in seen_matched:
                        continue
                    matched_facts.append(normalized)
                    seen_matched.add(normalized)
                    continue

                if normalized in seen_unmatched:
                    continue
                unmatched_facts.append(normalized)
                seen_unmatched.add(normalized)

        facts = matched_facts or unmatched_facts

        if not facts:
            lines = [
                "I could not find enough verified facts in the current report memory to answer that reliably.",
                "Please try a narrower question or add more simulation evidence.",
            ]
            if checked_queries:
                lines.append(f"Checked queries: {', '.join(checked_queries[:3])}")
            return "\n".join(lines)

        if len(facts) == 1:
            return facts[0]

        lines = ["Based only on retrieved facts:"]
        for fact in facts[:6]:
            lines.append(f"- {fact}")
        if len(facts) > 6:
            lines.append(f"- … {len(facts) - 6} more retrieved facts omitted")
        return "\n".join(lines)

    def _render_tool_result_for_llm(
        self,
        tool_name: str,
        result: Any,
        raw_text: str,
        evidence_summary: Dict[str, Any],
    ) -> str:
        if self.graph_backend != "cognee" or tool_name not in {"insight_forge", "panorama_search", "quick_search"}:
            return raw_text

        grounded_lines = self._extract_grounded_tool_lines(tool_name, result)
        if not grounded_lines:
            return raw_text

        evidence_sources = evidence_summary.get("evidence_sources") or []
        lines = [f"Verified runtime evidence from {tool_name}:"]
        if evidence_sources:
            lines.append(f"Evidence sources: {', '.join(evidence_sources)}")
        lines.append("Grounded facts:")
        for line in grounded_lines[:8]:
            lines.append(f"- {line}")
        lines.append("Use only the grounded facts above for narrative synthesis.")
        return "\n".join(lines)

    def _build_tool_evidence_summary(self, tool_name: str, result: Any) -> Dict[str, Any]:
        """为工具结果提取一个轻量 evidence summary，用于 section grounding guard。"""
        diagnostics = getattr(result, "diagnostics", None)
        if not isinstance(diagnostics, dict):
            diagnostics = {}

        evidence_sources = [
            str(source) for source in (diagnostics.get("evidence_sources") or []) if source
        ]
        summary: Dict[str, Any] = {
            "tool_name": tool_name,
            "evidence_sources": evidence_sources,
            "meaningful": False,
        }
        grounded_lines = self._extract_grounded_tool_lines(tool_name, result)
        summary["grounded_line_count"] = len(grounded_lines)

        if tool_name == "insight_forge":
            semantic_facts = list(getattr(result, "semantic_facts", []) or [])
            entity_insights = list(getattr(result, "entity_insights", []) or [])
            relationship_chains = list(getattr(result, "relationship_chains", []) or [])
            summary.update({
                "semantic_fact_count": len(semantic_facts),
                "entity_count": len(entity_insights),
                "relationship_chain_count": len(relationship_chains),
            })
            summary["meaningful"] = bool(grounded_lines) if self.graph_backend == "cognee" else bool(semantic_facts or relationship_chains)
            return summary

        if tool_name == "panorama_search":
            active_facts = list(getattr(result, "active_facts", []) or [])
            all_nodes = list(getattr(result, "all_nodes", []) or [])
            all_edges = list(getattr(result, "all_edges", []) or [])
            summary.update({
                "active_fact_count": len(active_facts),
                "node_count": len(all_nodes),
                "edge_count": len(all_edges),
            })
            summary["meaningful"] = bool(grounded_lines) if self.graph_backend == "cognee" else bool(active_facts or all_edges)
            return summary

        if tool_name == "quick_search":
            facts = list(getattr(result, "facts", []) or [])
            nodes = list(getattr(result, "nodes", []) or [])
            edges = list(getattr(result, "edges", []) or [])
            summary.update({
                "fact_count": len(facts),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "total_count": getattr(result, "total_count", None),
            })
            summary["meaningful"] = bool(grounded_lines) if self.graph_backend == "cognee" else bool(facts or edges)
            return summary

        if tool_name == "interview_agents":
            interviews = list(getattr(result, "interviews", []) or [])
            summary["interview_count"] = len(interviews)
            summary["meaningful"] = bool(interviews)
            return summary

        text = self._serialize_tool_result(result).strip()
        summary["text_length"] = len(text)
        summary["meaningful"] = bool(
            text
            and not text.startswith("Unknown tool:")
            and not text.startswith("Tool execution failed:")
        )
        return summary

    def _execute_tool_payload(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        report_context: str = "",
    ) -> Dict[str, Any]:
        """执行工具并返回文本结果和 evidence summary。"""
        logger.info(f"执行工具: {tool_name}, 参数: {parameters}")

        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                kwargs = dict(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx,
                )
                if self.graph_backend == "cognee":
                    kwargs["simulation_id"] = self.simulation_id
                result = self.zep_tools.insight_forge(**kwargs)
            elif tool_name == "panorama_search":
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ["true", "1", "yes"]
                kwargs = dict(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired,
                )
                if self.graph_backend == "cognee":
                    kwargs["simulation_id"] = self.simulation_id
                result = self.zep_tools.panorama_search(**kwargs)
            elif tool_name == "quick_search":
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                kwargs = dict(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit,
                )
                if self.graph_backend == "cognee":
                    kwargs["simulation_id"] = self.simulation_id
                result = self.zep_tools.quick_search(**kwargs)
            elif tool_name == "interview_agents":
                if not self._supports_interview_agents():
                    text = json.dumps(
                        {"summary": "interview_agents is unavailable for the current graph backend."},
                        ensure_ascii=False,
                        indent=2,
                    )
                    return {
                        "text": text,
                        "evidence": {
                            "tool_name": tool_name,
                            "evidence_sources": [],
                            "meaningful": False,
                        },
                    }
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents,
                )
            elif tool_name == "search_graph":
                logger.info("search_graph 已重定向到 quick_search")
                return self._execute_tool_payload("quick_search", parameters, report_context)
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name,
                )
            elif tool_name == "get_simulation_context":
                logger.info("get_simulation_context 已重定向到 insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool_payload("insight_forge", {"query": query}, report_context)
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type,
                )
                result = [n.to_dict() for n in nodes]
            else:
                text = (
                    f"Unknown tool: {tool_name}. Use one of these tools instead: "
                    "insight_forge, panorama_search, quick_search"
                )
                return {
                    "text": text,
                    "evidence": {
                        "tool_name": tool_name,
                        "evidence_sources": [],
                        "meaningful": False,
                    },
                }

            raw_text = self._serialize_tool_result(result)
            evidence = self._build_tool_evidence_summary(tool_name, result)
            return {
                "text": self._render_tool_result_for_llm(tool_name, result, raw_text, evidence),
                "log_text": raw_text,
                "evidence": evidence,
                "chat_facts": self._extract_chat_tool_lines(tool_name, result),
            }
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}, error: {str(e)}")
            return {
                "text": f"Tool execution failed: {str(e)}",
                "log_text": f"Tool execution failed: {str(e)}",
                "evidence": {
                    "tool_name": tool_name,
                    "evidence_sources": [],
                    "meaningful": False,
                    "error": str(e),
                },
                "chat_facts": [],
            }

    def _build_grounded_runtime_fallback(
        self,
        tool_evidence_records: List[Dict[str, Any]],
    ) -> str:
        """当图检索没有产出有效证据时，退化为只基于 runtime actions 的 grounded 文本。"""
        runtime_evidence = self._get_runtime_evidence()
        action_facts = [
            str(item).strip() for item in (runtime_evidence.get("action_facts") or []) if str(item).strip()
        ]
        current_round = runtime_evidence.get("current_round", 0)
        total_actions = runtime_evidence.get("total_actions", 0)

        lines = [
            "Grounded evidence for this part of the report is limited because graph retrieval returned sparse results. The notes below rely only on verified runtime actions from the simulation trace.",
        ]

        if current_round or total_actions:
            lines.append(
                f"Verified runtime scope: round {current_round} observed with {total_actions} total actions recorded so far."
            )

        if action_facts:
            lines.append("")
            lines.append("Verified observations:")
            for fact in action_facts[:6]:
                lines.append(f"- {fact}")
            lines.append("")
            lines.append(
                "A stronger synthesis should be generated only after graph tools return non-empty facts, relationships, or search matches."
            )
            return "\n".join(lines).strip()

        lines.append(
            "The current trace does not expose enough verified action-level evidence to support a reliable analytical narrative yet. Please retry after more runtime evidence is available."
        )
        if tool_evidence_records:
            sparse_tools = ", ".join(record.get("tool_name", "unknown") for record in tool_evidence_records)
            lines.append(f"Sparse tool calls observed: {sparse_tools}.")
        return "\n".join(lines).strip()

    def _finalize_section_output(
        self,
        section: "ReportSection",
        section_index: int,
        content: str,
        tool_calls_count: int,
        tool_evidence_records: List[Dict[str, Any]],
    ) -> str:
        """在写入章节前做最后一道 grounding 检查。"""
        final_answer = (content or "").strip()
        meaningful_evidence = any(record.get("meaningful") for record in tool_evidence_records)

        if not meaningful_evidence:
            logger.warning(
                f"章节 {section.title} 缺少有效工具证据，使用 runtime grounded fallback "
                f"（tool_calls={tool_calls_count}）"
            )
            final_answer = self._build_grounded_runtime_fallback(tool_evidence_records)
            if self.report_logger:
                self.report_logger.log(
                    action="section_grounded_fallback",
                    stage="generating",
                    section_title=section.title,
                    section_index=section_index,
                    details={
                        "message": "All tool calls were empty or low-signal; replaced section with runtime-grounded fallback.",
                        "tool_calls_count": tool_calls_count,
                        "tool_evidence": tool_evidence_records,
                    },
                )

        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count,
            )
        return final_answer
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        从LLM响应中解析工具调用

        支持的格式（按优先级）：
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. 裸 JSON（响应整体或单行就是一个工具调用 JSON）
        """
        tool_calls = []

        # 格式1: XML风格（标准格式）
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # 格式2: 兜底 - LLM 直接输出裸 JSON（没包 <tool_call> 标签）
        # 只在格式1未匹配时尝试，避免误匹配正文中的 JSON
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # 响应可能包含思考文字 + 裸 JSON，尝试提取最后一个 JSON 对象
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """校验解析出的 JSON 是否是合法的工具调用"""
        # 支持 {"name": ..., "parameters": ...} 和 {"tool": ..., "params": ...} 两种键名
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self._available_tool_names():
            # 统一键名为 name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """生成工具描述文本"""
        desc_parts = ["Available tools:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parameters: {params_desc}")
        return "\n".join(desc_parts)

    def _should_use_native_tools(self) -> bool:
        checker = getattr(self.llm, "prefers_native_tools", None)
        return bool(checker()) if callable(checker) else False

    def _get_native_llm_tools(self) -> List[Dict[str, Any]]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "insight_forge",
                    "description": TOOL_DESC_INSIGHT_FORGE,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Question or topic to analyze in depth"},
                            "report_context": {"type": "string", "description": "Current report-section context"},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "panorama_search",
                    "description": TOOL_DESC_PANORAMA_SEARCH,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query used for relevance ranking"},
                            "include_expired": {"type": "boolean", "description": "Whether to include expired or historical content"},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "quick_search",
                    "description": TOOL_DESC_QUICK_SEARCH,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query string"},
                            "limit": {"type": "integer", "description": "Number of results to return"},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        if self._supports_interview_agents():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "interview_agents",
                        "description": TOOL_DESC_INTERVIEW_AGENTS,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "interview_topic": {"type": "string", "description": "Interview topic or evidence need"},
                                "max_agents": {"type": "integer", "description": "Maximum number of agents to interview"},
                            },
                            "required": ["interview_topic"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return tools

    def _get_llm_tool_kwargs(self) -> Dict[str, Any]:
        if not self._should_use_native_tools():
            return {}
        return {
            "tools": self._get_native_llm_tools(),
            "tool_choice": "auto",
        }
    
    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        规划报告大纲
        
        使用LLM分析模拟需求，规划报告的目录结构
        
        Args:
            progress_callback: 进度回调函数
            
        Returns:
            ReportOutline: 报告大纲
        """
        logger.info("开始规划报告大纲...")
        
        if progress_callback:
            progress_callback("planning", 0, "正在分析模拟需求...")
        
        # 首先获取模拟上下文
        context_kwargs = {
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
        }
        if self.graph_backend == "cognee":
            context_kwargs["simulation_id"] = self.simulation_id
        context = self.zep_tools.get_simulation_context(**context_kwargs)
        
        if progress_callback:
            progress_callback("planning", 30, "正在生成报告大纲...")
        
        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, "正在解析大纲结构...")
            
            # 解析大纲
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "模拟分析报告"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, "大纲规划完成")
            
            logger.info(f"大纲规划完成: {len(sections)} 个章节")
            return outline
            
        except Exception as e:
            logger.error(f"大纲规划失败: {str(e)}")
            # 返回默认大纲（3个章节，作为fallback）
            return ReportOutline(
                title="情景推演分析报告",
                summary="基于模拟推演的趋势、风险与机会分析",
                sections=[
                    ReportSection(title="情景与核心发现"),
                    ReportSection(title="参与方行为分析"),
                    ReportSection(title="趋势、风险与机会")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        使用ReACT模式生成单个章节内容
        
        ReACT循环：
        1. Thought（思考）- 分析需要什么信息
        2. Action（行动）- 调用工具获取信息
        3. Observation（观察）- 分析工具返回结果
        4. 重复直到信息足够或达到最大次数
        5. Final Answer（最终回答）- 生成章节内容
        
        Args:
            section: 要生成的章节
            outline: 完整大纲
            previous_sections: 之前章节的内容（用于保持连贯性）
            progress_callback: 进度回调
            section_index: 章节索引（用于日志记录）
            
        Returns:
            章节内容（Markdown格式）
        """
        logger.info(f"ReACT生成章节: {section.title}")
        
        # 记录章节开始日志
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )

        # 构建用户prompt - 每个已完成章节各传入最大4000字
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # 每个章节最多4000字
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(This is the first section.)"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # ReACT循环
        tool_calls_count = 0
        max_iterations = 5  # 最大迭代轮数
        min_tool_calls = 3  # 最少工具调用次数
        conflict_retries = 0  # 工具调用与Final Answer同时出现的连续冲突次数
        used_tools = set()  # 记录已调用过的工具名
        tool_evidence_records: List[Dict[str, Any]] = []
        all_tools = self._available_tool_names()

        # 报告上下文，用于InsightForge的子问题生成
        report_context = f"Section title: {section.title}\nSimulation requirement: {self.simulation_requirement}"
        llm_tool_kwargs = self._get_llm_tool_kwargs()
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    f"深度检索与撰写中 ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )
            
            # 调用LLM
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096,
                **llm_tool_kwargs,
            )

            # 检查 LLM 返回是否为 None（API 异常或内容为空）
            if response is None:
                logger.warning(f"章节 {section.title} 第 {iteration + 1} 次迭代: LLM 返回 None")
                # 如果还有迭代次数，添加消息并重试
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(Empty response.)"})
                    messages.append({"role": "user", "content": "Continue generating the section content."})
                    continue
                # 最后一次迭代也返回 None，跳出循环进入强制收尾
                break

            logger.debug(f"LLM响应: {response[:200]}...")

            # 解析一次，复用结果
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── 冲突处理：LLM 同时输出了工具调用和 Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"章节 {section.title} 第 {iteration+1} 轮: "
                    f"LLM 同时输出工具调用和 Final Answer（第 {conflict_retries} 次冲突）"
                )

                if conflict_retries <= 2:
                    # 前两次：丢弃本次响应，要求 LLM 重新回复
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[Format error] Your reply included both a tool call and Final Answer, which is not allowed.\n"
                            "Each reply may do exactly one of the following:\n"
                            "- call one tool (output one <tool_call> block and do not include Final Answer)\n"
                            "- output the final section content (start with 'Final Answer:' and do not include <tool_call>)\n"
                            "Reply again and do only one of those two actions."
                        ),
                    })
                    continue
                else:
                    # 第三次：降级处理，截断到第一个工具调用，强制执行
                    logger.warning(
                        f"章节 {section.title}: 连续 {conflict_retries} 次冲突，"
                        "降级为截断执行第一个工具调用"
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # 记录 LLM 响应日志
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── 情况1：LLM 输出了 Final Answer ──
            if has_final_answer:
                # 工具调用次数不足，拒绝并要求继续调工具
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f" Unused tools so far: {', '.join(unused_tools)}." if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # 正常结束
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"章节 {section.title} 生成完成（工具调用: {tool_calls_count}次）")
                return self._finalize_section_output(
                    section=section,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count,
                    tool_evidence_records=tool_evidence_records,
                )

            # ── 情况2：LLM 尝试调用工具 ──
            if has_tool_calls:
                # 工具额度已耗尽 → 明确告知，要求输出 Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # 只执行第一个工具调用
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"LLM 尝试调用 {len(tool_calls)} 个工具，只执行第一个: {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                payload = self._execute_tool_payload(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )
                result = payload.get("text", "")
                log_result = payload.get("log_text", result)
                tool_evidence_records.append(payload.get("evidence") or {
                    "tool_name": call["name"],
                    "meaningful": False,
                })

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=log_result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # 构建未使用工具提示
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list=", ".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # ── 情况3：既没有工具调用，也没有 Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # 工具调用次数不足，推荐未用过的工具
                unused_tools = all_tools - used_tools
                unused_hint = f" Unused tools so far: {', '.join(unused_tools)}." if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # 工具调用已足够，LLM 输出了内容但没带 "Final Answer:" 前缀
            # 这时依然需要经过 grounding guard，避免空工具结果直接放行
            logger.info(f"章节 {section.title} 未检测到 'Final Answer:' 前缀，尝试以当前内容完成章节（工具调用: {tool_calls_count}次）")
            final_answer = response.strip()
            return self._finalize_section_output(
                section=section,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count,
                tool_evidence_records=tool_evidence_records,
            )
        
        # 达到最大迭代次数，强制生成内容
        logger.warning(f"章节 {section.title} 达到最大迭代次数，强制生成")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        # 检查强制收尾时 LLM 返回是否为 None
        if response is None:
            logger.error(f"章节 {section.title} 强制收尾时 LLM 返回 None，使用默认错误提示")
            final_answer = "(This section could not be generated because the LLM returned an empty response. Please retry later.)"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        return self._finalize_section_output(
            section=section,
            section_index=section_index,
            content=final_answer,
            tool_calls_count=tool_calls_count,
            tool_evidence_records=tool_evidence_records,
        )
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        生成完整报告（分章节实时输出）
        
        每个章节生成完成后立即保存到文件夹，不需要等待整个报告完成。
        文件结构：
        reports/{report_id}/
            meta.json       - 报告元信息
            outline.json    - 报告大纲
            progress.json   - 生成进度
            section_01.md   - 第1章节
            section_02.md   - 第2章节
            ...
            full_report.md  - 完整报告
        
        Args:
            progress_callback: 进度回调函数 (stage, progress, message)
            report_id: 报告ID（可选，如果不传则自动生成）
            
        Returns:
            Report: 完整报告
        """
        import uuid
        
        # 如果没有传入 report_id，则自动生成
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # 已完成的章节标题列表（用于进度追踪）
        completed_section_titles = []
        
        try:
            # 初始化：创建报告文件夹并保存初始状态
            ReportManager._ensure_report_folder(report_id)
            
            # 初始化日志记录器（结构化日志 agent_log.jsonl）
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # 初始化控制台日志记录器（console_log.txt）
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, "初始化报告...",
                completed_sections=[]
            )
            ReportManager.save_report(report)

            self._ensure_generation_readiness()
            
            # 阶段1: 规划大纲
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "开始规划报告大纲...",
                completed_sections=[]
            )
            
            # 记录规划开始日志
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, "开始规划报告大纲...")
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # 记录规划完成日志
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # 保存大纲到文件
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"大纲规划完成，共{len(outline.sections)}个章节",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(f"大纲已保存到文件: {report_id}/outline.json")
            
            # 阶段2: 逐章节生成（分章节保存）
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # 保存内容用于上下文
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # 更新进度
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"正在生成章节: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )
                
                if progress_callback:
                    progress_callback(
                        "generating", 
                        base_progress, 
                        f"正在生成章节: {section.title} ({section_num}/{total_sections})"
                    )
                
                # 生成主章节内容
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # 保存章节
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # 记录章节完成日志
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"章节已保存: {report_id}/section_{section_num:02d}.md")
                
                # 更新进度
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    f"章节 {section.title} 已完成",
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # 阶段3: 组装完整报告
            if progress_callback:
                progress_callback("generating", 95, "正在组装完整报告...")
            
            ReportManager.update_progress(
                report_id, "generating", 95, "正在组装完整报告...",
                completed_sections=completed_section_titles
            )
            
            # 使用ReportManager组装完整报告
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # 计算总耗时
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # 记录报告完成日志
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # 保存最终报告
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "报告生成完成",
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, "报告生成完成")
            
            logger.info(f"报告生成完成: {report_id}")
            
            # 关闭控制台日志记录器
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(f"报告生成失败: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # 记录错误日志
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # 保存失败状态
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"报告生成失败: {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # 忽略保存失败的错误
            
            # 关闭控制台日志记录器
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        与Report Agent对话
        
        在对话中Agent可以自主调用检索工具来回答问题
        
        Args:
            message: 用户消息
            chat_history: 对话历史
            
        Returns:
            {
                "response": "Agent回复",
                "tool_calls": [调用的工具列表],
                "sources": [信息来源]
            }
        """
        logger.info(f"Report Agent对话: {message[:50]}...")
        
        chat_history = chat_history or []
        
        # 获取已生成的报告内容
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # 限制报告长度，避免上下文过长
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [报告内容已截断] ..."
        except Exception as e:
            logger.warning(f"获取报告内容失败: {e}")
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(No report available yet.)",
            tools_description=self._get_tools_description(),
        )

        # 构建消息
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加历史对话
        for h in chat_history[-10:]:  # 限制历史长度
            messages.append(h)
        
        # 添加用户消息
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # ReACT循环（简化版）
        tool_calls_made = []
        tool_fact_records: List[Dict[str, Any]] = []
        max_iterations = 2  # 减少迭代轮数
        llm_tool_kwargs = self._get_llm_tool_kwargs()
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                **llm_tool_kwargs,
            )
            
            # 解析工具调用
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                if self.graph_backend == "cognee" and not tool_fact_records:
                    auto_call = {
                        "name": "quick_search",
                        "parameters": {"query": self._build_chat_search_query(message), "limit": 8},
                    }
                    payload = self._execute_tool_payload(auto_call["name"], auto_call["parameters"])
                    tool_calls_made.append(auto_call)
                    tool_fact_records.append({
                        "tool": auto_call["name"],
                        "query": auto_call["parameters"]["query"],
                        "facts": payload.get("chat_facts") or [],
                    })
                if self.graph_backend == "cognee" and tool_fact_records:
                    return {
                        "response": self._build_fact_bound_chat_response(tool_fact_records),
                        "tool_calls": tool_calls_made,
                        "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made],
                    }

                return {
                    "response": self._clean_chat_response(response),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # 执行工具调用（限制数量）
            tool_results = []
            for call in tool_calls[:1]:  # 每轮最多执行1次工具调用
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                payload = self._execute_tool_payload(call["name"], call.get("parameters", {}))
                result = payload.get("text", "")
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # 限制结果长度
                })
                tool_calls_made.append(call)
                tool_fact_records.append({
                    "tool": call["name"],
                    "query": call.get("parameters", {}).get("query", ""),
                    "facts": payload.get("chat_facts") or [],
                })
            
            # 将结果添加到消息
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[{r['tool']} result]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # 达到最大迭代，获取最终响应
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        if self.graph_backend == "cognee" and tool_fact_records:
            return {
                "response": self._build_fact_bound_chat_response(tool_fact_records),
                "tool_calls": tool_calls_made,
                "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
            }
        
        return {
            "response": self._clean_chat_response(final_response),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    报告管理器
    
    负责报告的持久化存储和检索
    
    文件结构（分章节输出）：
    reports/
      {report_id}/
        meta.json          - 报告元信息和状态
        outline.json       - 报告大纲
        progress.json      - 生成进度
        section_01.md      - 第1章节
        section_02.md      - 第2章节
        ...
        full_report.md     - 完整报告
    """
    
    # 报告存储目录
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """确保报告根目录存在"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """获取报告文件夹路径"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """确保报告文件夹存在并返回路径"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """获取报告元信息文件路径"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """获取完整报告Markdown文件路径"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """获取大纲文件路径"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """获取进度文件路径"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """获取章节Markdown文件路径"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """获取 Agent 日志文件路径"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """获取控制台日志文件路径"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        获取控制台日志内容
        
        这是报告生成过程中的控制台输出日志（INFO、WARNING等），
        与 agent_log.jsonl 的结构化日志不同。
        
        Args:
            report_id: 报告ID
            from_line: 从第几行开始读取（用于增量获取，0 表示从头开始）
            
        Returns:
            {
                "logs": [日志行列表],
                "total_lines": 总行数,
                "from_line": 起始行号,
                "has_more": 是否还有更多日志
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # 保留原始日志行，去掉末尾换行符
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # 已读取到末尾
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        获取完整的控制台日志（一次性获取全部）
        
        Args:
            report_id: 报告ID
            
        Returns:
            日志行列表
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        获取 Agent 日志内容
        
        Args:
            report_id: 报告ID
            from_line: 从第几行开始读取（用于增量获取，0 表示从头开始）
            
        Returns:
            {
                "logs": [日志条目列表],
                "total_lines": 总行数,
                "from_line": 起始行号,
                "has_more": 是否还有更多日志
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # 跳过解析失败的行
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # 已读取到末尾
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        获取完整的 Agent 日志（用于一次性获取全部）
        
        Args:
            report_id: 报告ID
            
        Returns:
            日志条目列表
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        保存报告大纲
        
        在规划阶段完成后立即调用
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"大纲已保存: {report_id}")
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        保存单个章节

        在每个章节生成完成后立即调用，实现分章节输出

        Args:
            report_id: 报告ID
            section_index: 章节索引（从1开始）
            section: 章节对象

        Returns:
            保存的文件路径
        """
        cls._ensure_report_folder(report_id)

        # 构建章节Markdown内容 - 清理可能存在的重复标题
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # 保存文件
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"章节已保存: {report_id}/{file_suffix}")
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        清理章节内容
        
        1. 移除内容开头与章节标题重复的Markdown标题行
        2. 将所有 ### 及以下级别的标题转换为粗体文本
        
        Args:
            content: 原始内容
            section_title: 章节标题
            
        Returns:
            清理后的内容
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 检查是否是Markdown标题行
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # 检查是否是与章节标题重复的标题（跳过前5行内的重复）
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # 将所有级别的标题（#, ##, ###, ####等）转换为粗体
                # 因为章节标题由系统添加，内容中不应有任何标题
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # 添加空行
                continue
            
            # 如果上一行是被跳过的标题，且当前行为空，也跳过
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # 移除开头的空行
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # 移除开头的分隔线
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # 同时移除分隔线后的空行
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        更新报告生成进度
        
        前端可以通过读取progress.json获取实时进度
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """获取报告生成进度"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        获取已生成的章节列表
        
        返回所有已保存的章节文件信息
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 从文件名解析章节索引
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        组装完整报告
        
        从已保存的章节文件组装完整报告，并进行标题清理
        """
        folder = cls._get_report_folder(report_id)
        
        # 构建报告头部
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # 按顺序读取所有章节文件
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # 后处理：清理整个报告的标题问题
        md_content = cls._post_process_report(md_content, outline)
        
        # 保存完整报告
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"完整报告已组装: {report_id}")
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        后处理报告内容
        
        1. 移除重复的标题
        2. 保留报告主标题(#)和章节标题(##)，移除其他级别的标题(###, ####等)
        3. 清理多余的空行和分隔线
        
        Args:
            content: 原始报告内容
            outline: 报告大纲
            
        Returns:
            处理后的内容
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # 收集大纲中的所有章节标题
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # 检查是否是标题行
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # 检查是否是重复标题（在连续5行内出现相同内容的标题）
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # 跳过重复标题及其后的空行
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # 标题层级处理：
                # - # (level=1) 只保留报告主标题
                # - ## (level=2) 保留章节标题
                # - ### 及以下 (level>=3) 转换为粗体文本
                
                if level == 1:
                    if title == outline.title:
                        # 保留报告主标题
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # 章节标题错误使用了#，修正为##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # 其他一级标题转为粗体
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # 保留章节标题
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # 非章节的二级标题转为粗体
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # ### 及以下级别的标题转换为粗体文本
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # 跳过标题后紧跟的分隔线
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # 标题后只保留一个空行
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # 清理连续的多个空行（保留最多2个）
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """保存报告元信息和完整报告"""
        cls._ensure_report_folder(report.report_id)
        
        # 保存元信息JSON
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # 保存大纲
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # 保存完整Markdown报告
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(f"报告已保存: {report.report_id}")
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """获取报告"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # 兼容旧格式：检查直接存储在reports目录下的文件
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建Report对象
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # 如果markdown_content为空，尝试从full_report.md读取
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """根据模拟ID获取报告"""
        cls._ensure_reports_dir()
        reports = []
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # 新格式：文件夹
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    reports.append(report)
            # 兼容旧格式：JSON文件
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    reports.append(report)

        if not reports:
            return None

        completed_reports = [r for r in reports if r.status == ReportStatus.COMPLETED]
        if completed_reports:
            completed_reports.sort(
                key=lambda r: ((r.completed_at or r.created_at or ""), (r.created_at or ""), r.report_id),
                reverse=True,
            )
            return completed_reports[0]

        reports.sort(
            key=lambda r: ((r.created_at or ""), r.report_id),
            reverse=True,
        )
        return reports[0]
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """列出报告"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # 新格式：文件夹
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # 兼容旧格式：JSON文件
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # 按创建时间倒序
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """删除报告（整个文件夹）"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # 新格式：删除整个文件夹
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"报告文件夹已删除: {report_id}")
            return True
        
        # 兼容旧格式：删除单独的文件
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
