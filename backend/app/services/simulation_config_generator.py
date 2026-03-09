"""
模拟配置智能生成器
使用LLM根据模拟需求、文档内容、图谱信息自动生成细致的模拟参数
实现全程自动化，无需人工设置参数

采用分步生成策略，避免一次性生成过长内容导致失败：
1. 生成时间配置
2. 生成事件配置
3. 分批生成Agent配置
4. 生成平台配置
"""

import json
import math
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from .graph_entities import EntityNode

logger = get_logger('mirofish.simulation_config')

# 默认活动节律配置（可按世界观调整）
DEFAULT_ACTIVITY_RHYTHM_CONFIG = {
    # 深夜时段（几乎无人活动）
    "dead_hours": [0, 1, 2, 3, 4, 5],
    # 早间时段（逐渐醒来）
    "morning_hours": [6, 7, 8],
    # 工作时段
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # 晚间高峰（最活跃）
    "peak_hours": [19, 20, 21, 22],
    # 夜间时段（活跃度下降）
    "night_hours": [23],
    # 活跃度系数
    "activity_multipliers": {
        "dead": 0.05,      # 凌晨几乎无人
        "morning": 0.4,    # 早间逐渐活跃
        "work": 0.7,       # 工作时段中等
        "peak": 1.5,       # 晚间高峰
        "night": 0.5       # 深夜下降
    }
}


@dataclass
class AgentActivityConfig:
    """单个Agent的活动配置"""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str
    
    # 活跃度配置 (0.0-1.0)
    activity_level: float = 0.5  # 整体活跃度
    
    # 发言频率（每小时预期发言次数）
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0
    
    # 活跃时间段（24小时制，0-23）
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))
    
    # 响应速度（对热点事件的反应延迟，单位：模拟分钟）
    response_delay_min: int = 5
    response_delay_max: int = 60
    
    # 情感倾向 (-1.0到1.0，负面到正面)
    sentiment_bias: float = 0.0
    
    # 立场（对特定话题的态度）
    stance: str = "neutral"  # supportive, opposing, neutral, observer
    
    # 影响力权重（决定其发言被其他Agent看到的概率）
    influence_weight: float = 1.0


@dataclass  
class TimeSimulationConfig:
    """时间模拟配置（默认昼夜节律，可按世界观调整）"""
    # 模拟总时长（模拟小时数）
    total_simulation_hours: int = 72  # 默认模拟72小时（3天）
    
    # 每轮代表的时间（模拟分钟）- 默认60分钟（1小时），加快时间流速
    minutes_per_round: int = 60
    
    # 每小时激活的Agent数量范围
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20
    
    # 高峰时段（默认晚间19-22点，可按世界观调整）
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5
    
    # 低谷时段（默认凌晨0-5点，几乎无人活动）
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05  # 凌晨活跃度极低
    
    # 早间时段
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4
    
    # 工作时段
    work_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """事件配置"""
    # 初始事件（模拟开始时的触发事件）
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)
    
    # 定时事件（在特定时间触发的事件）
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # 热点话题关键词
    hot_topics: List[str] = field(default_factory=list)
    
    # 舆论引导方向
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """平台特定配置"""
    platform: str  # twitter or reddit
    
    # 推荐算法权重
    recency_weight: float = 0.4  # 时间新鲜度
    popularity_weight: float = 0.3  # 热度
    relevance_weight: float = 0.3  # 相关性
    
    # 病毒传播阈值（达到多少互动后触发扩散）
    viral_threshold: int = 10
    
    # 回声室效应强度（相似观点聚集程度）
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """完整的模拟参数配置"""
    # 基础信息
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str
    
    # 时间配置
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)
    
    # Agent配置列表
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)
    
    # 事件配置
    event_config: EventConfig = field(default_factory=EventConfig)
    
    # 平台配置
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None
    
    # LLM配置
    llm_model: str = ""
    llm_base_url: str = ""
    
    # 生成元数据
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # LLM的推理说明
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    模拟配置智能生成器
    
    使用LLM分析模拟需求、文档内容、图谱实体信息，
    自动生成最佳的模拟参数配置
    
    采用分步生成策略：
    1. 生成时间配置和事件配置（轻量级）
    2. 分批生成Agent配置（每批10-20个）
    3. 生成平台配置
    """
    
    # 上下文最大字符数
    MAX_CONTEXT_LENGTH = 50000
    # 每批生成的Agent数量
    AGENTS_PER_BATCH = 15
    
    # 各步骤的上下文截断长度（字符数）
    TIME_CONFIG_CONTEXT_LENGTH = 10000   # 时间配置
    EVENT_CONFIG_CONTEXT_LENGTH = 8000   # 事件配置
    ENTITY_SUMMARY_LENGTH = 300          # 实体摘要
    AGENT_SUMMARY_LENGTH = 300           # Agent配置中的实体摘要
    ENTITIES_PER_TYPE_DISPLAY = 20       # 每类实体显示数量
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.get_stage_model('sim_config')
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationParameters:
        """
        智能生成完整的模拟配置（分步生成）
        
        Args:
            simulation_id: 模拟ID
            project_id: 项目ID
            graph_id: 图谱ID
            simulation_requirement: 模拟需求描述
            document_text: 原始文档内容
            entities: 过滤后的实体列表
            enable_twitter: 是否启用Twitter
            enable_reddit: 是否启用Reddit
            progress_callback: 进度回调函数(current_step, total_steps, message)
            
        Returns:
            SimulationParameters: 完整的模拟参数
        """
        logger.info(f"开始智能生成模拟配置: simulation_id={simulation_id}, 实体数={len(entities)}")
        
        # 计算总步骤数
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches  # 时间配置 + 事件配置 + N批Agent + 平台配置
        current_step = 0
        
        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")
        
        # 1. 构建基础上下文信息
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities
        )
        
        reasoning_parts = []
        
        # ========== 步骤1: 生成时间配置 ==========
        report_progress(1, "生成时间配置...")
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(f"Time configuration: {time_config_result.get('reasoning', 'success')}")
        
        # ========== 步骤2: 生成事件配置 ==========
        report_progress(2, "生成事件配置和热点话题...")
        event_config_result = self._generate_event_config(context, simulation_requirement, entities)
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(f"Event configuration: {event_config_result.get('reasoning', 'success')}")
        
        # ========== 步骤3-N: 分批生成Agent配置 ==========
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]
            
            report_progress(
                3 + batch_idx,
                f"生成Agent配置 ({start_idx + 1}-{end_idx}/{len(entities)})..."
            )
            
            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement
            )
            all_agent_configs.extend(batch_configs)
        
        reasoning_parts.append(f"Agent configuration: generated {len(all_agent_configs)} entries successfully")
        
        # ========== 为初始帖子分配发布者 Agent ==========
        logger.info("为初始帖子分配合适的发布者 Agent...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len([p for p in event_config.initial_posts if p.get("poster_agent_id") is not None])
        reasoning_parts.append(f"Initial post assignment: {assigned_count} posts assigned to poster agents")
        
        # ========== 最后一步: 生成平台配置 ==========
        report_progress(total_steps, "生成平台配置...")
        twitter_config = None
        reddit_config = None
        
        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5
            )
        
        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6
            )
        
        # 构建最终参数
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts)
        )
        
        logger.info(f"模拟配置生成完成: {len(params.agent_configs)} 个Agent配置")
        
        return params
    
    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode]
    ) -> str:
        """构建LLM上下文，截断到最大长度"""
        
        # 实体摘要
        entity_summary = self._summarize_entities(entities)
        
        # 构建上下文
        context_parts = [
            f"## Simulation Requirement\n{simulation_requirement}",
            f"\n## Entity Information ({len(entities)} entities)\n{entity_summary}",
        ]
        
        current_length = sum(len(p) for p in context_parts)
        remaining_length = self.MAX_CONTEXT_LENGTH - current_length - 500  # 留500字符余量
        
        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(document truncated)"
            context_parts.append(f"\n## Source Document Content\n{doc_text}")
        
        return "\n".join(context_parts)
    
    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """生成实体摘要"""
        lines = []
        
        # 按类型分组
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)
        
        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)} entities)")
            # 使用配置的显示数量和摘要长度
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (e.summary[:summary_len] + "...") if len(e.summary) > summary_len else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... {len(type_entities) - display_count} more")
        
        return "\n".join(lines)
    
    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """带重试的LLM调用，包含JSON修复逻辑"""
        import re
        
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)  # 每次重试降低温度
                    # 不设置max_tokens，让LLM自由发挥
                )
                
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                
                # 检查是否被截断
                if finish_reason == 'length':
                    logger.warning(f"LLM输出被截断 (attempt {attempt+1})")
                    content = self._fix_truncated_json(content)
                
                # 尝试解析JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON解析失败 (attempt {attempt+1}): {str(e)[:80]}")
                    
                    # 尝试修复JSON
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed
                    
                    last_error = e
                    
            except Exception as e:
                logger.warning(f"LLM调用失败 (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(2 * (attempt + 1))
        
        raise last_error or Exception("LLM调用失败")
    
    def _fix_truncated_json(self, content: str) -> str:
        """修复被截断的JSON"""
        content = content.strip()
        
        # 计算未闭合的括号
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        # 检查是否有未闭合的字符串
        if content and content[-1] not in '",}]':
            content += '"'
        
        # 闭合括号
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content
    
    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """尝试修复配置JSON"""
        import re
        
        # 修复被截断的情况
        content = self._fix_truncated_json(content)
        
        # 提取JSON部分
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            
            # 移除字符串中的换行符
            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s
            
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)
            
            try:
                return json.loads(json_str)
            except:
                # 尝试移除所有控制字符
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass
        
        return None
    
    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """生成时间配置"""
        # 使用配置的上下文截断长度
        context_truncated = context[:self.TIME_CONFIG_CONTEXT_LENGTH]
        
        # 计算最大允许值（80%的agent数）
        max_agents_allowed = max(1, int(num_entities * 0.9))
        
        prompt = f"""Generate the time simulation configuration based on the scenario below.

{context_truncated}

## Task
Generate the time configuration as JSON.

### Guiding principles (reference only; adapt them to the world, event type, and participating groups)
- Follow the worldbuilding, role identities, institutions, and daily rhythms whenever they are explicitly defined
- If the world rules are unclear, you may fall back to a generic day-night activity pattern
- Default pattern: low activity from 00:00-05:00, increasing activity from 06:00-08:00, moderate daytime activity from 09:00-18:00, peak activity from 19:00-22:00, then decline after 23:00
- Important: the example values below are only reference values. Adjust them based on the event and the participating groups.
  - Example: students or apprentices may peak around 21:00-23:00; media or messenger channels may stay active nearly all day; official institutions may speak mostly during office hours.
  - Example: nocturnal species, wartime conditions, religious rituals, festivals, or sudden crises may break the generic day-night rhythm.

### Return JSON only (no markdown)

Example:
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "Why this time configuration fits the scenario"
}}

Field guide:
- total_simulation_hours (int): total duration of the simulation, usually 24-168 hours
- minutes_per_round (int): duration of each round, usually 30-120 minutes, with 60 as a common default
- agents_per_hour_min (int): minimum active agents per hour (range: 1-{max_agents_allowed})
- agents_per_hour_max (int): maximum active agents per hour (range: 1-{max_agents_allowed})
- peak_hours (int array): peak activity hours adjusted to the participating groups
- off_peak_hours (int array): low activity hours, usually late night and early morning
- morning_hours (int array): morning period
- work_hours (int array): work or institutional hours
- reasoning (string): brief explanation of why this configuration fits"""

        system_prompt = "You are an expert in scenario simulation rhythm planning. Return pure JSON. Follow the worldbuilding and role rhythms first; only fall back to a generic day-night pattern when the scenario is underspecified."
        
        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"时间配置LLM生成失败: {e}, 使用默认配置")
            return self._get_default_time_config(num_entities)
    
    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """获取默认时间配置（通用昼夜节律）"""
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,  # 每轮1小时，加快时间流速
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "Used the default day-night rhythm configuration (1 hour per round)."
        }

    def _coerce_int(self, value: Any, default: int) -> int:
        """Best-effort integer coercion for LLM-produced config values."""
        if isinstance(value, bool):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _split_compact_hour_string(self, raw: str) -> List[int]:
        """Split compact hour strings like '012345' or '10111213' into hours."""
        digits = re.sub(r'[^0-9]', '', raw or '')
        if not digits:
            return []

        candidates: List[List[int]] = []

        def backtrack(index: int, parts: List[int]):
            if index == len(digits):
                candidates.append(parts.copy())
                return

            for width in (1, 2):
                if index + width > len(digits):
                    continue
                token = digits[index:index + width]
                if width == 2 and token.startswith('0'):
                    continue
                value = int(token)
                if 0 <= value <= 23:
                    parts.append(value)
                    backtrack(index + width, parts)
                    parts.pop()

        backtrack(0, [])
        if not candidates:
            return []

        def score(sequence: List[int]) -> tuple[int, int]:
            discontinuities = sum(
                1 for i in range(1, len(sequence))
                if sequence[i] - sequence[i - 1] != 1
            )
            return discontinuities, len(sequence)

        return min(candidates, key=score)

    def _normalize_hour_list(self, value: Any, default: List[int]) -> List[int]:
        """Normalize hour arrays produced by the LLM into integer 0-23 lists."""
        normalized: List[int] = []

        def append_hours(raw: Any):
            if isinstance(raw, bool) or raw is None:
                return
            if isinstance(raw, list):
                for item in raw:
                    append_hours(item)
                return
            if isinstance(raw, int):
                parsed = [raw] if 0 <= raw <= 23 else self._split_compact_hour_string(str(raw))
            elif isinstance(raw, float) and raw.is_integer():
                whole = int(raw)
                parsed = [whole] if 0 <= whole <= 23 else self._split_compact_hour_string(str(whole))
            elif isinstance(raw, str):
                text = raw.strip()
                if not text:
                    return
                tokens = re.findall(r'\d+', text)
                parsed = []
                if len(tokens) > 1:
                    for token in tokens:
                        parsed.extend(self._split_compact_hour_string(token))
                else:
                    parsed = self._split_compact_hour_string(text)
            else:
                return

            for hour in parsed:
                if 0 <= hour <= 23 and hour not in normalized:
                    normalized.append(hour)

        append_hours(value)
        return normalized or list(default)
    
    def _parse_time_config(self, result: Dict[str, Any], num_entities: int) -> TimeSimulationConfig:
        """解析时间配置结果，并验证agents_per_hour值不超过总agent数"""
        # 获取原始值
        agents_per_hour_min = self._coerce_int(
            result.get("agents_per_hour_min", max(1, num_entities // 15)),
            max(1, num_entities // 15)
        )
        agents_per_hour_max = self._coerce_int(
            result.get("agents_per_hour_max", max(5, num_entities // 5)),
            max(5, num_entities // 5)
        )
        
        # 验证并修正：确保不超过总agent数
        if agents_per_hour_min > num_entities:
            logger.warning(f"agents_per_hour_min ({agents_per_hour_min}) 超过总Agent数 ({num_entities})，已修正")
            agents_per_hour_min = max(1, num_entities // 10)
        
        if agents_per_hour_max > num_entities:
            logger.warning(f"agents_per_hour_max ({agents_per_hour_max}) 超过总Agent数 ({num_entities})，已修正")
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)
        
        # 确保 min < max
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(f"agents_per_hour_min >= max，已修正为 {agents_per_hour_min}")
        
        return TimeSimulationConfig(
            total_simulation_hours=self._coerce_int(result.get("total_simulation_hours", 72), 72),
            minutes_per_round=self._coerce_int(result.get("minutes_per_round", 60), 60),  # 默认每轮1小时
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=self._normalize_hour_list(result.get("peak_hours"), [19, 20, 21, 22]),
            off_peak_hours=self._normalize_hour_list(result.get("off_peak_hours"), [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,  # 凌晨几乎无人
            morning_hours=self._normalize_hour_list(result.get("morning_hours"), [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=self._normalize_hour_list(result.get("work_hours"), list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5
        )
    
    def _generate_event_config(
        self, 
        context: str, 
        simulation_requirement: str,
        entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """生成事件配置"""
        
        # 获取可用的实体类型列表，供 LLM 参考
        entity_types_available = list(set(
            e.get_entity_type() or "Unknown" for e in entities
        ))
        
        # 为每种类型列出代表性实体名称
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)
        
        type_info = "\n".join([
            f"- {t}: {', '.join(examples)}" 
            for t, examples in type_examples.items()
        ])
        
        # 使用配置的上下文截断长度
        context_truncated = context[:self.EVENT_CONFIG_CONTEXT_LENGTH]
        
        prompt = f"""Generate the event configuration for the following scenario.

Simulation requirement: {simulation_requirement}

{context_truncated}

## Available entity types and examples
{type_info}

## Task
Generate the event configuration as JSON:
- extract the core topics or hot keywords
- describe the likely narrative direction or public storyline
- design initial messages, statements, or posts, and **every post must include a poster_type**

Important: poster_type must be selected from the "available entity types" list above so that each initial post can be assigned to an appropriate agent.
For example: official statements should come from Official, University, or GovernmentAgency types; news or announcements should come from MediaOutlet or a similar type; personal viewpoints should come from Student, Person, Leader, or a similar type.

Return JSON only (no markdown):
{{
    "hot_topics": ["topic 1", "topic 2", ...],
    "narrative_direction": "<description of the scenario direction or public narrative>",
    "initial_posts": [
        {{"content": "post content", "poster_type": "entity type (must be chosen from the available types)"}},
        ...
    ],
    "reasoning": "<brief explanation>"
}}"""

        system_prompt = "You are an expert in scenario event design. Return pure JSON. poster_type must exactly match one of the available entity types and fit that role's identity." 
        
        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"事件配置LLM生成失败: {e}, 使用默认配置")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "Used the default event configuration."
            }
    
    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """解析事件配置结果"""
        return EventConfig(
            initial_posts=self._normalize_initial_posts(result.get("initial_posts", [])),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", "")
        )

    def _normalize_initial_posts(self, value: Any) -> List[Dict[str, Any]]:
        """Normalize LLM-produced initial_posts into a list of post dicts."""
        if value is None:
            return []

        raw_posts = value if isinstance(value, list) else [value]
        normalized_posts: List[Dict[str, Any]] = []

        for post in raw_posts:
            if isinstance(post, str):
                content = post.strip()
                if content:
                    normalized_posts.append({
                        "content": content,
                        "poster_type": "Unknown",
                    })
                continue

            if not isinstance(post, dict):
                continue

            content = (
                post.get("content")
                or post.get("message")
                or post.get("text")
                or post.get("body")
                or ""
            )
            content = str(content).strip()
            if not content:
                continue

            poster_type = post.get("poster_type") or post.get("author_type") or "Unknown"
            normalized_posts.append({
                **post,
                "content": content,
                "poster_type": str(poster_type).strip() or "Unknown",
            })

        return normalized_posts
    
    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        为初始帖子分配合适的发布者 Agent
        
        根据每个帖子的 poster_type 匹配最合适的 agent_id
        """
        if not event_config.initial_posts:
            return event_config

        def normalize_label(value: str) -> str:
            return re.sub(r'[^a-z0-9]+', '', (value or '').strip().lower())

        def tokenize_label(value: str) -> List[str]:
            return re.findall(r'[a-z0-9]+', (value or '').strip().lower())

        def build_name_variants(value: str) -> set[str]:
            tokens = tokenize_label(value)
            if not tokens:
                return set()

            variants = {
                normalize_label(value),
                normalize_label(' '.join(tokens)),
            }

            meaningful_tokens = [token for token in tokens if token not in {'the', 'a', 'an'}]
            if meaningful_tokens:
                variants.add(normalize_label(' '.join(meaningful_tokens)))

            if 'the' in tokens:
                split_idx = tokens.index('the')
                before_the = [token for token in tokens[:split_idx] if token not in {'the', 'a', 'an'}]
                after_the = [token for token in tokens[split_idx + 1:] if token not in {'the', 'a', 'an'}]
                if before_the and after_the:
                    variants.add(normalize_label(' '.join(after_the + before_the)))
                    variants.add(normalize_label(' '.join(before_the + after_the)))

            if len(meaningful_tokens) == 2:
                variants.add(normalize_label(' '.join(reversed(meaningful_tokens))))

            return {variant for variant in variants if variant}

        non_poster_types = {
            'documentchunk',
            'textdocument',
            'entitytype',
            'textsummary',
            'schema',
        }

        poster_eligible_agents = [
            agent for agent in agent_configs
            if normalize_label(agent.entity_type) not in non_poster_types
        ] or agent_configs

        def choose_agent(candidates: List[AgentActivityConfig], bucket_key: str) -> Optional[AgentActivityConfig]:
            if not candidates:
                return None
            idx = used_indices.get(bucket_key, 0) % len(candidates)
            used_indices[bucket_key] = idx + 1
            return candidates[idx]
        
        # 按实体类型建立 agent 索引
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        agents_by_name: Dict[str, List[AgentActivityConfig]] = {}
        agent_name_tokens: List[tuple[set[str], AgentActivityConfig]] = []
        for agent in poster_eligible_agents:
            etype = normalize_label(agent.entity_type)
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)

            for entity_name_key in build_name_variants(agent.entity_name):
                if entity_name_key not in agents_by_name:
                    agents_by_name[entity_name_key] = []
                agents_by_name[entity_name_key].append(agent)

            meaningful_tokens = {
                token for token in tokenize_label(agent.entity_name)
                if token not in {'the', 'a', 'an'}
            }
            if meaningful_tokens:
                agent_name_tokens.append((meaningful_tokens, agent))
        
        # 类型映射表（处理 LLM 可能输出的不同格式）
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person", "actor"],
            "professor": ["professor", "expert", "teacher", "actor"],
            "alumni": ["alumni", "person", "actor"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni", "actor", "character"],
            "actor": ["actor", "person", "character", "student", "alumni", "professor"],
        }
        
        # 记录每种类型已使用的 agent 索引，避免重复使用同一个 agent
        used_indices: Dict[str, int] = {}
        normalized_posts = self._normalize_initial_posts(event_config.initial_posts)
        
        updated_posts = []
        for post in normalized_posts:
            poster_type = post.get("poster_type", "")
            poster_type_key = normalize_label(poster_type)
            content = post.get("content", "")
            content_key = normalize_label(content)
            
            # 尝试找到匹配的 agent
            matched_agent_id = None

            # 1. poster_type 实际上可能是实体名称，先做名称精确匹配
            if poster_type_key in agents_by_name:
                matched_agent = choose_agent(agents_by_name[poster_type_key], f"name:{poster_type_key}")
                matched_agent_id = matched_agent.agent_id if matched_agent else None

            # 2. 从内容中识别被点名的实体（优先选择名字更长、更具体的匹配）
            if matched_agent_id is None and content_key:
                content_name_matches = []
                for agent_name_key, candidates in agents_by_name.items():
                    if agent_name_key and agent_name_key in content_key:
                        content_name_matches.append((len(agent_name_key), agent_name_key, candidates))
                if content_name_matches:
                    _, matched_name_key, candidates = max(content_name_matches, key=lambda item: item[0])
                    matched_agent = choose_agent(candidates, f"content:{matched_name_key}")
                    matched_agent_id = matched_agent.agent_id if matched_agent else None

            # 2.5. 使用 token 集合匹配变体名，例如 "Captain Aria" -> "Aria the Captain"
            if matched_agent_id is None and content:
                content_tokens = set(tokenize_label(content))
                token_matches = []
                for candidate_tokens, agent in agent_name_tokens:
                    if len(candidate_tokens) >= 2 and candidate_tokens.issubset(content_tokens):
                        token_matches.append((len(candidate_tokens), len(''.join(sorted(candidate_tokens))), agent))

                if token_matches:
                    _, _, matched_agent = max(token_matches, key=lambda item: (item[0], item[1]))
                    matched_agent_id = matched_agent.agent_id
            
            # 3. 类型直接匹配
            if matched_agent_id is None and poster_type_key in agents_by_type:
                matched_agent = choose_agent(agents_by_type[poster_type_key], f"type:{poster_type_key}")
                matched_agent_id = matched_agent.agent_id if matched_agent else None
            elif matched_agent_id is None:
                # 4. 使用别名匹配
                for alias_key, aliases in type_aliases.items():
                    if poster_type_key in aliases or alias_key == poster_type_key:
                        for alias in aliases:
                            if alias in agents_by_type:
                                matched_agent = choose_agent(agents_by_type[alias], f"alias:{alias}")
                                matched_agent_id = matched_agent.agent_id if matched_agent else None
                                break
                    if matched_agent_id is not None:
                        break
            
            # 5. 如果仍未找到，使用影响力最高的 poster-eligible agent
            if matched_agent_id is None:
                logger.warning(f"未找到类型 '{poster_type_key}' 的匹配 Agent，使用影响力最高的 Agent")
                if poster_eligible_agents:
                    # 按影响力排序，选择影响力最高的
                    sorted_agents = sorted(poster_eligible_agents, key=lambda a: a.influence_weight, reverse=True)
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0
            
            updated_posts.append({
                "content": content,
                "poster_type": post.get("poster_type", "Unknown"),
                "poster_agent_id": matched_agent_id
            })
            
            logger.info(f"初始帖子分配: poster_type='{poster_type}' -> agent_id={matched_agent_id}")
        
        event_config.initial_posts = updated_posts
        return event_config
    
    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> List[AgentActivityConfig]:
        """分批生成Agent配置"""
        
        # 构建实体信息（使用配置的摘要长度）
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": e.summary[:summary_len] if e.summary else ""
            })
        
        prompt = f"""Generate public activity configurations for each entity using the scenario information below.

Simulation requirement: {simulation_requirement}

## Entity list
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Task
Generate an activity configuration for each entity. Follow these rules:
- **Respect worldbuilding and identity logic first**: if the scenario defines clear routines, institutions, religious rhythms, wartime conditions, or species traits, follow those rules before using generic assumptions.
- **If information is limited, fall back to a generic day-night pattern**: low activity from 00:00-05:00 and relatively high activity from 19:00-22:00.
- **Institutional or official roles** (such as University, GovernmentAgency, Organization): lower activity, more restrained tone, slower response, but higher influence.
- **Media, announcement, or messenger channels** (such as MediaOutlet): medium-to-high activity, wider active-hour coverage, faster response, higher influence.
- **Personal roles** (such as Student, Person, Alumni): medium-to-high activity, but still adjusted to identity, class, duty, and world rules.
- **Leaders, experts, or representatives**: medium activity, medium-to-high influence, and stronger position-taking.

Return JSON only (no markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <must match the input exactly>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <posting frequency>,
            "comments_per_hour": <comment frequency>,
            "active_hours": [<active hours adjusted to world rules and role routine>],
            "response_delay_min": <minimum response delay in minutes>,
            "response_delay_max": <maximum response delay in minutes>,
            "sentiment_bias": <-1.0 to 1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <influence weight>
        }},
        ...
    ]
}}"""

        system_prompt = "You are an expert in configuring agent behavior for scenario simulations. Return pure JSON. Follow worldbuilding, identity, and institutional constraints first; only fall back to a generic day-night rhythm when necessary."
        
        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning(f"Agent配置批次LLM生成失败: {e}, 使用规则生成")
            llm_configs = {}
        
        # 构建AgentActivityConfig对象
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})
            
            # 如果LLM没有生成，使用规则生成
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)
            
            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=self._normalize_hour_list(
                    cfg.get("active_hours"),
                    list(range(9, 23))
                ),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            configs.append(config)
        
        return configs
    
    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        """基于规则生成单个Agent配置（通用活动节律）"""
        entity_type = (entity.get_entity_type() or "Unknown").lower()
        
        if entity_type in ["university", "governmentagency", "ngo"]:
            # 官方机构：工作时间活动，低频率，高影响力
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),  # 9:00-17:59
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0
            }
        elif entity_type in ["mediaoutlet"]:
            # 媒体：全天活动，中等频率，高影响力
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),  # 7:00-23:59
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5
            }
        elif entity_type in ["professor", "expert", "official"]:
            # 专家/教授：工作+晚间活动，中等频率
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),  # 8:00-21:59
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0
            }
        elif entity_type in ["student"]:
            # 学生：晚间为主，高频率
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # 上午+晚间
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8
            }
        elif entity_type in ["alumni"]:
            # 校友：晚间为主
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],  # 午休+晚间
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
        else:
            # 普通人：晚间高峰
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # 白天+晚间
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
    

