"""
本体生成服务
接口1：分析文本内容，生成适合社会模拟的实体和关系类型定义
"""

import json
import re
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient
from ..config import Config


# Ontology-generation system prompt
ONTOLOGY_SYSTEM_PROMPT = """You are an expert knowledge-graph ontology designer. Analyze the supplied text and simulation requirement, then design entity types and relationship types suitable for **scenario simulation / social projection simulation**.

Important: you must return valid JSON and nothing else.

## Core task background

We are building a simulation system grounded in worldbuilding and scenario logic. In this system:
- every entity should be a subject that can speak, act, represent a position, spread information, or influence others
- these subjects may be individuals, organizations, factions, institutions, representative channels, or other publicly expressive actors
- the world may be modern, historical, alternate, fantasy, or science-fiction; do not force it into a modern setting

Therefore, entity types should prioritize subjects that can be projected as actors, organizations, or representative channels.

When the material is narrative, fantasy, historical-fiction, or lore-heavy, prefer broad reusable ontology anchors over overly narrow title/profession classes.

## What can count as an entity type
- specific people such as participants, leaders, scholars, journalists, knights, priests, merchants, explorers, and similar roles
- organizations such as universities, royal courts, religious orders, guilds, legions, companies, media outlets, councils, academies, and similar bodies
- representative groups or channels with public expressive capacity
- subjects that can generate public actions, attitudes, commands, responses, propagation, or coordination in the scenario

## What must not count as an entity type
- abstract concepts such as public opinion, emotion, trend, or fate
- topics such as academic integrity, education reform, or resource crisis
- positions such as supporters or opponents
- pure background elements that cannot be mapped to an acting subject or representative channel

For narrative/lore settings, also avoid these failure modes unless the source strongly justifies them as stable reusable classes:
- do not make top-level entity types that are only specific ranks or titles such as `Queen`, `Prince`, `Captain`, or `Archbishop`
- do not make top-level entity types that are only professions such as `Alchemist`, `Blacksmith`, or `Merchant`
- instead prefer broader classes such as `Character`, `Faction`, `Location`, `Kingdom`, `Court`, `Guild`, `Artifact`, `Institution`, or `Rumor` when grounded in the source
- treat ranks, titles, professions, bloodlines, and social roles as attributes or subtype hints whenever possible
- only use metaphysical or cosmic types such as `Void`, `Fate`, or `Chaos` if they behave like concrete places, forces, beings, or objects in the world rather than abstract themes

## Output format

Return JSON in this structure:

```json
{
    "entity_types": [
        {
            "name": "Entity type name in English PascalCase",
            "description": "Short English description under 100 characters",
            "attributes": [
                {
                    "name": "attribute_name in English snake_case",
                    "type": "text",
                    "description": "English attribute description"
                }
            ],
            "examples": ["Example entity 1", "Example entity 2"]
        }
    ],
    "edge_types": [
        {
            "name": "RELATIONSHIP_NAME in English UPPER_SNAKE_CASE",
            "description": "Short English description under 100 characters",
            "source_targets": [
                {"source": "Source entity type", "target": "Target entity type"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Brief English analysis summary of the source text"
}
```

## Design guide

### 1. Entity type design - must follow strictly

You must return exactly 10 entity types.

The 10 entity types must include both specific types and fallback types.

A. Fallback types (must be included as the last 2 items):
   - `Person`: fallback for any individual not covered by a more specific person type
   - `Organization`: fallback for any organization not covered by a more specific organization type

B. Specific types (8 items derived from the text):
   - design more specific types for the major roles present in the source material
   - example for academic scenarios: `Student`, `Professor`, `University`
   - example for business scenarios: `Company`, `CEO`, `Employee`

Specific-type principles:
- identify high-frequency or high-importance role categories from the text
- each specific type should have a clear boundary and avoid unnecessary overlap
- the description should clearly distinguish the type from its fallback category
- for narrative/lore source material, favor reusable worldbuilding classes over one-off title buckets
- prefer `Character` over narrow person-title classes unless the title defines a durable institution-level category
- prefer `Court`, `Guild`, `Faction`, `Kingdom`, or `Institution` over specific named bodies when a broader class is sufficient

### 2. Relationship type design

- return 6 to 10 relationship types
- relationships should reflect concrete ties, representation, influence paths, cooperation, conflict, command, propagation, or response patterns in the scenario
- make sure source_targets cover the entity types you define

### 3. Attribute design

- each entity type should have 1 to 3 key attributes
- do not use reserved names such as `name`, `uuid`, `group_id`, `created_at`, or `summary`
- preferred attribute names include `full_name`, `title`, `role`, `position`, `location`, and `description`

## Reference entity types

Specific person-like types:
- Student
- Professor
- Journalist
- Celebrity
- Executive
- Official
- Lawyer
- Doctor

Fallback person-like type:
- Person

Specific organization-like types:
- University
- Company
- GovernmentAgency
- MediaOutlet
- Hospital
- School
- NGO

Fallback organization-like type:
- Organization

## Reference relationship types

- WORKS_FOR
- STUDIES_AT
- AFFILIATED_WITH
- REPRESENTS
- REGULATES
- REPORTS_ON
- COMMENTS_ON
- RESPONDS_TO
- SUPPORTS
- OPPOSES
- COLLABORATES_WITH
- COMPETES_WITH

## Narrative/lore-friendly anchors

When the source text is clearly narrative or fantasy, strong candidate entity types often include:
- Character
- Faction
- Location
- Kingdom
- Court
- Guild
- Artifact
- Rumor

In those settings, strong candidate relationship types often include:
- RULES
- LEADS
- MEMBER_OF
- ALLIED_WITH
- RIVAL_OF
- RESIDES_IN
- POSSESSES
- SEEKS
- SPREADS
- KNOWS_ABOUT
"""


NARRATIVE_CONTEXT_RE = re.compile(
    r"\b("
    r"narrative|story|lore|chapter|act|scene|character|quest|king|queen|prince|princess|"
    r"kingdom|realm|court|guild|faction|throne|succession|artifact|relic|magic|mage|"
    r"wizard|witch|alchemist|prophecy|curse|void|empire|clan|temple|oracle|rumor"
    r")\b",
    re.IGNORECASE,
)


class OntologyGenerator:
    """
    本体生成器
    分析文本内容，生成实体和关系类型定义
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient(model=Config.get_stage_model('ontology'))
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成本体定义
        
        Args:
            document_texts: 文档文本列表
            simulation_requirement: 模拟需求描述
            additional_context: 额外上下文
            
        Returns:
            本体定义（entity_types, edge_types等）
        """
        # 构建用户消息
        user_message = self._build_user_message(
            document_texts, 
            simulation_requirement,
            additional_context
        )
        
        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # 调用LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        
        # 验证和后处理
        result = self._validate_and_process(result)
        
        return result
    
    # 传给 LLM 的文本最大长度（5万字）
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """构建用户消息"""
        
        # 合并文本
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        
        # 如果文本超过5万字，截断（仅影响传给LLM的内容，不影响图谱构建）
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(source text length: {original_length} characters; truncated to the first {self.MAX_TEXT_LENGTH_FOR_LLM} characters for ontology analysis)..."
        
        message = f"""## Simulation Requirement

{simulation_requirement}

## Source Documents

{combined_text}
"""
        
        if additional_context:
            message += f"""
## Additional Context

{additional_context}
"""

        narrative_context = self._looks_like_narrative_context(
            combined_text=combined_text,
            simulation_requirement=simulation_requirement,
            additional_context=additional_context,
        )
        
        message += """
Design entity types and relationship types suitable for scenario simulation and social projection.

Rules you must follow:
1. Return exactly 10 entity types
2. The last 2 types must be the fallback types Person and Organization
3. The first 8 types must be specific types derived from the source text
4. Prioritize subjects that can act as speaking actors, representative actors, or organization channels
5. The setting may be modern, historical, fantasy, science-fiction, or alternate; do not force it into a modern platform-native framing
6. Do not turn abstract concepts, pure topics, or pure emotions into entity types
7. Do not use reserved attribute names like name, uuid, or group_id; prefer names such as full_name or org_name instead
"""

        if narrative_context:
            message += """

Additional guidance for this source material:
8. This source appears narrative/lore-heavy, so prefer broad reusable worldbuilding classes over narrow title or profession buckets
9. Prefer `Character` over types like `Queen`, `Prince`, or `Alchemist` unless the narrower class is clearly a durable ontology category across the whole world
10. Prefer grounded lore anchors such as `Faction`, `Location`, `Kingdom`, `Court`, `Guild`, `Artifact`, or `Rumor` when supported by the source
11. Treat ranks, titles, professions, and bloodlines as attributes whenever possible instead of making each one a top-level entity type
12. Only use metaphysical types like `Void` if the source treats them as concrete world entities, places, or forces rather than abstract themes
"""
        
        return message

    def _looks_like_narrative_context(
        self,
        *,
        combined_text: str,
        simulation_requirement: str,
        additional_context: Optional[str],
    ) -> bool:
        combined = "\n".join([
            str(simulation_requirement or ""),
            str(additional_context or ""),
            str(combined_text or ""),
        ])
        return bool(NARRATIVE_CONTEXT_RE.search(combined))
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证和后处理结果"""
        
        # 确保必要字段存在
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        # 验证实体类型
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # 确保description不超过100字符
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        # 验证关系类型
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        
        # Zep API 限制：最多 10 个自定义实体类型，最多 10 个自定义边类型
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10
        
        # 兜底类型定义
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        # 检查是否已有兜底类型
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        
        # 需要添加的兜底类型
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            
            # 如果添加后会超过 10 个，需要移除一些现有类型
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # 计算需要移除多少个
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # 从末尾移除（保留前面更重要的具体类型）
                result["entity_types"] = result["entity_types"][:-to_remove]
            
            # 添加兜底类型
            result["entity_types"].extend(fallbacks_to_add)
        
        # 最终确保不超过限制（防御性编程）
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]
        
        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        将本体定义转换为Python代码（类似ontology.py）
        
        Args:
            ontology: 本体定义
            
        Returns:
            Python代码字符串
        """
        code_lines = [
            '"""',
            '自定义实体类型定义',
            '由MiroFish自动生成，用于社会舆论模拟',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== 实体类型定义 ==============',
            '',
        ]
        
        # 生成实体类型
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== 关系类型定义 ==============')
        code_lines.append('')
        
        # 生成关系类型
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # 转换为PascalCase类名
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        # 生成类型字典
        code_lines.append('# ============== 类型配置 ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        # 生成边的source_targets映射
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)

