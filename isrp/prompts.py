"""
ISRP Prompt 模板管理

统一管理框架内所有 LLM 调用的 Prompt 模板：

  预处理阶段:
    - 风格识别 (Style Recognition)
    - 查询压缩 (Query Compression)

  大纲树生成阶段:
    - 根节点建立 (Root Establishment) — 意图发散 + 串行反思
    - DFS 节点扩展 (DFS Expansion) — 纵向扩展 + 横向探针
    - 章节级检查点 (Chapter-Level Checkpoint) — 即时验证 + 补丁修复
    - 全局验证 (Global Validation) — 全文结构校验与修复

  大纲树优化阶段:
    - 描述优化 (Description Optimization) — 批量精修节点描述
    - BFS 字数分配 (Word Count Allocation) — 逐层权重评估

  Write 阶段:
    - 正文生成模板 (中/英文)

通过 PromptTemplates 类统一加载，按 language 参数切换中/英文。

"""
from typing import Dict


class PromptTemplates:
    """ISRP Prompt 模板管理"""

    _instance = None
    _templates_cache = {}

    def __new__(cls, language: str = "zh"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, language: str = "zh"):
        if language in self._templates_cache:
            self._load_from_cache(language)
        else:
            self._load_builtin_templates()
        self._language = language

    def _load_from_cache(self, language: str):
        templates = self._templates_cache[language]
        self._load_from_dict(templates)
        self._language = language

    def _load_from_dict(self, templates: Dict[str, str]):
        for key, value in templates.items():
            setattr(self, key, value)

    def _load_builtin_templates(self):
        """加载内置模板"""
        # 查询风格识别（在根节点建立之前执行）
        self.PROMPT_0_STYLE_ZH = """你是一个专业的写作风格分析师。请分析用户的写作指令，提取其中的风格需求。

【用户指令】
{user_prompt}

请分析以下维度并输出 JSON 格式：
{{
  "writing_type": "写作类型（自媒体文章/科普文章/学术论文/作文/报刊文章/小说/报告/其他）",
  "style_keywords": ["风格关键词1", "风格关键词2", "风格关键词3"],
  "tone_guidance": "语气语调指导",
  "structure_preference": "结构偏好说明",
  "language_style": "语言风格说明",
  "target_audience": "目标读者群体（可选）",
  "requires_real_facts": true或false,
  "fact_type": "事实类型（如果requires_real_facts为true）",
  "key_entities": ["关键实体1", "关键实体2"],
  "key_requirements": ["必须满足的要求1", "必须满足的要求2"],
  "core_subject": "核心主题（用户要写的具体主题内容，如'节能建筑'、'单细胞数据分析'、'党性分析'）",
  "theme_keywords": ["主题关键词1", "主题关键词2", "主题关键词3"],
  "subject_domain": "学科领域（如'建筑学'、'生物医学'、'政治学'、'计算机科学'）",
  "user_provided_facts": ["用户提供的事实1（如时间、地点、人物、数据等）", "事实2", "事实3"],
  "explicit_chapter_frame": {{
    "has_explicit_frame": true或false,
    "confidence": 0.0-1.0,
    "chapters": [
      {{
        "title": "章节标题",
        "needs_expansion": true或false,
        "estimated_weight": 1-10,
        "is_fixed_format": true或false
      }}
    ],
    "frame_source": "用户原文中描述框架的部分",
    "word_allocation_hint": "字数分配提示（如有）"
  }}
}}

【专业写作类型识别】
识别以下需要真实事实的写作类型：
- "True Crime" / "真实犯罪纪实" / "真实罪案" → requires_real_facts=true, fact_type="True Crime"
- "新闻报道" / "News Report" / "新闻稿" → requires_real_facts=true, fact_type="News"
- "学术论文" / "Academic Paper" / "研究论文" → requires_real_facts=true, fact_type="Academic"
- "案例分析" / "Case Study" / "案例研究" → requires_real_facts=true, fact_type="Case Study"
- "实习报告" / "工作报告" / "实习总结" → requires_real_facts=true, fact_type="Report"
- "入党申请书" / "辞职信" / "求职信" → requires_real_facts=true, fact_type="Application"

【文体类型识别】
根据用户指令识别文体类型：
- 如果用户要求"写小说/故事/章节/短篇/长篇" → writing_type="小说"，必须生成叙事性文本
- 如果用户要求"写论文/研究报告/毕业论文" → writing_type="论文"，需包含摘要、参考文献等
- 如果用户要求"写自媒体文章/公众号文章/推文" → writing_type="自媒体文章"，必须有吸引人的标题
- 如果用户要求"写报告/实习报告/调研报告" → writing_type="报告"，需按正式报告格式
- 如果用户要求"写发言稿/讲话稿/演讲稿/发言用" → writing_type="发言稿"，需使用口语化、互动性强的语言风格

关键判断规则：
1. 如果要求写"小说"，必须生成叙事性故事，而非评论、分析或元虚构
2. 如果要求写"文章"，必须生成完整文章，而非大纲或提示
3. 如果要求写"评论"或"分析"，则生成评论性文章，而非故事
4. 如果要求写"发言稿"，必须使用口语化、演讲式语言，而非学术论文风格

【实体上下文识别】
根据上下文判断实体类型：
1. 如果实体出现在"角色"、"人物"、"主角"、"反派"、"配角"等列表中 → 视为人物名，添加到 key_entities
2. 如果实体出现在"地点"、"城市"、"国家"列表中 → 视为地名
3. 如果存在歧义（如"班加罗尔"既可是城市也可是角色名）：
   - 检查相邻实体的类型
   - 如果相邻实体都是角色名，则该实体也应视为角色名

示例：
- "主角：昊京。反派boss：云天月。其余角色：班加罗尔、寻宝猎犬"
  → "班加罗尔"应识别为角色名，添加到 key_entities

【关键信息追踪】
识别用户指令中的关键信息：
1. 人名、地名、机构名等专有名词 → 添加到 key_entities
2. 用户明确要求的格式（如"吸引人的标题"、"分章结构"）→ 添加到 key_requirements
3. 用户提供的特定数据（如日期、数字）→ 需在生成内容中使用

【主题提取】
从用户指令中提取核心主题信息：
1. core_subject（核心主题）：用户要写的具体主题内容
   - "写一篇关于节能建筑的论文" → core_subject="节能建筑"
   - "帮我写单细胞数据分析报告" → core_subject="单细胞数据分析"
   - "党性分析报告" → core_subject="党性分析"
   - 如果指令中没有明确主题，从关键词推断
2. theme_keywords（主题关键词）：与核心主题相关的关键概念
   - "节能建筑" → theme_keywords=["绿色建筑", "节能技术", "可持续发展"]
   - "单细胞数据分析" → theme_keywords=["测序技术", "细胞异质性", "生物信息学"]
3. subject_domain（学科领域）：主题所属的学科或行业
   - "节能建筑" → subject_domain="建筑学"
   - "单细胞数据分析" → subject_domain="生物医学"
   - "党性分析" → subject_domain="政治学"

【用户事实提取】
从用户指令中提取用户提供的具体事实信息，这些信息必须在生成内容中被使用：
1. 时间信息：日期、年份、时间段（如"2022年6月-7月"、"30天"、"2024年"）
2. 地点信息：城市、学校、公司、场所名称（如"合肥市第六十九中学"、"紫金矿业"）
3. 人物信息：姓名、角色、职称（如"张老师"、"李明"）
4. 数据信息：数字、百分比、统计值（如"3000字"、"15篇"、"增长20%"）
5. 事件信息：活动、任务、经历（如"支教"、"调研"、"实习"）

示例：
输入："2022年6月-7月，我在合肥市第六十九中学进行了为期30天的支教活动..."
输出：
{{
  "user_provided_facts": [
    "时间：2022年6月-7月",
    "地点：合肥市第六十九中学",
    "事件：支教活动",
    "时长：30天"
  ]
}}

分析要点：
1. 从指令中的关键词推断写作类型（如"公众号"、"科普"、"论文"等）
2. 根据写作类型推断典型风格特征
3. 识别是否有明确的风格要求（如"接地气"、"严谨"等）
4. 推断目标读者群体
5. 【重要】如果用户请求涉及真实事件、真实案件、真实案例，必须设置 requires_real_facts=true
6. 【重要】确保所有关键实体和要求被准确提取，避免遗漏

【章节框架识别】
检测用户是否提供了明确的章节/部分结构要求：

**第一优先级：用户明确列举章节**
检测模式：
1. "包含以下几个部分：A、B、C、D"
2. "第一部分...第二部分...第三部分..."
3. "第一章...第二章...第三章..."
4. 章节名称列表（3个以上，逗号/顿号分隔）

**第二优先级：文体类型暗示**
当 writing_type 为以下类型时，自动填充标准结构：
- "论文" / "学术论文" / "研究论文" / "毕业论文" → 摘要、引言、文献综述、研究方法、研究结果、讨论、结论、参考文献
- "实习报告" / "实习总结" → 前言、实习单位简介、实习内容、收获与反思、结语
- "调研报告" / "调查报告" → 调研背景、调研方法、调研发现、问题分析、建议
- "开题报告" → 选题背景、研究现状、研究内容、研究方法、预期成果
- "党性分析" → 存在问题、原因分析、整改措施
- "工作报告" / "年终总结" / "述职报告" → 工作概述、主要工作内容、工作成果、存在问题、下一步计划

**第三优先级：结构描述关键词**
检测模式：
- "文章结构：首段...中间段...文末..."
- "开头...中间...结尾..."
- "按照...的结构"

判断标准：
- has_explicit_frame=true:
  - 用户明确列出 3+ 章节名称，OR
  - 写作类型暗示标准结构（如"论文"），OR
  - 包含结构描述关键词
- confidence:
  - 明确列举章节：0.9-1.0
  - 文体暗示（无用户结构）：0.8-0.9
  - 结构描述关键词：0.7-0.8
- needs_expansion=false: 标题页、参考文献、附录等固定格式章节
- estimated_weight: 根据章节重要性预估
- is_fixed_format=true: 不需要展开内容的固定格式章节

示例1（明确列举）：
输入："一篇论文通常包含以下几个部分：标题页、摘要、引言、方法、结果、讨论、结论、参考文献"
输出：
{{
  "has_explicit_frame": true,
  "confidence": 0.95,
  "chapters": [
    {{"title": "标题页", "needs_expansion": false, "estimated_weight": 1, "is_fixed_format": true}},
    {{"title": "摘要", "needs_expansion": false, "estimated_weight": 3, "is_fixed_format": true}},
    {{"title": "引言", "needs_expansion": true, "estimated_weight": 7, "is_fixed_format": false}},
    {{"title": "方法", "needs_expansion": true, "estimated_weight": 8, "is_fixed_format": false}},
    {{"title": "结果", "needs_expansion": true, "estimated_weight": 9, "is_fixed_format": false}},
    {{"title": "讨论", "needs_expansion": true, "estimated_weight": 8, "is_fixed_format": false}},
    {{"title": "结论", "needs_expansion": true, "estimated_weight": 6, "is_fixed_format": false}},
    {{"title": "参考文献", "needs_expansion": false, "estimated_weight": 2, "is_fixed_format": true}}
  ],
  "frame_source": "一篇论文通常包含以下几个部分：标题页、摘要、引言、方法、结果、讨论、结论、参考文献",
  "word_allocation_hint": ""
}}

示例2（文体暗示）：
输入："帮我写一篇5000字的关于唐朝民风对诗人创作影响的论文"
输出：
{{
  "has_explicit_frame": true,
  "confidence": 0.85,
  "chapters": [
    {{"title": "摘要", "needs_expansion": false, "estimated_weight": 3, "is_fixed_format": true}},
    {{"title": "引言", "needs_expansion": true, "estimated_weight": 7, "is_fixed_format": false}},
    {{"title": "文献综述", "needs_expansion": true, "estimated_weight": 8, "is_fixed_format": false}},
    {{"title": "研究方法", "needs_expansion": true, "estimated_weight": 8, "is_fixed_format": false}},
    {{"title": "研究结果", "needs_expansion": true, "estimated_weight": 9, "is_fixed_format": false}},
    {{"title": "讨论", "needs_expansion": true, "estimated_weight": 8, "is_fixed_format": false}},
    {{"title": "结论", "needs_expansion": true, "estimated_weight": 6, "is_fixed_format": false}},
    {{"title": "参考文献", "needs_expansion": false, "estimated_weight": 2, "is_fixed_format": true}}
  ],
  "frame_source": "【文体推断】学术论文标准结构",
  "word_allocation_hint": "按章节重要性分配"
}}"""

        self.PROMPT_0_STYLE_EN = """You are a professional writing style analyst. Please analyze the user's writing instruction and extract style requirements.

[User Instruction]
{user_prompt}

Please analyze the following dimensions and output in JSON format:
{{
  "writing_type": "Writing type (social media article/popular science article/academic paper/essay/newspaper article/novel/report/other)",
  "style_keywords": ["keyword1", "keyword2", "keyword3"],
  "tone_guidance": "Tone guidance",
  "structure_preference": "Structure preference description",
  "language_style": "Language style description",
  "target_audience": "Target audience (optional)",
  "requires_real_facts": true or false,
  "fact_type": "Fact type (if requires_real_facts is true)",
  "key_entities": ["key entity 1", "key entity 2"],
  "key_requirements": ["required requirement 1", "required requirement 2"],
  "explicit_chapter_frame": {{
    "has_explicit_frame": true or false,
    "confidence": 0.0-1.0,
    "chapters": [
      {{
        "title": "Chapter title",
        "needs_expansion": true or false,
        "estimated_weight": 1-10,
        "is_fixed_format": true or false
      }}
    ],
    "frame_source": "Original text describing the frame",
    "word_allocation_hint": "Word allocation hint (if any)"
  }}
}}

[Professional Writing Type Recognition]
Identify the following writing types that require real facts:
- "True Crime" / "真实犯罪纪实" / "真实罪案" → requires_real_facts=true, fact_type="True Crime"
- "News Report" / "新闻报道" / "新闻稿" → requires_real_facts=true, fact_type="News"
- "Academic Paper" / "学术论文" / "研究论文" → requires_real_facts=true, fact_type="Academic"
- "Case Study" / "案例分析" / "案例研究" → requires_real_facts=true, fact_type="Case Study"
- "Internship Report" / "工作报告" / "实习总结" → requires_real_facts=true, fact_type="Report"
- "Application Letter" / "入党申请书" / "辞职信" / "求职信" → requires_real_facts=true, fact_type="Application"

[Writing Type Recognition]
Identify the writing type based on user instruction:
- If user requests "novel/story/chapter/short story/long story" → writing_type="novel", MUST generate narrative text
- If user requests "paper/research paper/thesis" → writing_type="paper", should include abstract, references
- If user requests "social media article/blog post" → writing_type="social media article", must have attractive title
- If user requests "report/internship report/survey report" → writing_type="report", follow formal report format

Key Rules:
1. If user requests "novel", MUST generate narrative story, NOT commentary or analysis
2. If user requests "article", MUST generate complete article, NOT outline or prompt
3. If user requests "commentary" or "analysis", generate commentary article, NOT story

[Entity Context Recognition]
Determine entity type based on context:
1. If entity appears in "character/person/protagonist/antagonist/supporting character" lists → treat as person name, add to key_entities
2. If entity appears in "location/city/country" lists → treat as place name
3. If ambiguous (e.g., "Bangalore" could be city or character name):
   - Check adjacent entity types
   - If adjacent entities are all character names, treat this entity as character name too

Example:
- "Protagonist: Wu Jing. Antagonist boss: Yun Tianyue. Other characters: Bangalore, Bloodhound"
  → "Bangalore" should be recognized as character name, add to key_entities

[Key Information Tracking]
Identify key information in user instruction:
1. Proper nouns (names, places, organizations) → add to key_entities
2. Explicit format requirements (e.g., "attractive title", "chapter structure") → add to key_requirements
3. Specific data provided by user (dates, numbers) → must be used in generated content

Analysis points:
1. Infer writing type from keywords in the instruction
2. Infer typical style characteristics based on writing type
3. Identify explicit style requirements
4. Infer target audience
5. [IMPORTANT] If the user request involves real events, real cases, or real examples, you MUST set requires_real_facts=true
6. [IMPORTANT] Ensure all key entities and requirements are accurately extracted, avoid omissions

[Chapter Frame Recognition]
Detect if user provided explicit chapter/section structure requirements:
1. Explicit enumeration: "contains the following parts: A, B, C, D"
2. Structured list: "Part one... Part two..."
3. Standard frame implication: "paper format", "report format" (implies standard structure)

Judgment criteria:
- has_explicit_frame=true: User explicitly listed 3+ chapter names
- confidence: Score based on description clarity (0.0-1.0)
- needs_expansion=false: Fixed format chapters like title page, references, appendix
- estimated_weight: Estimate based on chapter importance (e.g., abstract is short=3, main chapters are longer=7-9)
- is_fixed_format=true: Fixed format chapters that don't need content expansion

Example:
Input: "A paper typically contains the following parts: title page, abstract, introduction, method, results, discussion, conclusion, references"
Output:
{{
  "has_explicit_frame": true,
  "confidence": 0.95,
  "chapters": [
    {{"title": "Title Page", "needs_expansion": false, "estimated_weight": 1, "is_fixed_format": true}},
    {{"title": "Abstract", "needs_expansion": false, "estimated_weight": 3, "is_fixed_format": true}},
    {{"title": "Introduction", "needs_expansion": true, "estimated_weight": 7, "is_fixed_format": false}},
    {{"title": "Method", "needs_expansion": true, "estimated_weight": 8, "is_fixed_format": false}},
    {{"title": "Results", "needs_expansion": true, "estimated_weight": 9, "is_fixed_format": false}},
    {{"title": "Discussion", "needs_expansion": true, "estimated_weight": 8, "is_fixed_format": false}},
    {{"title": "Conclusion", "needs_expansion": true, "estimated_weight": 6, "is_fixed_format": false}},
    {{"title": "References", "needs_expansion": false, "estimated_weight": 2, "is_fixed_format": true}}
  ],
  "frame_source": "A paper typically contains the following parts: title page, abstract, introduction, method, results, discussion, conclusion, references",
  "word_allocation_hint": ""
}}"""

        # 根节点建立 — 意图发散
        self.PROMPT_1_DIVERGENCE_ZH = """你是一个顶级的长内容架构师。请针对用户的写作指令，提供一个独特、深刻且具有高度可执行性的全文切入角度/实现思路。

【用户指令】
{user_prompt}

{style_context}

【事实约束】
如果用户指令中包含具体的事实信息（人名、地名、事件、数据、专业术语等）：
1. 必须严格基于用户提供的信息展开，不得编造矛盾内容
2. 对于用户提供的关键词和专有名词，不得擅自替换为其他相似词汇
3. 如果涉及特定品牌、人物、事件，必须使用用户指定的准确名称
4. 对于不确定的信息，应保持谨慎态度，避免过度演绎

【文体风格约束】
根据写作类型调整语言风格：
1. 实习报告/调研报告/工作报告：
   - 语言简洁明了，避免过度学术化
   - 使用第一人称"我"，增强真实感
   - 避免使用"结构性认知约束"等晦涩术语
   - 保持专业性但不牺牲可读性
2. 申请书（入党申请书、求职信等）：
   - 采用标准书信格式（标题+称呼+正文+结尾+署名+日期）
   - 语言真诚朴实，避免套话空话
   - 结构清晰，逻辑连贯
3. 小说/故事创作：
   - 注重情节设计和人物塑造
   - 允许虚构和想象
4. 学术论文/科普文章：
   - 论证严谨，引用准确
   - 专业术语使用恰当

【专业写作类型处理】
{fact_type_guidance}

请严格按照以下 JSON 格式输出（不要包含其他任何文字）：
{{
  "angle_name": "切入角度名称",
  "rationale": "为什么选择这个角度，它的优势是什么",
  "high_level_structure": "宏观的架构简述"
}}"""

        
        self.PROMPT_1_FRAME_VALIDATE_ZH = """你是一个写作结构专家。用户提供了以下明确的章节框架，请验证其合理性。

【写作主题】
{user_prompt}

【用户提供的章节框架】
{chapters}

【文体类型】
{writing_type}

请评估：
1. 章节框架是否完整（缺少关键章节请指出）
2. 章节顺序是否合理
3. 是否需要补充或调整

请严格按照以下 JSON 格式输出（不要包含其他任何文字）：
{{
  "is_valid": true或false,
  "missing_chapters": ["缺少的章节1", "缺少的章节2"],
  "order_issues": ["顺序问题说明"],
  "suggestions": ["改进建议1", "改进建议2"],
  "adjusted_chapters": ["调整后的章节列表"]
}}"""

        self.PROMPT_1_FRAME_VALIDATE_EN = """You are a writing structure expert. The user provided the following explicit chapter frame, please validate its reasonableness.

[Writing Topic]
{user_prompt}

[User Provided Chapter Frame]
{chapters}

[Writing Type]
{writing_type}

Please evaluate:
1. Is the chapter frame complete (point out missing key chapters)
2. Is the chapter order reasonable
3. Any additions or adjustments needed

Please output strictly in the following JSON format (no other text):
{{
  "is_valid": true or false,
  "missing_chapters": ["missing chapter 1", "missing chapter 2"],
  "order_issues": ["order issue description"],
  "suggestions": ["suggestion 1", "suggestion 2"],
  "adjusted_chapters": ["adjusted chapter list"]
}}"""

        self.PROMPT_1_DIVERGENCE_EN = """You are a top-tier long-form content architect. Please provide a unique, profound, and highly executable angle/approach for the user's writing instruction.

[User Instruction]
{user_prompt}

{style_context}

[Factual Constraints]
If the user instruction contains specific factual information (names, places, events, data, terminology, etc.):
1. You MUST strictly base your content on the information provided by the user, do not fabricate contradictory content
2. For keywords and proper nouns provided by the user, do NOT replace them with similar alternatives
3. If specific brands, people, or events are mentioned, you must use the exact names specified by the user
4. For uncertain information, maintain a cautious attitude and avoid over-interpretation

[Writing Style Constraints]
Adjust language style according to writing type:
1. Internship Report/Survey Report/Work Report:
   - Use clear and concise language, avoid over-academic style
   - Use first person "I" to enhance authenticity
   - Avoid obscure jargon like "structural cognitive constraints"
   - Maintain professionalism without sacrificing readability
2. Application Letter (Party Application, Job Application, etc.):
   - Use standard letter format (Title + Salutation + Body + Closing + Signature + Date)
   - Use sincere and simple language, avoid empty rhetoric
   - Clear structure and logical flow
3. Novel/Story Creation:
   - Focus on plot design and character development
   - Allow fiction and imagination
4. Academic Paper/Popular Science Article:
   - Rigorous argumentation and accurate citations
   - Appropriate use of technical terms

[Professional Writing Type Handling]
{fact_type_guidance}

Please output strictly in the following JSON format (do not include any other text):
{{
  "angle_name": "Name of the angle",
  "rationale": "Why choose this angle, what are its advantages",
  "high_level_structure": "Brief overview of the macro structure"
}}"""

        # 根节点建立 — 串行反思与评估
        self.PROMPT_1_REFINE_ZH = """你是一个严苛的评审专家。请使用 6 维评分细则对当前方案进行评估。

【用户原始指令】
{user_prompt}

{style_context}

【当前方案】
角度: {angle_name}
理由: {rationale}
架构: {high_level_structure}

【6 维评分细则】
1. Relevance (相关性 1-10): 是否紧扣用户指令
2. Accuracy (准确性 1-10): 逻辑是否合理，是否有事实错误
3. Coherence (连贯性 1-10): 结构是否清晰，逻辑是否连贯
4. Clarity (清晰度 1-10): 表达是否清晰易懂
5. Breadth_Depth (广度深度 1-10): 内容覆盖是否全面深入
6. Reading_Experience (阅读体验 1-10): 是否具有可读性

请输出 JSON 格式：
{{
  "analysis": "综合评估分析",
  "relevance": 1-10,
  "accuracy": 1-10,
  "coherence": 1-10,
  "clarity": 1-10,
  "breadth_depth": 1-10,
  "reading_experience": 1-10,
  "is_perfect": false,
  "refined_angle_name": "深化后的角度名称（如需修改）",
  "refined_rationale": "深化后的理由（如需修改）",
  "refined_structure": "深化后的结构（如需修改）"
}}

注意：如果方案已经完善（均分 >= 9.5），设置 is_perfect = true，并保持 refined 字段为空。"""

        self.PROMPT_1_REFINE_EN = """You are a strict reviewer. Please evaluate the current proposal using the 6-dimension rubric.

[Original User Instruction]
{user_prompt}

{style_context}

[Current Proposal]
Angle: {angle_name}
Rationale: {rationale}
Structure: {high_level_structure}

[6-Dimension Rubric]
1. Relevance (1-10): How relevant to user instruction
2. Accuracy (1-10): Logical soundness and factual accuracy
3. Coherence (1-10): Structural clarity and logical flow
4. Clarity (1-10): Expression clarity
5. Breadth_Depth (1-10): Coverage breadth and depth
6. Reading_Experience (1-10): Readability

Output in JSON format:
{{
  "analysis": "Comprehensive evaluation analysis",
  "relevance": 1-10,
  "accuracy": 1-10,
  "coherence": 1-10,
  "clarity": 1-10,
  "breadth_depth": 1-10,
  "reading_experience": 1-10,
  "is_perfect": false,
  "refined_angle_name": "Refined angle name (if modification needed)",
  "refined_rationale": "Refined rationale (if modification needed)",
  "refined_structure": "Refined structure (if modification needed)"
}}

Note: If the proposal is already perfect (average >= 9.5), set is_perfect = true and leave refined fields empty."""

        # 全局摘要
        self.PROMPT_2_ABSTRACT_ZH = """基于确立的核心方向，请为这篇文章撰写全局摘要（Abstract）。

{style_context}

【核心执行角度】
{core_angle}

【核心执行架构】
{core_structure}

【目标总字数】
{total_words}

请输出 JSON 格式：
{{
  "global_abstract": "文章的核心主旨、起承转合基调、以及主要任务（300字左右）",
  "total_word_budget": {total_words}
}}"""

        self.PROMPT_2_ABSTRACT_EN = """Based on the established core direction, please write a global abstract for this article.

{style_context}

[Core Angle]
{core_angle}

[Core Structure]
{core_structure}

[Target Word Count]
{total_words}

Output in JSON format:
{{
  "global_abstract": "Core theme, tone, and main tasks of the article (around 300 words)",
  "total_word_budget": {total_words}
}}"""

        # DFS 节点生成
        self.PROMPT_3_GENERATE_ZH = """你是一个顶级内容架构师。请根据以下上下文，生成当前节点的子节点列表。

{context}

{style_context}

【全局字数预算】
总字数: {total_words} 字
当前层级深度: Level {level} (最大 Level 2)

【当前待处理节点】
层级编号: {node_index}
标题: {node_title}
描述: {node_description}

【风格一致性检查】
生成内容时需确保：
1. 语言风格与要求的写作类型一致
2. 不偏离指定文体特征（如小说需叙事、论文需具备学术规范）
3. 保持人物设定一致（如有角色名需前后一致）
4. 若指定风格（如“斯坦李风格”），需体现对应特征

【关键实体追踪】
{key_entities_context}

【任务】
为当前节点生成 {min_width}-{max_width} 个子节点。每个子节点必须：
1. 提供清晰标题与详细描述（≥100字）
2. 提取 2-3 个核心概念标签
3. 在生成时直接判断是否需要继续向下拆分（needs_vertical_expansion）
4. 结合全局字数预算，控制当前层级信息密度

【needs_vertical_expansion 判定规则】
1. 复杂度  
- 多子主题 / 机制 / 阶段 → true  
- 单一事实 / 事件 / 观点 → false  

2. 类型  
- 框架 / 分类 / 流程 / 系统 → true  
- 案例 / 情节 / 具体描述 → false  

3. 抽象程度  
- 抽象概念 → true  
- 具体实例 → false  

4. 可展开性 
- 可清晰预判下一层结构 → true  
- 无法明确拆分 → false  

【总原则】
仅当“值得展开 + 可以展开”同时满足时，设 needs_vertical_expansion 为 true，否则为 false。

【内容差异化要求】
1. 每个子节点必须提供信息增量，避免重复
2. 若核心概念与已有内容重叠 >50%，需调整方向
3. 同级节点保持主题独立，避免交叉

请输出 JSON：
{{
  "thinking_process": "说明子节点划分逻辑、篇幅规划，以及每个节点 needs_vertical_expansion 的判断依据，并说明如何避免重复",
  "children": [
    {{
      "title": "子节点标题",
      "description": "详细描述（≥100字）",
      "core_concepts": ["概念1", "概念2"],
      "needs_vertical_expansion": true
    }}
  ]
}}

【注意】
- 若当前为 Level 2，则所有 needs_vertical_expansion 必须为 false
- 标题无需编号前缀
- 子节点需逻辑连贯
- 必须避免概念重复，确保信息增量
- 若写作类型为“小说”，必须输出叙事内容，而非分析"""

        self.PROMPT_3_GENERATE_EN = """You are a top-tier content architect. Please generate a list of child nodes for the current node based on the following context.
{context}
{style_context}
【Global Word Count Budget】
Total words: {total_words}
Current level depth: Level {level} (Maximum Level 2)
【Current Node to Process】
Node ID: {node_index}
Title: {node_title}
Description: {node_description}
【Style Consistency Check】
When generating content, ensure the following:
1. The writing style aligns with the required genre
2. Do not deviate from specified stylistic characteristics (e.g., narrative for fiction, academic standards for research papers)
3. Character settings remain consistent (e.g., character names must be consistent throughout)
4. If a specific style is specified (e.g., “Stan Lee style”), corresponding characteristics must be reflected
【Key Entity Tracking】
{key_entities_context}
【Task】
Generate {min_width}-{max_width} sub-nodes for the current node. Each sub-node must:
1. Provide a clear title and detailed description (≥100 characters)
2. Extract 2–3 core concept tags
3. Determine during generation whether further vertical expansion is needed (needs_vertical_expansion)
4. Control the information density at the current level based on the overall word count budget
【needs_vertical_expansion Determination Rules】
1. Complexity  
- Multiple subtopics / mechanisms / stages → true  
- Single fact / event / viewpoint → false  
2. Type  
- Framework / classification / process / system → true  
- Case study / plot / specific description → false  
3. Level of Abstraction  
- Abstract concepts → true  
- Concrete examples → false  
4. Expandability 
- The next-level structure can be clearly predicted → true  
- Cannot be clearly broken down → false  
【General Principle】
Set `needs_vertical_expansion` to `true` only when both “worth expanding” and “can be expanded” are satisfied; otherwise, set it to `false`.
【Content Differentiation Requirements】
1. Each child node must provide incremental information; avoid repetition
2. If the core concept overlaps with existing content by more than 50%, adjust the direction
3. Nodes at the same level must maintain thematic independence; avoid overlap
Please output JSON:
{{
  "thinking_process": "Explain the logic behind child node segmentation, content planning, and the criteria for determining `needs_vertical_expansion` for each node, as well as how to avoid repetition",
  "children": [
    {{
      "title": "Subnode Title",
      "description": "Detailed description (>=100 words)",
      "core_concepts": ["Concept 1", "Concept 2"],
      "needs_vertical_expansion": true
    }}
  ]
}}
【Note】
- If the current level is Level 2, all `needs_vertical_expansion` values must be `false`
- Titles do not require a numbered prefix
- Subnodes must be logically coherent
- Duplicate concepts must be avoided; ensure information is incremental
- If the writing type is “novel,” the output must consist of narrative content, not analysis
"""

        # 章节检查点
        self.PROMPT_4_CHECKPOINT_ZH = """你是一个专业的文章审核编辑。请对当前章节进行全面评估，并在发现问题时提供修复补丁。

【用户原始指令】
{user_prompt}

{style_context}

【全局基调】
{global_abstract}

【前序章节摘要】
{prev_chapter_abstract}

【当前章节内容】
{chapter_content}

【结构健康检查】
在评估前，请先检查以下结构问题：
1. "结论"、"总结"、"结语"等章节是否出现在正确位置？
   - 如果这类章节出现在中间位置（非最后一章），必须触发 restructure 补丁
2. 章节顺序是否符合逻辑递进？
3. 是否存在"前言"或"引言"出现在中间的异常？
4. 是否存在明显的重复内容或矛盾论述？

请按照 6 维评分细则进行评估，并输出 JSON 格式：
{{
  "evaluation": {{
    "relevance": 评分(1-10),
    "accuracy": 评分(1-10),
    "coherence": 评分(1-10),
    "clarity": 评分(1-10),
    "breadth_depth": 评分(1-10),
    "reading_experience": 评分(1-10)
  }},
  "issues": ["发现的问题1", "发现的问题2"],
  "suggestions": ["改进建议1", "改进建议2"],
  "needs_revision": true或false,
  "chapter_abstract": "本章核心内容摘要（100字左右）",
  "transition_quality": "与前序章节的衔接评价",
  "structure_issues": ["结构问题1", "结构问题2（如结论位置异常等）"],
  "patches": [
    // 补丁列表，用于修复发现的问题。支持的类型：
    // 1. reorder: 调整子节点顺序
    //    {{"action": "reorder", "new_order": ["节点ID1", "节点ID2", ...]}}
    // 2. add_bridge: 在两个节点之间添加桥梁节点
    //    {{"action": "add_bridge", "position": "after:节点ID", "node": {{"title": "标题", "description": "描述", "core_concepts": ["概念"]}}}}
    // 3. modify: 修改节点属性
    //    {{"action": "modify", "node_id": "节点ID", "title": "新标题", "description": "新描述"}}
    // 4. remove: 删除冗余节点
    //    {{"action": "remove", "node_id": "节点ID"}}
    // 5. restructure: 结构重组（调整章节顺序）
    //    {{"action": "restructure", "reason": "原因说明", "target_position": "建议移动到的位置"}}
    // 如果不需要修复，patches为空数组 []
  ]
}}

注意：
1. 只有在确实需要修复时才提供补丁
2. 补丁中的节点ID必须来自当前章节内容的节点
3. 添加桥梁节点时，position格式为 "after:节点ID" 或 "before:节点ID"
4. 如果章节质量良好（平均分>=8），patches可以为空
5. 【重要】如果发现"结论"类章节出现在中间位置，必须触发 restructure 补丁
6. 【严格约束】补丁的 action 字段只能是以下五种之一：reorder, add_bridge, modify, remove, restructure
   - 禁止使用其他类型如 content_insertion, content_expansion, symbolic_reinforcement 等
   - 如果需要插入内容，使用 add_bridge
   - 如果需要扩展内容，使用 modify
   - 如果需要删除内容，使用 remove
7. 【格式规范】每个补丁必须包含 action 字段，格式如下：
   - reorder: {{"action": "reorder", "new_order": ["节点ID1", "节点ID2"]}}
   - add_bridge: {{"action": "add_bridge", "position": "after:节点ID", "node": {{"title": "标题", "description": "描述"}}}}
   - modify: {{"action": "modify", "node_id": "节点ID", "description": "新描述"}}
   - remove: {{"action": "remove", "node_id": "节点ID"}}
   - restructure: {{"action": "restructure", "reason": "原因", "target_position": 位置}}"""

        self.PROMPT_4_CHECKPOINT_EN = """You are a professional article review editor. Please evaluate the current chapter comprehensively and provide repair patches if issues are found.

[User Original Instruction]
{user_prompt}

{style_context}

[Global Tone]
{global_abstract}

[Previous Chapter Abstracts]
{prev_chapter_abstract}

[Current Chapter Content]
{chapter_content}

[Structure Health Check]
Before evaluation, please check the following structural issues:
1. Are "Conclusion", "Summary", "Final Thoughts" chapters in the correct position?
   - If such chapters appear in the middle (not the last chapter), a restructure patch MUST be triggered
2. Does the chapter order follow logical progression?
3. Are there anomalies like "Introduction" or "Preface" appearing in the middle?
4. Are there obvious content duplications or contradictory statements?

Please evaluate according to the 6-dimensional scoring criteria and output in JSON format:
{{
  "evaluation": {{
    "relevance": score(1-10),
    "accuracy": score(1-10),
    "coherence": score(1-10),
    "clarity": score(1-10),
    "breadth_depth": score(1-10),
    "reading_experience": score(1-10)
  }},
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1", "suggestion2"],
  "needs_revision": true or false,
  "chapter_abstract": "Chapter abstract (about 100 words)",
  "transition_quality": "Transition quality from previous chapters",
  "structure_issues": ["structure issue1", "structure issue2 (e.g., conclusion in wrong position)"],
  "patches": [
    // Patch list for fixing issues. Supported types:
    // 1. reorder: Reorder children
    //    {{"action": "reorder", "new_order": ["node_id1", "node_id2", ...]}}
    // 2. add_bridge: Add bridge node between sections
    //    {{"action": "add_bridge", "position": "after:node_id", "node": {{"title": "title", "description": "desc", "core_concepts": ["concept"]}}}}
    // 3. modify: Modify node attributes
    //    {{"action": "modify", "node_id": "node_id", "title": "new title", "description": "new desc"}}
    // 4. remove: Remove redundant node
    //    {{"action": "remove", "node_id": "node_id"}}
    // 5. restructure: Structural reorganization (adjust chapter order)
    //    {{"action": "restructure", "reason": "reason", "target_position": "suggested position"}}
    // If no fix needed, patches should be empty array []
  ]
}}

Notes:
1. Only provide patches when repair is actually needed
2. Node IDs in patches must come from current chapter content
3. Position format for bridge nodes: "after:node_id" or "before:node_id"
4. If chapter quality is good (avg score >= 8), patches can be empty
5. [IMPORTANT] If "conclusion" type chapters appear in the middle, a restructure patch MUST be triggered
6. [STRICT] The "action" field in patches MUST be one of: reorder, add_bridge, modify, remove, restructure
   - Do NOT use other types like content_insertion, content_expansion, symbolic_reinforcement
   - For inserting content, use add_bridge
   - For expanding content, use modify
   - For removing content, use remove
7. [FORMAT] Each patch MUST contain an action field, format examples:
   - reorder: {{"action": "reorder", "new_order": ["node_id1", "node_id2"]}}
   - add_bridge: {{"action": "add_bridge", "position": "after:node_id", "node": {{"title": "title", "description": "desc"}}}}
   - modify: {{"action": "modify", "node_id": "node_id", "description": "new desc"}}
   - remove: {{"action": "remove", "node_id": "node_id"}}
   - restructure: {{"action": "restructure", "reason": "reason", "target_position": position}}"""

        # 章节节点优化
        self.PROMPT_REFINE_CHAPTER_ZH = """你是一个专业的长文架构师。现在需要基于前序章节的摘要，优化当前章节的节点设置，以增强章节间的衔接流畅度。

【用户原始指令】
{user_prompt}

{style_context}

【全局基调】
{global_abstract}

【前序章节摘要】
{prev_abstracts}

【当前章节节点】
标题: {chapter_title}
描述: {chapter_description}
核心概念: {chapter_concepts}

请分析当前章节与前序章节的衔接情况，判断是否需要优化，并输出 JSON 格式：
{{
  "thinking": "分析当前章节与前序章节的内容衔接、逻辑递进关系",
  "needs_update": true或false,
  "transition_quality": "衔接质量评价（流畅/基本合理/需要改进）",
  "update_reason": "如果需要更新，说明原因",
  "new_title": "优化后的标题（如不需要更新则留空）",
  "new_description": "优化后的描述（如不需要更新则留空）",
  "additional_concepts": ["建议新增的核心概念"],
  "suggested_leads": ["建议的开头要点，用于衔接前文"]
}}

注意：
1. 只有在确实需要增强衔接时才更新
2. 标题更新应保持与前序章节的风格一致性
3. 新增概念应有助于平滑过渡，而非改变章节主旨
4. 如果衔接已经流畅，needs_update 设为 false"""

        self.PROMPT_REFINE_CHAPTER_EN = """You are a professional long-form content architect. Now you need to optimize the current chapter node based on previous chapter abstracts to enhance chapter transition flow.

[User Original Instruction]
{user_prompt}

{style_context}

[Global Tone]
{global_abstract}

[Previous Chapter Abstracts]
{prev_abstracts}

[Current Chapter Node]
Title: {chapter_title}
Description: {chapter_description}
Core Concepts: {chapter_concepts}

Please analyze the transition between current and previous chapters, determine if optimization is needed, and output in JSON format:
{{
  "thinking": "Analyze the content connection and logical progression between current and previous chapters",
  "needs_update": true or false,
  "transition_quality": "Transition quality assessment (smooth/reasonable/needs improvement)",
  "update_reason": "If update needed, explain why",
  "new_title": "Optimized title (leave empty if no update needed)",
  "new_description": "Optimized description (leave empty if no update needed)",
  "additional_concepts": ["Suggested additional core concepts"],
  "suggested_leads": ["Suggested lead points to connect with previous content"]
}}

Notes:
1. Only update when transition enhancement is genuinely needed
2. Title updates should maintain style consistency with previous chapters
3. New concepts should help smooth transition, not change the chapter's main theme
4. If transition is already smooth, set needs_update to false"""

        # 字数分配 — LLM 权重评估
        self.PROMPT_5_WEIGHT_ZH = """你是一个专业的内容权重评估师。请根据各节点的内容广度和深度，结合同级兄弟节点评估各节点的相对重要性。

{style_context}

【父节点】
标题: {parent_title}
描述: {parent_description}

【待评估的兄弟节点列表】
{siblings_list}

请分析各节点的重要性，并输出 JSON 格式：
{{
  "thinking": "结合各节点的重要性，整体分析考虑当前节点的内容广度和深度，以及与父节点的联系",
  "allocations": [
    {{
      "node_id": "保持与输入一致的节点ID",
      "reasoning": "为什么给这个权重？简要说明理由",
      "relative_weight": 权重值(1-10的整数)
    }}
  ]
}}

评估标准：
- 内容深度：该节点是否需要深入展开
- 内容广度：该节点覆盖的范围大小
- 重要性：对整体文章的贡献程度
- 复杂度：该主题的复杂程度

注意：权重值必须在 1-10 之间，且各节点权重应反映其相对重要性。最终这个权重值将与该节点的字数篇幅成正比。"""

        self.PROMPT_5_WEIGHT_EN = """You are a professional content weight evaluator. Please evaluate the relative importance of sibling nodes based on their content breadth and depth.

{style_context}

[Parent Node]
Title: {parent_title}
Description: {parent_description}

[Sibling Nodes to Evaluate]
{siblings_list}

Please analyze the importance of the current node and output in JSON format:
{{
  "thinking": "Combining the importance of each node, overall analyze the importance of each node considering its content breadth and depth, as well as its relationship with the parent node",
  "allocations": [
    {{
      "node_id": "Node ID consistent with input",
      "reasoning": "Why this weight? Brief explanation",
      "relative_weight": Weight value (integer between 1-10)
    }}
  ]
}}

Evaluation criteria:
- Content depth: Whether the node needs in-depth expansion
- Content breadth: The scope covered by the node
- Importance: Contribution to the overall article
- Complexity: Complexity of the topic

Note: Weight values must be between 1-10 and should reflect relative importance of the current node.This weight will be proportional to the length of the node."""

        # 字数分配 — 子节点浓缩
        self.PROMPT_5_CONDENSE_ZH = """你是一个内容结构优化专家。当前节点的字数预算不足以支撑其子节点的展开，需要你进行结构浓缩。

【父节点】
标题: {parent_title}
描述: {parent_description}
目标字数: {target_words}字

【当前子节点列表】
{children_list}

请分析并决定如何浓缩这些子节点，输出 JSON 格式：
{{
  "thinking": "分析哪些子节点可以合并，如何保持逻辑连贯性",
  "action": "merge_children 或 keep",
  "merged_node_ids": ["需要合并的子节点ID列表"],
  "new_title": "合并后的新标题（如果选择合并）",
  "new_description": "合并后的新描述"
}}

注意：
- 合并时要保持逻辑连贯性
- 合并后的描述应该融合原有内容的关键点
- 如果子节点都很重要无法合并，选择 action: "keep" """

        self.PROMPT_5_CONDENSE_EN = """You are a content structure optimization expert. The current node's word budget is insufficient to support its children, so you need to condense the structure.

[Parent Node]
Title: {parent_title}
Description: {parent_description}
Target Words: {target_words}

[Current Children List]
{children_list}

Please analyze and decide how to condense these children, output in JSON format:
{{
  "thinking": "Analyze which children can be merged while maintaining logical coherence",
  "action": "merge_children or keep",
  "merged_node_ids": ["List of child IDs to merge"],
  "new_title": "New title after merge (if merging)",
  "new_description": "New description after merge"
}}

Note:
- Maintain logical coherence when merging
- The merged description should integrate key points from original content
- If all children are important and cannot be merged, choose action: "keep" """

        # 字数分配 — 智能合并
        self.PROMPT_5_SMART_MERGE_ZH = """你是一个内容整合专家。需要将两个相关节点合并为一个更完整、更精炼的节点。

【节点1】
标题: {title1}
描述: {description1}
核心概念: {concepts1}

【节点2】
标题: {title2}
描述: {description2}
核心概念: {concepts2}

请将这两个节点合并为一个新节点，输出 JSON 格式：
{{
  "thinking": "分析两个节点的关联性，说明合并后的逻辑结构",
  "new_title": "合并后的新标题（简洁、精炼，不超过30字，不使用&符号）",
  "new_description": "合并后的新描述（融合两个节点的关键内容，200字以上）",
  "new_concepts": ["合并后的核心概念1", "核心概念2"]
}}

注意：
- 新标题要体现两个节点的共同主题，不要简单拼接
- 新描述要有机融合两个节点的内容，不要机械堆砌
- 保留两个节点的关键信息和逻辑关系"""

        self.PROMPT_5_SMART_MERGE_EN = """You are a content integration expert. You need to merge two related nodes into one more complete and refined node.

[Node 1]
Title: {title1}
Description: {description1}
Core Concepts: {concepts1}

[Node 2]
Title: {title2}
Description: {description2}
Core Concepts: {concepts2}

Please merge these two nodes into a new node, output in JSON format:
{{
  "thinking": "Analyze the relationship between the two nodes and explain the logical structure after merging",
  "new_title": "New title after merge (concise, refined, no more than 30 words, do not use & symbol)",
  "new_description": "New description after merge (integrate key content from both nodes, 200+ words)",
  "new_concepts": ["Merged core concept 1", "Core concept 2"]
}}

Note:
- The new title should reflect the common theme of both nodes, do not simply concatenate
- The new description should organically integrate content from both nodes, do not mechanically pile up
- Preserve key information and logical relationships from both nodes"""

        # 字数分配 — 节点拆分
        self.PROMPT_5_SPLIT_ZH = """你是一个内容结构优化专家。当前叶子节点的字数过多，需要拆分为多个要点。

【当前节点】
标题: {node_title}
描述: {node_description}
当前字数: {current_words}字
建议每个要点: {target_words_per_node}字

请将当前节点拆分为 2-3 个逻辑连贯的要点，输出 JSON 格式：
{{
  "thinking": "分析如何拆分才能保持逻辑连贯性",
  "new_nodes": [
    {{
      "title": "拆分后的标题1",
      "description": "详细描述",
      "allocated_words": 字数,
      "core_concepts": ["概念1", "概念2"]
    }},
    {{
      "title": "拆分后的标题2",
      "description": "详细描述",
      "allocated_words": 字数,
      "core_concepts": ["概念1", "概念2"]
    }}
  ]
}}

注意：
- 拆分后的要点应该有明确的逻辑顺序
- 每个要点的字数总和应该接近原节点的字数
- 保持内容的完整性和连贯性 """

        self.PROMPT_5_SPLIT_EN = """You are a content structure optimization expert. The current leaf node has too many words and needs to be split into multiple points.

[Current Node]
Title: {node_title}
Description: {node_description}
Current Words: {current_words}
Suggested words per point: {target_words_per_node}

Please split the current node into 2-3 logically coherent points, output in JSON format:
{{
  "thinking": "Analyze how to split while maintaining logical coherence",
  "new_nodes": [
    {{
      "title": "Title after split 1",
      "description": "Detailed description",
      "allocated_words": word_count,
      "core_concepts": ["concept1", "concept2"]
    }},
    {{
      "title": "Title after split 2",
      "description": "Detailed description",
      "allocated_words": word_count,
      "core_concepts": ["concept1", "concept2"]
    }}
  ]
}}

Note:
- Split points should have clear logical order
- Total word count of split points should be close to original node
- Maintain content completeness and coherence """

        # 全局校验与修复
        self.PROMPT_VALIDATE_OUTLINE_ZH = """你是一个专业的内容编辑。以下大纲存在一些问题，需要进行校验和修复。

【当前大纲】
{outline_text}

【检测到的问题】
{issues}

【校验要求】
1. 修复所有章节编号问题，确保编号连续、不重复、格式统一
2. 统一标题风格，保持一致性
3. 确保章节顺序符合逻辑
4. 标题要简洁有力，能概括内容主旨

【重要】
- 输出的标题**不要包含任何编号前缀**（如"第一章"、"1."、"一、"等）
- 编号会在格式化时自动添加
- 只输出纯标题文本

请输出修复后的大纲结构，JSON 格式：
{{
  "needs_repair": true/false,
  "analysis": "问题分析和修复思路（50字内）",
  "chapters": [
    {{
      "title": "修复后的章节标题（不含编号）",
      "description": "简短描述（可选）",
      "children": [
        {{
          "title": "修复后的子节点标题（不含编号）"
        }}
      ]
    }}
  ]
}}

注意：
- 如果大纲没有问题，设置 needs_repair: false
- chapters 数量必须与原文档一致（共 {chapter_count} 章）
- 只修复问题，不要改变内容的核心含义"""

        self.PROMPT_VALIDATE_OUTLINE_EN = """You are a professional content editor. The following outline has some issues that need validation and repair.

[Current Outline]
{outline_text}

[Detected Issues]
{issues}

[Validation Requirements]
1. Fix all chapter numbering issues, ensure numbers are continuous, non-repeating, and uniformly formatted
2. Unify title style, maintain consistency
3. Ensure chapter order is logical
4. Titles should be concise and capture the main point

[IMPORTANT]
- Output titles **without any numbering prefix** (like "Chapter 1", "1.", "I.", etc.)
- Numbers will be added automatically during formatting
- Only output pure title text

Please output the repaired outline structure in JSON format:
{{
  "needs_repair": true/false,
  "analysis": "Problem analysis and repair approach (within 50 words)",
  "chapters": [
    {{
      "title": "Repaired chapter title (without number)",
      "description": "Brief description (optional)",
      "children": [
        {{
          "title": "Repaired child node title (without number)"
        }}
      ]
    }}
  ]
}}

Note:
- If the outline has no issues, set needs_repair: false
- The number of chapters must match the original document (total {chapter_count} chapters)
- Only fix issues, do not change the core meaning of the content"""

        # Prompt 智能压缩模板
        self.PROMPT_COMPRESS_ZH = """你是一个信息提取专家。请从以下用户写作指令中提取关键信息并压缩。

【用户原始指令】
{user_prompt}

【任务要求】
1. 提取核心写作主题和目标
2. 提取关键约束条件（字数、风格、格式等）
3. 提取必须包含的关键信息（人名、地名、事件、数据等）
4. 压缩原始指令，保留所有关键信息

请输出 JSON 格式：
{{
  "core_topic": "核心主题（一句话概括）",
  "key_requirements": ["关键要求1", "关键要求2", ...],
  "constraints": {{
    "word_count": 字数要求（如有）,
    "style": "风格要求（如有）",
    "format": "格式要求（如有）"
  }},
  "must_include": ["必须包含的信息1", "信息2", ...],
  "compressed_prompt": "压缩后的完整指令（保留所有关键信息，目标长度 {target_length} 字符以内）"
}}

注意：
- compressed_prompt 必须保留原始指令的所有关键信息，不能遗漏重要细节
- 压缩时应保留核心任务、关键约束、必须包含的信息
- 如果原始指令中包含具体的人名、地名、事件、数据等，必须在 must_include 和 compressed_prompt 中保留"""

        self.PROMPT_COMPRESS_EN = """You are an information extraction expert. Please extract key information from the following user writing instruction and compress it.

[Original User Instruction]
{user_prompt}

[Task Requirements]
1. Extract the core writing topic and goal
2. Extract key constraints (word count, style, format, etc.)
3. Extract must-include information (names, places, events, data, etc.)
4. Compress the original instruction while preserving all key information

Please output in JSON format:
{{
  "core_topic": "Core topic (one sentence summary)",
  "key_requirements": ["Key requirement 1", "Key requirement 2", ...],
  "constraints": {{
    "word_count": Word count requirement (if any),
    "style": "Style requirement (if any)",
    "format": "Format requirement (if any)"
  }},
  "must_include": ["Must-include info 1", "Info 2", ...],
  "compressed_prompt": "Compressed complete instruction (preserving all key information, target length {target_length} characters)"
}}

Note:
- compressed_prompt must preserve all key information from the original instruction
- When compressing, preserve core tasks, key constraints, and must-include information
- If the original instruction contains specific names, places, events, data, etc., they must be preserved in must_include and compressed_prompt"""

        # ========================================================================
        # Evaluator prompts — ISRP 6-dimension
        # ========================================================================

        self.PROMPT_EVAL_REFINE_ZH = """[ISRPEvaluator] 你是一个严苛的评审专家。请使用 6 维评分细则对当前方案进行评估。

【用户原始指令】
{user_prompt}

{style_context}

【当前方案】
角度: {angle_name}
理由: {rationale}
架构: {high_level_structure}

【6 维评分细则】
1. Relevance (相关性 1-10): 是否紧扣用户指令
2. Accuracy (准确性 1-10): 逻辑是否合理，是否有事实错误
3. Coherence (连贯性 1-10): 结构是否清晰，逻辑是否连贯
4. Clarity (清晰度 1-10): 表达是否清晰易懂
5. Breadth_Depth (广度深度 1-10): 内容覆盖是否全面深入
6. Reading_Experience (阅读体验 1-10): 是否具有可读性

请输出 JSON 格式：
{{
  "analysis": "综合评估分析",
  "relevance": 1-10,
  "accuracy": 1-10,
  "coherence": 1-10,
  "clarity": 1-10,
  "breadth_depth": 1-10,
  "reading_experience": 1-10,
  "is_perfect": false,
  "refined_angle_name": "深化后的角度名称（如需修改）",
  "refined_rationale": "深化后的理由（如需修改）",
  "refined_structure": "深化后的结构（如需修改）"
}}

注意：如果方案已经完善（均分 >= 9.5），设置 is_perfect = true，并保持 refined 字段为空。"""

        self.PROMPT_EVAL_REFINE_EN = """[ISRPEvaluator] You are a strict reviewer. Please evaluate the current proposal using the 6-dimension rubric.

[Original User Instruction]
{user_prompt}

{style_context}

[Current Proposal]
Angle: {angle_name}
Rationale: {rationale}
Structure: {high_level_structure}

[6-Dimension Rubric]
1. Relevance (1-10): How relevant to user instruction
2. Accuracy (1-10): Logical soundness and factual accuracy
3. Coherence (1-10): Structural clarity and logical flow
4. Clarity (1-10): Expression clarity
5. Breadth_Depth (1-10): Coverage breadth and depth
6. Reading_Experience (1-10): Readability

Output in JSON format:
{{
  "analysis": "Comprehensive evaluation analysis",
  "relevance": 1-10,
  "accuracy": 1-10,
  "coherence": 1-10,
  "clarity": 1-10,
  "breadth_depth": 1-10,
  "reading_experience": 1-10,
  "is_perfect": false,
  "refined_angle_name": "Refined angle name (if modification needed)",
  "refined_rationale": "Refined rationale (if modification needed)",
  "refined_structure": "Refined structure (if modification needed)"
}}

Note: If the proposal is already perfect (average >= 9.5), set is_perfect = true and leave refined fields empty."""

        self.PROMPT_EVAL_CHECKPOINT_ZH = """[ISRPEvaluator] 你是一个专业的文章审核编辑。请对当前章节进行全面评估，并在发现问题时提供修复补丁。

【用户原始指令】
{user_prompt}

{style_context}

【全局基调】
{global_abstract}

【前序章节摘要】
{prev_chapter_abstract}

【当前章节内容】
{chapter_content}

【结构健康检查】
在评估前，请先检查以下结构问题：
1. "结论"、"总结"、"结语"等章节是否出现在正确位置？
   - 如果这类章节出现在中间位置（非最后一章），必须触发 restructure 补丁
2. 章节顺序是否符合逻辑递进？
3. 是否存在"前言"或"引言"出现在中间的异常？
4. 是否存在明显的重复内容或矛盾论述？

请按照 6 维评分细则进行评估，并输出 JSON 格式：
{{
  "evaluation": {{
    "relevance": 评分(1-10),
    "accuracy": 评分(1-10),
    "coherence": 评分(1-10),
    "clarity": 评分(1-10),
    "breadth_depth": 评分(1-10),
    "reading_experience": 评分(1-10)
  }},
  "issues": ["发现的问题1", "发现的问题2"],
  "suggestions": ["改进建议1", "改进建议2"],
  "needs_revision": true或false,
  "chapter_abstract": "本章核心内容摘要（100字左右）",
  "transition_quality": "与前序章节的衔接评价",
  "structure_issues": ["结构问题1", "结构问题2（如结论位置异常等）"],
  "patches": []
}}

注意：
1. 只有在确实需要修复时才提供补丁
2. 如果章节质量良好（平均分>=8），patches可以为空
3. 【重要】如果发现"结论"类章节出现在中间位置，必须触发 restructure 补丁
4. 【严格约束】补丁的 action 字段只能是以下五种之一：reorder, add_bridge, modify, remove, restructure
   - 禁止使用其他类型如 content_insertion, content_expansion, symbolic_reinforcement, remove_and_relocate, revise_scope 等
   - 如果需要插入内容，使用 add_bridge
   - 如果需要扩展内容，使用 modify
   - 如果需要删除内容，使用 remove"""

        self.PROMPT_EVAL_CHECKPOINT_EN = """[ISRPEvaluator] You are a professional article review editor. Please evaluate the current chapter comprehensively and provide repair patches if issues are found.

[User Original Instruction]
{user_prompt}

{style_context}

[Global Tone]
{global_abstract}

[Previous Chapter Abstracts]
{prev_chapter_abstract}

[Current Chapter Content]
{chapter_content}

[Structure Health Check]
Before evaluation, please check the following structural issues:
1. Are "Conclusion", "Summary", "Final Thoughts" chapters in the correct position?
   - If such chapters appear in the middle (not the last chapter), a restructure patch MUST be triggered
2. Does the chapter order follow logical progression?
3. Are there anomalies like "Introduction" or "Preface" appearing in the middle?
4. Are there obvious content duplications or contradictory statements?

Please evaluate according to the 6-dimensional scoring criteria and output in JSON format:
{{
  "evaluation": {{
    "relevance": score(1-10),
    "accuracy": score(1-10),
    "coherence": score(1-10),
    "clarity": score(1-10),
    "breadth_depth": score(1-10),
    "reading_experience": score(1-10)
  }},
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1", "suggestion2"],
  "needs_revision": true or false,
  "chapter_abstract": "Chapter abstract (about 100 words)",
  "transition_quality": "Transition quality from previous chapters",
  "structure_issues": ["structure issue1", "structure issue2 (e.g., conclusion in wrong position)"],
  "patches": []
}}

Notes:
1. Only provide patches when repair is actually needed
2. If chapter quality is good (avg score >= 8), patches can be empty
3. [IMPORTANT] If "conclusion" type chapters appear in the middle, a restructure patch MUST be triggered
4. [STRICT] The action field in patches must be one of these five values: reorder, add_bridge, modify, remove, restructure
   - Other types like content_insertion, content_expansion, symbolic_reinforcement, remove_and_relocate, revise_scope are NOT allowed
   - Use add_bridge to insert content
   - Use modify to expand content
   - Use remove to delete content"""

        self.PROMPT_EVAL_DFS_ZH = """[ISRPEvaluator] 你是一个严格的内容评审专家。请评估当前节点的内容质量。

{context}

【当前节点】
标题: {node_title}
描述: {node_description}

【评估标准】
请从以下 6 个维度进行评估（1-10 分）：
1. Relevance (相关性): 内容是否与主题相关
2. Accuracy (准确性): 信息是否准确，逻辑是否合理
3. Coherence (连贯性): 内容是否连贯，结构是否清晰
4. Clarity (清晰度): 表达是否清晰易懂
5. Breadth_Depth (广度深度): 内容是否足够深入和全面
6. Reading_Experience (阅读体验): 是否具有良好的可读性

请输出 JSON 格式：
{{
  "analysis": "综合评估分析",
  "relevance": 1-10,
  "accuracy": 1-10,
  "coherence": 1-10,
  "clarity": 1-10,
  "breadth_depth": 1-10,
  "reading_experience": 1-10,
  "needs_improvement": true或false,
  "improvement_suggestions": ["改进建议1", "改进建议2"]
}}"""

        self.PROMPT_EVAL_DFS_EN = """[ISRPEvaluator] You are a strict content reviewer. Please evaluate the quality of the current node.

{context}

[Current Node]
Title: {node_title}
Description: {node_description}

[Evaluation Criteria]
Please evaluate from the following 6 dimensions (1-10 score):
1. Relevance: Is the content relevant to the topic
2. Accuracy: Is the information accurate and logical
3. Coherence: Is the content coherent and well-structured
4. Clarity: Is the expression clear and easy to understand
5. Breadth_Depth: Is the content deep and comprehensive enough
6. Reading_Experience: Does it have good readability

Please output in JSON format:
{{
  "analysis": "Comprehensive evaluation analysis",
  "relevance": 1-10,
  "accuracy": 1-10,
  "coherence": 1-10,
  "clarity": 1-10,
  "breadth_depth": 1-10,
  "reading_experience": 1-10,
  "needs_improvement": true or false,
  "improvement_suggestions": ["suggestion1", "suggestion2"]
}}"""

        self.PROMPT_EVAL_FINAL_ZH = """你是一个专业的文本质量评估专家。请对以下生成的文章进行全面评估。

【用户原始指令】
{user_prompt}

【生成的文章】
{generated_text}

【评估标准】
请从以下 6 个维度进行评估（1-10 分）：

1. Relevance (相关性): 内容是否紧扣用户指令的主题，是否回答了用户的问题
2. Accuracy (准确性): 信息是否准确，事实是否正确，逻辑是否合理
3. Coherence (连贯性): 文章结构是否清晰，段落之间逻辑是否连贯，是否有良好的过渡
4. Clarity (清晰度): 表达是否清晰易懂，是否有歧义或晦涩的表达
5. Breadth and Depth (广度与深度): 内容覆盖是否全面，是否深入探讨了主题
6. Reading Experience (阅读体验): 文章是否流畅自然，是否具有良好的可读性

【重要说明】
- 字数要求不在质量评估范围内，请勿因字数多少而扣分
- 字数评估由独立的长度评估模块负责
- 请仅关注内容质量本身进行评分

【输出格式】
请输出 JSON 格式：
{{
  "Relevance": 1-10,
  "Accuracy": 1-10,
  "Coherence": 1-10,
  "Clarity": 1-10,
  "Breadth and Depth": 1-10,
  "Reading Experience": 1-10,
  "Analysis": "综合评估分析（简要说明各维度评分理由）"
}}

注意：请确保输出的是有效的 JSON 格式，所有分数为 1-10 的整数。"""

        self.PROMPT_EVAL_FINAL_EN = """You are a professional text quality evaluation expert. Please evaluate the following generated article comprehensively.

[Original User Instruction]
{user_prompt}

[Generated Article]
{generated_text}

[Evaluation Criteria]
Please evaluate from the following 6 dimensions (1-10 score):

1. Relevance: Does the content closely follow the user instruction's theme, does it answer the user's question
2. Accuracy: Is the information accurate, are facts correct, is the logic sound
3. Coherence: Is the article structure clear, is the logic between paragraphs coherent, are there good transitions
4. Clarity: Is the expression clear and easy to understand, are there ambiguous or obscure expressions
5. Breadth and Depth: Is the content coverage comprehensive, is the topic explored in depth
6. Reading Experience: Is the article smooth and natural, does it have good readability

[IMPORTANT NOTES]
- Word count requirements are NOT within the scope of quality evaluation, please do not deduct points for word count
- Word count evaluation is handled by a separate length evaluation module
- Please focus only on content quality itself when scoring

[Output Format]
Please output in JSON format:
{{
  "Relevance": 1-10,
  "Accuracy": 1-10,
  "Coherence": 1-10,
  "Clarity": 1-10,
  "Breadth and Depth": 1-10,
  "Reading Experience": 1-10,
  "Analysis": "Comprehensive evaluation analysis (briefly explain the reasoning for each dimension score)"
}}

Note: Please ensure the output is valid JSON format, all scores are integers from 1-10."""

        # ========================================================================
        # Evaluator prompts — WritingBench dynamic checklist
        # ========================================================================

        self.PROMPT_EVAL_WB_REFINE_ZH = """[WritingBenchEvaluator] 你是一个严苛的评审专家。请使用以下评估标准对当前方案进行评估。

【用户原始指令】
{user_prompt}

{style_context}

【当前方案】
角度: {angle_name}
理由: {rationale}
架构: {high_level_structure}

【评估标准】
{criteria_text}

请输出 JSON 格式：
{{
  "analysis": "综合评估分析",
  "scores": {{
    "标准名称1": 1-10,
    "标准名称2": 1-10,
    ...
  }},
  "average_score": 平均分,
  "is_perfect": false,
  "refined_angle_name": "深化后的角度名称（如需修改）",
  "refined_rationale": "深化后的理由（如需修改）",
  "refined_structure": "深化后的结构（如需修改，必须是字符串格式，不要使用数组）"
}}

注意：
1. 如果方案已经完善（均分 >= 9.5），设置 is_perfect = true，并保持 refined 字段为空
2. refined_structure 必须是字符串类型，用换行符分隔各部分，不要使用数组"""

        self.PROMPT_EVAL_WB_REFINE_EN = """[WritingBenchEvaluator] You are a strict reviewer. Please evaluate the current proposal using the following criteria.

[Original User Instruction]
{user_prompt}

{style_context}

[Current Proposal]
Angle: {angle_name}
Rationale: {rationale}
Structure: {high_level_structure}

[Evaluation Criteria]
{criteria_text}

Output in JSON format:
{{
  "analysis": "Comprehensive evaluation analysis",
  "scores": {{
    "criterion1": 1-10,
    "criterion2": 1-10,
    ...
  }},
  "average_score": average_score,
  "is_perfect": false,
  "refined_angle_name": "Refined angle name (if modification needed)",
  "refined_rationale": "Refined rationale (if modification needed)",
  "refined_structure": "Refined structure (if modification needed, MUST be a string, NOT an array)"
}}

Notes:
1. If the proposal is already perfect (average >= 9.5), set is_perfect = true and leave refined fields empty
2. refined_structure MUST be a string type with newline separators, NOT an array"""

        self.PROMPT_EVAL_WB_CHECKPOINT_ZH = """[WritingBenchEvaluator] 你是一个专业的文章审核编辑。请对当前章节进行全面评估。

【用户原始指令】
{user_prompt}

{style_context}

【全局基调】
{global_abstract}

【前序章节摘要】
{prev_chapter_abstract}

【当前章节内容】
{chapter_content}

【评估标准】
{criteria_text}

请输出 JSON 格式：
{{
  "analysis": "综合评估分析",
  "scores": {{
    "标准名称1": 1-10,
    "标准名称2": 1-10,
    ...
  }},
  "average_score": 平均分,
  "needs_revision": true或false,
  "chapter_abstract": "本章核心内容摘要（100字左右）",
  "issues": ["发现的问题1", "发现的问题2"],
  "suggestions": ["改进建议1", "改进建议2"],
  "patches": []
}}"""

        self.PROMPT_EVAL_WB_CHECKPOINT_EN = """[WritingBenchEvaluator] You are a professional article review editor. Please evaluate the current chapter comprehensively.

[User Original Instruction]
{user_prompt}

{style_context}

[Global Tone]
{global_abstract}

[Previous Chapter Abstracts]
{prev_chapter_abstract}

[Current Chapter Content]
{chapter_content}

[Evaluation Criteria]
{criteria_text}

Output in JSON format:
{{
  "analysis": "Comprehensive evaluation analysis",
  "scores": {{
    "criterion1": 1-10,
    "criterion2": 1-10,
    ...
  }},
  "average_score": average_score,
  "needs_revision": true or false,
  "chapter_abstract": "Chapter abstract (about 100 words)",
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1", "suggestion2"],
  "patches": []
}}"""

        self.PROMPT_EVAL_WB_DFS_ZH = """[WritingBenchEvaluator] 你是一个严格的内容评审专家。请评估当前节点的内容质量。

{context}

【当前节点】
标题: {node_title}
描述: {node_description}

【评估标准】
{criteria_text}

请输出 JSON 格式：
{{
  "analysis": "综合评估分析",
  "scores": {{
    "标准名称1": 1-10,
    "标准名称2": 1-10,
    ...
  }},
  "average_score": 平均分,
  "needs_improvement": true或false,
  "improvement_suggestions": ["改进建议1", "改进建议2"]
}}"""

        self.PROMPT_EVAL_WB_DFS_EN = """[WritingBenchEvaluator] You are a strict content reviewer. Please evaluate the quality of the current node.

{context}

[Current Node]
Title: {node_title}
Description: {node_description}

[Evaluation Criteria]
{criteria_text}

Output in JSON format:
{{
  "analysis": "Comprehensive evaluation analysis",
  "scores": {{
    "criterion1": 1-10,
    "criterion2": 1-10,
    ...
  }},
  "average_score": average_score,
  "needs_improvement": true or false,
  "improvement_suggestions": ["suggestion1", "suggestion2"]
}}"""

        self.PROMPT_EVAL_WB_FINAL_ZH = """你是一个专业的文本质量评估专家。请对以下生成的文章进行全面评估。

【用户原始指令】
{user_prompt}

【生成的文章】
{generated_text}

【评估标准】
{criteria_text}

【重要说明】
- 字数要求不在质量评估范围内，请勿因字数多少而扣分
- 字数评估由独立的长度评估模块负责
- 请仅关注内容质量本身进行评分

【输出格式】
请输出 JSON 格式：
{{
  "scores": {{
    "标准名称1": 1-10,
    "标准名称2": 1-10,
    ...
  }},
  "average_score": 平均分,
  "Analysis": "综合评估分析（简要说明各维度评分理由）"
}}

注意：
1. scores 对象中应包含所有评估标准的分数
2. 所有分数为 1-10 的整数
3. 请确保输出的是有效的 JSON 格式"""

        self.PROMPT_EVAL_WB_FINAL_EN = """You are a professional text quality evaluation expert. Please evaluate the following generated article comprehensively.

[Original User Instruction]
{user_prompt}

[Generated Article]
{generated_text}

[Evaluation Criteria]
{criteria_text}

[IMPORTANT NOTES]
- Word count requirements are NOT within the scope of quality evaluation, please do not deduct points for word count
- Word count evaluation is handled by a separate length evaluation module
- Please focus only on content quality itself when scoring

[Output Format]
Please output in JSON format:
{{
  "scores": {{
    "criterion1": 1-10,
    "criterion2": 1-10,
    ...
  }},
  "average_score": average_score,
  "Analysis": "Comprehensive evaluation analysis (briefly explain the reasoning for each dimension score)"
}}

Note:
1. The scores object should contain scores for all evaluation criteria
2. All scores are integers from 1-10
3. Please ensure the output is valid JSON format"""

        # ========================================================================
        # Evaluation script prompts
        # ========================================================================

        self.PROMPT_EVAL_OUTLINE_ZH = """你是一位专业的学术写作大纲评审专家。请对以下大纲样本进行质量评估。

【用户原始指令】
{user_prompt}

【大纲内容】
{outline}

【评估标准】（10分制）
1. 主题相关性（relevance，1-10）：每个章节描述是否紧扣用户指定的主题？
2. 内容具体性（specificity，1-10）：描述是否清晰说明该章节要写什么具体内容？
3. 结构指导性（guidance，1-10）：描述是否为写作提供明确的方向和要点？
4. 语言专业性（language，1-10）：语言是否专业、流畅、符合文体要求？
5. 主题覆盖率（coverage，1-10）：大纲是否全面覆盖用户要求的各个方面？
6. 语义正确性（semantic_correctness，1-10）：
   - 分包/合同关系理解是否正确（如 Package A/B/C 是否正确识别为施工/监理/验收）
   - 角色/实体关系是否合理（如投标人、招标人、监理方的职责界定）
   - 关键信息是否被准确理解（如金额、数量、时间等约束条件）
   - 若存在语义理解错误，此维度应扣分（如误解分包含义扣3-5分）

【输出格式】
请严格输出标准JSON格式（不要使用代码块）：
{{"dimensions": {{"relevance": 9, "specificity": 7, "guidance": 8, "language": 9, "coverage": 8, "semantic_correctness": 9}}, "strengths": ["优点"], "weaknesses": ["不足"], "suggestions": ["建议"], "low_quality_descriptions": [], "semantic_errors": ["语义错误列表，如：分包理解错误、角色关系混淆等"]}}

注意：只需输出六个维度分数，最终得分将由系统根据六维度平均值自动计算。"""

        self.PROMPT_EVAL_PAIRWISE_OUTLINE_EN = """You are an expert judge for outline comparison.

User Prompt:
{user_prompt}

Evaluate the two candidate outlines using only these static dimensions:
{dimensions_text}

Return JSON with:
- winner
- scores.A
- scores.B
- average_scores.A
- average_scores.B
- reason
- confidence

Result A:
{result_a}

Result B:
{result_b}"""

        self.PROMPT_EVAL_PAIRWISE_WRITE_EN = """You are an expert judge for writing comparison.

User Prompt:
{user_prompt}

Evaluate the two candidate writings using only this checklist:
{criteria_text}

Return JSON with:
- winner
- scores.A
- scores.B
- average_scores.A
- average_scores.B
- reason
- confidence

Result A:
{result_a}

Result B:
{result_b}"""

        self._language = "zh"

    def get_template(self, name: str, language: str = None) -> str:
        """
        获取模板

        Args:
            name: 模板名称（不含语言后缀，如 "PROMPT_0_STYLE"）
            language: 语言代码（"zh" 或 "en"），默认使用实例语言

        Returns:
            模板字符串
        """
        lang = language or self._language
        suffix = "_ZH" if lang == "zh" else "_EN"
        return getattr(self, f"{name}{suffix}", getattr(self, name, ""))


    # ========================================================================
    # Write 阶段模板
    # ========================================================================

    # 中文写作模板
    PROMPT_WRITE_ZH = """你是一个优秀的写作助手。我会给你一个原始的写作指令和我计划的写作步骤。我也会提供我已经写好的文本。请根据写作指令、写作步骤和已写好的文本，帮我继续写下一段。

写作指令：

$INST$

写作步骤（共 $TOTAL_STEPS$ 步，当前需要写第 $CURRENT_INDEX$ 步）：

$PLAN$

已写好的文本：

$TEXT$

【当前任务 - 非常重要】
你现在需要写的是第 $CURRENT_INDEX$ 步，内容如下：

$STEP$

【严格禁止】
1. 绝对不能跳过当前步骤去写其他步骤的内容！
2. 绝对不能重复已写好的内容！
3. 必须严格按照上面指定的步骤内容来写！

【写作要求】
1. 严格遵循每个步骤中指定的字数要求（如"Word Count: 450 words"）。
2. 不要超过指定的字数。字数是目标值，而非最低值。
3. 尽量接近要求（±5%以内可接受）。
4. 对于中文内容，"Word Count: 450 words" 表示精确的 450 个中文字符。
5. 简洁优于冗长。注重质量和相关性，而非长度。

请只输出你写的当前段落内容（第 $CURRENT_INDEX$ 步），不要重复已写好的文本。如果需要，可以在开头添加章节小标题。"""

    # 英文写作模板
    PROMPT_WRITE_EN = """You are an excellent writing assistant. I will give you an original writing instruction and my planned writing steps. I will also provide you with the text I have already written. Please help me continue writing to the next paragraph based on the writing instruction, writing steps, and the already written text.

Writing instruction:

$INST$

Writing steps (Total $TOTAL_STEPS$ steps, currently writing step $CURRENT_INDEX$):

$PLAN$

Already written text:

$TEXT$

【CURRENT TASK - VERY IMPORTANT】
You need to write step $CURRENT_INDEX$, content as follows:

$STEP$

【STRICTLY FORBIDDEN】
1. You MUST NOT skip the current step to write content for other steps!
2. You MUST NOT repeat already written content!
3. You MUST strictly follow the step content specified above!

【CRITICAL REQUIREMENTS】
1. STRICTLY follow the word count specified in each step (e.g., "Word Count: 450 words").
2. Do NOT exceed the specified word count. The word count is a TARGET, not a minimum.
3. Aim to meet the requirement as closely as possible (within ±5% is acceptable).
4. For Chinese content, "Word Count: 450 words" means 450 Chinese characters exactly.
   For English content, "Word Count: 450 words" means 450 English words exactly.
5. Conciseness is valued over verbosity. Focus on quality and relevance, not length.

Please only output the current paragraph you write (step $CURRENT_INDEX$), without repeating already written text. If needed, you can add a small subtitle at the beginning."""


    # ========================================================================
    # engines.py 辅助 prompt
    # ========================================================================

    # 概念黑名单提示
    PROMPT_CONCEPT_BLACKLIST = "【已覆盖概念（建议探索新方向，避免重复）】\n{concepts_str}"

    # 节点扩展分析提示
    PROMPT_NODE_EXPAND_ZH = """{context}

【当前节点】
标题: {title}
描述: {description}
层级: {level}

请分析当前节点，判断：
1. 是否需要纵向延伸（细分为下级子节点）？
2. 是否需要横向延伸（补充同级节点）？

注意：
- 最大层级限制为 3（当前: {level}）
- 已达 Level 3 则必须停止纵向延伸
- 如果当前节点内容已足够具体，无需再细分"""

    PROMPT_NODE_EXPAND_EN = """{context}

[Current Node]
Title: {title}
Description: {description}
Level: {level}

Please analyze the current node and determine:
1. Whether vertical expansion (splitting into child nodes) is needed?
2. Whether horizontal expansion (adding sibling nodes) is needed?

Note:
- Maximum depth is 3 (current: {level})
- If at Level 3, vertical expansion must stop
- If the current node is specific enough, no further splitting is needed"""

    # ========================================================================
    # description_optimize.py prompt
    # ========================================================================

    PROMPT_DESC_OPTIMIZE_ZH = """请优化以下章节描述，使其更具体、更有指导性。

【用户原始需求】
{user_prompt}

【待优化的章节】
{nodes_str}

【优化目标】
1. specificity >= 8: 明确说明要写什么具体内容
2. guidance >= 8: 为写作提供清晰的方向
3. 避免模板化语言（"需涵盖"、"应包括"等）
4. 自然融入用户具体信息

【输出格式】JSON
{{
  "章节标题": "优化后的描述",
  ...
}}

只输出 JSON，不要其他文字。"""

    PROMPT_DESC_OPTIMIZE_EN = """Please optimize the following chapter descriptions to make them more specific and actionable.

【User Request】
{user_prompt}

【Chapters to Optimize】
{nodes_str}

【Optimization Goals】
1. specificity >= 8: Clearly state what specific content to write
2. guidance >= 8: Provide clear direction for writing
3. Avoid template language like "should include", "need to cover"
4. Naturally incorporate user-specific information

【Output Format】JSON
{{
  "章节标题": "Optimized description",
  ...
}}

Only output the JSON, no other text."""

    # ========================================================================
    # dfs_expansion_engine.py prompt
    # ========================================================================

    PROMPT_HORIZONTAL_PROBE_ZH = """你是一个内容架构师。请判断当前层级是否需要补充更多同级节点。

【用户指令】
{effective_prompt}

【全局摘要】
{global_abstract}

【父节点】
标题: {parent_title}
描述: {parent_desc}

【已有的同级节点】（共 {sibling_count} 个）
{sibling_list}

【已覆盖的核心概念】
{covered_concepts}

【最后处理的节点】
标题: {last_child_title}

请判断是否需要补充同级节点，输出 JSON 格式：
{{
  "analysis": "简短分析（50字内）",
  "coverage_score": 1-10,
  "needs_more_siblings": true/false,
  "suggested_sibling": {{
    "title": "如需补充，提供新节点标题（不含编号）",
    "description": "简要描述（50字内）"
  }}
}}"""

    PROMPT_CONTENT_SUPPLEMENT_ZH = """你是一个内容架构专家。当前大纲存在"框架过重、内容缺失"的问题，需要补充生成主体内容章节。

【用户指令】
{effective_prompt}

【现有章节】
{existing_summary}

【风格上下文】
{style_context}

【问题诊断】
当前大纲主要以引言、理论框架、脉络概述等"框架性内容"为主，缺乏实质性的主题内容章节。

【任务要求】
请补充生成2-4个主体内容章节，确保：
1. 章节内容是实质性的分析/论述/案例，而非框架性描述
2. 章节标题明确，描述具体，有可操作性
3. 与现有框架章节形成有机衔接

输出JSON格式：
{{
  "analysis": "简述当前问题及补充策略（50字内）",
  "new_chapters": [
    {{
      "title": "章节标题（不含编号）",
      "description": "章节具体内容描述（100字内，明确要阐述的核心观点或内容）",
      "core_concepts": ["核心概念1", "核心概念2"],
      "insert_after": 0
    }}
  ]
}}"""


    # ========================================================================
    # E2E planner prompt
    # ========================================================================

    PROMPT_E2E_PLAN = """I need you to help me break down the following long-form writing instruction into multiple subtasks. Each subtask will guide the writing of one paragraph in the essay, and should include the main points and word count requirements for that paragraph.

The writing instruction is as follows:

$INST$

Please break it down in the following format, with each subtask taking up one line:

Paragraph 1 - Main Point: [Describe the main point of the paragraph, in detail] - Word Count: [Word count requirement, e.g., 400 words]

Paragraph 2 - Main Point: [Describe the main point of the paragraph, in detail] - Word Count: [word count requirement, e.g. 1000 words].

...

Make sure that each subtask is clear and specific, and that all subtasks cover the entire content of the writing instruction. Do not split the subtasks too finely; each subtask's paragraph should be no less than 200 words and no more than 1000 words. Do not output any other content."""


# 常量导出
MAX_ITERATIONS = 3  # 串行迭代上限
