"""
ISRP 模板库

统一管理节点描述模板，支持不同文体的差异化输出：
  - 基础模板：通用章节描述格式
  - 增强模板：5 类文体（论文、报告、自媒体、小说、发言稿）的专用模板

在根节点建立和节点扩展阶段动态匹配文体类型，选用对应模板。

"""
import re
from typing import List, Dict, Optional


# ============================================================================
# 基础模板
# ============================================================================

BASE_TEMPLATES = {
    # 开头类
    "引言": "开篇系统介绍{subject}的研究背景、目的与全文结构，交代研究缘起与核心关切。",
    "前言": "概述{subject}的写作背景、目的与全文框架，引出核心议题。",
    "绪论": "系统阐述{subject}的研究背景、意义、目的与方法，构建全文的理论框架与研究路径。",
    "摘要": "概括{subject}的核心内容，包括研究问题、主要方法、关键发现与核心结论，语言精炼、信息完整。",

    # 背景类
    "背景": "阐述{subject}的发展背景，介绍历史起源、演进脉络与当前状况，结合具体数据或案例说明。",
    "研究背景": "系统阐述{subject}的研究背景，说明问题产生的现实动因与理论依据，引用权威数据或典型案例支撑。",
    "选题背景": "说明{subject}选题的背景，交代选题来源、现实意义与研究必要性。",

    # 综述类
    "文献综述": "系统梳理{subject}领域的国内外研究成果，分析现有研究的贡献与不足，确立研究的理论起点。",
    "研究现状": "综述{subject}的国内外研究现状，梳理主要学术观点与研究进展，指出研究空白与本研究切入点。",

    # 方法类
    "方法": "详细阐述{subject}的研究方法与技术路线，说明研究设计、数据来源与分析工具。",
    "研究方法": "系统介绍{subject}的研究设计与分析方法，说明样本选取、数据收集与分析技术。",
    "技术路线": "明确{subject}研究的技术路径与实施步骤，说明研究过程的关键节点与操作流程。",

    # 内容类
    "主要内容": "详细阐述{subject}的核心内容，展开深入分析与论述，明确关键要点与论述逻辑。",
    "研究内容": "系统展开{subject}的研究内容，分析关键问题与核心观点，构建研究的主体框架。",

    # 结果类
    "结果": "呈现{subject}研究的主要发现与数据分析结果，客观展示研究产出，使用图表辅助说明。",
    "研究结果": "系统呈现{subject}的研究结果，包括数据发现、统计分析与关键结论。",
    "研究发现": "总结{subject}研究的核心发现，提炼具有理论价值与实践意义的结论，结合具体证据支撑。",

    # 讨论类
    "讨论": "对{subject}的研究结果进行深入讨论，阐释其理论内涵与实践启示。",
    "分析与讨论": "对{subject}的核心问题进行深入分析与讨论，揭示内在逻辑与理论对话。",

    # 结论类
    "结论": "总结{subject}研究的核心观点与贡献，指出研究局限与未来方向。",
    "总结": "对{subject}研究进行系统总结，归纳核心观点与主要结论。",
    "结语": "作为结语，总结{subject}研究的要点，展望未来发展趋势或研究方向。",
    "研究结论": "系统总结{subject}的研究结论，阐明理论贡献与实践价值。",

    # 参考文献类
    "参考文献": "列出研究过程中引用的所有文献资料，按学术规范格式编排。",

    # 建议类
    "建议": "针对{subject}提出针对性建议，为实践应用与政策制定提供参考。",
    "对策建议": "针对{subject}提出具体对策与建议，明确实施路径与关键措施。",
    "措施": "针对{subject}制定具体改进措施，确保问题得到有效解决。",

    # 目标类
    "目标": "明确{subject}研究的目标与预期成果，为后续工作指明方向。",
    "研究目标": "阐明{subject}研究的核心目标与具体任务，明确研究的价值定位。",

    # 意义类
    "意义": "阐述{subject}研究的理论意义与实践价值，论证研究的重要性与必要性。",
    "目的与意义": "阐明{subject}研究的目的与意义，说明研究的价值与必要性。",

    # 活动类
    "活动计划": "详细规划活动安排，明确活动目标、时间安排与实施步骤。",
    "活动内容": "具体描述活动内容，说明活动形式、主要环节与执行细节。",
    "主要成果": "总结主要成果，说明量化成果、质性成效与典型案例。",

    # 创新类
    "创新点": "突出创新之处，说明理论、方法或应用层面的独特贡献。",
    "可行性分析": "分析可行性，评估技术可行性、经济可行性与风险评估。",

    # 报告类
    "主要工作及成效": "系统总结工作与成效，说明工作任务完成情况与主要成绩。",
    "存在问题及原因分析": "深入分析问题与原因，揭示问题表现、影响程度与根本原因。",
    "整改措施": "提出整改措施，明确整改目标、具体措施与责任主体。",
    "下一步工作": "规划下一步工作，明确工作重点、目标任务与实施路径。",

    # 分析类
    "分析": "深入分析{subject}，揭示内在逻辑与实践启示。",
    "数据分析": "进行{subject}数据分析，说明数据来源、统计方法与核心数据结论。",
    "问题分析": "剖析{subject}问题，揭示问题具体表现、影响范围与成因。",
    "原因分析": "分析{subject}原因，梳理直接原因、深层原因与根本原因。",

    # 研究类
    "案例研究": "进行{subject}案例研究，说明案例选取依据与分析框架应用。",
    "实证研究": "开展{subject}实证研究，说明研究假设、变量设计与数据检验方法。",
    "理论研究": "进行{subject}理论研究，界定核心概念并构建理论框架。",

    # 理论类
    "理论框架": "构建{subject}理论框架，界定核心概念与变量关系。",
    "概念界定": "界定{subject}核心概念，明确概念内涵外延与相关概念的区别。",
    "理论基础": "阐述{subject}理论基础，说明所依据的基础理论及其指导意义。",

    # 设计类
    "系统设计": "阐述{subject}系统设计，说明系统架构、功能模块与关键技术选型。",
    "方案设计": "提出{subject}方案设计，说明设计思路、方案内容与技术路线图。",
    "模型设计": "构建{subject}分析模型，说明模型假设、变量定义与参数设置。",

    # 实践类
    "实践探索": "总结{subject}实践经验，说明实践背景、探索过程与经验教训。",
    "应用实践": "展示{subject}应用实践，说明应用场景、实施过程与效果评估。",
    "实践应用": "阐述{subject}实践应用，说明应用领域、操作方法与推广价值。",

    # 案例类
    "案例分析": "深入剖析案例，揭示案例背景、分析过程与关键发现。",

    # 比较类
    "比较分析": "对{subject}进行比较分析，明确比较维度、异同点与规律总结。",
    "对比分析": "对{subject}进行对比分析，从管理模式、核心策略、运作机制等方面展开，揭示关键差异与成功经验。",
    "异同分析": "分析{subject}的异同点，明确相同之处、差异之处及其成因。",

    # 发展类
    "发展趋势": "分析{subject}发展趋势，介绍历史演变、当前特征与未来方向。",
    "发展历程": "梳理{subject}发展历程，介绍起源阶段、发展阶段与成熟阶段。",
    "前景": "展望{subject}发展前景，说明发展机遇、面临挑战与未来趋势。",
    "展望": "进行{subject}前景展望，说明技术演进趋势与应用拓展方向。",
    "机遇": "分析{subject}机遇与挑战，说明外部机遇、内部优势与潜在风险。",
    "挑战与对策": "剖析{subject}挑战与对策，说明挑战表现与应对策略。",
    "面临的挑战": "剖析{subject}面临挑战，说明挑战类型与解决思路。",
    "关键技术与架构": "阐述{subject}关键技术，说明核心技术原理与技术架构。",
    "应用场景与产业": "阐述{subject}应用场景，说明典型应用领域与产业融合模式。",

    # 对策类
    "对策": "针对{subject}提出针对性的解决策略，明确实施路径与关键措施。",

    # 概述类
    "概述": "对{subject}进行总体介绍，明确研究范围与主要内容。",
    "总体概述": "对{subject}进行宏观介绍，构建整体认知框架。",

    # 党性分析类（党政公文风格）
    "党性分析": "对照党章党规，从政治信仰、党员意识、担当作为、理论学习、政治素质等方面进行{subject}自我剖析。",
    "党性认识": "对照党章党规，从政治信仰、党员意识、担当作为等方面进行{subject}自我评价。",
    "自我评价": "对照党员标准，客观评定{subject}在党性修养方面的总体表现与不足。",
    "存在问题": "坚持问题导向，深入查摆{subject}在理论学习、政治素质、担当作为、为民服务等方面存在的具体问题。",
    "努力方向": "明确{subject}今后的努力方向，提出强化理论武装、锤炼党性修养等具体目标。",
    "整改": "针对{subject}存在问题制定整改措施，明确时间表与责任人。",

    # 实习/社会实践类
    "实习单位简介": "介绍{subject}实习单位基本情况，说明单位性质、规模、主营业务与行业地位。",
    "实践单位简介": "介绍{subject}实践单位基本情况，说明单位背景、规模特点与主要职能。",
    "实习内容": "详细记录{subject}实习工作内容，说明岗位职责、参与项目与具体任务。",
    "实习过程": "按阶段记录{subject}实习过程，说明各阶段时间、主要任务与完成情况。",
    "实践内容": "详细描述{subject}实践内容与过程，说明实践目标、具体活动与阶段性成果。",
    "收获与体会": "总结{subject}实习收获，说明专业技能提升、职业认知转变与个人成长。",
    "收获与反思": "总结{subject}收获与反思，说明能力提升、认知变化与改进方向。",
    "问题与反思": "反思{subject}问题与不足，说明遇到的问题、原因分析与经验教训。",
    "实习总结": "系统总结{subject}实习经历，说明主要收获、能力变化与职业规划。",
    "实践总结": "总结{subject}实践经历，说明实践成果、个人成长与社会认知。",

    # 自媒体/新媒体类（活泼生动风格）
    "开篇": "设计{subject}吸引人的开篇，设置悬念、营造画面感、引导读者兴趣。",
    "悬念": "设置{subject}悬念，提出问题、揭示矛盾、激发好奇心。",
    "热议": "展示{subject}热议观点，呈现典型观点与多元立场对比。",
    "舆论": "分析{subject}舆论反响，揭示舆论焦点与不同声音。",
    "心声": "描写{subject}心理活动，表达真实情感与内心矛盾。",
    "心理": "深入描写{subject}心理，展现心理变化过程与情感转折点。",
    "共鸣": "设计{subject}共鸣内容，挖掘普遍情感与共同经历。",
    "互动": "设计{subject}互动内容，设置讨论话题与读者参与方式。",
    "科普": "进行{subject}科普说明，将专业知识通俗化、实用化。",
    "干货": "提供{subject}实用干货，说明具体方法、操作步骤与实用技巧。",
    "结尾": "设计引发讨论的{subject}结尾，重申核心观点、升华话题。",
    "升华": "升华{subject}全文观点，提炼核心价值、启发读者思考。",
    "实践路径": "梳理{subject}实践路径，介绍典型做法与成功案例。",
    "多元": "多维度分析{subject}，呈现不同视角与多元观点。",
    "生存": "剖析{subject}生存困境，揭示困境表现与应对策略。",
    "困境": "分析{subject}困境表现，说明困境类型与突破思路。",
    "两难": "探讨{subject}两难处境，揭示矛盾焦点与平衡之道。",
    "缺位": "揭示{subject}制度缺位，说明缺位表现与完善建议。",
    "期盼": "梳理{subject}各方期盼，介绍利益相关方诉求与政策建议。",
    "现实": "揭示{subject}现实张力，说明理想与现实差距及解决思路。",
    "拷问": "进行{subject}深度拷问，追问核心问题、引发读者思考。",

    # 调研报告类（政府公文风格）
    "调研背景": "说明{subject}调研背景，交代调研动因、现实需求与政策依据。",
    "调研方法": "说明{subject}调研方法，介绍调研时间、对象、方式与样本情况。",
    "调研情况": "概述{subject}调研情况，说明调研范围、对象特征与数据来源。",
    "现状分析": "分析{subject}现状，介绍基本情况、主要特点与存在问题。",
    "调研结论": "总结{subject}调研结论，归纳主要发现与核心结论。",
    "改进建议": "提出{subject}改进建议，明确问题导向与具体方案。",
}


# ============================================================================
# 增强模板（5类文体）
# ============================================================================

ENHANCED_TEMPLATES = {
    # 党性分析报告
    "party_analysis": {
        "党性分析": "对照党章党规，从政治信仰、党员意识、担当作为、理论学习、政治素质等方面进行{subject}自我剖析。",
        "存在问题": "坚持问题导向，深入查摆{subject}在理论学习、政治素质、担当作为、为民服务等方面存在的具体问题。",
        "原因分析": "从理想信念、宗旨意识、党性修养、纪律观念等方面剖析{subject}问题产生的深层原因。",
        "整改措施": "针对{subject}查摆出的问题，制定切实可行的整改清单与具体举措，明确时间表与责任人。",
        "努力方向": "明确{subject}今后的努力方向，提出强化理论武装、锤炼党性修养等具体目标。",
        "自我评价": "对照党员标准，客观评定{subject}在党性修养方面的总体表现与不足。",
        "_keywords": ["党性", "党员", "党章", "党性分析", "自我剖析"],
        "_fact_patterns": ["政治", "理论", "担当", "作风", "纪律", "信念"]
    },

    # 交通分析报告
    "traffic_report": {
        "措施": "具体实施{subject}，包括{numbered_measures}等措施。",
        "交通措施": "针对交通拥堵问题，采取{numbered_measures}等措施，确保交通顺畅。",
        "对策建议": "针对交通问题，提出{numbered_measures}等对策建议。",
        "_keywords": ["交通", "拥堵", "措施", "限行", "公交", "HOV"],
        "_extract_numbered": True
    },

    # 发言稿
    "speech": {
        "开篇": "以亲切问候开场，引出{subject}的核心议题，营造轻松互动的氛围。",
        "主体": "围绕{subject}展开论述，结合具体案例与数据支撑观点，语言生动有感染力。",
        "结尾": "总结{subject}要点，提出展望与号召，引发听众共鸣与思考。",
        "_keywords": ["发言稿", "讲话稿", "演讲稿", "发言用", "致辞"],
        "_style": "口语化、互动性强"
    },

    # 英文论文
    "english_paper": {
        "Introduction": "This section introduces {subject}, outlining the research background, objectives, and paper structure.",
        "Literature Review": "This section reviews existing literature on {subject}, identifying research gaps and establishing the theoretical foundation.",
        "Methodology": "This section describes the research methodology for {subject}, including data collection methods, analysis techniques, and research design.",
        "Results": "This section presents the main findings of the {subject} study, with supporting data and statistical analysis.",
        "Discussion": "This section discusses the implications of the {subject} findings, relating them to existing theory and practice.",
        "Conclusion": "This section summarizes the key contributions and limitations of the {subject} study, suggesting directions for future research.",
        "Abstract": "This abstract summarizes the {subject} research, including objectives, methods, key findings, and conclusions.",
        "References": "List all cited works following the required academic citation format.",
        "_keywords": ["paper", "essay", "research", "study", "dissertation", "thesis", "academic"],
        "_language": "en"
    },

    # 自媒体文章
    "social_media": {
        "开篇": "用热点话题或悬念切入，引发读者对{subject}的关注与思考，制造阅读期待。",
        "悬念": "设置{subject}悬念，提出问题、揭示矛盾、激发读者好奇心与探索欲。",
        "热议": "展示{subject}热议观点，呈现典型观点与多元立场对比，引发讨论。",
        "干货": "提供{subject}实用干货，说明具体方法、操作步骤与实用技巧，增强价值感。",
        "结尾": "设计引发讨论的{subject}结尾，重申核心观点、升华话题，留下思考空间。",
        "_keywords": ["自媒体", "公众号", "新媒体", "爆款", "文章"],
        "_style": "活泼生动、引人入胜"
    }
}


# ============================================================================
# 辅助函数
# ============================================================================

def _extract_numbered_measures(facts: List[str]) -> List[str]:
    """
    提取编号措施（如 1、xxx 2、xxx）

    Args:
        facts: 用户事实列表

    Returns:
        提取的措施列表
    """
    measures = []
    for fact in facts:
        # 匹配 "1、xxx" 或 "1.xxx" 或 "1．xxx" 格式
        matches = re.findall(r'\d[、.．]\s*([^0-9]+?)(?=\d[、.．]|$)', fact)
        measures.extend([m.strip() for m in matches if m.strip()])
    return measures


def _fill_template(template: str, subject: str, user_facts: List[str],
                   template_config: dict) -> str:
    """
    填充模板变量

    Args:
        template: 模板字符串
        subject: 主题
        user_facts: 用户事实列表
        template_config: 模板配置（用于判断是否需要提取编号措施）

    Returns:
        填充后的字符串
    """
    result = template.replace("{subject}", subject)

    # 处理编号措施
    if "{numbered_measures}" in result and user_facts:
        measures = _extract_numbered_measures(user_facts)
        if measures:
            result = result.replace("{numbered_measures}", "、".join(measures[:5]))
        else:
            result = result.replace("{numbered_measures}", "相关具体措施")

    return result


def get_template(chapter_title: str, subject: str, writing_type: str = None,
                 user_facts: List[str] = None) -> str:
    """
    获取模板（优先增强版，其次基础版）

    Args:
        chapter_title: 章节标题
        subject: 父节点主题
        writing_type: 文体类型
        user_facts: 用户事实列表

    Returns:
        填充后的描述字符串
    """
    if user_facts is None:
        user_facts = []

    # 清理章节标题中的编号
    clean_title = re.sub(r'^[\d.]+\s*', '', chapter_title).strip()

    # 1. 尝试匹配增强模板
    for template_type, templates in ENHANCED_TEMPLATES.items():
        keywords = templates.get("_keywords", [])
        if writing_type and any(kw in writing_type for kw in keywords):
            # 检查章节标题是否在模板中
            for key in templates:
                if not key.startswith("_") and (key in clean_title or key in chapter_title):
                    template = templates[key]
                    return _fill_template(template, subject, user_facts, templates)

    # 2. 尝试匹配基础模板
    for key, template in BASE_TEMPLATES.items():
        if key in clean_title or key in chapter_title:
            return template.replace("{subject}", subject)

    # 3. 默认模板
    return f"阐述{subject}的相关内容，展开详细分析与论述。"


def detect_writing_type(text: str) -> Optional[str]:
    """
    检测文体类型

    Args:
        text: 文本内容

    Returns:
        检测到的文体类型，如果未检测到则返回 None
    """
    for template_type, templates in ENHANCED_TEMPLATES.items():
        keywords = templates.get("_keywords", [])
        if any(kw in text for kw in keywords):
            return template_type
    return None