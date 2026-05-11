"""
ISRP 内部评估器

实现 ISRP 标准 6 维评分体系：
  - Relevance (相关性)
  - Accuracy (准确性)
  - Coherence (连贯性)
  - Clarity (清晰度)
  - Breadth/Depth (广度与深度)
  - Reading Experience (阅读体验)

用于迭代优化过程中的节点级即时评估与优化信号。
支持三种评估模式：refine（节点优化）、checkpoint（章节验证）、dfs_eval（扩展评估）。
通过 LLM-as-a-Judge 范式，每次评估均提供评分理由以增强可解释性。

"""
import json
import re
from typing import Dict, List, Any, Optional
from .base import BaseEvaluator, EvaluationResult, EvalType
from ..prompts import PromptTemplates


class ISRPEvaluator(BaseEvaluator):
    """ISRP 6 维评估器"""

    # 评分维度
    DIMENSIONS = [
        "relevance",
        "accuracy",
        "coherence",
        "clarity",
        "breadth_depth",
        "reading_experience"
    ]

    # 维度本地化名称
    DIMENSION_NAMES_ZH = {
        "relevance": "相关性",
        "accuracy": "准确性",
        "coherence": "连贯性",
        "clarity": "清晰度",
        "breadth_depth": "广度深度",
        "reading_experience": "阅读体验"
    }

    DIMENSION_NAMES_EN = {
        "relevance": "Relevance",
        "accuracy": "Accuracy",
        "coherence": "Coherence",
        "clarity": "Clarity",
        "breadth_depth": "Breadth & Depth",
        "reading_experience": "Reading Experience"
    }

    def __init__(self, threshold: float = 8.0):
        """
        初始化 ISRP 评估器

        Args:
            threshold: 通过阈值（10分制，默认 8.0）
        """
        super().__init__(threshold)
        self._prompt_templates = PromptTemplates()
        self._load_prompts()  # deprecated: prompts now in PromptTemplates, kept for subclass compat

    def _load_prompts(self):
        """[deprecated] 提示词已迁移至 isrp.prompts.PromptTemplates，保留空方法以兼容子类"""
        pass

    def get_refine_prompt(
        self,
        user_prompt: str,
        style_context: str,
        angle_name: str,
        rationale: str,
        high_level_structure: str,
        language: str = "zh"
    ) -> str:
        """生成根节点建立反思评估提示词"""
        template = self._prompt_templates.get_template("PROMPT_EVAL_REFINE", language)
        return template.format(
            user_prompt=user_prompt,
            style_context=style_context,
            angle_name=angle_name,
            rationale=rationale,
            high_level_structure=high_level_structure
        )

    def get_checkpoint_prompt(
        self,
        user_prompt: str,
        style_context: str,
        global_abstract: str,
        prev_chapter_abstract: str,
        chapter_content: str,
        language: str = "zh"
    ) -> str:
        """生成章节检查点评估提示词"""
        template = self._prompt_templates.get_template("PROMPT_EVAL_CHECKPOINT", language)
        return template.format(
            user_prompt=user_prompt,
            style_context=style_context,
            global_abstract=global_abstract,
            prev_chapter_abstract=prev_chapter_abstract,
            chapter_content=chapter_content
        )

    def get_dfs_eval_prompt(
        self,
        node_title: str,
        node_description: str,
        context: str,
        language: str = "zh"
    ) -> str:
        """生成 DFS 节点评估提示词"""
        template = self._prompt_templates.get_template("PROMPT_EVAL_DFS", language)
        return template.format(
            node_title=node_title,
            node_description=node_description,
            context=context
        )

    def get_final_eval_prompt(
        self,
        user_prompt: str,
        generated_text: str,
        language: str = "zh"
    ) -> str:
        """生成最终输出评估提示词"""
        template = self._prompt_templates.get_template("PROMPT_EVAL_FINAL", language)
        return template.format(
            user_prompt=user_prompt,
            generated_text=generated_text
        )

    def parse_final_result(self, llm_output: str) -> Dict[str, Any]:
        """解析最终评估结果"""
        data = self._parse_json(llm_output)

        if data is None:
            return {
                "scores": {},
                "average_score": 0.0,
                "Analysis": "JSON 解析失败"
            }

        # 提取分数（使用英文维度名）
        scores = {}
        dimension_mapping = {
            "Relevance": "relevance",
            "Accuracy": "accuracy",
            "Coherence": "coherence",
            "Clarity": "clarity",
            "Breadth and Depth": "breadth_depth",
            "Reading Experience": "reading_experience"
        }

        for display_name, internal_name in dimension_mapping.items():
            if display_name in data:
                scores[display_name] = int(data[display_name])
            elif internal_name in data:
                scores[display_name] = int(data[internal_name])

        # 计算平均分
        if scores:
            average_score = sum(scores.values()) / len(scores)
        else:
            average_score = 0.0

        return {
            "scores": scores,
            "average_score": average_score,
            "Analysis": data.get("Analysis", data.get("analysis", ""))
        }

    def parse_result(self, llm_output: str, eval_type: EvalType) -> EvaluationResult:
        """解析 LLM 输出为评估结果"""
        # 尝试解析 JSON
        data = self._parse_json(llm_output)

        if data is None:
            return EvaluationResult(
                analysis="JSON 解析失败",
                average_score=0.0,
                passed=False,
                raw_scores={}
            )

        # 提取分数
        raw_scores = {}
        total_score = 0
        score_count = 0

        # 对于 checkpoint 类型，分数在 evaluation 子对象中
        if eval_type == EvalType.CHECKPOINT and "evaluation" in data:
            eval_data = data["evaluation"]
        else:
            eval_data = data

        for dim in self.DIMENSIONS:
            if dim in eval_data:
                score = int(eval_data[dim])
                raw_scores[dim] = score
                total_score += score
                score_count += 1

        # 计算平均分
        average_score = total_score / score_count if score_count > 0 else 0.0

        # 判断是否通过
        passed = average_score >= self.threshold

        return EvaluationResult(
            analysis=data.get("analysis", ""),
            average_score=average_score,
            passed=passed,
            raw_scores=raw_scores,
            extra=data  # 保存原始数据供后续使用
        )

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """解析 JSON 文本"""
        # 剥离 Markdown 代码块
        cleaned = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)

        # 尝试定位 JSON 对象
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    def get_dimensions(self) -> List[str]:
        """获取评分维度列表"""
        return self.DIMENSIONS.copy()

    def get_dimension_names(self, language: str = "zh") -> Dict[str, str]:
        """获取评分维度的本地化名称"""
        return self.DIMENSION_NAMES_ZH if language == "zh" else self.DIMENSION_NAMES_EN
