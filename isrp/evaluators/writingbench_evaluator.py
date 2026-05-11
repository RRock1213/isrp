"""
WritingBench 外部评估器

使用 WritingBench 数据集的动态 checklist 机制进行评估。
每个样本的评估标准从 checklist 字段动态加载（5 条实例化 rubric），
评估结果与已有基准方法对齐，用于最终实验结果报告。
同样支持 refine / checkpoint / dfs_eval 三种评估模式。

"""
import json
import re
from typing import Dict, List, Any, Optional
from .base import BaseEvaluator, EvaluationResult, EvalType
from ..prompts import PromptTemplates


class WritingBenchEvaluator(BaseEvaluator):
    """WritingBench checklist 评估器"""

    def __init__(
        self,
        checklist: List[Dict[str, Any]],
        threshold: float = 8.0
    ):
        """
        初始化 WritingBench 评估器

        Args:
            checklist: 样本特定的评估标准列表
                [{"name": "...", "description": "...", "rubric": [...]}, ...]
            threshold: 通过阈值（10分制）
        """
        super().__init__(threshold)
        self.checklist = checklist
        self._validate_checklist()
        self._dimension_names = self._extract_dimension_names()
        self._prompt_templates = PromptTemplates()

    def _validate_checklist(self):
        """验证 checklist 格式"""
        if not self.checklist or len(self.checklist) < 1:
            raise ValueError("Checklist cannot be empty")

        for i, criterion in enumerate(self.checklist):
            if "name" not in criterion:
                raise ValueError(f"Checklist item {i} missing 'name' field")

    def _extract_dimension_names(self) -> List[str]:
        """从 checklist 提取维度名称"""
        return [c.get("name", f"Criterion {i+1}")
                for i, c in enumerate(self.checklist)]

    def _format_checklist(self, language: str = "zh") -> str:
        """格式化 checklist 为提示词文本"""
        lines = []
        for i, criterion in enumerate(self.checklist, 1):
            name = criterion.get("name", f"标准{i}")
            desc = criterion.get("description", "")

            if language == "zh":
                lines.append(f"{i}. {name}: {desc}")
            else:
                lines.append(f"{i}. {name}: {desc}")

        return "\n".join(lines)

    def _format_rubric(self, criterion: Dict[str, Any], language: str = "zh") -> str:
        """格式化评分标准"""
        rubric = criterion.get("rubric", [])
        if not rubric:
            return ""

        lines = []
        for item in rubric:
            score = item.get("score", "")
            desc = item.get("description", "")
            if score and desc:
                lines.append(f"  - {score}分: {desc}")

        return "\n".join(lines)

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
        criteria_text = self._format_checklist(language)
        template = self._prompt_templates.get_template("PROMPT_EVAL_WB_REFINE", language)
        return template.format(
            user_prompt=user_prompt,
            style_context=style_context,
            angle_name=angle_name,
            rationale=rationale,
            high_level_structure=high_level_structure,
            criteria_text=criteria_text
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
        criteria_text = self._format_checklist(language)
        template = self._prompt_templates.get_template("PROMPT_EVAL_WB_CHECKPOINT", language)
        return template.format(
            user_prompt=user_prompt,
            style_context=style_context,
            global_abstract=global_abstract,
            prev_chapter_abstract=prev_chapter_abstract,
            chapter_content=chapter_content,
            criteria_text=criteria_text
        )

    def get_dfs_eval_prompt(
        self,
        node_title: str,
        node_description: str,
        context: str,
        language: str = "zh"
    ) -> str:
        """生成 DFS 节点评估提示词"""
        criteria_text = self._format_checklist(language)
        template = self._prompt_templates.get_template("PROMPT_EVAL_WB_DFS", language)
        return template.format(
            node_title=node_title,
            node_description=node_description,
            context=context,
            criteria_text=criteria_text
        )

    def parse_result(self, llm_output: str, eval_type: EvalType) -> EvaluationResult:
        """解析 LLM 输出为评估结果"""
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
        scores_data = data.get("scores", {})

        for dim_name in self._dimension_names:
            if dim_name in scores_data:
                raw_scores[dim_name] = int(scores_data[dim_name])

        # 使用 average_score 字段或计算平均分
        if "average_score" in data:
            average_score = float(data["average_score"])
        elif raw_scores:
            average_score = sum(raw_scores.values()) / len(raw_scores)
        else:
            average_score = 0.0

        # 判断是否通过
        passed = average_score >= self.threshold

        return EvaluationResult(
            analysis=data.get("analysis", ""),
            average_score=average_score,
            passed=passed,
            raw_scores=raw_scores,
            extra=data
        )

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """解析 JSON 文本"""
        cleaned = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)

        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    def get_dimensions(self) -> List[str]:
        """获取评分维度列表"""
        return self._dimension_names

    def get_dimension_names(self, language: str = "zh") -> Dict[str, str]:
        """获取评分维度的本地化名称"""
        return {name: name for name in self._dimension_names}

    def get_final_eval_prompt(
        self,
        user_prompt: str,
        generated_text: str,
        language: str = "zh"
    ) -> str:
        """生成最终输出评估提示词"""
        criteria_text = self._format_checklist(language)
        template = self._prompt_templates.get_template("PROMPT_EVAL_WB_FINAL", language)
        return template.format(
            user_prompt=user_prompt,
            generated_text=generated_text,
            criteria_text=criteria_text
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

        # 提取分数
        scores = {}
        scores_data = data.get("scores", {})

        for dim_name in self._dimension_names:
            if dim_name in scores_data:
                scores[dim_name] = int(scores_data[dim_name])

        # 计算平均分
        if "average_score" in data:
            average_score = float(data["average_score"])
        elif scores:
            average_score = sum(scores.values()) / len(scores)
        else:
            average_score = 0.0

        return {
            "scores": scores,
            "average_score": average_score,
            "Analysis": data.get("Analysis", data.get("analysis", ""))
        }