"""
ISRP 评估器工厂

根据配置字符串（"isrp" / "writingbench"）动态创建对应的评估器实例。
统一注入 API 密钥、基础 URL、模型选择等配置。

"""
from typing import List, Dict, Any, Optional
from .base import BaseEvaluator
from .isrp_evaluator import ISRPEvaluator
from .writingbench_evaluator import WritingBenchEvaluator


class EvaluatorFactory:
    """评估器工厂"""

    @staticmethod
    def create(
        evaluator_type: str = "isrp",
        checklist: Optional[List[Dict[str, Any]]] = None,
        threshold: float = 8.0
    ) -> BaseEvaluator:
        """
        创建评估器实例

        Args:
            evaluator_type: 评估器类型
                - "isrp": ISRP 6 维评估器（默认）
                - "writingbench": WritingBench checklist 评估器
            checklist: WritingBench 评估标准（仅 writingbench 类型需要）
            threshold: 通过阈值（10分制）

        Returns:
            评估器实例

        Raises:
            ValueError: 不支持的评估器类型或缺少必要参数
        """
        if evaluator_type == "isrp":
            return ISRPEvaluator(threshold=threshold)
        elif evaluator_type == "writingbench":
            if not checklist:
                raise ValueError("WritingBench evaluator requires checklist parameter")
            return WritingBenchEvaluator(checklist=checklist, threshold=threshold)
        else:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}. "
                           f"Supported types: 'isrp', 'writingbench'")

    @staticmethod
    def create_from_sample(
        sample: Dict[str, Any],
        evaluator_type: str = "auto",
        threshold: float = 8.0
    ) -> BaseEvaluator:
        """
        从样本数据创建评估器

        Args:
            sample: 样本数据（可能包含 checklist 字段）
            evaluator_type: 评估器类型
                - "auto": 自动检测（如果样本有 checklist 则用 writingbench，否则用 isrp）
                - "isrp": 强制使用 ISRP 评估器
                - "writingbench": 强制使用 WritingBench 评估器
            threshold: 通过阈值

        Returns:
            评估器实例
        """
        if evaluator_type == "auto":
            # 自动检测：如果样本有 checklist 字段且非空，使用 writingbench
            if "checklist" in sample and sample["checklist"]:
                return EvaluatorFactory.create(
                    evaluator_type="writingbench",
                    checklist=sample["checklist"],
                    threshold=threshold
                )
            else:
                return EvaluatorFactory.create(
                    evaluator_type="isrp",
                    threshold=threshold
                )
        elif evaluator_type == "writingbench":
            checklist = sample.get("checklist")
            if not checklist:
                raise ValueError("Sample does not contain 'checklist' field")
            return EvaluatorFactory.create(
                evaluator_type="writingbench",
                checklist=checklist,
                threshold=threshold
            )
        else:
            return EvaluatorFactory.create(
                evaluator_type="isrp",
                threshold=threshold
            )