"""
AgentWrite E2E (End-to-End) 模块

端到端大纲生成方式：
- 一次性生成完整大纲
- 简单直接，适合快速生成

与 ISRP 的区别：
- ISRP: 深入实现大纲，通过迭代细化每个段落
- E2E: 端到端一次性生成完整大纲

使用方法：
    from e2e.planner import process_single_file

    process_single_file(
        input_file="data/WritingBench-120.jsonl",
        output_file="outputs/plan_e2e_wb.jsonl"
    )
"""

from .planner import (
    generate_plan,
    process_single_file,
    process_parallel,
    load_template,
)

__all__ = [
    "generate_plan",
    "process_single_file",
    "process_parallel",
    "load_template",
]
