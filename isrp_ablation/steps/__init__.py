"""
ISRP Ablation — 步骤处理器模块

包含九个步骤处理器，分为三个阶段：
  Phase I — 预处理: QueryStyleRecognition
  Phase II — 大纲树生成: RootNodeEstablishment, GlobalAbstract,
              DFSExpansionEngine, ChapterLevelValidation
  Phase III — 大纲树优化: DescriptionOptimize, WordCountAllocation,
              GlobalValidation, Format
"""

import sys


def flush_stdout():
    """强制刷新 stdout"""
    sys.stdout.flush()


def print_and_flush(msg: str):
    """打印并立即刷新输出"""
    print(msg)
    sys.stdout.flush()


from .query_style_recognition import QueryStyleRecognition
from .root_node_establishment import RootNodeEstablishment
from .global_abstract import GlobalAbstract
from .dfs_expansion_engine import DFSExpansionEngine
from .chapter_level_validation import ChapterLevelValidation
from .description_optimize import DescriptionOptimize
from .word_count_allocation import WordCountAllocation
from .global_validation import GlobalValidation
from .format import Format

__all__ = [
    'QueryStyleRecognition',
    'RootNodeEstablishment',
    'GlobalAbstract',
    'DFSExpansionEngine',
    'ChapterLevelValidation',
    'DescriptionOptimize',
    'WordCountAllocation',
    'GlobalValidation',
    'Format',
    'flush_stdout',
    'print_and_flush',
]
