"""
ISRP 管线步骤模块

包含九个独立处理器，按三阶段组织：

  预处理:
    QueryStyleRecognition      风格识别 + 显式章节框架检测（快捷路径）

  大纲树生成:
    RootNodeEstablishment      根节点建立（意图发散 + 串行反思）
    GlobalAbstract             全局摘要提取
    DFSExpansionEngine         深度优先节点扩展（纵向扩展 + 横向探针 + 概念覆盖池）
    ChapterLevelValidation     章节级检查点（完成一级章节后即时触发）
    GlobalValidation           全局校验与修复

  大纲树优化:
    DescriptionOptimize        节点描述批量精修
    WordCountAllocation        BFS 逐层字数分配
    Format                     标准化格式输出

"""
import sys


# ============================================================================
# 实时输出辅助函数
# ============================================================================

def flush_stdout():
    """强制刷新 stdout，确保实时输出"""
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
