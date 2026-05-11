"""
消融实验配置模块（对应论文 Table 3）

通过环境变量控制各消融开关：
  - ABLATION_ITERATION: 禁用迭代评估（w/o Iteration Evaluation）
  - ABLATION_ANTI_FORGET: 禁用全局记忆注入（w/o Global Memory Injection）
  - ABLATION_DELAYED_CONSTRAINT: 使用均匀预分配（w/o Word Count Allocation）
  - RAG 禁用通过 run_ablation_test.py 的 use_rag=False 控制（w/o Local Context Injection）
"""

import os


def is_ablation_iteration_enabled() -> bool:
    """检查迭代评估消融是否启用"""
    return os.environ.get('ABLATION_ITERATION', '').lower() in ('true', '1', 'yes')


def is_ablation_anti_forget_enabled() -> bool:
    """检查全局记忆注入消融是否启用"""
    return os.environ.get('ABLATION_ANTI_FORGET', '').lower() in ('true', '1', 'yes')


def is_ablation_delayed_constraint_enabled() -> bool:
    """检查字数分配消融是否启用（使用均匀预分配代替 BFS 权重分配）"""
    return os.environ.get('ABLATION_DELAYED_CONSTRAINT', '').lower() in ('true', '1', 'yes')


def set_ablation_config(ablation_type: str):
    """
    根据消融类型设置环境变量

    Args:
        ablation_type: 消融类型
            iteration_eval  — w/o Iteration Evaluation
            global_memory   — w/o Global Memory Injection
            word_allocation — w/o Word Count Allocation
            full            — Full ISRP（对照组）
    """
    for key in ['ABLATION_ITERATION', 'ABLATION_ANTI_FORGET', 'ABLATION_DELAYED_CONSTRAINT']:
        os.environ.pop(key, None)

    if ablation_type == 'iteration_eval':
        os.environ['ABLATION_ITERATION'] = 'true'
    elif ablation_type == 'global_memory':
        os.environ['ABLATION_ANTI_FORGET'] = 'true'
    elif ablation_type == 'word_allocation':
        os.environ['ABLATION_DELAYED_CONSTRAINT'] = 'true'
