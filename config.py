"""
AgentWrite 配置文件

支持多模型 API 配置，包括：
- OpenAI API (官方)
- 阿里云百炼 API (DashScope)

使用方法：
    1. 在下方 API_KEYS 区域填入你的 API Key
    2. 在 PROVIDER_CONFIGS 区域配置各平台的基础信息
    3. 在 MODEL_CONFIGS 区域选择使用的模型
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================================
# API Keys - 在这里集中配置所有密钥
# ============================================================================

API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "dashscope": os.getenv("DASHSCOPE_API_KEY", ""),
}

# ============================================================================
# Provider 基础配置 - 各平台的 API 端点
# ============================================================================

PROVIDER_CONFIGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
}

# ============================================================================
# 模型列表 - 各平台可用的模型
# ============================================================================

MODELS = {
    "openai": {
        "gpt-4o": "gpt-4o-2024-05-13",
        "gpt-4-turbo": "gpt-4-turbo",
    },
    "dashscope": {
        "deepseek-v4-pro": "deepseek-v4-pro",
        "qwen3-max": "qwen3-max-2026-01-23",
        "glm-5": "glm-5",
        "kimi-k2.5": "kimi-k2.5",
    },
}

# ============================================================================
# 通用配置
# ============================================================================

INPUT_FILE = "data/WritingBench-120.jsonl"
OUTPUT_PLAN_FILE = "outputs/plan.jsonl"
OUTPUT_WRITE_FILE = "outputs/write.jsonl"
CACHE_FILE = "outputs/write_cache.jsonl"

NUM_PROCESSES = 1
RESUME_FROM_CHECKPOINT = True

# ============================================================================
# 模型配置 — 大纲生成
# ============================================================================

PLANNER_PROVIDER = "dashscope"
PLANNER_MODEL = os.getenv("AGENTWRITE_PLANNER_MODEL", "qwen3-max")
PLANNER_TEMPERATURE = 0.5

PLANNER_CONFIG = {
    "provider": PLANNER_PROVIDER,
    "openai": {
        "api_key": API_KEYS["openai"],
        "base_url": PROVIDER_CONFIGS["openai"]["base_url"],
        "model": MODELS["openai"]["gpt-4o"],
    },
    "dashscope": {
        "api_key": API_KEYS["dashscope"],
        "base_url": PROVIDER_CONFIGS["dashscope"]["base_url"],
        "model": MODELS["dashscope"].get(PLANNER_MODEL, PLANNER_MODEL),
        "temperature": PLANNER_TEMPERATURE,
    },
    "custom": {
        "api_key": "",
        "base_url": "",
        "model": "",
        "temperature": 0.5,
    },
}

# ============================================================================
# 模型配置 — 长文生成
# ============================================================================

WRITER_CONFIGS = {
    "qwen3-max-preview": {
        "provider": "dashscope",
        "api_key": API_KEYS["dashscope"],
        "base_url": PROVIDER_CONFIGS["dashscope"]["base_url"],
        "model": MODELS["dashscope"]["qwen3-max"],
        "temperature": 0.5,
    },
    "deepseek-v4-pro": {
        "provider": "dashscope",
        "api_key": API_KEYS["dashscope"],
        "base_url": PROVIDER_CONFIGS["dashscope"]["base_url"],
        "model": MODELS["dashscope"]["deepseek-v4-pro"],
        "temperature": 0.5,
    },
    "qwen3-max-2026-01-23": {
        "provider": "dashscope",
        "api_key": API_KEYS["dashscope"],
        "base_url": PROVIDER_CONFIGS["dashscope"]["base_url"],
        "model": MODELS["dashscope"]["qwen3-max"],
        "temperature": 0.5,
    },
    "kimi-k2.5": {
        "provider": "dashscope",
        "api_key": API_KEYS["dashscope"],
        "base_url": PROVIDER_CONFIGS["dashscope"]["base_url"],
        "model": MODELS["dashscope"]["kimi-k2.5"],
        "temperature": 0.5,
    },
}

# 默认使用的 Writer 模型
DEFAULT_WRITER = os.getenv("AGENTWRITE_DEFAULT_WRITER", "qwen3-max-2026-01-23")

# ============================================================================
# 评估模型配置
# ============================================================================

EVALUATOR_PROVIDER = "dashscope"
EVALUATOR_MODEL = "glm-5"
EVALUATOR_TEMPERATURE = 0.1

EVALUATOR_CONFIG = {
    "provider": EVALUATOR_PROVIDER,
    "dashscope": {
        "api_key": API_KEYS["dashscope"],
        "base_url": PROVIDER_CONFIGS["dashscope"]["base_url"],
        "model_options": {
            "glm-5": MODELS["dashscope"]["glm-5"],
        },
        "default_model": MODELS["dashscope"][EVALUATOR_MODEL],
        "temperature": EVALUATOR_TEMPERATURE,
    },
}

# ============================================================================
# 数据后处理配置
# ============================================================================

POST_PROCESSING = {
    # 是否去除段落前缀（如 "Paragraph 1:"）
    "remove_paragraph_prefix": True,

    # 是否过滤太短的输出（字数）
    "min_output_length": 100,

    # 是否去除生成的多余空行
    "remove_extra_newlines": True,

    # 是否验证格式
    "validate_format": True,
}

# ============================================================================
# 日志配置
# ============================================================================

LOGGING = {
    "enable": True,
    "log_file": "outputs/log.txt",
    "print_frequency": 10,  # 每处理 N 条打印一次进度
}

# ============================================================================
# RAG 模块配置
# ============================================================================

RAG_CONFIG = {
    # 基础开关
    "enabled": True,                    # 是否启用 RAG

    # 切片器配置
    "chunker_type": "recursive",        # recursive / semantic / sentence
    "chunk_size": 512,                  # 目标切片大小
    "chunk_overlap": 50,                # 切片重叠

    # Embedding 配置
    "embedding_model": "text-embedding-v4",
    "embedding_dimension": 1024,
    "embedding_batch_size": 10,

    # 向量存储配置
    "vector_db_path": "rag_cache/vectors",
    "cache_dir": "rag_cache",

    # 检索配置
    "top_k": 3,                         # 检索返回数量
    "retrieval_level": "all",           # all / level1 / level0
    "min_prompt_length": 2000,          # 触发 RAG 的最小 prompt 长度
    "similarity_threshold": 0.3,        # 相似度阈值

    # API 配置（使用全局 API_KEYS）
    "api_key": API_KEYS.get("dashscope", ""),
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}
