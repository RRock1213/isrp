"""
统一的 API 调用客户端（重构版）

支持多种 API 提供商的可插拔式调用策略：
- DashScope (阿里云百炼)
- OpenAI 兼容 API (硅基流动等)

使用策略模式实现，支持运行时动态切换平台
"""

from typing import Optional, List, Dict, Any
try:
    from .api_strategies import StrategyFactory, APICallStrategy
except ImportError:
    from api_strategies import StrategyFactory, APICallStrategy


class UnifiedAPIClient:
    """统一的 API 调用客户端（使用策略模式）"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 API 客户端

        参数：
            config: 配置字典，包含 provider, api_key, base_url, model 等
        """
        self.provider = config.get("provider", "openai_compatible")
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "")
        self.model = config.get("model", "")
        self.temperature = config.get("temperature", 0.5)
        self.thinking = config.get("thinking", None)

        # 创建策略实例
        self.strategy = StrategyFactory.create_strategy(self.provider, config)

    def call_api(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
    ) -> str:
        """
        调用 API 生成文本

        参数：
            prompt: 用户提示词
            max_new_tokens: 最大生成 token 数（覆盖配置）
            temperature: 温度参数（覆盖配置）
            stop: 停止词
            system_prompt: 系统提示词
            messages: 完整的 messages 列表（覆盖 prompt 和 system_prompt）

        返回：
            生成的文本内容
        """
        return self.strategy.call(
            prompt=prompt,
            temperature=temperature,
            stop=stop,
            system_prompt=system_prompt,
            messages=messages,
        )

    def switch_provider(self, provider: str, config: Dict[str, Any]) -> None:
        """
        切换 API 提供商

        参数：
            provider: 提供商名称
            config: 新的配置字典
        """
        self.provider = provider
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "")
        self.model = config.get("model", "")
        self.temperature = config.get("temperature", 0.5)
        self.thinking = config.get("thinking", None)

        # 创建新的策略实例
        self.strategy = StrategyFactory.create_strategy(provider, config)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型配置信息"""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
        }


def create_client_from_config(config_key: str, config_dict: Dict[str, Any]) -> UnifiedAPIClient:
    """
    从配置创建 API 客户端

    参数：
        config_key: 配置键名（如 "qwen3-8b"）
        config_dict: 配置字典（如 WRITER_CONFIGS[config_key]）

    返回：
        UnifiedAPIClient 实例
    """
    return UnifiedAPIClient(config_dict)


# 便捷函数：从 config.py 加载配置
def get_planner_client():
    """获取大纲生成用的 API 客户端"""
    try:
        from .config import PLANNER_CONFIG
    except ImportError:
        from config import PLANNER_CONFIG

    provider = PLANNER_CONFIG["provider"]
    if provider not in PLANNER_CONFIG:
        raise ValueError(f"未知的 provider: {provider}")

    config = PLANNER_CONFIG[provider].copy()
    config["provider"] = provider

    return UnifiedAPIClient(config)


def get_writer_client(model_name: str = None):
    """
    获取长文生成用的 API 客户端

    参数：
        model_name: 模型名称（如 "qwen3-8b"），如果为 None 则使用默认模型

    返回：
        UnifiedAPIClient 实例
    """
    try:
        from .config import WRITER_CONFIGS, DEFAULT_WRITER
    except ImportError:
        from config import WRITER_CONFIGS, DEFAULT_WRITER

    if model_name is None:
        model_name = DEFAULT_WRITER

    if model_name not in WRITER_CONFIGS:
        raise ValueError(f"未知的模型: {model_name}。可用模型: {list(WRITER_CONFIGS.keys())}")

    config = WRITER_CONFIGS[model_name]

    return UnifiedAPIClient(config)


def get_evaluator_client(model_name: str = None, provider: str = None):
    """
    获取评估用的 API 客户端

    参数：
        model_name: 模型名称（如 "glm-5"），如果为 None 则使用默认模型
        provider: 提供商名称（如 "dashscope"），如果为 None 则使用默认提供商

    返回：
        UnifiedAPIClient 实例
    """
    try:
        from .config import EVALUATOR_CONFIG
    except ImportError:
        from config import EVALUATOR_CONFIG

    # 确定使用的提供商
    if provider is None:
        provider = EVALUATOR_CONFIG.get("provider", "dashscope")

    # 检查提供商配置是否存在
    if provider not in EVALUATOR_CONFIG:
        raise ValueError(f"未知的 provider: {provider}。可用: {list(EVALUATOR_CONFIG.keys())}")

    # 确定使用的模型
    if model_name is None:
        model_name = EVALUATOR_CONFIG[provider]["default_model"]

    model_options = EVALUATOR_CONFIG[provider]["model_options"]
    if model_name not in model_options:
        raise ValueError(f"未知的评估模型: {model_name}。可用模型: {list(model_options.keys())}")

    # 构建配置
    config = {
        "provider": provider,
        "api_key": EVALUATOR_CONFIG[provider]["api_key"],
        "model": model_options[model_name],
        "temperature": EVALUATOR_CONFIG[provider].get("temperature", 0.1),
    }

    # 如果不是 dashscope，需要添加 base_url
    if provider != "dashscope":
        config["base_url"] = EVALUATOR_CONFIG[provider].get("base_url", "")

    return UnifiedAPIClient(config)
