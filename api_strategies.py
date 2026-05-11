"""
API 调用策略模式实现

支持多种 API 提供商的可插拔式调用策略：
- DashScope (阿里云百炼)
- OpenAI 兼容 API
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import time


class APICallStrategy(ABC):
    """API 调用策略基类"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略

        参数：
            config: 配置字典，包含 api_key, model, temperature 等
        """
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "")
        self.temperature = config.get("temperature", 0.5)
        self.thinking = config.get("thinking", None)
        self.extra_body = config.get("extra_body", None)

    @abstractmethod
    def call(
        self,
        prompt: str,

        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
    ) -> str:
        """
        调用 API 生成文本

        参数：
            prompt: 用户提示词
            temperature: 温度参数（覆盖配置）
            stop: 停止词
            system_prompt: 系统提示词
            messages: 完整的 messages 列表（覆盖 prompt 和 system_prompt）

        返回：
            生成的文本内容
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """获取提供商名称"""
        pass

    def validate_config(self) -> None:
        """验证配置"""
        if not self.api_key:
            raise ValueError(f"API Key 未配置（provider: {self.get_provider_name()}）")
        if not self.model:
            raise ValueError(f"Model 未配置（provider: {self.get_provider_name()}）")


class DashScopeStrategy(APICallStrategy):
    """阿里云百炼 (DashScope) 调用策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.max_retries = config.get("max_retries", 3)  # 最大重试次数
        self.retry_delay = config.get("retry_delay", 1.0)  # 重试延迟（秒）
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except ImportError:
            raise ImportError("请先安装 openai: pip install openai")
        self.validate_config()

    def call(
        self,
        prompt: str,

        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
    ) -> str:
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        # 重试机制
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                # print(f"--- [DEBUG] Text 长度: {len(prompt)} 字符")

                request_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature if temperature is not None else self.temperature,
                }

                if stop:
                    request_params["stop"] = stop

                extra_body = dict(self.extra_body or {})
                if "enable_thinking" not in extra_body and self.model.startswith("qwen3"):
                    extra_body["enable_thinking"] = False
                if extra_body:
                    request_params["extra_body"] = extra_body

                response = self.client.chat.completions.create(**request_params)

                if response.choices and len(response.choices) > 0:
                    message = response.choices[0].message

                    if hasattr(message, 'content') and message.content:
                        return message.content

                    if hasattr(message, 'reasoning_content') and message.reasoning_content:
                        return message.reasoning_content

                    return str(message)
                else:
                    raise Exception(f"DashScope API 响应格式异常: {response}")

            except Exception as e:
                last_exception = e
                error_msg = str(e)
                
                # 判断是否应该重试
                should_retry = False
                retry_reason = ""
                
                # 检查是否为可重试的错误
                if "data_inspection_failed" in error_msg:
                    # 内容审核失败 - 可能是误报，可以重试
                    should_retry = True
                    retry_reason = "内容审核失败（可能是误报）"
                elif "400" in error_msg and "InternalError" in error_msg:
                    # 内部错误 - 可以重试
                    should_retry = True
                    retry_reason = "API 内部错误"
                elif "500" in error_msg or "502" in error_msg or "503" in error_msg:
                    # 服务器错误 - 应该重试
                    should_retry = True
                    retry_reason = "服务器错误"
                elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                    # 超时或连接错误 - 应该重试
                    should_retry = True
                    retry_reason = "网络超时或连接错误"
                
                # 如果是最后一次尝试，不再重试
                if attempt == self.max_retries - 1:
                    should_retry = False
                
                if should_retry:
                    import traceback
                    traceback.print_exc()
                    
                    delay = self.retry_delay * (2 ** attempt)  # 指数退避
                    print(f"  [DashScope] 第 {attempt + 1}/{self.max_retries} 次调用失败: {retry_reason}")
                    print(f"  [DashScope] {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                else:
                    # 不应该重试或已达到最大重试次数，抛出异常
                    import traceback
                    traceback.print_exc()
                    raise Exception(f"DashScope API 调用失败: {e}")
        
        # 所有重试都失败
        raise Exception(f"DashScope API 调用失败（已重试 {self.max_retries} 次）: {last_exception}")

    def get_provider_name(self) -> str:
        return "dashscope"


class OpenAICompatibleStrategy(APICallStrategy):
    """OpenAI 兼容 API 调用策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "")
        if not self.base_url:
            raise ValueError("Base URL 未配置（provider: openai_compatible）")
        self.validate_config()

    def call(
        self,
        prompt: str,

        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
    ) -> str:
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        import requests

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if stop:
            payload["stop"] = stop

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        endpoint = self.base_url.rstrip('/')
        if "chat/completions" not in endpoint:
            endpoint = f"{endpoint}/chat/completions"

        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=600
                )

                if response.status_code != 200:
                    raise Exception(f"API 返回错误: {response.status_code} - {response.text}")

                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    return content
                else:
                    raise Exception(f"API 响应格式异常: {result}")

            except KeyboardInterrupt as e:
                raise e

            except Exception as e:
                error_msg = str(e)

                if "maximum context length" in error_msg or "maximum_tokens" in error_msg:
                    raise e
                elif "content management policy" in error_msg or "triggering" in error_msg:
                    return 'Trigger content management policy'

                print(f"API 调用失败（尝试 {attempt + 1}/{max_retries}）: {error_msg}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1
                    time.sleep(wait_time)

        print("API 调用失败：达到最大重试次数")
        return ""

    def get_provider_name(self) -> str:
        return "openai_compatible"


class StrategyFactory:
    """策略工厂类"""

    _strategies = {
        "dashscope": DashScopeStrategy,
        "openai_compatible": OpenAICompatibleStrategy,
    }

    @classmethod
    def create_strategy(cls, provider: str, config: Dict[str, Any]) -> APICallStrategy:
        """
        创建策略实例

        参数：
            provider: 提供商名称（dashscope, openai_compatible）
            config: 配置字典

        返回：
            APICallStrategy 实例
        """
        if provider not in cls._strategies:
            raise ValueError(f"未知的 provider: {provider}。可用: {list(cls._strategies.keys())}")

        strategy_class = cls._strategies[provider]
        return strategy_class(config)

    @classmethod
    def register_strategy(cls, provider: str, strategy_class: type) -> None:
        """
        注册新策略

        参数：
            provider: 提供商名称
            strategy_class: 策略类
        """
        cls._strategies[provider] = strategy_class

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """获取所有可用的提供商"""
        return list(cls._strategies.keys())
