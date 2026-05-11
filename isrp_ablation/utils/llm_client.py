"""
ISRP — LLM 调用封装

统一 LLM 调用接口，支持:
  - 基础调用：任意 prompt 文本 -> 响应
  - 带重试的调用：失败自动重试（可配置次数）
  - 带 Schema 解析的调用：Pydantic BaseModel 自动验证与实例化
  - 调用统计：累计调用次数、总耗时、Token 消耗
  - 实时输出刷新：在终端实时显示生成进度

"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from ..schemas import robust_json_parse

def _flush_stdout():
    """强制刷新 stdout，确保实时输出"""
    sys.stdout.flush()

def _print_and_flush(msg: str):
    """打印并立即刷新输出"""
    print(msg)
    _flush_stdout()

class LLMClient:
    """
    LLM 调用封装

    提供统一的 LLM 调用接口，支持重试、Schema 解析和持久化
    """

    # 类级别的关闭检查器，可由外部设置
    _shutdown_checker: callable = None

    @classmethod
    def set_shutdown_checker(cls, checker: callable) -> None:
        """
        设置关闭检查器（用于优雅中断）

        Args:
            checker: 无参函数，返回 bool 表示是否请求关闭
        """
        cls._shutdown_checker = checker

    @classmethod
    def clear_shutdown_checker(cls) -> None:
        """清除关闭检查器"""
        cls._shutdown_checker = None

    def __init__(
        self,
        client,
        persist_file: str = None,
        enable_logging: bool = True
    ):
        """
        初始化 LLM 客户端

        Args:
            client: API 客户端（UnifiedAPIClient 实例）
            persist_file: LLM 调用持久化文件路径
            enable_logging: 是否启用日志
        """
        self.client = client
        self.persist_file = persist_file
        self.enable_logging = enable_logging

        # 统计信息
        self.call_count = 0
        self.total_time = 0.0

    def call(
        self,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 4096
    ) -> str:
        """
        同步调用 LLM

        Args:
            prompt: 提示词
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            LLM 响应字符串
        """
        # 检查是否请求关闭（用于优雅中断）
        # 使用类名访问类属性，避免实例绑定问题
        if LLMClient._shutdown_checker is not None:
            if LLMClient._shutdown_checker():
                raise KeyboardInterrupt("用户请求退出")

        start_time = time.time()

        response = self.client.call_api(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )

        elapsed = time.time() - start_time
        self.call_count += 1
        self.total_time += elapsed
        # 持久化
        if self.persist_file:
            self._persist_call(prompt, response, elapsed, max_tokens, temperature)

        return response

    def call_with_retry(
        self,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 4096,
        max_retries: int = 2
    ) -> str:
        """
        带重试机制的 LLM 调用

        Args:
            prompt: 提示词
            temperature: 温度参数
            max_tokens: 最大 token 数
            max_retries: 最大重试次数

        Returns:
            LLM 响应字符串

        Raises:
            Exception: 所有重试都失败后抛出最后一个异常
        """
        last_error = None
        current_temp = temperature

        for attempt in range(max_retries + 1):
            try:
                response = self.call(prompt, current_temp, max_tokens)
                if response and response.strip():
                    return response
                else:
                    last_error = Exception("Empty LLM response")
            except Exception as e:
                last_error = e

            if attempt < max_retries:
                if self.enable_logging:
                    print(f"    [Retry {attempt + 1}/{max_retries}] LLM call failed, retrying...")
                current_temp = min(current_temp + 0.1, 1.0)

        raise Exception(f"LLM call failed after {max_retries + 1} attempts. Last error: {last_error}")

    def call_with_schema(
        self,
        prompt: str,
        schema: type,
        temperature: float = 0.5,
        max_tokens: int = 4096,
        max_retries: int = 3
    ) -> Any:
        """
        调用 LLM 并解析为 Pydantic Schema

        Args:
            prompt: 提示词
            schema: 目标 Pydantic 模型
            temperature: 温度参数
            max_tokens: 最大 token 数
            max_retries: 最大重试次数

        Returns:
            解析后的 Pydantic 模型实例

        Raises:
            Exception: 所有重试都失败后抛出最后一个异常
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.call(prompt, temperature, max_tokens)
                result = robust_json_parse(response, schema)
                return result
            except Exception as e:
                last_error = e
                if self.enable_logging and attempt < max_retries - 1:
                    print(f"    [Retry {attempt + 1}/{max_retries}] Parse error: {str(e)}...")

        # 所有重试都失败，抛出最后一个错误
        raise Exception(f"Failed after {max_retries} retries. Last error: {last_error}")

    def parse_json_with_retry(
        self,
        response: str,
        target_type: type,
        max_retries: int = 2,
        regenerate_fn=None,
        regenerate_args: dict = None
    ) -> Any:
        """
        带重试机制的 JSON 解析

        Args:
            response: LLM 响应字符串
            target_type: 目标类型（dict 或 Pydantic 模型）
            max_retries: 最大重试次数
            regenerate_fn: 重试时重新生成的函数（可选）
            regenerate_args: 重新生成函数的参数（可选）

        Returns:
            解析后的对象
        """
        last_error = None
        current_response = response

        for attempt in range(max_retries + 1):
            try:
                if isinstance(target_type, type) and issubclass(target_type, BaseModel):
                    return robust_json_parse(current_response, target_type)
                else:
                    return robust_json_parse(current_response, target_type)
            except Exception as e:
                last_error = e

                if attempt < max_retries:
                    if self.enable_logging:
                        print(f"    [Retry {attempt + 1}/{max_retries}] JSON parse failed: {str(e)}")

                    # 如果有重新生成函数，尝试重新生成
                    if regenerate_fn and regenerate_args:
                        try:
                            current_response = regenerate_fn(**regenerate_args)
                        except:
                            pass

        raise Exception(f"JSON parse failed after {max_retries + 1} attempts. Last error: {last_error}")

    def get_stats(self) -> Dict[str, Any]:
        """获取调用统计"""
        return {
            "total_calls": self.call_count,
            "total_time": round(self.total_time, 3),
            "avg_time": round(self.total_time / self.call_count, 3) if self.call_count > 0 else 0
        }

    def _persist_call(
        self,
        prompt: str,
        response: str,
        elapsed: float,
        max_tokens: int,
        temperature: float
    ) -> None:
        """
        持久化 LLM 调用记录（带错误处理）

        Args:
            prompt: 提示词
            response: 响应
            elapsed: 耗时
            max_tokens: 最大 token 数
            temperature: 温度参数
        """
        if not self.persist_file:
            return

        try:
            import os
            os.makedirs(os.path.dirname(self.persist_file), exist_ok=True)

            record = {
                "timestamp": datetime.now().isoformat(),
                "call_id": self.call_count,
                "elapsed_time": round(elapsed, 3),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "prompt": prompt,
                "response": response
            }

            # 带重试的文件写入（处理 Windows 文件锁）
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with open(self.persist_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    break  # 成功则退出重试
                except (IOError, OSError) as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.1)  # 等待 100ms 后重试
                    else:
                        # 重试失败，记录警告但不抛出异常
                        if self.enable_logging:
                            print(f"  [Warning] 无法写入日志文件（重试 {max_retries} 次后失败）: {e}")

        except Exception as e:
            # 日志写入失败不应影响主流程
            if self.enable_logging:
                print(f"  [Warning] 日志持久化失败（不影响主流程）: {e}")