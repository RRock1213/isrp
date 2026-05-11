"""
ISRP 全量测试脚本

使用方法：
    python run_plan.py                           # 使用默认配置（ISRP 评估器）
    python run_plan.py --input data/WritingBench-120.jsonl --output outputs/plan.jsonl
    python run_plan.py --evaluator writingbench  # 使用 WritingBench 评估器
    python run_plan.py --no_resume               # 禁用断点续传（默认启用）
    python run_plan.py --limit 10                # 仅处理前10条
    python run_plan.py --no_use_llm_word_count   # 禁用 LLM 预估字数（默认启用）
    python run_plan.py --no_log_console          # 禁用终端日志持久化（默认启用）
    python run_plan.py --no_rag                  # 禁用 RAG 检索增强（默认启用）

并发安全说明：
    当同时运行多个实例时，建议使用不同的输出文件：
    python run_plan.py --start 0 --limit 10 --output outputs/plan_part1.jsonl
    python run_plan.py --start 10 --limit 10 --output outputs/plan_part2.jsonl

    或者使用 --output_suffix 参数自动生成唯一输出文件：
    python run_plan.py --start 0 --limit 10 --output_suffix "_part1"
    python run_plan.py --start 10 --limit 10 --output_suffix "_part2"
"""

import sys
import time
import json
import argparse
import os
import signal
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Any, Optional, List


# ============================================================================
# 终端输出持久化 - Tee 类
# ============================================================================

class TeeOutput:
    """
    将输出同时写入终端和文件

    用法：
        tee = TeeOutput(log_file_path)
        sys.stdout = tee
        print("这条信息会同时显示在终端和写入文件")
    """

    def __init__(self, log_file_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'a', encoding='utf-8')
        self.log_file_path = log_file_path

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()  # 立即刷新终端
        self.log_file.write(message)
        self.log_file.flush()  # 立即刷新文件

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# 全局 Tee 实例
_tee_output = None


def setup_console_logging(log_file_path: str):
    """
    设置终端输出持久化

    Args:
        log_file_path: 日志文件路径
    """
    global _tee_output
    _tee_output = TeeOutput(log_file_path)
    sys.stdout = _tee_output
    print(f"[Console Logging] Terminal output will be saved to: {log_file_path}")


def close_console_logging():
    """关闭终端输出持久化"""
    global _tee_output
    if _tee_output:
        print(f"[Console Logging] Log saved to: {_tee_output.log_file_path}")
        _tee_output.close()
        sys.stdout = _tee_output.terminal
        _tee_output = None


# ============================================================================
# Windows 控制台 UTF-8 编码设置
# ============================================================================

# Windows 控制台 UTF-8 编码设置
# 解决 UnicodeEncodeError: 'gbk' codec can't encode character 问题
if sys.platform == 'win32':
    try:
        # 方法1：设置环境变量
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        # 方法2：重新配置 stdout/stderr（使用更安全的方式）
        if hasattr(sys.stdout, 'buffer'):
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')
    except Exception as e:
        # 静默失败，使用 errors='replace' 策略处理无法编码的字符
        pass

# 跨平台文件锁支持
if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api_client import get_planner_client

# 导入 ISRP 模块
from isrp import SyncISRPPlanner, MAX_ITERATIONS, MAX_DEPTH, MAX_WIDTH, LLMClient
from isrp.engines import extract_word_count
from isrp.evaluators import EvaluatorFactory, BaseEvaluator
from isrp.utils import generate_sample_id  # 统一样本 ID 生成

# 导入 RAG 服务
try:
    from rag_module.rag_service import create_rag_service
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


# ============================================================================
# 优雅退出支持
# ============================================================================

# 全局标志：是否收到中断信号
_shutdown_requested = False


def _signal_handler(signum, frame):
    """信号处理器：优雅退出"""
    global _shutdown_requested
    if _shutdown_requested:
        print("\n\n强制退出...")
        sys.exit(1)
    _shutdown_requested = True
    print("\n\n收到中断信号，正在优雅退出...")
    print("（再次按 Ctrl+C 可强制退出）")


def is_shutdown_requested() -> bool:
    """检查是否请求退出"""
    return _shutdown_requested


# ============================================================================
# 询问类指令检测
# ============================================================================

def is_query_only_instruction(prompt: str) -> bool:
    """
    检测是否为询问类指令（非写作任务）

    询问类指令特征：
    - 结尾有"吗？"、"能吗"、"可以吗"等
    - 以"可以"、"能否"、"是否"开头或包含
    - 没有明确的写作主题或字数要求

    Args:
        prompt: 用户指令

    Returns:
        True 表示是询问类指令，应跳过处理
    """
    import re

    # 清理 prompt
    prompt_lower = prompt.lower().strip()

    # 检测询问关键词结尾
    query_endings = ['吗？', '吗?', '可以吗', '能吗', '能否', '是否可以', '能做到吗']
    for ending in query_endings:
        if prompt_lower.endswith(ending):
            # 进一步确认没有明确的写作要求
            # 检查是否包含写作相关关键词
            writing_keywords = ['写', '撰写', '论文', '文章', '报告', '字', '千字', '万字']
            has_writing_keyword = any(kw in prompt for kw in writing_keywords)

            # 如果没有写作关键词，且主要内容是询问
            if not has_writing_keyword:
                return True

            # 即使有写作关键词，如果主要内容是询问而非任务
            # 例如："可以做到吗？请你帮我写论文大纲" -> 这是有效任务
            # 例如："你会写论文吗？" -> 这是询问
            if len(prompt) < 50 and not any(kw in prompt for kw in ['帮我', '请', '要求', '字数']):
                return True

    # 检测纯询问模式（没有实质内容）
    # 例如："你会写论文吗？" -> 只有询问，没有任务
    pure_query_patterns = [
        r'^你会[写撰].*[吗？]',
        r'^你能[写撰].*[吗？]',
        r'^可以[帮写].*[吗？]',
        r'^是否[能可].*[吗？]',
    ]
    for pattern in pure_query_patterns:
        if re.search(pattern, prompt):
            return True

    return False


# 注册信号处理器
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# 注册关闭检查器到 LLMClient
LLMClient.set_shutdown_checker(is_shutdown_requested)


# ============================================================================
# 跨平台文件锁实现
# ============================================================================

def acquire_file_lock(file_obj, max_retries=3):
    """获取文件锁（跨平台）"""
    import time
    for attempt in range(max_retries):
        try:
            if sys.platform == 'win32':
                msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            return True
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.1)  # 等待 100ms 后重试
            else:
                return False
    return False


def release_file_lock(file_obj):
    """释放文件锁（跨平台）"""
    try:
        if sys.platform == 'win32':
            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
    except (IOError, OSError):
        pass  # 忽略释放锁时的错误


def safe_write_line(file_obj, line: str, max_retries: int = 5):
    """安全写入一行数据（带文件锁和错误处理）"""
    import time

    for attempt in range(max_retries):
        try:
            # 尝试获取文件锁
            lock_acquired = acquire_file_lock(file_obj, max_retries=1)
            try:
                file_obj.write(line)
                file_obj.flush()
                return True  # 写入成功
            finally:
                if lock_acquired:
                    release_file_lock(file_obj)
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.2 * (attempt + 1))  # 递增等待时间
            else:
                print(f"  [Warning] 文件写入失败（重试 {max_retries} 次后）: {e}")
                return False
        except Exception as e:
            print(f"  [Warning] 文件写入发生意外错误: {e}")
            return False

    return False


# ============================================================================
# LLM 字数预估
# ============================================================================

def estimate_word_count_with_llm(client, prompt: str, language: str = "zh") -> int:
    """
    使用 LLM 预估合适的字数（仅在格式化提取失败时调用）

    Args:
        client: API 客户端
        prompt: 用户指令
        language: 语言代码

    Returns:
        预估的字数
    """
    if language == "zh":
        estimation_prompt = f"""分析以下写作任务，预估一个合理的文章字数。

【写作指令】
{prompt}

请输出 JSON 格式：
{{
  "task_type": "任务类型（如：小说、论文、报告等）",
  "suggested_word_count": 建议字数（整数）,
  "reasoning": "预估理由"
}}

注意：
- 小型任务：1500-3000字
- 中等任务：3000-5000字
- 复杂任务：5000-10000字

只输出 JSON，不要其他内容。"""
    else:
        estimation_prompt = f"""Analyze the following writing task and estimate a reasonable word count.

[Writing Instruction]
{prompt}

Please output in JSON format:
{{
  "task_type": "Task type (e.g., novel, paper, report, etc.)",
  "suggested_word_count": suggested word count (integer),
  "reasoning": "Reasoning for the estimate"
}}

Note:
- Small tasks: 1500-3000 words
- Medium tasks: 3000-5000 words
- Complex tasks: 5000-10000 words

Output only JSON, no other content."""

    try:
        result = client.call_api(estimation_prompt, temperature=0.3)

        # 解析 JSON
        import re
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            word_count = data.get("suggested_word_count", 5000)
            # 验证范围
            if 1000 <= word_count <= 20000:
                return word_count
    except Exception as e:
        print(f"  [LLM 字数预估失败] {e}")

    return 5000


# ============================================================================
# 测试函数
# ============================================================================

def run_full_test(
    input_file: str,
    output_file: str,
    evaluator_type: str = "isrp",
    use_llm_word_count: bool = True,
    resume: bool = True,
    start: int = 0,
    limit: int = None,
    max_depth: int = MAX_DEPTH,
    max_iterations: int = MAX_ITERATIONS,
    enable_logging: bool = True,
    log_console: bool = True,
    console_log_file: str = None,
    use_rag: bool = True,
    rag_chunk_size: int = 512,
    rag_top_k: int = 3,
    rag_min_prompt_length: int = 2000,
    rag_chunker_type: str = "recursive",
    rag_log_dir: str = None,
    max_width: int = None,
    model: str = None
):
    """
    运行 ISRP 全量测试

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        evaluator_type: 评估器类型 ("isrp" 或 "writingbench")
        use_llm_word_count: 是否使用 LLM 预估字数
        resume: 是否启用断点续传
        start: 开始索引
        limit: 处理条数限制
        max_depth: 最大树深度
        max_iterations: 串行迭代上限
        enable_logging: 是否启用日志
        log_console: 是否将终端输出持久化到日志文件（默认启用）
        console_log_file: 终端日志文件路径（可选）
        use_rag: 是否启用 RAG 检索增强（默认启用）
        rag_chunk_size: RAG 切片大小
        rag_top_k: RAG 检索返回数量
        rag_min_prompt_length: 触发 RAG 的最小 prompt 长度
        rag_chunker_type: 切片器类型
        rag_log_dir: RAG 日志目录
        max_width: DFS 扩展每层最大子节点数，None 表示使用 engines.py 默认值
    """
    # 终端输出持久化
    if log_console:
        if console_log_file is None:
            # 自动生成日志文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path(output_file).parent / "console_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            console_log_file = str(log_dir / f"console_{timestamp}.log")
        setup_console_logging(console_log_file)

    try:
        print("=" * 70)
        print("ISRP 全量测试：动态评估器支持")
        print("=" * 70)
        print(f"评估器类型: {evaluator_type}")
        print(f"LLM 字数预估: {'启用' if use_llm_word_count else '禁用'}")
        if max_width is not None:
            print(f"生成宽度上限: {max_width}")
        if log_console:
            print(f"终端日志持久化: 启用 -> {console_log_file}")

        print("\n正在初始化 API 客户端...")
        try:
            if model:
                from config import PLANNER_CONFIG, MODELS
                provider = PLANNER_CONFIG["provider"]
                resolved = MODELS.get(provider, {}).get(model, model)
                PLANNER_CONFIG[provider]["model"] = resolved
                print(f"模型覆盖: {model} -> {resolved}")
            client = get_planner_client()
            print(f"API 客户端初始化成功：{client.get_model_info()}")
        except Exception as e:
            print(f"API 客户端初始化失败：{e}")
            return

        # RAG 服务初始化
        rag_service = None
        if use_rag and RAG_AVAILABLE:
            if rag_log_dir is None:
                rag_log_dir = str(Path(output_file).parent / "rag_logs")

            print("\n正在初始化 RAG 服务...")
            try:
                rag_service = create_rag_service(
                    enabled=True,
                    chunk_size=rag_chunk_size,
                    top_k=rag_top_k,
                    min_prompt_length=rag_min_prompt_length,
                    chunker_type=rag_chunker_type,
                    cache_dir=str(Path(output_file).parent / "rag_cache"),
                    log_dir=rag_log_dir  # 传递日志目录
                )
                print(f"RAG 服务初始化成功")
                print(f"  - 切片器: {rag_chunker_type}, 切片大小: {rag_chunk_size}")
                print(f"  - 检索数量: {rag_top_k}, 最小 prompt 长度: {rag_min_prompt_length}")
                print(f"  - 日志目录: {rag_log_dir}")
            except Exception as e:
                print(f"RAG 服务初始化失败：{e}")
                rag_service = None
        elif use_rag and not RAG_AVAILABLE:
            print("\n[警告] RAG 模块不可用，跳过 RAG 初始化")

        print(f"\n读取输入文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]

        total_count = len(data)
        print(f"共 {total_count} 条指令")

        # 应用 start 和 limit 参数
        if start > 0:
            if start >= total_count:
                print(f"错误：start 索引 {start} 超出数据范围（总数：{total_count}）")
                return
            data = data[start:]
            print(f"从第 {start + 1} 条开始处理（跳过前 {start} 条）")

        if limit:
            data = data[:limit]
            print(f"限制处理 {limit} 条")

        print(f"实际处理：{len(data)} 条（索引 {start} 到 {start + len(data) - 1}）")

        # 断点续传
        processed_prompts = set()
        if resume and os.path.exists(output_file):
            print(f"检测到输出文件，启用断点续传...")
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if 'prompt' in item:
                            processed_prompts.add(item['prompt'])
            print(f"已处理 {len(processed_prompts)} 条")

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(output_file).parent / f"logs_{evaluator_type}"
        log_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        fail_count = 0
        skip_count = 0
        total_stats = {
            'total_calls': 0,
            'total_time': 0.0,
            'llm_word_count_used': 0
        }

        output_file_obj = open(output_file, 'a', encoding='utf-8')

        # 使用 ascii=True 和 ncols 参数提高 Windows 兼容性
        progress_bar = tqdm(
            data,
            desc=f"生成大纲 (ISRP - {evaluator_type})",
            ascii=False,  # 使用 Unicode 字符
            ncols=100,    # 固定宽度，避免超出终端
            mininterval=1.0  # 最小刷新间隔（秒）
        )

        for idx, item in enumerate(progress_bar):
            # 检查是否请求退出
            if is_shutdown_requested():
                print(f"\n\n优雅退出：已处理 {success_count} 条，跳过 {skip_count} 条")
                break

            # 支持 'query' 作为 'prompt' 的别名
            prompt = item.get('prompt', '') or item.get('query', '')
            if not prompt:
                # 统计空 prompt 的数量
                total_stats['empty_prompt'] = total_stats.get('empty_prompt', 0) + 1
                continue

            if prompt in processed_prompts:
                skip_count += 1
                continue

            # 检测询问类指令
            if is_query_only_instruction(prompt):
                skip_count += 1
                total_stats['query_skipped'] = total_stats.get('query_skipped', 0) + 1
                if enable_logging:
                    progress_bar.write(f"  [跳过询问类指令] {prompt[:50]}...")
                continue

            # 提取字数
            target_words = extract_word_count(prompt, default=None)

            # 如果无法提取且启用了 LLM 预估
            if target_words is None:
                if use_llm_word_count:
                    # 检测语言
                    import re
                    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', prompt))
                    language = "zh" if chinese_chars / max(len(prompt), 1) > 0.3 else "en"

                    target_words = estimate_word_count_with_llm(client, prompt, language)
                    total_stats['llm_word_count_used'] += 1
                else:
                    target_words = 5000

            # 创建评估器
            try:
                if evaluator_type == "writingbench":
                    checklist = item.get('checklist', [])
                    if checklist:
                        evaluator = EvaluatorFactory.create(
                            evaluator_type="writingbench",
                            checklist=checklist
                        )
                    else:
                        # checklist 为空时回退到 ISRP 评估器
                        print(f"\n  [Warning] Empty checklist, falling back to ISRP evaluator")
                        evaluator = EvaluatorFactory.create(evaluator_type="isrp")
                else:
                    evaluator = EvaluatorFactory.create(evaluator_type="isrp")
            except Exception as e:
                print(f"\n  [评估器创建失败] {e}")
                # 创建失败时回退到 ISRP 评估器而不是 None
                try:
                    evaluator = EvaluatorFactory.create(evaluator_type="isrp")
                    print(f"  [Fallback] Using ISRP evaluator instead")
                except Exception as fallback_e:
                    print(f"  [Error] Failed to create fallback evaluator: {fallback_e}")
                    evaluator = None

            # 使用原始索引生成日志文件名
            original_idx = start + idx
            persist_file = log_dir / f"llm_calls_{timestamp}_{original_idx:04d}.jsonl"

            # 生成样本 ID
            sample_id = generate_sample_id(prompt, index=original_idx)

            # 创建规划器（传递评估器和 RAG 服务）
            planner = SyncISRPPlanner(
                client=client,
                max_depth=max_depth,
                max_iterations=max_iterations,
                enable_logging=enable_logging,
                persist_file=str(persist_file),
                evaluator=evaluator,
                rag_service=rag_service,  # RAG 服务
                max_width=max_width  # 宽度上限
            )

            start_time = time.time()

            try:
                # 传入 sample_id
                outline = planner.generate_plan(prompt, target_words, sample_id=sample_id)
                elapsed_time = time.time() - start_time
                stats = planner.get_stats()

                total_stats['total_calls'] += stats['total_calls']
                total_stats['total_time'] += stats['total_time']

                # 检查 RAG 是否实际启用
                rag_actually_used = rag_service is not None and rag_service.is_enabled() and rag_service.should_use_rag(prompt)

                result = {
                    "prompt": prompt,
                    "outline": outline,
                    "target_words": target_words,
                    "elapsed_time": elapsed_time,
                    "llm_calls": stats['total_calls'],
                    "llm_time": stats['total_time'],
                    "planner_version": "ISRP",
                    "evaluator_type": evaluator_type,
                    "sample_id": sample_id,
                    "rag_enabled": rag_actually_used
                }

                # 保留完整 checklist（含评分标准）
                if 'checklist' in item:
                    result['checklist'] = item['checklist']

                # 保留其他原始字段
                for key, value in item.items():
                    if key not in ['prompt', 'checklist']:
                        result[key] = value

                safe_write_line(output_file_obj, json.dumps(result, ensure_ascii=False) + '\n')
                success_count += 1

            except KeyboardInterrupt:
                # 用户按下 Ctrl+C，优雅退出
                print(f"\n\n收到中断信号，正在保存进度并退出...")
                print(f"已处理 {success_count} 条，失败 {fail_count} 条，跳过 {skip_count} 条")
                error_result = {
                    "prompt": prompt,
                    "error": "KeyboardInterrupt - 用户中断",
                    "elapsed_time": time.time() - start_time,
                    "planner_version": "ISRP",
                    "evaluator_type": evaluator_type
                }
                safe_write_line(output_file_obj, json.dumps(error_result, ensure_ascii=False) + '\n')
                output_file_obj.close()
                sys.exit(0)

            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"\n错误处理第 {original_idx + 1} 条 (索引 {original_idx}): {prompt[:50]}...")
                print(f"错误信息: {e}")
                fail_count += 1

                error_result = {
                    "prompt": prompt,
                    "error": str(e),
                    "elapsed_time": elapsed_time,
                    "planner_version": "ISRP",
                    "evaluator_type": evaluator_type
                }
                safe_write_line(output_file_obj, json.dumps(error_result, ensure_ascii=False) + '\n')

            finally:
                # 保存 RAG 日志并释放内存，但保留缓存以支持断点续传复用
                if rag_service and sample_id:
                    try:
                        # clear_cache=False: 保留磁盘缓存以支持断点续传复用
                        rag_service.clear_sample(sample_id, clear_cache=False)
                    except Exception as e:
                        if enable_logging:
                            progress_bar.write(f"  [Warning] RAG cleanup failed: {e}")

        output_file_obj.close()

        # 写入完成信号
        done_signal_file = output_file + ".done"
        with open(done_signal_file, 'w', encoding='utf-8') as f:
            f.write(f"completed_at: {datetime.now().isoformat()}\n")
            f.write(f"evaluator_type: {evaluator_type}\n")
            f.write(f"success: {success_count}\n")
            f.write(f"failed: {fail_count}\n")
            f.write(f"skipped: {skip_count}\n")
            f.write(f"llm_word_count_used: {total_stats['llm_word_count_used']}\n")

        print("\n" + "=" * 70)
        print("ISRP 测试完成！")
        print("=" * 70)
        print(f"  评估器类型: {evaluator_type}")
        print(f"  成功: {success_count}")
        print(f"  失败: {fail_count}")
        print(f"  跳过（已处理）: {skip_count}")
        # 显示空 prompt 和询问类跳过的统计
        if total_stats.get('empty_prompt', 0) > 0:
            print(f"  跳过（空 prompt/query）: {total_stats['empty_prompt']}")
        if total_stats.get('query_skipped', 0) > 0:
            print(f"  跳过（询问类指令）: {total_stats['query_skipped']}")
        print(f"  LLM 字数预估次数: {total_stats['llm_word_count_used']}")
        print(f"  总 LLM 调用次数: {total_stats['total_calls']}")
        print(f"  总 LLM 耗时: {total_stats['total_time']:.1f}s ({total_stats['total_time']/60:.1f}min)")
        print(f"  输出文件: {output_file}")
        print(f"  完成信号: {done_signal_file}")
        print("=" * 70)

    finally:
        # 关闭终端日志持久化
        if log_console:
            close_console_logging()


def main():
    parser = argparse.ArgumentParser(description="ISRP 全量测试（RAG 集成版）")
    parser.add_argument("--input", type=str, default="data/WritingBench-120.jsonl",
                       help="输入文件路径")
    parser.add_argument("--output", type=str, default="outputs/plan.jsonl",
                       help="输出文件路径")
    parser.add_argument("--output_suffix", type=str, default=None,
                       help="输出文件后缀（并发运行时自动生成唯一文件名，如 '_part1'）")
    parser.add_argument("--evaluator", type=str, default="isrp",
                       choices=["isrp", "writingbench", "auto"],
                       help="评估器类型 (默认 isrp)")
    parser.add_argument("--use_llm_word_count", action="store_true", default=True,
                       help="使用 LLM 预估字数（对于无明确字数的样本，默认启用）")
    parser.add_argument("--no_use_llm_word_count", action="store_true",
                       help="禁用 LLM 字数预估")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="启用断点续传（默认启用）")
    parser.add_argument("--no_resume", action="store_true",
                       help="禁用断点续传")
    parser.add_argument("--start", type=int, default=0,
                       help="开始索引（从0开始），默认从头开始")
    parser.add_argument("--limit", type=int, default=None,
                       help="处理条数限制，默认处理到末尾")
    parser.add_argument("--max_depth", type=int, default=MAX_DEPTH,
                       help="最大树深度 (默认使用 engines.py 中的配置)")
    parser.add_argument("--max_iterations", type=int, default=MAX_ITERATIONS,
                       help="串行迭代上限 (默认使用 engines.py 中的配置)")
    # 宽度上限参数
    parser.add_argument("--max-width", "--max_width", type=int, default=None,
                       help="DFS 扩展宽度上限（每层最大子节点数，默认使用 engines.py MAX_WIDTH）")
    parser.add_argument("--quiet", action="store_true",
                       help="禁用详细日志")

    # 终端输出持久化参数（默认启用）
    parser.add_argument("--log_console", action="store_true", default=True,
                       help="将终端输出持久化到日志文件（默认启用）")
    parser.add_argument("--no_log_console", action="store_true",
                       help="禁用终端输出持久化")
    parser.add_argument("--console_log_file", type=str, default=None,
                       help="指定终端日志文件路径（可选，默认自动生成）")

    # RAG 相关参数（默认启用）
    parser.add_argument("--use_rag", action="store_true", default=True,
                       help="启用 RAG 检索增强（默认启用）")
    parser.add_argument("--no_rag", action="store_true",
                       help="禁用 RAG 检索增强")
    parser.add_argument("--rag_chunk_size", type=int, default=512,
                       help="RAG 切片大小（默认512）")
    parser.add_argument("--rag_top_k", type=int, default=3,
                       help="RAG 检索返回数量（默认3）")
    parser.add_argument("--rag_min_prompt_length", type=int, default=2000,
                       help="触发 RAG 的最小 prompt 长度（默认2000）")
    parser.add_argument("--rag_chunker_type", type=str, default="recursive",
                       choices=["recursive", "semantic", "sentence", "token", "fast"],
                       help="切片器类型（默认recursive）")
    parser.add_argument("--rag_log_dir", type=str, default=None,
                       help="RAG 日志目录（默认在输出目录下创建 rag_logs）")
    parser.add_argument("--model", type=str, default=None,
                       help="覆盖规划器模型（如 kimi-k2.5, qwen3-max, deepseek-v4-pro 等）")

    args = parser.parse_args()

    # 处理输出文件后缀（用于并发运行）
    output_file = args.output
    if args.output_suffix:
        # 在文件扩展名前添加后缀
        base, ext = os.path.splitext(args.output)
        output_file = f"{base}{args.output_suffix}{ext}"
        print(f"[并发模式] 使用独立输出文件: {output_file}")

    run_full_test(
        input_file=args.input,
        output_file=output_file,
        evaluator_type=args.evaluator,
        use_llm_word_count=not args.no_use_llm_word_count,
        resume=not args.no_resume,
        start=args.start,
        limit=args.limit,
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        enable_logging=not args.quiet,
        log_console=not args.no_log_console,
        console_log_file=args.console_log_file,
        model=args.model,
        # RAG 参数
        use_rag=not args.no_rag,
        rag_chunk_size=args.rag_chunk_size,
        rag_top_k=args.rag_top_k,
        rag_min_prompt_length=args.rag_min_prompt_length,
        rag_chunker_type=args.rag_chunker_type,
        rag_log_dir=args.rag_log_dir,
        max_width=args.max_width
    )


if __name__ == "__main__":
    main()
