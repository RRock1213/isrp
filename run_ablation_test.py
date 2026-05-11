"""
消融实验测试脚本（对应论文 Table 3）

使用方法：
    python run_ablation_test.py --ablation iteration_eval --input data/WritingBench-120.jsonl --limit 2
    python run_ablation_test.py --ablation global_memory --input data/WritingBench-120.jsonl --limit 2
    python run_ablation_test.py --ablation local_context --input data/WritingBench-120.jsonl --limit 2
    python run_ablation_test.py --ablation word_allocation --input data/WritingBench-120.jsonl --limit 2
    python run_ablation_test.py --ablation full --input data/WritingBench-120.jsonl --limit 2  # 对照组

消融参数说明（对应论文 Table 3）：
    iteration_eval:  w/o Iteration Evaluation — 禁用迭代评估（串行反思循环）
    global_memory:   w/o Global Memory Injection — 禁用全局记忆注入
    local_context:   w/o Local Context Injection — 禁用 RAG 局部上下文注入
    word_allocation: w/o Word Count Allocation — 使用均匀预分配代替 BFS 权重分配
    full:            Full ISRP — 完整功能（对照组）
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
    def __init__(self, log_file_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'a', encoding='utf-8')
        self.log_file_path = log_file_path

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


_tee_output = None


def setup_console_logging(log_file_path: str):
    global _tee_output
    _tee_output = TeeOutput(log_file_path)
    sys.stdout = _tee_output
    print(f"[Console Logging] Terminal output will be saved to: {log_file_path}")


def close_console_logging():
    global _tee_output
    if _tee_output:
        print(f"[Console Logging] Log saved to: {_tee_output.log_file_path}")
        _tee_output.close()
        sys.stdout = _tee_output.terminal
        _tee_output = None


# ============================================================================
# Windows 控制台 UTF-8 编码设置
# ============================================================================

if sys.platform == 'win32':
    try:
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        if hasattr(sys.stdout, 'buffer'):
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')
    except Exception:
        pass

# 跨平台文件锁支持
if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api_client import get_planner_client

# ============================================================================
# 【核心修改】根据消融类型选择导入模块
# ============================================================================

ABLATION_TYPE = None  # 全局消融类型，在 main() 中设置

def set_ablation_type(ablation_type: str):
    """设置消融类型，用于控制各消融开关"""
    global ABLATION_TYPE
    ABLATION_TYPE = ablation_type

def is_ablation_enabled(ablation_name: str) -> bool:
    """检查指定消融项是否启用"""
    global ABLATION_TYPE
    if ABLATION_TYPE == "full":
        return False  # 对照组不启用任何消融
    if ABLATION_TYPE == ablation_name:
        return True
    # "all" 类型启用所有消融（可选功能）
    if ABLATION_TYPE == "all":
        return True
    return False


def get_planner_module(ablation_type: str):
    """根据消融类型返回对应的规划器模块"""
    if ablation_type in ["iteration_eval", "global_memory", "local_context", "word_allocation"]:
        # 使用消融版本模块
        print(f"[消融实验] 使用 isrp_ablation 模块，消融类型: {ablation_type}")
        # 先导入消融配置模块并设置消融类型
        from isrp_ablation.ablation_config import set_ablation_config
        set_ablation_config(ablation_type)
        print(f"[消融实验] 已设置消融配置: {ablation_type}")
        # 然后导入其他模块
        from isrp_ablation import SyncISRPPlanner, MAX_ITERATIONS, MAX_DEPTH, LLMClient
        from isrp_ablation.engines import extract_word_count
        from isrp_ablation.evaluators import EvaluatorFactory, BaseEvaluator
        from isrp_ablation.utils import generate_sample_id
        return SyncISRPPlanner, MAX_ITERATIONS, MAX_DEPTH, LLMClient, extract_word_count, EvaluatorFactory, BaseEvaluator, generate_sample_id
    else:
        # 使用原版模块（包括 rag 消融、对照组 full）
        print(f"[消融实验] 使用原版 isrp 模块，消融类型: {ablation_type}")
        from isrp import SyncISRPPlanner, MAX_ITERATIONS, MAX_DEPTH, LLMClient
        from isrp.engines import extract_word_count
        from isrp.evaluators import EvaluatorFactory, BaseEvaluator
        from isrp.utils import generate_sample_id
        return SyncISRPPlanner, MAX_ITERATIONS, MAX_DEPTH, LLMClient, extract_word_count, EvaluatorFactory, BaseEvaluator, generate_sample_id


# ============================================================================
# 优雅退出支持
# ============================================================================

_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        print("\n\n强制退出...")
        sys.exit(1)
    _shutdown_requested = True
    print("\n\n收到中断信号，正在优雅退出...")
    print("（再次按 Ctrl+C 可强制退出）")

def is_shutdown_requested() -> bool:
    return _shutdown_requested

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================================
# 跨平台文件锁实现
# ============================================================================

def acquire_file_lock(file_obj, max_retries=3):
    import time
    for attempt in range(max_retries):
        try:
            if sys.platform == 'win32':
                msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            return True
        except (IOError, OSError):
            if attempt < max_retries - 1:
                time.sleep(0.1)
            else:
                return False
    return False

def release_file_lock(file_obj):
    try:
        if sys.platform == 'win32':
            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
    except (IOError, OSError):
        pass

def safe_write_line(file_obj, line: str, max_retries: int = 5):
    import time
    for attempt in range(max_retries):
        try:
            lock_acquired = acquire_file_lock(file_obj, max_retries=1)
            try:
                file_obj.write(line)
                file_obj.flush()
                return True
            finally:
                if lock_acquired:
                    release_file_lock(file_obj)
        except (IOError, OSError):
            if attempt < max_retries - 1:
                time.sleep(0.2 * (attempt + 1))
            else:
                print(f"  [Warning] 文件写入失败（重试 {max_retries} 次后）")
                return False
    return False


# ============================================================================
# LLM 字数预估
# ============================================================================

def estimate_word_count_with_llm(client, prompt: str, language: str = "zh") -> int:
    if language == "zh":
        estimation_prompt = f"""分析以下写作任务，预估一个合理的文章字数。

【写作指令】
{prompt}

请输出 JSON 格式：
{
  "task_type": "任务类型",
  "suggested_word_count": 建议字数（整数）,
  "reasoning": "预估理由"
}

注意：
- 小型任务：1500-3000字
- 中等任务：3000-5000字
- 复杂任务：5000-10000字

只输出 JSON，不要其他内容。"""
    else:
        estimation_prompt = f"""Analyze the following writing task and estimate a reasonable word count.

[Writing Instruction]
{prompt}

Please output in JSON format.

Output only JSON, no other content."""

    try:
        result = client.call_api(estimation_prompt, temperature=0.3)
        import re
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            word_count = data.get("suggested_word_count", 5000)
            if 1000 <= word_count <= 20000:
                return word_count
    except Exception as e:
        print(f"  [LLM 字数预估失败] {e}")

    return 5000


# ============================================================================
# 测试函数
# ============================================================================

def run_ablation_test(
    input_file: str,
    output_file: str,
    ablation_type: str = "full",
    evaluator_type: str = "isrp",
    use_llm_word_count: bool = True,
    resume: bool = True,
    start: int = 0,
    limit: int = None,
    max_depth: int = 2,
    max_iterations: int = 3,
    enable_logging: bool = True,
    log_console: bool = True,
    console_log_file: str = None,
    use_rag: bool = True,
    rag_chunk_size: int = 512,
    rag_top_k: int = 3,
    rag_min_prompt_length: int = 2000,
    rag_chunker_type: str = "recursive",
    rag_log_dir: str = None
):
    """
    运行消融实验测试
    """
    # 设置全局消融类型
    set_ablation_type(ablation_type)

    # 根据消融类型获取对应模块
    SyncISRPPlanner, MAX_ITERATIONS, MAX_DEPTH, LLMClient, extract_word_count, EvaluatorFactory, BaseEvaluator, generate_sample_id = get_planner_module(ablation_type)

    # 注册关闭检查器
    LLMClient.set_shutdown_checker(is_shutdown_requested)

    # 消融日志输出
    print(f"\n{'=' * 70}")
    print(f"消融实验: {ablation_type}")
    print(f"{'=' * 70}")

    ablation_messages = {
        "iteration_eval": "w/o Iteration Evaluation — 禁用迭代评估",
        "global_memory": "w/o Global Memory Injection — 禁用全局记忆注入",
        "local_context": "w/o Local Context Injection — 禁用 RAG 局部上下文",
        "word_allocation": "w/o Word Count Allocation — 使用均匀预分配",
        "full": "Full ISRP — 完整功能（对照组）"
    }
    print(f"消融内容: {ablation_messages.get(ablation_type, '未知')}")
    print(f"{'=' * 70}\n")

    # 【local_context 消融】自动禁用 RAG
    if ablation_type == "local_context":
        use_rag = False
        print("[消融-LocalContext] RAG 功能已禁用")

    # 从输出文件名提取标识符（去掉 .jsonl 后缀），用于 console log 和 llm_calls
    # 例如：output_file = "iteration_1-20.jsonl" → file_basename = "iteration_1-20"
    output_basename = os.path.splitext(os.path.basename(output_file))[0]

    # 设置输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 设置终端日志（使用输出文件标识符，避免并发冲突）
    if log_console:
        if not console_log_file:
            console_log_file = os.path.join(output_dir or "outputs", f"console_{output_basename}.log")
        setup_console_logging(console_log_file)

    # 创建 API 客户端
    client = get_planner_client()

    # 注意：WritingBench 评估器需要 checklist，在样本处理时动态创建
    # 不在全局创建评估器，而是在每个样本处理时根据 checklist 动态创建

    # 【RAG 服务】仅在非 rag 消融且启用 RAG 时创建
    rag_service = None
    if use_rag and ablation_type != "local_context":
        try:
            from rag_module.rag_service import create_rag_service
            rag_service = create_rag_service(
                enabled=True,
                chunk_size=rag_chunk_size,
                top_k=rag_top_k,
                min_prompt_length=rag_min_prompt_length,
                chunker_type=rag_chunker_type,
                log_dir=rag_log_dir or os.path.join(output_dir or "outputs", "rag_logs")
            )
            print(f"[RAG] 服务已初始化: chunk_size={rag_chunk_size}, top_k={rag_top_k}")
        except ImportError:
            print("[RAG] rag_module 不可用，跳过 RAG 服务")

    # 创建规划器（不传入评估器，在样本处理时动态设置）
    planner = SyncISRPPlanner(
        client,
        max_depth=max_depth,
        max_iterations=max_iterations,
        enable_logging=enable_logging,
        evaluator=None,  # 不在全局创建评估器
        rag_service=rag_service
    )

    # LLM 调用持久化（使用输出文件标识符，避免并发冲突）
    llm_calls_file = None
    if output_dir:
        llm_calls_file = os.path.join(output_dir, f"llm_calls_{output_basename}.jsonl")

    if llm_calls_file:
        planner.llm.persist_file = llm_calls_file

    # 加载输入数据
    print(f"\n加载输入文件: {input_file}")
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    total_samples = len(samples)
    print(f"总样本数: {total_samples}")

    # 加载已处理样本（断点续传）
    processed_ids = set()
    if resume and os.path.exists(output_file):
        print(f"检测到已有输出文件，正在加载已处理样本...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'sample_id' in data:
                        processed_ids.add(data['sample_id'])
        print(f"已处理样本数: {len(processed_ids)}")

    # 处理范围
    if limit is None:
        end_idx = total_samples
    else:
        end_idx = min(start + limit, total_samples)

    samples_to_process = samples[start:end_idx]
    print(f"待处理样本: {len(samples_to_process)} (索引 {start}-{end_idx-1})")

    # 统计信息
    success_count = 0
    skip_count = 0
    fail_count = 0
    start_time = time.time()

    # 打开输出文件
    output_f = open(output_file, 'a', encoding='utf-8')

    # 处理样本
    progress_bar = tqdm(
        samples_to_process,
        desc=f"Planning ({ablation_type})",
        ascii=False,
        ncols=100,
        mininterval=1.0
    )

    def refresh_progress() -> None:
        progress_bar.set_postfix(
            ok=success_count,
            fail=fail_count,
            skip=skip_count,
            refresh=False
        )

    for idx, sample in enumerate(progress_bar):
        original_idx = start + idx
        refresh_progress()
        if is_shutdown_requested():
            progress_bar.write("\nShutdown requested, saving progress and exiting...")
            break

        # 获取 prompt（支持 prompt 或 query 字段）
        prompt = sample.get('prompt', sample.get('query', sample.get('instruction', '')))
        if not prompt:
            progress_bar.write(f"  [Warning] sample index {original_idx} has no prompt/query, skipped")
            skip_count += 1
            refresh_progress()
            continue

        # 获取样本 ID
        sample_id = sample.get('index', sample.get('sample_id', idx))
        sample_id_str = f"sample_{sample_id}" if isinstance(sample_id, int) else sample_id

        # 检查是否已处理
        if sample_id_str in processed_ids:
            progress_bar.write(f"  [Resume] {sample_id_str} already processed, skipped")
            skip_count += 1
            refresh_progress()
            continue

        # 获取目标字数
        target_words = sample.get('word_count', sample.get('target_words', 5000))
        if isinstance(target_words, str):
            target_words = extract_word_count(target_words) or 5000

        # 使用 LLM 预估字数（可选）
        if use_llm_word_count and target_words <= 0:
            language = "zh" if any(c in prompt for c in "写撰论文报告") else "en"
            target_words = estimate_word_count_with_llm(client, prompt, language)
            progress_bar.write(f"  [LLM word count] {sample_id_str}: {target_words}")

        # 确保字数在合理范围
        target_words = max(1000, min(target_words, 20000))

        # 【关键修改】动态创建 WritingBench 评估器
        checklist = sample.get('checklist')
        if checklist:
            evaluator = EvaluatorFactory.create('writingbench', checklist=checklist)
            planner.planner.evaluator = evaluator  # 设置到内部 planner
        else:
            # 如果没有 checklist，使用 ISRP 评估器
            evaluator = EvaluatorFactory.create('isrp')
            planner.planner.evaluator = evaluator

        try:
            # 生成大纲
            outline = planner.generate_plan(
                prompt,
                total_words=target_words,
                sample_id=sample_id_str
            )

            if outline:
                # 构建输出数据
                output_data = {
                    'sample_id': sample_id_str,
                    'prompt': prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    'target_words': target_words,
                    'outline': outline,
                    'ablation_type': ablation_type,
                    'evaluator_type': evaluator_type,
                    'rag_enabled': use_rag and ablation_type != "local_context"
                }

                # WritingBench 格式：保留 checklist
                if 'checklist' in sample:
                    output_data['checklist'] = sample['checklist']

                # 写入输出
                safe_write_line(output_f, json.dumps(output_data, ensure_ascii=False) + '\n')
                success_count += 1
                refresh_progress()

        except Exception as e:
            progress_bar.write(f"  [Error] {sample_id_str} failed: {e}")
            fail_count += 1
            error_data = {
                'sample_id': sample_id_str,
                'prompt': prompt[:500] + "..." if len(prompt) > 500 else prompt,
                'target_words': target_words,
                'error': str(e),
                'ablation_type': ablation_type,
                'evaluator_type': evaluator_type,
                'rag_enabled': use_rag and ablation_type != "local_context"
            }
            if 'checklist' in sample:
                error_data['checklist'] = sample['checklist']
            safe_write_line(output_f, json.dumps(error_data, ensure_ascii=False) + '\n')
            refresh_progress()

    # 关闭输出文件
    output_f.close()

    # 输出统计
    elapsed_time = time.time() - start_time
    done_signal_file = output_file + ".done"
    with open(done_signal_file, 'w', encoding='utf-8') as f:
        f.write(f"completed_at: {datetime.now().isoformat()}\n")
        f.write(f"ablation_type: {ablation_type}\n")
        f.write(f"evaluator_type: {evaluator_type}\n")
        f.write(f"success: {success_count}\n")
        f.write(f"failed: {fail_count}\n")
        f.write(f"skipped: {skip_count}\n")
        f.write(f"elapsed_seconds: {elapsed_time:.2f}\n")

    print(f"\n{'=' * 70}")
    print(f"消融实验完成: {ablation_type}")
    print(f"{'=' * 70}")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print(f"耗时: {elapsed_time:.2f} 秒")
    print(f"输出: {output_file}")
    print(f"Done signal: {done_signal_file}")
    print(f"{'=' * 70}")

    # 关闭终端日志
    if log_console:
        close_console_logging()


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="消融实验测试脚本")

    # 消融参数（核心）
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["iteration_eval", "global_memory", "local_context", "word_allocation", "full"],
                        help="消融类型（默认 full=对照组）")

    # 基础参数
    parser.add_argument("--input", type=str, default="data/WritingBench-120.jsonl",
                        help="输入文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径（默认自动生成）")
    parser.add_argument("--evaluator", type=str, default="writingbench",
                        choices=["isrp", "writingbench"],
                        help="评估器类型")
    parser.add_argument("--limit", type=int, default=None,
                        help="处理条数限制")
    parser.add_argument("--start", type=int, default=0,
                        help="开始索引")
    parser.add_argument("--no_resume", action="store_true",
                        help="禁用断点续传")

    # 日志参数
    parser.add_argument("--log_console", action="store_true", default=True,
                        help="终端日志持久化（默认启用）")
    parser.add_argument("--no_log_console", action="store_true",
                        help="禁用终端日志")

    # RAG 参数
    parser.add_argument("--use_rag", action="store_true", default=True,
                        help="启用 RAG（默认启用）")
    parser.add_argument("--no_rag", action="store_true",
                        help="禁用 RAG")

    args = parser.parse_args()

    # 自动生成输出路径（用户未指定时使用默认路径）
    if args.output is None:
        output_dir = f"outputs/ablation/no_{args.ablation}" if args.ablation != "full" else "outputs/ablation/full_baseline"
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"plan_{args.ablation}.jsonl")

    # 运行消融测试
    run_ablation_test(
        input_file=args.input,
        output_file=args.output,
        ablation_type=args.ablation,
        evaluator_type=args.evaluator,
        limit=args.limit,
        start=args.start,
        resume=not args.no_resume,
        log_console=not args.no_log_console,
        use_rag=not args.no_rag
    )


if __name__ == "__main__":
    main()
