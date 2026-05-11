"""
评估生成质量 - 支持动态评估器

功能：
- 使用 LLM-as-a-Judge 评估质量
- 支持动态评估器（ISRP 6 维 / WritingBench checklist）
- 支持断点续传
- 实时持久化

使用方法：
    # WritingBench checklist 评估（自动从样本读取 checklist）
    python evaluation/eval_body.py --input outputs/write.jsonl --evaluator writingbench

    # 指定模型和提供商
    python evaluation/eval_body.py --input outputs/write.jsonl --model glm-5 --provider dashscope
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, List


# ============================================================================
# 实时输出辅助函数
# ============================================================================

def _flush_stdout():
    """强制刷新 stdout，确保实时输出"""
    sys.stdout.flush()


def _print_and_flush(msg: str):
    """打印并立即刷新输出"""
    print(msg)
    sys.stdout.flush()

# ============================================================================
# 配置区域 - 在这里修改参数，然后直接运行脚本
# ============================================================================

# 输入输出配置
INPUT_FILE = "outputs/write.jsonl"
OUTPUT_FILE = "outputs/eval_quality.jsonl"

# 评估配置
EVALUATOR_TYPE = "writingbench"  # 可选: "isrp", "writingbench"
MODEL_NAME = "glm-5"
PROVIDER = "dashscope" 
START_INDEX = 0
MAX_SAMPLES = None
RESUME = True
RETRY_FAILED = True
MAX_RETRIES = 3

# 日志配置
LOG_PARSE_FAILURES = True
PARSE_FAILURE_LOG = "outputs/evaluation/parse_failures.jsonl"

# ============================================================================
# 以下为代码实现
# ============================================================================


def get_parse_failure_logger(log_file: str):
    """获取解析失败日志记录器"""
    global _parse_failure_logger
    if _parse_failure_logger is None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _parse_failure_logger = open(log_file, 'a', encoding='utf-8')
    return _parse_failure_logger


_parse_failure_logger = None


def log_parse_failure(response: str, prompt: str = None, sample_idx: int = None,
                      error_reason: str = None, attempt: int = None):
    """记录解析失败的响应"""
    if not LOG_PARSE_FAILURES:
        return

    try:
        logger = get_parse_failure_logger(PARSE_FAILURE_LOG)
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "sample_idx": sample_idx,
            "attempt": attempt,
            "error_reason": error_reason,
            "response_length": len(response) if response else 0,
            "response": response,
        }
        if prompt:
            log_entry["prompt"] = prompt

        logger.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        logger.flush()
    except Exception as e:
        _print_and_flush(f"警告：无法写入解析失败日志: {e}")


def close_parse_failure_logger():
    """关闭日志文件"""
    global _parse_failure_logger
    if _parse_failure_logger is not None:
        _parse_failure_logger.close()
        _parse_failure_logger = None


def detect_language(text: str) -> str:
    """检测文本语言"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    if chinese_chars / max(len(text), 1) > 0.3:
        return "zh"
    return "en"


def evaluate_sample(
    client,
    evaluator,
    user_prompt: str,
    generated_text: str,
    max_retries: int = 5,
    sample_idx: int = None
) -> Optional[Dict[str, Any]]:
    """
    评估单个样本

    Args:
        client: API 客户端
        evaluator: 评估器实例
        user_prompt: 用户原始指令
        generated_text: 生成的文本
        max_retries: 最大重试次数
        sample_idx: 样本索引

    Returns:
        评估结果字典，包含 scores, average_score, Analysis
    """
    # 检测语言
    language = detect_language(user_prompt + generated_text)

    # 生成评估提示词
    eval_prompt = evaluator.get_final_eval_prompt(
        user_prompt=user_prompt,
        generated_text=generated_text,
        language=language
    )

    # 多次尝试
    for attempt in range(max_retries):
        try:
            # 调用 API
            response = client.call_api(
                prompt=eval_prompt,
                temperature=0.1,
                max_new_tokens=65536
            )

            if not response:
                _print_and_flush(f"  尝试 {attempt + 1}/{max_retries}: API 返回空响应")
                log_parse_failure(
                    response=None,
                    prompt=user_prompt,
                    sample_idx=sample_idx,
                    error_reason="API 返回空响应",
                    attempt=attempt + 1
                )
                continue

            # 解析结果
            result = evaluator.parse_final_result(response)

            if not result.get("scores"):
                _print_and_flush(f"  尝试 {attempt + 1}/{max_retries}: 无法解析分数")
                _print_and_flush(f"  响应内容: {response[:200]}...")
                log_parse_failure(
                    response=response,
                    prompt=user_prompt,
                    sample_idx=sample_idx,
                    error_reason="无法解析分数",
                    attempt=attempt + 1
                )
                continue

            # 验证分数范围
            valid = True
            for dim, score in result["scores"].items():
                if not isinstance(score, (int, float)) or score < 1 or score > 10:
                    valid = False
                    _print_and_flush(f"  尝试 {attempt + 1}/{max_retries}: 无效分数 {dim}={score}")
                    log_parse_failure(
                        response=response,
                        prompt=user_prompt,
                        sample_idx=sample_idx,
                        error_reason=f"无效分数 {dim}={score}",
                        attempt=attempt + 1
                    )
                    break

            if valid:
                return result

        except Exception as e:
            _print_and_flush(f"评估失败 ({attempt + 1}/{max_retries}): {e}")
            log_parse_failure(
                response=None,
                prompt=user_prompt,
                sample_idx=sample_idx,
                error_reason=f"异常: {str(e)}",
                attempt=attempt + 1
            )
            import traceback
            traceback.print_exc()
            _flush_stdout()

    return None


def process_evaluation(
    input_file: str,
    output_file: str,
    evaluator_type: str = "writingbench",
    model_name: str = "glm-5",
    provider: str = "dashscope",
    start_index: int = 0,
    max_samples: int = None,
    resume: bool = True
):
    """
    执行评估任务

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        evaluator_type: 评估器类型 (isrp/writingbench)
        model_name: 评估模型名称
        provider: API 提供商
        start_index: 开始索引
        max_samples: 最大样本数
        resume: 是否断点续传
    """
    # 添加项目根目录到 Python 路径
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from api_client import get_evaluator_client
    from isrp.evaluators import EvaluatorFactory

    # 转换相对路径
    if not Path(input_file).is_absolute():
        input_file = str((project_root / input_file).resolve())
    if not Path(output_file).is_absolute():
        output_file = str((project_root / output_file).resolve())

    global PARSE_FAILURE_LOG
    if not Path(PARSE_FAILURE_LOG).is_absolute():
        PARSE_FAILURE_LOG = str((project_root / PARSE_FAILURE_LOG).resolve())

    # 创建 API 客户端
    _print_and_flush(f"正在初始化评估客户端 (模型: {model_name}, 提供商: {provider})...")
    client = get_evaluator_client(model_name, provider)
    _print_and_flush(f"使用模型: {client.model}")

    # 读取输入数据
    _print_and_flush(f"\n读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    total_count = len(data)
    _print_and_flush(f"共 {total_count} 条数据")

    # 应用索引参数
    if start_index > 0:
        if start_index >= total_count:
            _print_and_flush(f"错误：start_index {start_index} 超出数据范围")
            return
        data = data[start_index:]
        _print_and_flush(f"从第 {start_index + 1} 条开始处理")

    if max_samples and max_samples < len(data):
        data = data[:max_samples]
        _print_and_flush(f"限制处理 {max_samples} 条")

    _print_and_flush(f"实际处理：{len(data)} 条（索引 {start_index} 到 {start_index + len(data) - 1}）")

    # 创建输出目录
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # 断点续传：读取已处理的 prompt
    processed_prompts = set()
    if resume and os.path.exists(output_file):
        _print_and_flush(f"\n检测到输出文件，启用断点续传...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        p = item.get('prompt') or item.get('query')
                        if p:
                            processed_prompts.add(p)
                    except:
                        pass
        _print_and_flush(f"已处理 {len(processed_prompts)} 条")

    # 打开输出文件
    output_handle = open(output_file, 'a', encoding='utf-8')

    # 统计
    success_count = 0
    fail_count = 0
    skip_count = 0
    results = []

    # 处理数据
    for local_idx, item in enumerate(tqdm(data, desc="评估质量")):
        original_idx = start_index + local_idx
        prompt = item.get('prompt', '') or item.get('query', '')

        if not prompt:
            continue

        # 跳过已处理
        if prompt in processed_prompts:
            skip_count += 1
            continue

        # 获取生成文本
        generated_text = item.get('write_full', '')
        if not generated_text and 'write' in item:
            if isinstance(item['write'], list):
                generated_text = '\n\n'.join(item['write'])
            else:
                generated_text = item['write']

        # 检查是否被 SKIPPED（输入过长导致模型跳过）
        is_skipped = False
        skip_reason = ""
        if generated_text and '[SKIPPED]' in generated_text:
            is_skipped = True
            # 提取跳过原因
            skip_match = re.search(r'\[SKIPPED\]\s*([^\n]+)', generated_text)
            if skip_match:
                skip_reason = skip_match.group(1).strip()
            else:
                skip_reason = "输入超过模型限制"
        elif not generated_text:
            is_skipped = True
            skip_reason = "生成文本为空"

        if is_skipped:
            _print_and_flush(f"[{original_idx}] ⚠ SKIPPED: {skip_reason}")
            skip_count += 1
            # 记录跳过的样本
            skip_item = {
                "prompt": prompt,
                "scores": {},
                "average_score": 0,
                "Analysis": f"[SKIPPED] {skip_reason}",
                "evaluator": "SKIPPED",
                "dimensions": [],
                "skip_reason": skip_reason
            }
            # 保留原始字段
            for key in ['checklist', 'target_words', 'sample_id', 'index']:
                if key in item:
                    skip_item[key] = item[key]
            output_handle.write(json.dumps(skip_item, ensure_ascii=False) + '\n')
            output_handle.flush()
            continue

        try:
            # 创建评估器
            sample_evaluator = EvaluatorFactory.create_from_sample(
                sample=item,
                evaluator_type=evaluator_type
            )

            dimensions = sample_evaluator.get_dimensions()
            _print_and_flush(f"\n[{original_idx}] 使用评估器: {sample_evaluator.__class__.__name__}")
            _print_and_flush(f"  评估维度: {dimensions}")

            # 评估
            result = evaluate_sample(
                client=client,
                evaluator=sample_evaluator,
                user_prompt=prompt,
                generated_text=generated_text,
                sample_idx=original_idx
            )

            if result is None:
                _print_and_flush(f"[{original_idx}] 评估失败")
                fail_count += 1
                continue

            # 保存结果
            output_item = {
                "prompt": prompt,
                "scores": result["scores"],
                "average_score": result.get("average_score", 0.0),
                "Analysis": result.get("Analysis", ""),
                "evaluator": sample_evaluator.__class__.__name__,
                "dimensions": dimensions
            }

            # 保留原始字段
            for key in ['checklist', 'target_words', 'planner_version']:
                if key in item:
                    output_item[key] = item[key]

            output_handle.write(json.dumps(output_item, ensure_ascii=False) + '\n')
            output_handle.flush()

            results.append(output_item)
            success_count += 1

            # 打印结果 - 实时输出
            scores_str = ', '.join([f"{k}:{v}" for k, v in result["scores"].items()])
            avg = result.get("average_score", 0.0)
            _print_and_flush(f"[{original_idx}] 成功 | 平均分: {avg:.1f} | 分数: {scores_str}")

        except Exception as e:
            _print_and_flush(f"[{original_idx}] 错误: {e}")
            import traceback
            traceback.print_exc()
            _flush_stdout()
            fail_count += 1

    output_handle.close()
    close_parse_failure_logger()

    # 打印统计
    _print_and_flush(f"\n处理完成！")
    _print_and_flush(f"  成功: {success_count}")
    _print_and_flush(f"  失败: {fail_count}")
    _print_and_flush(f"  跳过: {skip_count}")

    # 计算维度平均分
    if results:
        dimension_totals = {}
        dimension_counts = {}

        for result in results:
            for dim, score in result["scores"].items():
                dimension_totals[dim] = dimension_totals.get(dim, 0) + score
                dimension_counts[dim] = dimension_counts.get(dim, 0) + 1

        _print_and_flush("\n各维度平均分：")
        for dim in dimension_totals:
            avg = dimension_totals[dim] / dimension_counts[dim]
            _print_and_flush(f"  {dim}: {avg:.2f}")

        # 总体平均分
        total_avg = sum(r.get("average_score", 0) for r in results) / len(results)
        _print_and_flush(f"\n总体平均分: {total_avg:.2f}")

        # 写入统计摘要（单独文件，不混入 JSONL）
        summary = {
            "total_samples": len(results),
            "dimension_averages": {
                dim: dimension_totals[dim] / dimension_counts[dim]
                for dim in dimension_totals
            },
            "overall_average": total_avg
        }
        summary_path = str(output_path).replace('.jsonl', '_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="评估生成质量 - 支持动态评估器")
    parser.add_argument("--input", type=str, default=INPUT_FILE,
                        help=f"输入文件（默认: {INPUT_FILE}）")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help=f"输出文件（默认: {OUTPUT_FILE}）")
    parser.add_argument("--evaluator", type=str, default=EVALUATOR_TYPE,
                        choices=["isrp", "writingbench"],
                        help=f"评估器类型（默认: {EVALUATOR_TYPE}）")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help=f"评估模型（默认: {MODEL_NAME}）")
    parser.add_argument("--provider", type=str, default=PROVIDER,
                        help=f"API 提供商（默认: {PROVIDER}）")
    parser.add_argument("--start", type=int, default=START_INDEX,
                        help=f"开始索引（默认: {START_INDEX}）")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES,
                        help="处理条数限制")
    parser.add_argument("--no-resume", action="store_true",
                        help="禁用断点续传")

    args = parser.parse_args()

    process_evaluation(
        input_file=args.input,
        output_file=args.output,
        evaluator_type=args.evaluator,
        model_name=args.model,
        provider=args.provider,
        start_index=args.start,
        max_samples=args.max_samples,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()