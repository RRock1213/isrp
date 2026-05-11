"""
AgentWrite E2E Plan: 端到端大纲生成（一次性生成完整大纲）

与 ISRP 的区别：
- ISRP: 深入实现大纲，通过迭代细化每个段落
- E2E: 端到端一次性生成完整大纲，更简单直接

功能：
- 读取用户指令文件
- 调用配置的大纲生成模型（Planner）
- 一次性生成完整大纲
- 保存到输出文件

使用方法：
    直接运行：python e2e/planner.py
    或命令行：python e2e/planner.py --input data/WritingBench-120.jsonl --output outputs/plan_e2e_wb.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from isrp.prompts import PromptTemplates

INPUT_FILE = "data/WritingBench-120.jsonl"
OUTPUT_FILE = "outputs/plan_e2e_wb.jsonl"
TEMPLATE_FILE = "prompts/plan.txt"

MAX_TOKENS = 64000
NUM_PROCESSES = 1
RESUME = True


def get_prompt_field(item: Dict[str, Any]) -> Optional[str]:
    """兼容获取 prompt 字段，尝试 'prompt' 和 'query' 两种字段名"""
    if 'prompt' in item:
        return item['prompt']
    elif 'query' in item:
        return item['query']
    return None


def load_template(template_path: str) -> str:
    """加载 Prompt 模板"""
    script_dir = Path(__file__).parent

    if not Path(template_path).is_absolute():
        template_path = script_dir / template_path

    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def _get_template(template_path: str) -> str:
    """优先使用内置模板，否则从文件加载"""
    default = (Path(__file__).parent / TEMPLATE_FILE).resolve()
    if not Path(template_path).is_absolute():
        resolved = (Path(__file__).parent / template_path).resolve()
    else:
        resolved = Path(template_path).resolve()
    if resolved == default:
        return PromptTemplates.PROMPT_E2E_PLAN
    return load_template(template_path)


def generate_plan(client, prompt: str, template: str, max_new_tokens: int = 4096) -> str:
    """
    生成大纲（端到端一次性生成）

    参数：
        client: UnifiedAPIClient 实例
        prompt: 用户指令
        template: Prompt 模板
        max_new_tokens: 最大生成 token 数

    返回：
        生成的完整大纲
    """
    full_prompt = template.replace('$INST$', prompt)

    response = client.call_api(
        prompt=full_prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.5
    )

    return response


def process_single_file(
    input_file: str,
    output_file: str,
    template_path: str,
    resume: bool = True,
    max_new_tokens: int = 4096,
    start: Optional[int] = None,
    limit: Optional[int] = None,
    model: str = None,
):
    """
    单文件处理模式

    参数：
        input_file: 输入文件路径
        output_file: 输出文件路径
        template_path: Prompt 模板路径
        resume: 是否从断点继续
        max_new_tokens: 最大生成 token 数
    """
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    if not Path(input_file).is_absolute():
        input_file = project_root / input_file
    if not Path(output_file).is_absolute():
        output_file = project_root / output_file
    
    from api_client import get_planner_client

    template = _get_template(template_path)

    # 模型覆盖
    if model:
        from config import PLANNER_CONFIG, MODELS
        provider = PLANNER_CONFIG["provider"]
        resolved = MODELS.get(provider, {}).get(model, model)
        PLANNER_CONFIG[provider]["model"] = resolved
        print(f"模型覆盖: {model} -> {resolved}")

    print("正在初始化 API 客户端...")
    client = get_planner_client()
    print(f"使用模型: {client.model}")
    print(f"API Provider: {client.provider}")

    print(f"\n读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"共 {len(data)} 条指令")

    # 分片处理：start / limit
    if start is not None or limit is not None:
        s = start or 0
        e = (s + limit) if limit is not None else len(data)
        data = data[s:e]
        print(f"分片处理: data[{s}:{e}] = {len(data)} 条")

    processed_prompts = set()
    if resume and os.path.exists(output_file):
        print(f"检测到输出文件，启用断点续传...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                p = get_prompt_field(item)
                if p:
                    processed_prompts.add(p)
        print(f"已处理 {len(processed_prompts)} 条")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    output_file_obj = open(output_file, 'a', encoding='utf-8')

    success_count = 0
    fail_count = 0

    for item in tqdm(data, desc="生成大纲"):
        prompt = get_prompt_field(item)

        if prompt is None:
            print(f"\n警告：数据项缺少 'prompt' 或 'query' 字段，跳过")
            continue

        if prompt in processed_prompts:
            continue

        try:
            plan = generate_plan(client, prompt, template, max_new_tokens)

            if not plan:
                print(f"\n警告：生成失败，跳过该条指令")
                fail_count += 1
                continue

            result = {
                "prompt": prompt,
                "plan": plan,
            }

            for key, value in item.items():
                if key not in ('prompt', 'query'):
                    result[key] = value

            output_file_obj.write(json.dumps(result, ensure_ascii=False) + '\n')
            output_file_obj.flush()

            success_count += 1

        except Exception as e:
            print(f"\n错误处理指令: {prompt[:50]}...")
            print(f"错误信息: {e}")
            fail_count += 1

    output_file_obj.close()

    print(f"\n处理完成！")
    print(f"  - 成功: {success_count}")
    print(f"  - 失败: {fail_count}")
    print(f"  - 输出文件: {output_file}")


def process_parallel(
    input_file: str,
    output_file: str,
    template_path: str,
    num_processes: int = 8,
    resume: bool = True,
    max_new_tokens: int = 4096
):
    """
    多进程并行处理模式

    参数：
        input_file: 输入文件路径
        output_file: 输出文件路径
        template_path: Prompt 模板路径
        num_processes: 进程数
        resume: 是否从断点继续
        max_new_tokens: 最大生成 token 数
    """
    import multiprocessing as mp
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    if not Path(input_file).is_absolute():
        input_file = project_root / input_file
    if not Path(output_file).is_absolute():
        output_file = project_root / output_file
    
    from api_client import create_client_from_config
    from config import PLANNER_CONFIG

    template = _get_template(template_path)

    provider = PLANNER_CONFIG["provider"]
    config = PLANNER_CONFIG[provider].copy()
    config["provider"] = provider

    print(f"\n读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"共 {len(data)} 条指令")

    processed_prompts = set()
    if resume and os.path.exists(output_file):
        print(f"检测到输出文件，启用断点续传...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                p = get_prompt_field(item)
                if p:
                    processed_prompts.add(p)
        print(f"已处理 {len(processed_prompts)} 条")

    pending_data = [item for item in data if get_prompt_field(item) not in processed_prompts]
    print(f"待处理: {len(pending_data)} 条")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    data_subsets = [pending_data[i::num_processes] for i in range(num_processes)]

    def worker(rank, subset):
        client = create_client_from_config("planner", config)

        with open(output_file, 'a', encoding='utf-8') as f:
            for item in tqdm(subset, desc=f"进程 {rank}"):
                try:
                    prompt = get_prompt_field(item)
                    if not prompt:
                        continue

                    full_prompt = template.replace('$INST$', prompt)
                    plan = client.call_api(prompt=full_prompt, max_new_tokens=max_new_tokens)

                    if not plan:
                        continue

                    result = {
                        "prompt": prompt,
                        "plan": plan,
                    }
                    for key, value in item.items():
                        if key not in ('prompt', 'query'):
                            result[key] = value

                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()

                except Exception as e:
                    print(f"\n进程 {rank} 错误: {e}")

    print(f"\n启动 {num_processes} 个进程...")
    processes = []
    for rank in range(num_processes):
        if len(data_subsets[rank]) > 0:
            p = mp.Process(target=worker, args=(rank, data_subsets[rank]))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    print(f"\n处理完成！输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="AgentWrite E2E Plan: 端到端大纲生成")
    parser.add_argument("--input", type=str, required=False,
                        default=INPUT_FILE,
                        help=f"输入的指令文件（默认: {INPUT_FILE}）")
    parser.add_argument("--output", type=str, required=False,
                        default=OUTPUT_FILE,
                        help=f"输出的大纲文件（默认: {OUTPUT_FILE}）")
    parser.add_argument("--template", type=str, required=False,
                        default=TEMPLATE_FILE,
                        help=f"Prompt 模板文件（默认: {TEMPLATE_FILE}）")
    parser.add_argument("--start", type=int, default=None,
                        help="起始索引（用于并发分片）")
    parser.add_argument("--limit", type=int, default=None,
                        help="处理条数上限（用于并发分片）")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS,
                        help=f"最大生成 token 数（默认: {MAX_TOKENS}）")
    parser.add_argument("--processes", type=int, default=NUM_PROCESSES,
                        help=f"并行进程数（默认: {NUM_PROCESSES}，使用多进程时建议 4-8）")
    parser.add_argument("--no_resume", action="store_true",
                        help="不使用断点续传（默认：启用）")
    parser.add_argument("--model", type=str, default=None,
                        help="覆盖规划器模型（如 kimi-k2.5, qwen3-max, deepseek-v4-pro 等）")

    args = parser.parse_args()

    if args.processes == 1:
        process_single_file(
            input_file=args.input,
            output_file=args.output,
            template_path=args.template,
            resume=not args.no_resume,
            max_new_tokens=args.max_tokens,
            start=args.start,
            limit=args.limit,
            model=args.model,
        )
    else:
        process_parallel(
            input_file=args.input,
            output_file=args.output,
            template_path=args.template,
            num_processes=args.processes,
            resume=not args.no_resume,
            max_new_tokens=args.max_tokens
        )


if __name__ == "__main__":
    main()
