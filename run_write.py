"""
AgentWrite — 流式长文生成（支持热更新大纲文件）

功能：
- 先处理已存在的大纲条目（库存）
- 持续监听大纲文件的新增内容（热更新）
- 增量读取，避免重复处理
- 智能终止机制（超时/信号文件）
- 兼容 plan 和 outline 两种字段名

使用方法：
    python run_write.py --plan outputs/plan.jsonl --output outputs/write.jsonl

    配合 run_plan.py 同时运行：
    终端1: python run_plan.py
    终端2: python run_write.py --plan outputs/plan.jsonl

终止条件：
1. 检测到 .done 信号文件（run_plan.py 完成后创建）
2. 超过指定时间没有新内容（--timeout 参数）
3. 用户手动 Ctrl+C

依赖：
    pip install requests tqdm
"""

import argparse
import json
import os
import re
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Any
from tqdm import tqdm

# ============================================================================
# 配置区域
# ============================================================================

from config import PLANNER_MODEL as _PLANNER_MODEL, MODELS, PLANNER_PROVIDER
MODEL_NAME = MODELS.get(PLANNER_PROVIDER, {}).get(_PLANNER_MODEL, _PLANNER_MODEL)
PLAN_FILE = "outputs/plan.jsonl"
OUTPUT_FILE = f"outputs/write_{MODEL_NAME}_stream.jsonl"
CACHE_FILE = f"outputs/write_cache_{MODEL_NAME}_stream.jsonl"
# 模板已内置在 isrp/prompts.py (PromptTemplates.PROMPT_WRITE_ZH/EN)，不再从文件加载
TEMPLATE_FILE = "prompts/write.txt"  # deprecated: kept for --template CLI compatibility
MIN_OUTPUT_LENGTH = 100
CLEAN_PREFIX = True
POLL_INTERVAL = 2.0
TIMEOUT = 600

# ============================================================================
# 以下为代码实现
# ============================================================================


def detect_language(text: str) -> str:
    """
    检测文本语言（中文/英文）
    
    Args:
        text: 输入文本
        
    Returns:
        "zh" 或 "en"
    """
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(re.sub(r'\s', '', text))
    
    if total_chars == 0:
        return "zh"
    
    chinese_ratio = chinese_chars / total_chars
    
    if chinese_ratio > 0.3:
        return "zh"
    else:
        return "en"


def load_template(template_path: str = None, language: str = None) -> str:
    """
    加载 Prompt 模板（支持多语言）

    Args:
        template_path: 模板文件路径（已弃用，仅保留兼容性）
        language: 语言代码 ("zh" 或 "en")，None 则默认返回英文模板

    Returns:
        模板内容

    Note:
        现在从 isrp/prompts.py 读取模板（PromptTemplates.PROMPT_WRITE_ZH/EN）
    """
    script_dir = Path(__file__).parent

    # 优先从 isrp/prompts.py 读取 (write.json 已废弃)
    import json as _json
    try:
        from isrp.prompts import PromptTemplates
        pt = PromptTemplates()
        if language and language.startswith("zh"):
            return pt.PROMPT_WRITE_ZH
        return pt.PROMPT_WRITE_EN
    except (ImportError, AttributeError):
        pass
    # 兼容性回退
    json_template_path = script_dir / "prompts" / "write.json"
    if json_template_path.exists():
        with open(json_template_path, 'r', encoding='utf-8') as f:
            templates = _json.load(f)
        if language and language in templates:
            return templates[language]
        if "en" in templates:
            return templates["en"]
    raise FileNotFoundError("Write template not found")


def parse_plan(plan: str) -> list:
    """解析大纲为步骤列表"""
    plan = plan.strip().replace('\n\n', '\n')
    steps = [line.strip() for line in plan.split('\n') if line.strip()]
    return steps


def extract_section_number(step: str) -> str:
    """
    从步骤中提取章节编号

    格式示例：
    [Paragraph 1 - Main Point : 【学术论文...】 - 【第1章...】 - 【1.1 ...】 - 【1.1.1 标题】 - 描述...]

    Returns:
        章节编号字符串，如 "1.1.1", "2.1.2"，未找到则返回空字符串
    """
    import re

    # 匹配最后一个【x.x.x】格式的编号
    # 注意：要匹配最具体的编号（最深层的）
    matches = re.findall(r'【(\d+(?:\.\d+)+)\s+[^】]+】', step)

    if matches:
        # 返回最后一个匹配（最具体的编号）
        return matches[-1]

    return ""


def extract_section_title(step: str) -> str:
    """
    从步骤中提取章节标题

    Returns:
        章节标题字符串，未找到则返回空字符串
    """
    import re

    # 匹配最后一个【x.x.x 标题】格式的标题
    match = re.search(r'【\d+(?:\.\d+)+\s+([^】]+)】\s*$', step)
    if match:
        return match.group(1).strip()

    # 备选：匹配最后一个编号节点的标题
    matches = re.findall(r'【\d+(?:\.\d+)+\s+([^】]+)】', step)
    if matches:
        return matches[-1].strip()

    return ""


def extract_chapter_info(step: str) -> tuple:
    """
    从步骤中提取章级信息

    格式示例：【第1章：引言：花呗接入央行征信的背景...】

    Args:
        step: 步骤字符串

    Returns:
        (章节号, 章节标题) 或 (None, None)
    """
    import re

    # 匹配"第N章：标题"格式
    match = re.search(r'【第(\d+)章[：:]\s*([^：:】]+)', step)
    if match:
        return match.group(1), match.group(2).strip()

    return None, None


def insert_chapter_headers(paragraphs: list, steps: list, global_abstract: str = "", is_paper: bool = False) -> list:
    """
    在段落前插入章节标题

    当检测到新章节时，插入"### 第N章 标题"
    如果是论文类文章且提供了全局摘要，在开头插入摘要部分

    Args:
        paragraphs: 段落列表
        steps: 对应的步骤列表
        global_abstract: 全局摘要（可选）
        is_paper: 是否为论文类文章

    Returns:
        包含章节标题的新段落列表
    """
    result = []

    # 仅在论文类文章时插入摘要
    if is_paper and global_abstract:
        result.append("### 摘要")
        result.append(global_abstract)

    current_chapter = None

    for para, step in zip(paragraphs, steps):
        chapter_num, chapter_title = extract_chapter_info(step)

        if chapter_num and chapter_num != current_chapter:
            # 新章节开始，插入标题
            result.append(f"### 第{chapter_num}章 {chapter_title}")
            current_chapter = chapter_num

        result.append(para)

    return result


def clean_paragraph(paragraph: str, remove_prefix: bool = True) -> str:
    """清理段落内容"""
    if not paragraph:
        return ""
    paragraph = re.sub(r'\n{3,}', '\n\n', paragraph)
    if remove_prefix:
        patterns = [
            r'^Paragraph\s+\d+\s*[:\-]?\s*',
            r'^第\s*\d+\s*段\s*[:：]?\s*',
        ]
        for pattern in patterns:
            paragraph = re.sub(pattern, '', paragraph, count=1, flags=re.IGNORECASE)
    return paragraph.strip()


def fix_section_numbering(paragraph: str, expected_number: str, expected_title: str = "") -> str:
    """
    修正段落中的章节编号

    如果段落开头的编号与期望不符，替换为正确编号

    Args:
        paragraph: 原始段落
        expected_number: 期望的章节编号（如 "2.1.1"）
        expected_title: 期望的章节标题

    Returns:
        修正后的段落
    """
    if not expected_number:
        return paragraph

    # 匹配段落开头的编号格式
    # 支持格式: "1.1 标题", "### 1.1 标题", "**1.1 标题**", "*1.1 标题*"
    match = re.match(r'^(\*{1,2}|#{1,3}\s*)?(\d+(?:\.\d+)*)\s+(.*)$', paragraph, re.DOTALL)

    if match:
        prefix = match.group(1) or ""
        current_number = match.group(2)
        rest = match.group(3)

        # 处理加粗格式：移除末尾的 *
        if prefix.startswith("**"):
            rest = re.sub(r'\*{1,2}$', '', rest.strip())
            prefix = ""  # 不保留 ** 包裹

        if current_number != expected_number:
            # 编号不正确，替换为正确编号
            if expected_title:
                # 如果 rest 以标题开头，替换它
                title_match = re.match(r'^([^\n]+)', rest)
                if title_match and len(title_match.group(1)) < 100:
                    # 可能是标题，替换为正确的标题
                    rest = expected_title + rest[title_match.end():]
            return f"{prefix}{expected_number} {rest}"
        else:
            # 编号正确，返回格式化后的内容（移除 ** 包裹）
            return f"{expected_number} {rest}"

    return paragraph


def load_cache(cache_file: str) -> dict:
    """加载缓存"""
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    prompt = item.get('prompt')
                    step = item.get('step')
                    response = item.get('response')
                    if prompt and step and response:
                        if prompt not in cache:
                            cache[prompt] = {}
                        cache[prompt][step] = response
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    # 指定具体异常类型：JSON解析错误、键错误、类型错误
                    continue
    return cache


def load_processed_prompts(output_file: str) -> set:
    """加载已处理的 prompt"""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if 'prompt' in item:
                        processed.add(item['prompt'])
                except (json.JSONDecodeError, KeyError) as e:
                    # 指定具体异常类型：JSON解析错误、键错误
                    continue
    return processed


def count_lines(file_path: str) -> int:
    """计算文件行数"""
    if not os.path.exists(file_path):
        return 0
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def read_line_at(file_path: str, line_index: int) -> dict:
    """读取指定行的数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == line_index:
                return json.loads(line)
    return None


def generate_paragraph(
    client,
    instruction: str,
    plan: str,
    already_written: str,
    current_step: str,
    template: str,
    current_index: int = 1,
    total_steps: int = 1,
    section_number: str = "",
    section_title: str = ""
) -> str:
    """生成单个段落

    Args:
        section_number: 当前章节编号（如 "1.1.1", "2.1.2"）
        section_title: 当前章节标题
    """
    full_prompt = template \
        .replace('$INST$', instruction) \
        .replace('$PLAN$', plan.strip()) \
        .replace('$TEXT$', already_written.strip()) \
        .replace('$STEP$', current_step.strip()) \
        .replace('$CURRENT_INDEX$', str(current_index)) \
        .replace('$TOTAL_STEPS$', str(total_steps))

    # 如果有章节编号，添加到 prompt 中
    if section_number:
        section_info = f"\n\n【当前章节编号】\n你正在编写的是第 {section_number} 节"
        if section_title:
            section_info += f"，标题为：{section_title}"
        section_info += "\n请在输出时使用这个编号（如 \"### {section_number} {section_title}\"），不要使用其他编号格式。"
        full_prompt += section_info

    response = client.call_api(
        prompt=full_prompt,

        temperature=0.5
    )
    return response


def get_plan_from_item(item: dict) -> str:
    """
    从数据项中提取大纲内容
    兼容 'plan' 和 'outline' 两种字段名
    """
    if 'plan' in item and item['plan']:
        return item['plan']
    if 'outline' in item and item['outline']:
        return item['outline']
    return ""


def process_item(
    item: dict,
    client,
    template: str,
    cache: dict,
    cache_file_obj,
    clean_prefix: bool,
    min_output_length: int,
    model_name: str
) -> dict:
    """
    处理单个大纲条目，生成完整文章

    返回：
        成功返回结果字典，失败返回 None
    """
    prompt = item.get('prompt', '')
    plan = get_plan_from_item(item)

    if not plan:
        print(f"\n警告：未找到大纲内容，跳过该条指令")
        return None

    steps = parse_plan(plan)
    if len(steps) > 50:
        print(f"\n警告：步数过多 ({len(steps)})，跳过该条指令")
        return None

    total_steps = len(steps)
    paragraphs = []
    written_text = ""

    for idx, step in enumerate(steps, 1):
        current_index = idx  # 当前是第几步（从1开始）

        # 提取当前步骤的章节编号和标题
        section_number = extract_section_number(step)
        section_title = extract_section_title(step)

        if prompt in cache and step in cache[prompt]:
            paragraph = cache[prompt][step]
            print(f"\n缓存命中：{prompt} -> {step}")
        else:
            paragraph = generate_paragraph(
                client, prompt, plan, written_text, step, template,
                current_index=current_index,
                total_steps=total_steps,
                section_number=section_number,
                section_title=section_title
            )

            if not paragraph:
                print(f"\n警告：段落生成失败，停止该指令")
                break

            cache_record = {
                "prompt": prompt,
                "step": step,
                "response": paragraph
            }
            cache_file_obj.write(json.dumps(cache_record, ensure_ascii=False) + '\n')
            cache_file_obj.flush()

            if prompt not in cache:
                cache[prompt] = {}
            cache[prompt][step] = paragraph

        paragraph = clean_paragraph(paragraph, remove_prefix=clean_prefix)

        # 修正章节编号
        if section_number:
            paragraph = fix_section_numbering(paragraph, section_number, section_title)

        if len(paragraph) < min_output_length:
            print(f"\n警告：段落太短 ({len(paragraph)} 字符)")

        paragraphs.append(paragraph)
        written_text += paragraph + '\n\n'

    if len(paragraphs) != len(steps):
        return None

    # 获取全局摘要和文体类型
    global_abstract = item.get('global_abstract', '') or item.get('abstract', '')
    is_paper = item.get('writing_type', '') == '学术论文' or '论文' in prompt

    # 插入章节标题
    paragraphs_with_headers = insert_chapter_headers(paragraphs, steps, global_abstract, is_paper)
    write_full_with_headers = '\n\n'.join(paragraphs_with_headers)

    result = {
        "prompt": prompt,
        "plan": plan,
        "write": paragraphs,
        "write_full": written_text.strip(),
        "write_with_headers": write_full_with_headers,  # 新增：带章节标题的完整文本
        "model": model_name,
        "num_paragraphs": len(paragraphs),
    }

    for key, value in item.items():
        if key not in ['prompt', 'plan', 'outline', 'write', 'write_full', 'model', 'num_paragraphs']:
            result[key] = value

    return result


def stream_process(
    plan_file: str,
    output_file: str,
    cache_file: str,
    template_path: str,
    model_name: str,
    clean_prefix: bool = True,
    min_output_length: int = 100,
    poll_interval: float = 2.0,
    timeout: float = 300.0,
    start: int = 0,
    limit: int = None
):
    """
    流式处理大纲文件

    流程：
    1. 先处理已存在的大纲（库存）
    2. 持续监听新增内容
    3. 智能终止

    Args:
        start: 起始索引（从0开始）
        limit: 处理条数上限，None 表示处理全部
    """
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from api_client import get_writer_client

    template = load_template(template_path)
    template_zh = load_template(template_path, language="zh")
    template_en = load_template(template_path, language="en")

    print(f"正在初始化 API 客户端 (模型: {model_name})...")
    client = get_writer_client(model_name)
    print(f"使用模型: {client.model}")
    print(f"API Provider: {client.provider}")

    cache = load_cache(cache_file)
    print(f"加载缓存: {len(cache)} 个指令, {sum(len(v) for v in cache.values())} 个步骤")

    processed_prompts = load_processed_prompts(output_file)
    print(f"已处理: {len(processed_prompts)} 条")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)

    output_file_obj = open(output_file, 'a', encoding='utf-8')
    cache_file_obj = open(cache_file, 'a', encoding='utf-8')

    success_count = 0
    skip_count = 0
    fail_count = 0
    too_short_count = 0

    last_new_item_time = time.time()
    last_line_count = 0
    is_done = False

    print(f"\n{'='*60}")
    print(f"开始流式处理")
    print(f"  大纲文件: {plan_file}")
    print(f"  输出文件: {output_file}")
    print(f"  轮询间隔: {poll_interval}s")
    print(f"  超时时间: {timeout}s")
    if start > 0:
        print(f"  起始索引: {start}")
    if limit is not None:
        print(f"  处理上限: {limit} 条")
    print(f"{'='*60}\n")

    # 初始化起始位置（支持 start 参数）
    # 如果 start > 0，则 last_line_count 从 start 开始
    last_line_count = start if start > 0 else 0
    processed_in_this_run = 0  # 本次运行已处理数量（用于 limit）

    while True:
        done_signal_file = plan_file + ".done"
        if os.path.exists(done_signal_file):
            print(f"\n检测到完成信号文件: {done_signal_file}")
            is_done = True

        current_line_count = count_lines(plan_file)
        
        if current_line_count > last_line_count:
            new_items = current_line_count - last_line_count
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 检测到内容: {last_line_count} -> {current_line_count} 条 (+{new_items})")
            
            for line_idx in range(last_line_count, current_line_count):
                item = read_line_at(plan_file, line_idx)
                if item is None:
                    continue
                
                prompt = item.get('prompt', '')
                
                if prompt in processed_prompts:
                    skip_count += 1
                    continue
                
                detected_lang = detect_language(prompt)
                current_template = template_zh if detected_lang == "zh" else template_en
                print(f"\n处理第 {line_idx + 1} 条大纲... (语言: {detected_lang})")
                
                try:
                    result = process_item(
                        item, client, current_template, cache, cache_file_obj,
                        clean_prefix, min_output_length, model_name
                    )
                    
                    if result:
                        output_file_obj.write(json.dumps(result, ensure_ascii=False) + '\n')
                        output_file_obj.flush()
                        processed_prompts.add(prompt)
                        success_count += 1
                        processed_in_this_run += 1
                        print(f"✓ 完成 (共 {result['num_paragraphs']} 段)")
                    else:
                        fail_count += 1
                        print(f"✗ 失败")

                except Exception as e:
                    print(f"✗ 错误: {e}")
                    fail_count += 1

                # 【limit 检查】如果达到处理上限，提前退出
                if limit is not None and processed_in_this_run >= limit:
                    print(f"\n已达到处理上限 ({limit} 条)，停止处理")
                    is_done = True
                    break

            last_line_count = current_line_count
            last_new_item_time = time.time()
        
        if is_done:
            print(f"\n大纲生成已完成，等待当前处理结束...")
            break
        
        time_since_last = time.time() - last_new_item_time
        if time_since_last > timeout:
            print(f"\n超过 {timeout}s 没有新内容，停止监听")
            break
        
        if current_line_count == last_line_count:
            remaining = int(timeout - time_since_last)
            if remaining > 0 and remaining % 30 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 等待新内容... (剩余超时: {remaining}s)")
        
        time.sleep(poll_interval)

    output_file_obj.close()
    cache_file_obj.close()

    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"  - 成功: {success_count}")
    print(f"  - 跳过（已处理）: {skip_count}")
    print(f"  - 失败: {fail_count}")
    print(f"  - 输出文件: {output_file}")
    print(f"{'='*60}")


def process_single_file(
    input_file: str,
    output_file: str,
    cache_file: str,
    template_path: str,
    model_name: str,

    resume: bool = True,
    clean_prefix: bool = True,
    min_output_length: int = 100,
    start: int = 0,
    limit: int = None
):
    """
    单文件处理模式

    用于一次性处理完整的大纲文件，不进行流式监听

    Args:
        start: 起始索引（从0开始）
        limit: 处理条数上限，None 表示处理全部
    """
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from api_client import get_writer_client

    template = load_template(template_path)
    template_zh = load_template(template_path, language="zh")
    template_en = load_template(template_path, language="en")

    print(f"正在初始化 API 客户端 (模型: {model_name})...")
    client = get_writer_client(model_name)
    print(f"使用模型: {client.model}")
    print(f"API Provider: {client.provider}")

    print(f"\n读取大纲文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 应用 start 和 limit 切片
    total_count = len(data)
    if start > 0:
        print(f"起始索引: {start}")
    if limit is not None:
        print(f"处理上限: {limit} 条")
    data = data[start:start + limit] if limit else data[start:]
    print(f"实际处理: {len(data)} 条 (共 {total_count} 条)")

    cache = load_cache(cache_file)
    print(f"加载缓存: {len(cache)} 个指令, {sum(len(v) for v in cache.values())} 个步骤")

    processed_prompts = set[Any]()
    if resume and os.path.exists(output_file):
        print(f"检测到输出文件，启用断点续传...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if 'prompt' in item:
                    processed_prompts.add(item['prompt'])
        print(f"已处理 {len(processed_prompts)} 条")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)

    output_file_obj = open(output_file, 'a', encoding='utf-8')
    cache_file_obj = open(cache_file, 'a', encoding='utf-8')

    success_count = 0
    skip_count = 0
    fail_count = 0
    too_short_count = 0

    for item in tqdm(data, desc="生成长文"):
        prompt = item.get('prompt', '')
        
        if prompt in processed_prompts:
            skip_count += 1
            continue

        detected_lang = detect_language(prompt)
        current_template = template_zh if detected_lang == "zh" else template_en

        try:
            result = process_item(
                item, client, current_template, cache, cache_file_obj,
                clean_prefix, min_output_length, model_name
            )

            if result:
                output_file_obj.write(json.dumps(result, ensure_ascii=False) + '\n')
                output_file_obj.flush()
                processed_prompts.add(prompt)
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"\n错误处理指令: {prompt[:50]}...")
            print(f"错误信息: {e}")
            fail_count += 1

    output_file_obj.close()
    cache_file_obj.close()

    print(f"\n处理完成！")
    print(f"  - 成功: {success_count}")
    print(f"  - 跳过（已处理）: {skip_count}")
    print(f"  - 失败: {fail_count}")
    print(f"  - 输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="AgentWrite — 流式长文生成（支持热更新）")
    parser.add_argument("--plan", type=str, default=PLAN_FILE,
                        help=f"大纲文件路径（默认: {PLAN_FILE}）")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help=f"输出文件路径（默认: {OUTPUT_FILE}）")
    parser.add_argument("--cache", type=str, default=CACHE_FILE,
                        help=f"缓存文件路径（默认: {CACHE_FILE}）")
    parser.add_argument("--template", type=str, default=TEMPLATE_FILE,
                        help=f"Prompt 模板文件（默认: {TEMPLATE_FILE}）")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help=f"使用的模型名称（默认: {MODEL_NAME}）")
    parser.add_argument("--min_length", type=int, default=MIN_OUTPUT_LENGTH,
                        help=f"最小段落长度（默认: {MIN_OUTPUT_LENGTH}）")
    parser.add_argument("--keep_prefix", action="store_true",
                        help="保留段落前缀（默认：清理）")
    parser.add_argument("--poll_interval", type=float, default=POLL_INTERVAL,
                        help=f"轮询间隔秒数（默认: {POLL_INTERVAL}）")
    parser.add_argument("--timeout", type=float, default=TIMEOUT,
                        help=f"无新内容超时秒数（默认: {TIMEOUT}）")
    parser.add_argument("--no_stream", action="store_true",
                        help="禁用流式模式，一次性处理完整文件")
    parser.add_argument("--no_resume", action="store_true",
                        help="不使用断点续传（默认：启用，仅非流式模式有效）")
    parser.add_argument("--start", type=int, default=0,
                        help="开始索引（从0开始），默认从头开始")
    parser.add_argument("--limit", type=int, default=None,
                        help="处理条数限制，默认处理到末尾")
    parser.add_argument("--output_suffix", type=str, default=None,
                        help="输出文件后缀（并发运行时自动生成唯一文件名，如 '_part1'）")

    args = parser.parse_args()

    # 处理输出文件后缀（用于并发运行）
    output_file = args.output
    cache_file = args.cache
    if args.output_suffix:
        # 在文件扩展名前添加后缀
        base, ext = os.path.splitext(args.output)
        output_file = f"{base}{args.output_suffix}{ext}"

        cache_base, cache_ext = os.path.splitext(args.cache)
        cache_file = f"{cache_base}{args.output_suffix}{cache_ext}"

        print(f"[并发模式] 使用独立输出文件: {output_file}")
        print(f"[并发模式] 使用独立缓存文件: {cache_file}")

    if args.no_stream:
        process_single_file(
            input_file=args.plan,
            output_file=output_file,
            cache_file=cache_file,
            template_path=args.template,
            model_name=args.model,
            resume=not args.no_resume,
            clean_prefix=not args.keep_prefix,
            min_output_length=args.min_length,
            start=args.start,
            limit=args.limit
        )
    else:
        stream_process(
            plan_file=args.plan,
            output_file=output_file,
            cache_file=cache_file,
            template_path=args.template,
            model_name=args.model,

            clean_prefix=not args.keep_prefix,
            min_output_length=args.min_length,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            start=args.start,
            limit=args.limit
        )


if __name__ == "__main__":
    main()
