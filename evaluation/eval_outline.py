#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评估大纲样本"""

import json
import os
import sys
import re
from pathlib import Path

# 添加项目根目录到路径
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))

from isrp.prompts import PromptTemplates

# 导入项目配置和API客户端
import config
from api_client import UnifiedAPIClient

_PROMPT_EVAL_OUTLINE = PromptTemplates().get_template("PROMPT_EVAL_OUTLINE", "zh")

def robust_parse_json(response):
    """鲁棒JSON解析"""
    if not response:
        return None

    try:
        return json.loads(response)
    except:
        pass

    text = response.strip()

    # 去除代码块
    if '```' in text:
        text = re.sub(r'^```(?:json)?\s*\n', '', text)
        text = re.sub(r'\n```$', '', text)
        text = text.strip()

    # 提取JSON
    try:
        start = text.find('{')
        if start >= 0:
            depth = 0
            end = start
            for i, c in enumerate(text[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end > start:
                return json.loads(text[start:end+1])
    except:
        pass

    # 提取分数 - 计算维度平均值作为 overall_score
    dims_match = re.search(r'dimensions["\s:]+\{[^}]+\}', text)
    if dims_match:
        try:
            dims_text = dims_match.group(0)
            # 尝试解析 dimensions
            dims_start = dims_text.find('{')
            if dims_start >= 0:
                dims_json = json.loads(dims_text[dims_start:])
                if dims_json:
                    # 计算六维度平均值作为 overall_score
                    dim_values = list(dims_json.values())
                    calc_score = sum(dim_values) / len(dim_values) if dim_values else 0
                    return {
                        "overall_score": round(calc_score, 2),
                        "dimensions": dims_json,
                        "strengths": [],
                        "weaknesses": ["JSON格式不完整"],
                        "suggestions": [],
                        "low_quality_descriptions": [],
                        "semantic_errors": []
                    }
        except:
            pass

    # 兼容旧格式：提取 overall_score
    score_match = re.search(r'overall_score["\s:]+(\d+\.?\d*)', text)
    if score_match:
        score = float(score_match.group(1))
        return {
            "overall_score": score,
            "dimensions": {
                "relevance": score,
                "specificity": max(score-1, 1),
                "guidance": max(score-1, 1),
                "language": score,
                "coverage": score,
                "semantic_correctness": score
            },
            "strengths": [],
            "weaknesses": ["JSON格式不完整"],
            "suggestions": [],
            "low_quality_descriptions": [],
            "semantic_errors": []
        }

    return None

def save_progress(results, output_path):
    """原子写入：先写临时文件，再替换，避免写入中断导致文件损坏"""
    tmp = str(output_path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(tmp, str(output_path))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='评估大纲质量')
    parser.add_argument('--input', type=str, default='outputs/plan.jsonl', help='输入文件路径')
    parser.add_argument('--output', type=str, default='outputs/eval_outline.jsonl', help='输出文件路径')
    parser.add_argument('--start', type=int, default=0, help='跳过起始样本的索引')
    parser.add_argument('--limit', type=int, default=0, help='限制处理的条数，0表示不限制')
    parser.add_argument('--resume', action='store_true', default=None,
                        help='断点续传：跳过已评估的样本')
    parser.add_argument('--no_resume', dest='resume', action='store_false',
                        help='禁用断点续传，重新评估全部样本')
    parser.set_defaults(resume=None)
    args = parser.parse_args()

    # 初始化API客户端
    api_config = {
        "provider": "dashscope",
        "api_key": config.API_KEYS["dashscope"],
        "base_url": config.PROVIDER_CONFIGS["dashscope"]["base_url"],
        "model": "glm-5",
        "max_tokens": 2048,
        "temperature": 0.1
    }
    llm = UnifiedAPIClient(api_config)

    # 读取大纲样本
    plan_file = Path(base_dir / args.input)
    if not plan_file.exists():
        print(f"文件不存在: {plan_file}")
        return

    outlines = []
    with open(plan_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                outlines.append(json.loads(line))
            except:
                pass

    print(f"大纲样本数: {len(outlines)}")

    # 断点续传：读取已有评估结果
    existing_evals = {}
    output_path = Path(base_dir / args.output)
    resume = args.resume if args.resume is not None else True
    if resume and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    dims = r.get('evaluation', {}).get('dimensions', {})
                    if any(v > 0 for v in dims.values()):
                        existing_evals[r['prompt']] = r
                except:
                    pass
        print(f"已有有效评估: {len(existing_evals)}")
    elif not resume:
        print("禁用断点续传，重新评估全部样本")

    results = list(existing_evals.values())
    need_eval = [o for o in outlines if o.get('prompt') not in existing_evals]
    if args.start > 0:
        need_eval = need_eval[args.start:]
        print(f"跳过前 {args.start} 个样本")
    if args.limit > 0:
        need_eval = need_eval[:args.limit]
        print(f"限制处理条数: {args.limit}")
    print(f"已评估: {len(results)}, 需要评估: {len(need_eval)}")

    if len(need_eval) == 0:
        print("所有样本已评估完成")
        return

    for i, sample in enumerate(need_eval):
        prompt = sample.get('prompt', '') or sample.get('query', '')
        outline = sample.get('outline', '') or sample.get('plan', '')

        if not outline:
            results.append({
                'prompt': prompt,
                'evaluation': {
                    "overall_score": 0,
                    "dimensions": {
                        "relevance": 0, "specificity": 0, "guidance": 0,
                        "language": 0, "coverage": 0, "semantic_correctness": 0
                    },
                    "strengths": [],
                    "weaknesses": ["大纲为空"],
                    "suggestions": [],
                    "low_quality_descriptions": [],
                    "semantic_errors": []
                }
            })
            continue

        print(f"  评估 {i+1}/{len(need_eval)}: {prompt[:40]}...")

        try:
            response = llm.call_api(_PROMPT_EVAL_OUTLINE.format(user_prompt=prompt, outline=outline))
            eval_result = robust_parse_json(response)

            if eval_result is None:
                eval_result = {
                    "overall_score": 0,
                    "dimensions": {
                        "relevance": 0, "specificity": 0, "guidance": 0,
                        "language": 0, "coverage": 0, "semantic_correctness": 0
                    },
                    "strengths": [],
                    "weaknesses": ["解析失败"],
                    "suggestions": [],
                    "low_quality_descriptions": [],
                    "semantic_errors": []
                }

            results.append({'prompt': prompt, 'evaluation': eval_result})

            # 每条即时保存进度（原子写入），支持断点续传
            save_progress(results, output_path)

        except Exception as e:
            print(f"    [错误] {e}")
            results.append({
                'prompt': prompt,
                'evaluation': {
                    "overall_score": 0,
                    "dimensions": {
                        "relevance": 0, "specificity": 0, "guidance": 0,
                        "language": 0, "coverage": 0, "semantic_correctness": 0
                    },
                    "strengths": [],
                    "weaknesses": [str(e)],
                    "suggestions": [],
                    "low_quality_descriptions": [],
                    "semantic_errors": []
                }
            })
            # 出错也即时保存
            save_progress(results, output_path)

    # 最终保存
    save_progress(results, output_path)

    # 统计 - 使用科学计算方法：先计算每个样本的维度平均值，再计算全局平均值
    # 筛选有效结果（至少有一个维度分数 > 0）
    valid_results = [r for r in results
                     if any(v > 0 for v in r.get('evaluation', {}).get('dimensions', {}).values())]

    if valid_results:
        # 计算每个样本的平均分（六维度平均值）
        sample_avg_scores = []
        for r in valid_results:
            dims = r['evaluation'].get('dimensions', {})
            dim_values = list(dims.values())
            if dim_values:
                sample_avg = sum(dim_values) / len(dim_values)
                sample_avg_scores.append(sample_avg)
                # 更新该样本的 overall_score 为计算值
                r['evaluation']['overall_score'] = round(sample_avg, 2)

        # 全局平均分 = 所有样本平均分的平均值
        avg_score = sum(sample_avg_scores) / len(sample_avg_scores)
        high_count = sum(1 for s in sample_avg_scores if s >= 8)

        # 各维度全局平均
        dims_global = {}
        for r in valid_results:
            for k, v in r['evaluation'].get('dimensions', {}).items():
                dims_global[k] = dims_global.get(k, 0) + v
        for k in dims_global:
            dims_global[k] /= len(valid_results)

        # 验证：维度全局平均值之和/6 应等于全局平均分
        dims_avg_calc = sum(dims_global.values()) / len(dims_global)

        print(f"\n评估完成!")
        print(f"  有效: {len(valid_results)}/{len(results)}")
        print(f"  全局平均分: {avg_score:.2f}")
        print(f"  验证（维度均值/6）: {dims_avg_calc:.2f}")
        print(f"  高分(>=8): {high_count}/{len(valid_results)} ({high_count/len(valid_results)*100:.1f}%)")
        print(f"\n各维度平均:")
        for k, v in dims_global.items():
            print(f"  {k}: {v:.2f}")

        # 保存统计
        stats_path = Path(str(output_path).replace('.jsonl', '_stats.json'))
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_samples': len(results),
                'valid_evaluations': len(valid_results),
                'errors': len(results) - len(valid_results),
                'average_score': round(avg_score, 2),
                'average_score_method': 'dimensions_mean_per_sample_then_global_mean',
                'verification': {
                    'dims_avg_div6': round(dims_avg_calc, 2),
                    'difference': round(abs(avg_score - dims_avg_calc), 4)
                },
                'high_score_count': high_count,
                'high_score_ratio': round(high_count/len(valid_results), 3),
                'dimensions_avg': {k: round(v, 2) for k, v in dims_global.items()}
            }, f, ensure_ascii=False, indent=2)

        print(f"\n输出文件: {output_path}")
        print(f"统计文件: {stats_path}")

if __name__ == '__main__':
    main()