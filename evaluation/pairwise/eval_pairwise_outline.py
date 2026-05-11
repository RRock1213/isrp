import argparse
import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from api_client import get_evaluator_client
from isrp.prompts import PromptTemplates
from evaluation.pairwise_eval_common import (
    aggregate_pair_record,
    compute_average_scores,
    load_aligned_pairs,
    load_jsonl,
    normalize_judge_result_structure,
    parse_judge_response,
    save_jsonl,
    summarize_pairs,
)


OUTLINE_DIMENSIONS = [
    "relevance",
    "specificity",
    "guidance",
    "language",
    "coverage",
    "semantic_correctness",
]


def build_outline_judge_prompt(user_prompt, result_a, result_b):
    dimensions_text = "\n".join(f"- {name}" for name in OUTLINE_DIMENSIONS)
    template = PromptTemplates().get_template("PROMPT_EVAL_PAIRWISE_OUTLINE", "en")
    return template.format(
        user_prompt=user_prompt,
        dimensions_text=dimensions_text,
        result_a=result_a,
        result_b=result_b
    )


def evaluate_round(client, prompt, left_text, right_text):
    response = client.call_api(
        prompt=build_outline_judge_prompt(prompt, left_text, right_text),
        temperature=0.1,
        max_new_tokens=65536,
    )
    parsed = parse_judge_response(response)
    parsed = normalize_judge_result_structure(parsed, OUTLINE_DIMENSIONS)
    if "average_scores" not in parsed and "scores" in parsed:
        parsed["average_scores"] = compute_average_scores(parsed["scores"])
    return parsed


def build_round_records(pairs, client, raw_output_path, resume=True):
    existing = {}
    if resume and Path(raw_output_path).exists():
        for item in load_jsonl(raw_output_path):
            existing[(item["prompt"], item["round_index"])] = item

    records = list(existing.values())
    for pair in pairs:
        for round_index, left_model, right_model in [
            (1, "isrp", "e2e"),
            (2, "e2e", "isrp"),
        ]:
            key = (pair["prompt"], round_index)
            if key in existing:
                continue
            left_text = pair[left_model]["plan"]
            right_text = pair[right_model]["plan"]
            judge_result = evaluate_round(client, pair["prompt"], left_text, right_text)
            record = {
                "prompt": pair["prompt"],
                "task_type": "outline",
                "round_index": round_index,
                "left_model": left_model,
                "right_model": right_model,
                "judge_result": judge_result,
            }
            records.append(record)
            existing[key] = record
            save_jsonl(sorted(records, key=lambda item: (item["prompt"], item["round_index"])), raw_output_path)
    return sorted(records, key=lambda item: (item["prompt"], item["round_index"]))


def aggregate_pairs(raw_records):
    grouped = {}
    for item in raw_records:
        grouped.setdefault(item["prompt"], {})[item["round_index"]] = item

    pair_records = []
    for prompt, rounds in grouped.items():
        if 1 not in rounds or 2 not in rounds:
            continue
        pair_records.append(
            aggregate_pair_record(
                prompt,
                rounds[1],
                rounds[2],
                "isrp",
                "e2e",
            )
        )
    return sorted(pair_records, key=lambda item: item["prompt"])


def main():
    parser = argparse.ArgumentParser(description="Pairwise outline evaluation with glm-5")
    parser.add_argument(
        "--isrp",
        default="outputs/plan.jsonl",
    )
    parser.add_argument(
        "--e2e",
        default="outputs/plan_e2e_wb.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/pairwise_eval/outline",
    )
    parser.add_argument("--model", default="glm-5")
    parser.add_argument("--provider", default="dashscope")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_output_path = output_dir / "raw_rounds.jsonl"
    pair_output_path = output_dir / "pairs.jsonl"
    summary_output_path = output_dir / "summary.json"

    pairs = load_aligned_pairs(args.isrp, args.e2e)
    if args.start:
        pairs = pairs[args.start:]
    if args.limit:
        pairs = pairs[: args.limit]
    client = get_evaluator_client(args.model, args.provider)

    raw_records = build_round_records(pairs, client, raw_output_path, resume=not args.no_resume)
    pair_records = aggregate_pairs(raw_records)
    save_jsonl(pair_records, pair_output_path)

    summary = summarize_pairs(pair_records, "isrp", "e2e")
    summary["task_type"] = "outline"
    summary["criteria_source"] = "static_outline_dimensions"
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"raw rounds: {raw_output_path}")
    print(f"pairs: {pair_output_path}")
    print(f"summary: {summary_output_path}")


if __name__ == "__main__":
    main()
