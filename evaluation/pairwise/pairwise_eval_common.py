import json
import random
import re
from pathlib import Path


def load_jsonl(path):
    items = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def index_by_prompt(items):
    return {item.get("prompt") or item.get("query"): item for item in items if item.get("prompt") or item.get("query")}


def save_jsonl(items, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def normalize_text(value):
    if isinstance(value, list):
        return "\n\n".join(str(item) for item in value)
    return str(value)


def load_aligned_pairs(primary_path, secondary_path):
    primary_by_prompt = index_by_prompt(load_jsonl(primary_path))
    secondary_by_prompt = index_by_prompt(load_jsonl(secondary_path))
    prompts = sorted(set(primary_by_prompt) & set(secondary_by_prompt))
    pairs = []
    for prompt in prompts:
        pairs.append(
            {
                "prompt": prompt,
                "isrp": primary_by_prompt[prompt],
                "e2e": secondary_by_prompt[prompt],
            }
        )
    return pairs


def parse_judge_response(text):
    if not text:
        raise ValueError("empty judge response")

    stripped = text.strip()
    if "```" in stripped:
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        stripped = stripped.strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(stripped[start : end + 1])
        else:
            raise

    if "scores" not in parsed and "scores.A" in parsed and "scores.B" in parsed:
        parsed["scores"] = {
            "A": parsed.pop("scores.A"),
            "B": parsed.pop("scores.B"),
        }
    if "average_scores" not in parsed and "average_scores.A" in parsed and "average_scores.B" in parsed:
        parsed["average_scores"] = {
            "A": parsed.pop("average_scores.A"),
            "B": parsed.pop("average_scores.B"),
        }
    if "winner" in parsed:
        parsed["winner"] = normalize_winner_token(parsed["winner"])
    return parsed


def compute_average_scores(scores):
    averages = {}
    for side, side_scores in scores.items():
        values = list(side_scores.values())
        averages[side] = round(sum(values) / len(values), 4) if values else 0.0
    return averages


def normalize_score_map(score_value, criterion_names=None):
    if isinstance(score_value, dict):
        return score_value
    if isinstance(score_value, list):
        if criterion_names and len(score_value) == len(criterion_names):
            return {
                criterion_names[index]: value
                for index, value in enumerate(score_value)
            }
        return {
            f"criterion_{index + 1}": value
            for index, value in enumerate(score_value)
        }
    raise ValueError(f"unsupported score payload: {type(score_value)}")


def normalize_scores_payload(scores, criterion_names=None):
    if not isinstance(scores, dict):
        raise ValueError(f"unsupported scores payload: {type(scores)}")

    if "A" in scores or "B" in scores:
        return {
            side: normalize_score_map(side_scores, criterion_names)
            for side, side_scores in scores.items()
        }

    criterion_first = {}
    for criterion_name, criterion_scores in scores.items():
        if not isinstance(criterion_scores, dict):
            raise ValueError(f"unsupported criterion score payload: {type(criterion_scores)}")
        for side, value in criterion_scores.items():
            normalized_side = normalize_winner_token(side)
            criterion_first.setdefault(normalized_side, {})[criterion_name] = value
    return criterion_first


def normalize_judge_result_structure(parsed, criterion_names=None):
    if "scores" in parsed:
        parsed["scores"] = normalize_scores_payload(parsed["scores"], criterion_names)
    if "average_scores" not in parsed and "scores" in parsed:
        parsed["average_scores"] = compute_average_scores(parsed["scores"])
    if "confidence" in parsed:
        parsed["confidence"] = normalize_confidence_value(parsed["confidence"])
    return parsed


def normalize_winner_token(winner):
    if winner is None:
        return "Tie"
    normalized = str(winner).strip().lower()
    normalized = re.sub(r"[^a-z]", "", normalized)
    if normalized in {"a", "resulta", "optionx", "x", "left"}:
        return "A"
    if normalized in {"b", "resultb", "optiony", "y", "right"}:
        return "B"
    if normalized in {"tie", "draw", "equal"}:
        return "Tie"
    raise ValueError(f"unknown winner value: {winner}")


def normalize_confidence_value(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    normalized = str(value).strip().lower()
    qualitative_map = {
        "veryhigh": 0.95,
        "high": 0.85,
        "medium": 0.6,
        "low": 0.35,
        "verylow": 0.15,
    }
    compact = re.sub(r"[^a-z0-9.]", "", normalized)
    if compact in qualitative_map:
        return qualitative_map[compact]
    try:
        return float(compact)
    except ValueError as exc:
        raise ValueError(f"unknown confidence value: {value}") from exc


def _normalize_round_winner(round_result, first_label, second_label):
    winner = normalize_winner_token(round_result.get("winner", "Tie"))
    if winner == "Tie":
        return None
    if winner == "A":
        return first_label
    if winner == "B":
        return second_label
    raise ValueError(f"unknown winner value: {winner}")


def aggregate_pair_outcome(round_one, round_two, first_label, second_label):
    score_by_model = {first_label: 0, second_label: 0}

    normalized_one = _normalize_round_winner(round_one, first_label, second_label)
    normalized_two = _normalize_round_winner(round_two, second_label, first_label)

    for normalized in (normalized_one, normalized_two):
        if normalized is None:
            continue
        loser = second_label if normalized == first_label else first_label
        score_by_model[normalized] += 1
        score_by_model[loser] -= 1

    true_score_by_model = {
        first_label: score_by_model[first_label] / 2.0,
        second_label: score_by_model[second_label] / 2.0,
    }

    if true_score_by_model[first_label] > true_score_by_model[second_label]:
        final_winner = first_label
    elif true_score_by_model[first_label] < true_score_by_model[second_label]:
        final_winner = second_label
    else:
        final_winner = "Tie"

    return {
        "normalized_winners": [normalized_one, normalized_two],
        "round_score_by_model": score_by_model,
        "true_score_by_model": true_score_by_model,
        "final_winner": final_winner,
    }


def normalize_round_result_by_model(round_result, left_model, right_model):
    round_result = normalize_judge_result_structure(dict(round_result))
    winner = normalize_winner_token(round_result["winner"])
    normalized_winner = "Tie"
    if winner == "A":
        normalized_winner = left_model
    elif winner == "B":
        normalized_winner = right_model

    normalized_scores = {
        left_model: round_result["scores"]["A"],
        right_model: round_result["scores"]["B"],
    }
    normalized_average_scores = {
        left_model: round_result["average_scores"]["A"],
        right_model: round_result["average_scores"]["B"],
    }

    return {
        "winner": normalized_winner,
        "scores_by_model": normalized_scores,
        "average_scores_by_model": normalized_average_scores,
        "reason": round_result.get("reason", ""),
        "confidence": round_result.get("confidence", 0.0),
    }


def aggregate_pair_record(prompt, round_one_raw, round_two_raw, first_label, second_label):
    round_one = normalize_round_result_by_model(
        round_one_raw["judge_result"],
        round_one_raw["left_model"],
        round_one_raw["right_model"],
    )
    round_two = normalize_round_result_by_model(
        round_two_raw["judge_result"],
        round_two_raw["left_model"],
        round_two_raw["right_model"],
    )

    outcome = aggregate_pair_outcome(
        round_one_raw["judge_result"],
        round_two_raw["judge_result"],
        first_label,
        second_label,
    )

    criterion_scores = {}
    for model_name in (first_label, second_label):
        merged = {}
        keys = set(round_one["scores_by_model"][model_name]) | set(round_two["scores_by_model"][model_name])
        for key in keys:
            values = []
            if key in round_one["scores_by_model"][model_name]:
                values.append(round_one["scores_by_model"][model_name][key])
            if key in round_two["scores_by_model"][model_name]:
                values.append(round_two["scores_by_model"][model_name][key])
            merged[key] = round(sum(values) / len(values), 4) if values else 0.0
        criterion_scores[model_name] = merged

    average_scores = {}
    confidences = []
    for model_name in (first_label, second_label):
        values = [round_one["average_scores_by_model"][model_name], round_two["average_scores_by_model"][model_name]]
        average_scores[model_name] = round(sum(values) / len(values), 4)
    confidences.extend([round_one.get("confidence", 0.0), round_two.get("confidence", 0.0)])

    return {
        "prompt": prompt,
        "round_1": round_one_raw,
        "round_2": round_two_raw,
        "criterion_scores": criterion_scores,
        "average_scores": average_scores,
        "true_score_by_model": outcome["true_score_by_model"],
        "final_winner": outcome["final_winner"],
        "confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
    }


def summarize_pairs(pair_records, first_label, second_label):
    total_pairs = len(pair_records)
    first_wins = sum(1 for item in pair_records if item["final_winner"] == first_label)
    second_wins = sum(1 for item in pair_records if item["final_winner"] == second_label)
    ties = sum(1 for item in pair_records if item["final_winner"] == "Tie")

    criterion_avgs = {first_label: {}, second_label: {}}
    for model_name in (first_label, second_label):
        keys = set()
        for item in pair_records:
            keys.update(item["criterion_scores"][model_name].keys())
        for key in keys:
            values = [item["criterion_scores"][model_name][key] for item in pair_records if key in item["criterion_scores"][model_name]]
            criterion_avgs[model_name][key] = round(sum(values) / len(values), 4) if values else 0.0

    avg_true_score = {
        first_label: round(sum(item["true_score_by_model"][first_label] for item in pair_records) / total_pairs, 4) if total_pairs else 0.0,
        second_label: round(sum(item["true_score_by_model"][second_label] for item in pair_records) / total_pairs, 4) if total_pairs else 0.0,
    }
    low_confidence_prompts = [item["prompt"] for item in pair_records if item.get("confidence", 0.0) < 0.6]

    return {
        "total_pairs": total_pairs,
        "isrp_true_wins": first_wins if first_label == "isrp" else second_wins,
        "e2e_true_wins": second_wins if second_label == "e2e" else first_wins,
        "ties": ties,
        "win_rate_by_model": {
            first_label: round(first_wins / total_pairs, 4) if total_pairs else 0.0,
            second_label: round(second_wins / total_pairs, 4) if total_pairs else 0.0,
        },
        "avg_true_score_by_model": avg_true_score,
        "criterion_average_scores": criterion_avgs,
        "low_confidence_prompts": low_confidence_prompts,
    }


def build_review_manifest(primary_records, secondary_records, task_type, left_first_choices=None):
    primary_by_prompt = index_by_prompt(primary_records)
    secondary_by_prompt = index_by_prompt(secondary_records)
    prompts = sorted(set(primary_by_prompt) & set(secondary_by_prompt))

    if left_first_choices is None:
        randomizer = random.Random(42)
        left_first_choices = [randomizer.choice([True, False]) for _ in prompts]

    manifest = []
    field = task_type
    for idx, prompt in enumerate(prompts):
        primary = primary_by_prompt[prompt]
        secondary = secondary_by_prompt[prompt]
        primary_left = left_first_choices[idx]

        left_model = "isrp" if primary_left else "e2e"
        right_model = "e2e" if primary_left else "isrp"
        left_text = normalize_text(primary[field] if primary_left else secondary[field])
        right_text = normalize_text(secondary[field] if primary_left else primary[field])

        manifest.append(
            {
                "pair_index": idx,
                "prompt": prompt,
                "task_type": task_type,
                "criteria": primary.get("checklist", []),
                "left_model": left_model,
                "right_model": right_model,
                "left_text": left_text,
                "right_text": right_text,
            }
        )

    return manifest
