"""
Evaluation phase — reads raw agent outputs from generation_results.json and
computes Jaccard similarity metrics against the golden dataset.

This phase is FAST (pure Python + pandas, no LLM calls). Re-run as many times
as you want to refine metrics, fix bugs, or add new analyses without paying
the LLM inference cost again.

Pipeline:
    python generate.py --runs N    # expensive: runs the LLM, writes generation_results.json
    python evaluate.py             # cheap: reads it, computes metrics, writes jaccard_results.json

Two primary metrics, both Jaccard similarity coefficients (higher = better,
1.0 = identical, 0.0 = disjoint):
    - jaccard_similarity_sql:    similarity between agent SQL tokens and golden SQL tokens
    - jaccard_similarity_output: similarity between agent's output values and golden's output values
                                 (values are flattened across rows and columns, making the metric
                                 schema-agnostic so extra/renamed columns don't hurt the score)

Plus pairwise Jaccard similarity between runs (determinism) when runs >= 2.
"""

import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd


HERE = Path(__file__).parent
GOLDEN_PATH = HERE / "golden_dataset.json"
GENERATION_PATH = HERE / "generation_result_claude.json"
OUTPUT_PATH = HERE / "jaccard_results_claude.json"


# ---------------------------------------------------------------------------
# Jaccard primitives
# ---------------------------------------------------------------------------

def jaccard_similarity(a: set, b: set) -> float:
    """Standard Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|.
    1.0 = identical sets, 0.0 = disjoint sets.
    Two empty sets are considered identical (1.0).
    """
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def precision(predicted: set, golden: set) -> float:
    """Fraction of predicted tokens/values that appear in the golden set.
    Empty predicted set with empty golden → 1.0; empty predicted with non-empty golden → 0.0.
    """
    if not predicted and not golden:
        return 1.0
    if not predicted:
        return 0.0
    return len(predicted & golden) / len(predicted)


def recall(predicted: set, golden: set) -> float:
    """Fraction of golden tokens/values found in the predicted set.
    Empty golden with empty predicted → 1.0; empty golden with non-empty predicted → 0.0.
    """
    if not predicted and not golden:
        return 1.0
    if not golden:
        return 0.0
    return len(predicted & golden) / len(golden)


def pairwise_jaccard_similarity(items: list[set]) -> float:
    pairs = list(combinations(items, 2))
    if not pairs:
        return 1.0
    return sum(jaccard_similarity(a, b) for a, b in pairs) / len(pairs)


def sql_to_tokens(sql: str) -> set:
    """Normalize SQL → set of tokens. Case- and whitespace-insensitive."""
    if not sql:
        return set()
    s = sql.lower().replace("`", " ").replace('"', " ")
    s = re.sub(r"\s+", " ", s).strip()
    return set(re.findall(r"[\w_.%]+|[<>=!]+|[(),;]", s))


def _normalize_cell(v) -> str:
    """Stringify a cell value, rounding floats to 2 decimal places so that
    minor precision differences (e.g. 7.241 vs 7.2413298) don't hurt the score.
    Genuinely different values (5.78 vs 5.80) still produce different strings.
    """
    try:
        return f"{float(v):.2f}"
    except (ValueError, TypeError):
        return str(v)


def output_to_value_set(rows) -> set:
    """Flatten ALL cell values across rows and columns into a single set.
    Schema-agnostic: rewards correct values regardless of column count, names,
    or ordering — robust to the agent returning extra explanatory columns.
    """
    if rows is None:
        return set()
    out = set()
    for row in rows:
        if isinstance(row, dict):
            cells = row.values()
        elif isinstance(row, (list, tuple)):
            cells = row
        else:
            cells = [row]
        for v in cells:
            out.add(_normalize_cell(v))
    return out


def _row_key(row) -> frozenset:
    """Normalize one row to a frozenset of its cell values (schema-agnostic)."""
    if isinstance(row, dict):
        cells = row.values()
    elif isinstance(row, (list, tuple)):
        cells = row
    else:
        cells = [row]
    return frozenset(_normalize_cell(v) for v in cells)


def rows_to_multiset(rows) -> Counter:
    """Convert a list of rows to a Counter of row keys, preserving duplicates."""
    if not rows:
        return Counter()
    return Counter(_row_key(r) for r in rows)


def row_precision(agent_rows: list, golden_rows: list) -> float:
    """Fraction of agent rows that match a golden row (row-level precision).
    Both empty → 1.0; agent empty with non-empty golden → 0.0.
    """
    if not agent_rows and not golden_rows:
        return 1.0
    if not agent_rows:
        return 0.0
    agent_ms = rows_to_multiset(agent_rows)
    golden_ms = rows_to_multiset(golden_rows)
    matched = sum((agent_ms & golden_ms).values())
    return matched / sum(agent_ms.values())


def row_recall(agent_rows: list, golden_rows: list) -> float:
    """Fraction of golden rows that the agent retrieved (row-level recall).
    Both empty → 1.0; golden empty with non-empty agent → 0.0.
    """
    if not agent_rows and not golden_rows:
        return 1.0
    if not golden_rows:
        return 0.0
    agent_ms = rows_to_multiset(agent_rows)
    golden_ms = rows_to_multiset(golden_rows)
    matched = sum((agent_ms & golden_ms).values())
    return matched / sum(golden_ms.values())


# ---------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------

def score_question(agent_runs: list[dict], golden_sql: str, expected_rows: list) -> dict:
    """Compute the two Jaccard similarity scores for one question, averaged across runs.
    Higher is better; 1.0 = identical sets, 0.0 = disjoint sets.
    """
    golden_sql_tokens = sql_to_tokens(golden_sql)
    golden_output_set = output_to_value_set(expected_rows)

    sqls = [r.get("sql", "") for r in agent_runs]
    rows = [r.get("rows", []) for r in agent_runs]
    elapsed = [r.get("elapsed_seconds", 0.0) for r in agent_runs]

    sql_token_sets = [sql_to_tokens(s) for s in sqls]
    output_value_sets = [output_to_value_set(r) for r in rows]

    sim_sql    = [jaccard_similarity(t, golden_sql_tokens) for t in sql_token_sets]
    sim_output = [jaccard_similarity(v, golden_output_set) for v in output_value_sets]
    prec_sql   = [precision(t, golden_sql_tokens) for t in sql_token_sets]
    rec_sql    = [recall(t, golden_sql_tokens) for t in sql_token_sets]
    prec_out   = [precision(v, golden_output_set) for v in output_value_sets]
    rec_out    = [recall(v, golden_output_set) for v in output_value_sets]
    row_prec   = [row_precision(r, expected_rows) for r in rows]
    row_rec    = [row_recall(r, expected_rows) for r in rows]

    n = len(agent_runs)
    if n >= 2:
        det_sql = pairwise_jaccard_similarity(sql_token_sets)
        det_output = pairwise_jaccard_similarity(output_value_sets)
    else:
        det_sql = None
        det_output = None

    return {
        "runs": n,
        "jaccard_similarity_sql":    sum(sim_sql) / n,
        "jaccard_similarity_output": sum(sim_output) / n,
        "precision_sql":             sum(prec_sql) / n,
        "recall_sql":                sum(rec_sql) / n,
        "precision_output":          sum(prec_out) / n,
        "recall_output":             sum(rec_out) / n,
        "row_precision_output":      sum(row_prec) / n,
        "row_recall_output":         sum(row_rec) / n,
        "determinism_sql_similarity":    det_sql,
        "determinism_output_similarity": det_output,
        "elapsed_seconds_per_run": [round(e, 2) for e in elapsed],
        "mean_elapsed_seconds": round(sum(elapsed) / n, 2) if n else 0.0,
        "agent_sqls": sqls,
        "agent_row_counts": [len(r) for r in rows],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not GENERATION_PATH.exists():
        print(f"❌ {GENERATION_PATH.name} not found. Run `python generate.py` first.")
        return

    with open(GENERATION_PATH) as f:
        gen = json.load(f)
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    golden_by_id = {item.get("id"): item for item in golden}

    runs_per_question = gen.get("runs_per_question", 1)
    print(f"Loaded {len(gen['per_question'])} questions from {GENERATION_PATH.name} "
          f"(model: {gen.get('model', 'unknown')}, runs: {runs_per_question})\n")

    # Score every question
    rows_for_df = []
    per_question_records = []
    for q in gen["per_question"]:
        qid = q["id"]
        gold = golden_by_id.get(qid, {})
        scored = score_question(
            q["runs"],
            gold.get("golden_query", ""),
            gold.get("expected_rows", []),
        )
        record = {"id": qid, "question": q["question"], **scored}
        per_question_records.append(record)
        rows_for_df.append({
            "id": qid,
            "sql_sim":    round(scored["jaccard_similarity_sql"], 3),
            "output_sim": round(scored["jaccard_similarity_output"], 3),
            "prec_sql":   round(scored["precision_sql"], 3),
            "rec_sql":    round(scored["recall_sql"], 3),
            "prec_out":   round(scored["precision_output"], 3),
            "rec_out":    round(scored["recall_output"], 3),
            "row_prec":   round(scored["row_precision_output"], 3),
            "row_rec":    round(scored["row_recall_output"], 3),
            "det_sql":    None if scored["determinism_sql_similarity"]    is None else round(scored["determinism_sql_similarity"],    3),
            "det_output": None if scored["determinism_output_similarity"] is None else round(scored["determinism_output_similarity"], 3),
            "elapsed_s": scored["mean_elapsed_seconds"],
            "agent_n": max(scored["agent_row_counts"]) if scored["agent_row_counts"] else 0,
        })

    df = pd.DataFrame(rows_for_df).sort_values("id").reset_index(drop=True)

    # Per-question table
    print("=" * 90)
    print("PER-QUESTION METRICS")
    print("=" * 90)
    print(df.to_string(index=False))
    print()

    # Aggregate summary
    summary = {
        "questions_evaluated": len(df),
        "runs_per_question": runs_per_question,
        "questions_total":      gen.get("questions_total"),
        "timed_out_ids":        gen.get("timed_out_ids", []),
        "mean_jaccard_similarity_sql":    df["sql_sim"].mean(),
        "mean_jaccard_similarity_output": df["output_sim"].mean(),
        "mean_precision_sql":    df["prec_sql"].mean(),
        "mean_recall_sql":       df["rec_sql"].mean(),
        "mean_precision_output":     df["prec_out"].mean(),
        "mean_recall_output":        df["rec_out"].mean(),
        "mean_row_precision_output": df["row_prec"].mean(),
        "mean_row_recall_output":    df["row_rec"].mean(),
        "mean_elapsed_seconds_per_question": round(df["elapsed_s"].mean(), 2),
        "total_elapsed_seconds": round(df["elapsed_s"].sum() * runs_per_question, 2),
    }
    if runs_per_question >= 2:
        summary["mean_determinism_sql_similarity"]    = df["det_sql"].dropna().mean()
        summary["mean_determinism_output_similarity"] = df["det_output"].dropna().mean()

    print("=" * 90)
    print(f"SUMMARY — {len(df)} questions × {runs_per_question} run(s)")
    print("=" * 90)
    print(f"Jaccard similarity (higher = better):")
    print(f"  SQL:    {summary['mean_jaccard_similarity_sql']:.3f}")
    print(f"  Output: {summary['mean_jaccard_similarity_output']:.3f}")
    print(f"Precision / Recall — SQL:")
    print(f"  Precision: {summary['mean_precision_sql']:.3f}   Recall: {summary['mean_recall_sql']:.3f}")
    print(f"Precision / Recall — Output (token-level):")
    print(f"  Precision: {summary['mean_precision_output']:.3f}   Recall: {summary['mean_recall_output']:.3f}")
    print(f"Precision / Recall — Output (row-level):")
    print(f"  Precision: {summary['mean_row_precision_output']:.3f}   Recall: {summary['mean_row_recall_output']:.3f}")
    if runs_per_question >= 2:
        print(f"Determinism (pairwise similarity, higher = more deterministic):")
        print(f"  SQL:    {summary['mean_determinism_sql_similarity']:.3f}")
        print(f"  Output: {summary['mean_determinism_output_similarity']:.3f}")
    print(f"Latency:     mean per question {summary['mean_elapsed_seconds_per_question']:.1f}s")
    if summary["timed_out_ids"]:
        print(f"Timed out:   {summary['timed_out_ids']}")

    # Persist full results (summary + per_question raw scores)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({**summary, "per_question": per_question_records}, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results → {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
