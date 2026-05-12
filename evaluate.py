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
from itertools import combinations
from pathlib import Path

import pandas as pd


HERE = Path(__file__).parent
GOLDEN_PATH = HERE / "golden_dataset.json"
GENERATION_PATH = HERE / "generation_result_2.json"
OUTPUT_PATH = HERE / "jaccard_results_2.json"


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
            out.add(str(v))
    return out


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

    sim_sql = [jaccard_similarity(sql_to_tokens(s), golden_sql_tokens) for s in sqls]
    sim_output = [jaccard_similarity(output_to_value_set(r), golden_output_set) for r in rows]

    n = len(agent_runs)
    if n >= 2:
        det_sql = pairwise_jaccard_similarity([sql_to_tokens(s) for s in sqls])
        det_output = pairwise_jaccard_similarity([output_to_value_set(r) for r in rows])
    else:
        det_sql = None
        det_output = None

    return {
        "runs": n,
        "jaccard_similarity_sql": sum(sim_sql) / n,
        "jaccard_similarity_output": sum(sim_output) / n,
        "determinism_sql_similarity": det_sql,
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
