"""Check the agent's answer against evidence computed directly in Python.

This module does NOT reuse the agent's tools. It recomputes each check with
plain pandas one-liners, so a bug or a hallucination in the agent shows up as
a mismatch instead of being trusted because the model "said so".

Every problem is reduced to a (check, column, row_index) triple:
  ("missing_values", "age", 2)
  ("unusual_values", "gpa", 13)
  ("duplicates", "*", 20)            "*" = whole-row duplicates
  ("duplicates", "student_id", 21)
"""

from __future__ import annotations

import hashlib

import pandas as pd

ALL_ROWS = "*"


def file_fingerprint(path: str) -> str:
    """SHA-256 of the file, used to prove the dataset was not modified."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _dup_key(columns) -> str:
    if not columns or columns in (ALL_ROWS, "all", "all columns", "null"):
        return ALL_ROWS
    if isinstance(columns, str):
        columns = [c.strip() for c in columns.split(",")]
    return ",".join(sorted(columns))


# ------------------------------------------------ direct Python ground truth
def direct_missing(df: pd.DataFrame, columns=None) -> set:
    triples = set()
    for col in columns or df.columns:
        empty = df[col].isna() | (df[col].astype(str).str.strip() == "")
        triples |= {("missing_values", col, int(i)) for i in df.index[empty]}
    return triples


def direct_unusual(df: pd.DataFrame, column: str) -> set:
    values = pd.to_numeric(df[column], errors="coerce")
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    outside = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    return {("unusual_values", column, int(i)) for i in df.index[outside]}


def direct_duplicates(df: pd.DataFrame, columns=None) -> set:
    subset = list(columns) if columns else None
    repeated = df.duplicated(subset=subset, keep=False)
    key = _dup_key(columns)
    return {("duplicates", key, int(i)) for i in df.index[repeated]}


def ground_truth_for_calls(csv_path: str, tool_calls: list[dict]) -> set:
    """Recompute, directly in pandas, every check the agent actually ran."""
    df = pd.read_csv(csv_path)
    truth = set()
    for call in tool_calls:
        if not call["result"].get("ok"):
            continue  # failed calls produced no evidence
        args = call.get("args") or {}
        if call["name"] == "check_missing_values":
            truth |= direct_missing(df, args.get("columns"))
        elif call["name"] == "check_unusual_values":
            truth |= direct_unusual(df, args["column"])
        elif call["name"] == "check_duplicates":
            truth |= direct_duplicates(df, args.get("columns"))
    return truth


# ------------------------------------------------ what the agent claimed
def claimed_triples(answer: dict) -> set:
    triples = set()
    for f in (answer or {}).get("findings", []):
        check = f.get("check")
        column = f.get("column")
        if check == "duplicates":
            column = _dup_key(column)
        for row in f.get("rows", []):
            triples.add((check, column, int(row)))
    return triples


def verify(csv_path: str, answer: dict | None, tool_calls: list[dict],
           fingerprint_before: str) -> dict:
    """Compare the agent's findings with direct Python checks."""
    truth = ground_truth_for_calls(csv_path, tool_calls)
    claimed = claimed_triples(answer)
    unchanged = file_fingerprint(csv_path) == fingerprint_before

    report = {
        "confirmed": sorted(claimed & truth, key=str),
        "unsupported_claims": sorted(claimed - truth, key=str),  # hallucinated / wrong
        "missed_problems": sorted(truth - claimed, key=str),     # tool found, agent left out
        "dataset_unchanged": unchanged,
        "answer_parsed": answer is not None,
    }
    report["passed"] = (
        report["answer_parsed"]
        and unchanged
        and not report["unsupported_claims"]
        and not report["missed_problems"]
    )
    return report


def full_direct_report(csv_path: str) -> dict:
    """Every check on every column - handy for checking the datasets by hand."""
    df = pd.read_csv(csv_path)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    truth = direct_missing(df)
    for col in numeric:
        truth |= direct_unusual(df, col)
    truth |= direct_duplicates(df)
    return {"rows": len(df), "numeric_columns": numeric, "problems": sorted(truth, key=str)}


if __name__ == "__main__":  # python -m dqi.verify data/students_messy.csv
    import sys

    for path in sys.argv[1:] or ["data/students_clean.csv", "data/students_messy.csv"]:
        report = full_direct_report(path)
        print(f"\n{path}: {report['rows']} rows, {len(report['problems'])} problems")
        for problem in report["problems"]:
            print("  ", problem)
