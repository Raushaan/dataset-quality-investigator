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


# ------------------------------------------------ coverage for full scans
def id_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.lower() == "id" or c.lower().endswith("_id")]


def full_scan_checks(df: pd.DataFrame) -> list[tuple]:
    """Every check a complete answer to "is this dataset clean?" needs.

    Returns (check, column) keys: missing values in all columns, unusual
    values in each numeric column, identical rows, and repeated IDs.
    """
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return ([("missing_values", ALL_ROWS)]
            + [("unusual_values", c) for c in numeric]
            + [("duplicates", ALL_ROWS)]
            + [("duplicates", c) for c in id_columns(df)])


def performed_checks(df: pd.DataFrame, tool_calls: list[dict]) -> set:
    """The (check, column) keys the agent actually ran successfully."""
    done = set()
    for call in tool_calls:
        if not call["result"].get("ok"):
            continue
        args = call.get("args") or {}
        if call["name"] == "check_missing_values":
            cols = args.get("columns")
            if not cols or set(cols) >= set(df.columns):
                done.add(("missing_values", ALL_ROWS))
        elif call["name"] == "check_unusual_values":
            done.add(("unusual_values", args.get("column")))
        elif call["name"] == "check_duplicates":
            done.add(("duplicates", _dup_key(args.get("columns"))))
    return done


def direct_check(df: pd.DataFrame, key: tuple) -> set:
    check, column = key
    if check == "missing_values":
        return direct_missing(df)
    if check == "unusual_values":
        return direct_unusual(df, column)
    return direct_duplicates(df, None if column == ALL_ROWS else [column])


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
           fingerprint_before: str, full_scan: bool = False) -> dict:
    """Compare the agent's findings with direct Python checks.

    full_scan=True is for general questions ("is this dataset clean?"):
    every check in full_scan_checks() must have been run, and problems from
    skipped checks count as missed.
    """
    truth = ground_truth_for_calls(csv_path, tool_calls)
    claimed = claimed_triples(answer)
    unchanged = file_fingerprint(csv_path) == fingerprint_before

    not_checked = []
    if full_scan:
        df = pd.read_csv(csv_path)
        done = performed_checks(df, tool_calls)
        for key in full_scan_checks(df):
            if key not in done:
                not_checked.append(key)
                truth |= direct_check(df, key)

    report = {
        "confirmed": sorted(claimed & truth, key=str),
        "unsupported_claims": sorted(claimed - truth, key=str),  # hallucinated / wrong
        "missed_problems": sorted(truth - claimed, key=str),     # tool found, agent left out
        "not_checked": not_checked,                               # skipped in a full scan
        "dataset_unchanged": unchanged,
        "answer_parsed": answer is not None,
    }
    report["passed"] = (
        report["answer_parsed"]
        and unchanged
        and not report["unsupported_claims"]
        and not report["missed_problems"]
        and not report["not_checked"]
    )
    return report


def full_direct_report(csv_path: str) -> dict:
    """Every check on every column - handy for checking the datasets by hand."""
    df = pd.read_csv(csv_path)
    truth = set()
    for key in full_scan_checks(df):
        truth |= direct_check(df, key)
    return {"rows": len(df), "checks": full_scan_checks(df), "problems": sorted(truth, key=str)}


if __name__ == "__main__":  # python -m dqi.verify data/students_messy.csv
    import sys

    for path in sys.argv[1:] or ["data/students_clean.csv", "data/students_messy.csv"]:
        report = full_direct_report(path)
        print(f"\n{path}: {report['rows']} rows, {len(report['problems'])} problem entries "
              "(check, column, row_index)")
        for problem in report["problems"]:
            print("  ", problem)
