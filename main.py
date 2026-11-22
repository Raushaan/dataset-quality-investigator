"""Run the Dataset Quality Investigator from the command line.

Example:
    python main.py --data data/students_messy.csv --question "Is this dataset clean?"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from dqi.agent import run_agent
from dqi.llm import DEFAULT_MODEL, LLMError, OllamaClient
from dqi.verify import file_fingerprint, verify

LINE = "-" * 72


def check_input_file(path: str) -> str | None:
    """Return an error message if the CSV can't be used, else None."""
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    if p.suffix.lower() != ".csv":
        return f"Expected a .csv file, got: {p.name}"
    try:
        df = pd.read_csv(p)
    except Exception as exc:
        return f"Could not read {p.name} as CSV: {exc}"
    if df.empty:
        return f"{p.name} has no data rows."
    return None


def print_answer(result) -> None:
    print(f"\n{LINE}\nFINAL ANSWER ({result.stopped_reason})\n{LINE}")
    if result.answer is None:
        print("No usable answer. Last model output:\n" + (result.raw_answer or "(none)"))
        return
    print(result.answer.get("summary", ""))
    findings = result.answer.get("findings", [])
    if findings:
        print("\nFindings:")
        for f in findings:
            column = f.get("column") or "whole row"
            print(f"  - {f['check']:<15} column={column:<18} rows={f['rows']}  {f.get('detail', '')}")
    else:
        print("\nFindings: none")
    if result.answer.get("could_not_answer"):
        print(f"\nCould not answer: {result.answer['could_not_answer']}")
    for w in result.warnings:
        print(f"  warning: {w}")


def print_verification(report: dict) -> None:
    print(f"\n{LINE}\nVERIFICATION (direct pandas checks, not the model's word)\n{LINE}")
    print(f"  dataset unchanged (SHA-256): {'YES' if report['dataset_unchanged'] else 'NO'}")
    print(f"  confirmed findings:          {len(report['confirmed'])}")
    for t in report["unsupported_claims"]:
        print(f"  UNSUPPORTED claim (not in evidence): {t}")
    for t in report["missed_problems"]:
        print(f"  MISSED (tool found it, answer left it out): {t}")
    print(f"  verdict: {'PASS' if report['passed'] else 'FAIL'}")


def investigate(data: str, question: str, model: str = DEFAULT_MODEL,
                max_steps: int = 12, save_dir: str = "runs") -> dict:
    """Run one investigation end to end and save the trace. Returns the record."""
    print(f"{LINE}\nDataset:  {data}\nQuestion: {question}\nModel:    {model}\n{LINE}")

    problem = check_input_file(data)
    if problem:
        print(f"ERROR: {problem}\nThe agent was not started.")
        record = {"data": data, "question": question, "error": problem}
    else:
        before = file_fingerprint(data)
        try:
            result = run_agent(question, data, OllamaClient(model=model), max_steps=max_steps)
        except LLMError as exc:
            print(f"ERROR: {exc}")
            record = {"data": data, "question": question, "error": str(exc)}
        else:
            print_answer(result)
            report = verify(data, result.answer, result.tool_calls, before)
            print_verification(report)
            record = {
                "data": data, "question": question, "model": model,
                "stopped_reason": result.stopped_reason, "answer": result.answer,
                "tool_calls": result.tool_calls, "trace": result.trace,
                "verification": report, "warnings": result.warnings,
            }

    Path(save_dir).mkdir(exist_ok=True)
    out = Path(save_dir) / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    out.write_text(json.dumps(record, indent=2, default=str))
    print(f"\nTrace saved to {out}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset Quality Investigator agent")
    parser.add_argument("--data", required=True, help="path to a CSV file")
    parser.add_argument("--question", default="Is this dataset clean? Check for any problems.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()
    record = investigate(args.data, args.question, args.model, args.max_steps)
    return 1 if "error" in record else 0


if __name__ == "__main__":
    sys.exit(main())
