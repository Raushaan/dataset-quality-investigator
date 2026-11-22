"""The agent loop: the model decides which check to run next, reads the
result, and decides again, until it can answer the question.

    question -> [model decides] -> tool call -> observation -> [model decides] ...
                                -> final JSON answer
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .tools import TOOL_SCHEMAS, run_tool

VALID_CHECKS = {"missing_values", "unusual_values", "duplicates"}

SYSTEM_PROMPT = """You are a Dataset Quality Investigator. You inspect one CSV file \
with tools and REPORT data-quality problems. You can never change, clean, delete or \
save data: you have no tool for that.

Tools:
- get_dataset_overview: columns, column types, row count. Use it first if you do not know the columns.
- check_missing_values: empty cells.
- check_unusual_values: outliers in ONE numeric column (IQR rule). Call it once per numeric column.
- check_duplicates: identical rows (no arguments), or repeated IDs (columns=["<id column>"]).

Rules:
1. Choose only the checks the question needs. A general question such as "is this dataset clean?" \
needs missing values, unusual values in every numeric column, and duplicates (both identical rows \
and repeated IDs if there is an ID column).
2. Read every tool result before choosing the next step. If a tool returns an error, adapt \
(for example use a column that exists) or explain the problem to the user.
3. Report only what tool results show. Never invent rows, values or columns.
4. If the user asks you to modify, fix, delete or save data, do not try: say you only report \
problems, and report what you found if it is useful.
5. Identify rows by row_index exactly as the tools return it. For duplicates, list every row_index \
in each duplicate group (the original and its copies).

When you are finished, reply with ONLY this JSON object and no other text:
{"summary": "<2-4 plain sentences for the user>",
 "findings": [{"check": "missing_values" | "unusual_values" | "duplicates",
               "column": "<column name; for identical-row duplicates use null>",
               "rows": [<row_index>, ...],
               "detail": "<short explanation with the values>"}],
 "could_not_answer": "<what you could not do and why, or null>"}
Use "findings": [] when the checks you ran found no problems."""

NUDGE_NO_TOOLS = (
    "You have not inspected the data yet. Call a tool, or, if the request cannot be "
    "handled by your tools, reply with the final JSON and explain in could_not_answer."
)
NUDGE_BAD_JSON = (
    "Your last reply was not the required JSON object. Reply again with ONLY the JSON "
    "object (summary, findings, could_not_answer) and no other text."
)


@dataclass
class AgentResult:
    question: str
    csv_path: str
    answer: dict | None = None          # parsed final JSON
    raw_answer: str = ""                # what the model actually wrote last
    tool_calls: list = field(default_factory=list)   # [{"step","name","args","result"}]
    trace: list = field(default_factory=list)        # human-readable events
    stopped_reason: str = ""            # "answered" | "max_steps" | "bad_final_answer"
    warnings: list = field(default_factory=list)


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def parse_final_answer(text: str) -> dict | None:
    """Pull the JSON object out of the model's reply (tolerates ```json fences)."""
    text = _strip_thinking(text).replace("```json", "").replace("```", "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and "summary" in data else None


def clean_findings(answer: dict, warnings: list) -> dict:
    """Keep only well-formed findings; record anything dropped as a warning."""
    good = []
    for f in answer.get("findings") or []:
        if not isinstance(f, dict) or f.get("check") not in VALID_CHECKS:
            warnings.append(f"Dropped malformed finding: {f}")
            continue
        try:
            f["rows"] = [int(r) for r in (f.get("rows") or [])]  # accept "2" as 2
        except (TypeError, ValueError):
            warnings.append(f"Dropped finding with invalid rows: {f}")
            continue
        good.append(f)
    answer["findings"] = good
    answer.setdefault("could_not_answer", None)
    return answer


def run_agent(question: str, csv_path: str, llm, max_steps: int = 12,
              log=print) -> AgentResult:
    result = AgentResult(question=question, csv_path=csv_path)

    def event(kind: str, text: str, **data):
        result.trace.append({"type": kind, "text": text, **data})
        log(text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Dataset file: {Path(csv_path).name}\nQuestion: {question}"},
    ]
    nudged_for_tools = nudged_for_json = False

    for step in range(1, max_steps + 1):
        message = llm.chat(messages, tools=TOOL_SCHEMAS)
        calls = message.get("tool_calls") or []
        content = message.get("content") or ""
        messages.append({"role": "assistant", "content": content,
                         **({"tool_calls": calls} if calls else {})})

        # ---- the model chose one or more tools: run them, feed results back
        if calls:
            for call in calls:
                fn = call.get("function", {})
                name, args = fn.get("name", "?"), fn.get("arguments") or {}
                event("decision", f"[step {step}] model chose tool: {name}({json.dumps(args)})",
                      step=step, tool=name, args=args)
                observation = run_tool(name, args, csv_path)
                result.tool_calls.append(
                    {"step": step, "name": name, "args": args if isinstance(args, dict) else {},
                     "result": observation})
                event("observation", f"[step {step}] result: {_short(observation)}",
                      step=step, tool=name, result=observation)
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(observation)})
            continue

        # ---- no tool call: the model wants to finish
        if not result.tool_calls and not nudged_for_tools:
            nudged_for_tools = True
            event("nudge", f"[step {step}] model tried to answer without evidence -> asked to use a tool")
            messages.append({"role": "user", "content": NUDGE_NO_TOOLS})
            continue

        answer = parse_final_answer(content)
        if answer is None and not nudged_for_json:
            nudged_for_json = True
            event("nudge", f"[step {step}] final answer was not valid JSON -> asked again")
            messages.append({"role": "user", "content": NUDGE_BAD_JSON})
            continue

        result.raw_answer = content
        if answer is None:
            result.stopped_reason = "bad_final_answer"
            event("stop", f"[step {step}] stopped: model never produced valid JSON")
        else:
            result.answer = clean_findings(answer, result.warnings)
            result.stopped_reason = "answered"
            event("final", f"[step {step}] model finished with a final answer")
        return result

    result.stopped_reason = "max_steps"
    event("stop", f"stopped: reached the limit of {max_steps} steps without a final answer")
    return result


def _short(observation: dict, limit: int = 220) -> str:
    text = json.dumps(observation)
    return text if len(text) <= limit else text[:limit] + " ..."
