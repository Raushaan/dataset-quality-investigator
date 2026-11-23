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
1. Decide which checks the question needs.
   - A GENERAL question ("Is this dataset clean?", "Check for any problems") needs the FULL set:
     check_missing_values once; check_unusual_values once for EACH numeric column listed by
     get_dataset_overview; check_duplicates with no arguments; and check_duplicates with the
     ID column (for example ["student_id"]) if there is one. Do not give the final answer
     until every one of these has been run.
   - A SPECIFIC question needs only the checks it asks about.
2. Never run the same tool with the same arguments twice. Earlier results stay above in the
   conversation; reuse them.
3. Read every tool result before choosing the next step. If a tool returns an error, adapt \
(for example use a column that exists) or explain the problem to the user.
4. Report only what tool results show. Never invent rows, values or columns. Every problem a \
tool found must appear in your findings.
5. If the user asks you to modify, fix, delete or save data, do not try: say you only report \
problems, and report what you found if it is useful.
6. Identify rows by row_index exactly as the tools return it. For duplicates, list every row_index \
in each duplicate group (the original and its copies).
7. Use only column names that get_dataset_overview returned. Never guess column names.

When you are finished, call the final_answer tool with:
  summary: 2-4 plain sentences for the user
  findings: a list of {"check": "missing_values" | "unusual_values" | "duplicates",
                       "column": "<column name; null for identical-row duplicates>",
                       "rows": [<row_index>, ...],
                       "detail": "<short explanation with the values>"}
  could_not_answer: what you could not do and why, or null
Use an empty findings list when the checks you ran found no problems."""

NUDGE_NO_TOOLS = (
    "You have not inspected the data yet. Call a tool, or, if the request cannot be "
    "handled by your tools, reply with the final JSON and explain in could_not_answer."
)
NUDGE_BAD_JSON = (
    "Your last reply was not a final answer. Call the final_answer tool now with summary, "
    "findings and could_not_answer (or reply with ONLY that JSON object)."
)
NUDGE_JSON_ONLY = (
    "Reply with ONLY a JSON object with the keys summary, findings and could_not_answer, "
    "based on the tool results above."
)

FINAL_TOOL = "final_answer"
FINAL_ANSWER_SCHEMA = {
    "type": "function",
    "function": {
        "name": FINAL_TOOL,
        "description": "Give your final report to the user. Call this once, after the checks "
                       "the question needs.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "2-4 plain sentences."},
                "findings": {
                    "type": "array",
                    "description": "One item per problem found by a tool.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "check": {"type": "string",
                                      "enum": ["missing_values", "unusual_values", "duplicates"]},
                            "column": {"type": ["string", "null"]},
                            "rows": {"type": "array", "items": {"type": "integer"}},
                            "detail": {"type": "string"},
                        },
                        "required": ["check", "rows"],
                    },
                },
                "could_not_answer": {"type": ["string", "null"],
                                     "description": "What you could not do, or null."},
            },
            "required": ["summary", "findings"],
        },
    },
}
AGENT_TOOLS = TOOL_SCHEMAS + [FINAL_ANSWER_SCHEMA]


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
    return answer_from_args(data)


def answer_from_args(args) -> dict | None:
    """Build an answer dict from final_answer arguments (or a parsed JSON reply)."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict) or not ({"summary", "findings"} & set(args)):
        return None
    findings = args.get("findings") or []
    if isinstance(findings, str):  # some models send the list as a JSON string
        try:
            findings = json.loads(findings)
        except json.JSONDecodeError:
            findings = []
    return {"summary": str(args.get("summary") or ""),
            "findings": findings if isinstance(findings, list) else [],
            "could_not_answer": args.get("could_not_answer")}


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


def run_agent(question: str, csv_path: str, llm, max_steps: int = 15,
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
    nudged_for_tools = False
    format_nudges = 0           # times we asked for a proper final answer (max 2)
    force_json = False          # last resort: Ollama strict JSON mode, no tools
    seen_calls = {}             # (tool, args) -> step it was first run

    for step in range(1, max_steps + 1):
        if force_json:
            force_json = False
            message = llm.chat(messages, json_mode=True)
        else:
            message = llm.chat(messages, tools=AGENT_TOOLS)
        calls = message.get("tool_calls") or []
        content = message.get("content") or ""
        messages.append({"role": "assistant", "content": content,
                         **({"tool_calls": calls} if calls else {})})

        # ---- run the tools the model chose; set aside any final answer
        final_args, ran_checks = None, False
        for call in calls:
            fn = call.get("function", {})
            name, args = fn.get("name", "?"), fn.get("arguments") or {}
            stuffed = (name != FINAL_TOOL and isinstance(args, dict)
                       and {"summary", "findings"} & set(args))
            if name == FINAL_TOOL or stuffed:
                if stuffed:
                    event("note", f"[step {step}] model put its final answer inside {name}(...) "
                                  "-> treated as final_answer", step=step)
                final_args = args
                messages.append({"role": "tool", "tool_name": name,
                                 "content": '{"ok": true, "result": "final answer received"}'})
                continue

            ran_checks = True
            event("decision", f"[step {step}] model chose tool: {name}({json.dumps(args)})",
                  step=step, tool=name, args=args)
            key = f"{name} {json.dumps(args, sort_keys=True)}"
            if key in seen_calls:
                observation = {"ok": False, "error": (
                    f"You already ran {name} with these arguments at step {seen_calls[key]}; "
                    "its result is above. Run a different check or call final_answer.")}
            else:
                seen_calls[key] = step
                observation = run_tool(name, args, csv_path)
            result.tool_calls.append(
                {"step": step, "name": name, "args": args if isinstance(args, dict) else {},
                 "result": observation})
            event("observation", f"[step {step}] result: {_short(observation)}",
                  step=step, tool=name, result=observation)
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(observation, separators=(",", ":"))})

        if ran_checks:
            if final_args is not None:  # answered before reading these new results
                event("nudge", f"[step {step}] final answer sent together with new checks "
                               "-> asked to read the results first", step=step)
                messages.append({"role": "user", "content":
                                 "Read the new tool results above, then call final_answer again."})
            continue

        # ---- the model wants to finish (final_answer call or plain text)
        if not result.tool_calls and not nudged_for_tools:
            nudged_for_tools = True
            event("nudge", f"[step {step}] model tried to answer without evidence -> asked to use a tool")
            messages.append({"role": "user", "content": NUDGE_NO_TOOLS})
            continue

        answer = answer_from_args(final_args) if final_args is not None \
            else parse_final_answer(content)
        if answer is None and format_nudges < 2:
            format_nudges += 1
            force_json = format_nudges == 2
            mode = " (strict JSON mode)" if force_json else ""
            event("nudge", f"[step {step}] no usable final answer -> asked again{mode}")
            messages.append({"role": "user",
                             "content": NUDGE_JSON_ONLY if force_json else NUDGE_BAD_JSON})
            continue

        result.raw_answer = content or json.dumps(final_args)
        if answer is None:
            result.stopped_reason = "bad_final_answer"
            event("stop", f"[step {step}] stopped: model never produced a usable final answer")
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
