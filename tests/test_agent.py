"""Agent-loop tests with a scripted fake model (no Ollama needed).

They prove the loop: runs the tool the model chooses, feeds the observation
back, recovers from bad tool calls and bad answers, and stops safely."""

import json

from dqi.agent import run_agent

MESSY = "data/students_messy.csv"


class FakeLLM:
    """Returns pre-written assistant messages in order and records what it saw."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.json_mode_calls = []

    def chat(self, messages, tools=None, json_mode=False):
        self.seen.append([dict(m) for m in messages])
        self.json_mode_calls.append(json_mode)
        return self.replies.pop(0)


def tool(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def finish(findings, could_not=None):
    """Final answer given the preferred way: through the final_answer tool."""
    return tool("final_answer", summary="done", findings=findings, could_not_answer=could_not)


def final(findings, could_not=None):
    """Final answer given as plain-text JSON (fallback path)."""
    return {"role": "assistant", "content": json.dumps(
        {"summary": "done", "findings": findings, "could_not_answer": could_not})}


def quiet(*_):
    pass


def test_tool_result_is_fed_back_to_model():
    llm = FakeLLM([tool("check_duplicates", columns=["student_id"]),
                   finish([{"check": "duplicates", "column": "student_id", "rows": [1, 20, 8, 21]}])])
    result = run_agent("Any repeated students?", MESSY, llm, log=quiet)
    assert result.stopped_reason == "answered"
    assert result.tool_calls[0]["name"] == "check_duplicates"
    last_seen = llm.seen[1][-1]                      # what the model saw on step 2
    assert last_seen["role"] == "tool" and "[8,21]" in last_seen["content"]
    assert result.answer["findings"][0]["rows"] == [1, 20, 8, 21]


def test_plain_text_json_answer_still_accepted():
    llm = FakeLLM([tool("check_duplicates"),
                   final([{"check": "duplicates", "column": None, "rows": [1, 20]}])])
    result = run_agent("duplicates?", MESSY, llm, log=quiet)
    assert result.stopped_reason == "answered" and result.answer["findings"][0]["rows"] == [1, 20]


def test_answer_stuffed_into_another_tool_is_recovered():
    llm = FakeLLM([tool("check_duplicates"),
                   tool("get_dataset_overview", summary="dups", findings=[
                       {"check": "duplicates", "column": None, "rows": [1, 20]}])])
    result = run_agent("duplicates?", MESSY, llm, log=quiet)
    assert result.stopped_reason == "answered"
    assert any(e["type"] == "note" for e in result.trace)
    assert len(result.tool_calls) == 1               # the stuffed call was not run as a check


def test_final_answer_sent_with_new_checks_is_postponed():
    both = {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "check_duplicates", "arguments": {}}},
        {"function": {"name": "final_answer", "arguments": {"summary": "x", "findings": []}}}]}
    llm = FakeLLM([both, finish([{"check": "duplicates", "column": None, "rows": [1, 20]}])])
    result = run_agent("duplicates?", MESSY, llm, log=quiet)
    assert result.answer["findings"][0]["rows"] == [1, 20]


def test_bad_tool_call_becomes_error_observation():
    llm = FakeLLM([tool("check_unusual_values", column="salary"),
                   finish([], could_not="There is no salary column.")])
    result = run_agent("Check salary outliers", MESSY, llm, log=quiet)
    assert result.tool_calls[0]["result"]["ok"] is False
    assert "Available columns" in llm.seen[1][-1]["content"]
    assert result.answer["could_not_answer"]


def test_answer_without_evidence_is_nudged_once():
    llm = FakeLLM([finish([]), tool("check_missing_values"), finish([])])
    result = run_agent("Is it clean?", MESSY, llm, log=quiet)
    assert any(e["type"] == "nudge" for e in result.trace)
    assert len(result.tool_calls) == 1


def test_bad_answers_retried_then_strict_json_mode():
    llm = FakeLLM([tool("check_duplicates"),
                   {"role": "assistant", "content": "There are duplicates."},
                   {"role": "assistant", "content": "Still prose."},
                   final([{"check": "duplicates", "column": None, "rows": [1, 20]}])])
    result = run_agent("duplicates?", MESSY, llm, log=quiet)
    assert result.stopped_reason == "answered"
    assert llm.json_mode_calls == [False, False, False, True]  # last try used strict JSON


def test_gives_up_after_three_bad_answers():
    prose = {"role": "assistant", "content": "no json here"}
    llm = FakeLLM([tool("check_duplicates"), prose, prose, prose])
    result = run_agent("duplicates?", MESSY, llm, log=quiet)
    assert result.stopped_reason == "bad_final_answer" and result.answer is None


def test_repeated_tool_call_is_blocked():
    llm = FakeLLM([tool("get_dataset_overview"), tool("get_dataset_overview"), finish([])])
    result = run_agent("overview twice", MESSY, llm, log=quiet)
    first, second = result.tool_calls
    assert first["result"]["ok"] and not second["result"]["ok"]
    assert "already ran" in second["result"]["error"]


def test_stops_at_max_steps():
    llm = FakeLLM([tool("check_unusual_values", column=c) for c in ["age", "gpa", "credits_completed"]])
    result = run_agent("loop forever", MESSY, llm, max_steps=3, log=quiet)
    assert result.stopped_reason == "max_steps" and result.answer is None
