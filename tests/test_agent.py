"""Agent-loop tests with a scripted fake model (no Ollama needed).

They prove the loop: runs the tool the model chooses, feeds the observation
back, recovers from bad tool calls, and stops safely."""

import json

from dqi.agent import run_agent

MESSY = "data/students_messy.csv"


class FakeLLM:
    """Returns pre-written assistant messages in order and records what it saw."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)


def tool(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def final(findings, could_not=None):
    return {"role": "assistant", "content": json.dumps(
        {"summary": "done", "findings": findings, "could_not_answer": could_not})}


def quiet(*_):
    pass


def test_tool_result_is_fed_back_to_model():
    llm = FakeLLM([tool("check_duplicates", columns=["student_id"]),
                   final([{"check": "duplicates", "column": "student_id", "rows": [1, 20, 8, 21]}])])
    result = run_agent("Any repeated students?", MESSY, llm, log=quiet)
    assert result.stopped_reason == "answered"
    assert result.tool_calls[0]["name"] == "check_duplicates"
    last_seen = llm.seen[1][-1]                      # what the model saw on step 2
    assert last_seen["role"] == "tool" and "[8, 21]" in last_seen["content"]


def test_bad_tool_call_becomes_error_observation():
    llm = FakeLLM([tool("check_unusual_values", column="salary"),
                   final([], could_not="There is no salary column.")])
    result = run_agent("Check salary outliers", MESSY, llm, log=quiet)
    assert result.tool_calls[0]["result"]["ok"] is False
    assert "Available columns" in llm.seen[1][-1]["content"]
    assert result.answer["could_not_answer"]


def test_answer_without_evidence_is_nudged_once():
    llm = FakeLLM([final([]), tool("check_missing_values"), final([])])
    result = run_agent("Is it clean?", MESSY, llm, log=quiet)
    assert any(e["type"] == "nudge" for e in result.trace)
    assert len(result.tool_calls) == 1


def test_invalid_json_is_retried_then_accepted():
    llm = FakeLLM([tool("check_duplicates"),
                   {"role": "assistant", "content": "There are duplicates."},
                   final([{"check": "duplicates", "column": None, "rows": [1, 20]}])])
    result = run_agent("duplicates?", MESSY, llm, log=quiet)
    assert result.stopped_reason == "answered"
    assert result.answer["findings"][0]["rows"] == [1, 20]


def test_stops_at_max_steps():
    llm = FakeLLM([tool("get_dataset_overview")] * 3)
    result = run_agent("loop forever", MESSY, llm, max_steps=3, log=quiet)
    assert result.stopped_reason == "max_steps" and result.answer is None
