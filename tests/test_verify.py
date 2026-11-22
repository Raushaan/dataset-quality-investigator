"""The verifier must catch hallucinated findings and missed problems."""

from dqi.verify import file_fingerprint, verify

MESSY = "data/students_messy.csv"
CALLS = [{"name": "check_missing_values", "args": {}, "result": {"ok": True}}]


def answer(findings):
    return {"summary": "test", "findings": findings, "could_not_answer": None}


def test_correct_answer_passes():
    a = answer([{"check": "missing_values", "column": "age", "rows": [2]},
                {"check": "missing_values", "column": "gpa", "rows": [6]},
                {"check": "missing_values", "column": "major", "rows": [10]}])
    report = verify(MESSY, a, CALLS, file_fingerprint(MESSY))
    assert report["passed"] and len(report["confirmed"]) == 3


def test_hallucinated_row_is_caught():
    a = answer([{"check": "missing_values", "column": "age", "rows": [2, 5]},
                {"check": "missing_values", "column": "gpa", "rows": [6]},
                {"check": "missing_values", "column": "major", "rows": [10]}])
    report = verify(MESSY, a, CALLS, file_fingerprint(MESSY))
    assert not report["passed"]
    assert ("missing_values", "age", 5) in report["unsupported_claims"]


def test_missed_problem_is_caught():
    a = answer([{"check": "missing_values", "column": "age", "rows": [2]}])
    report = verify(MESSY, a, CALLS, file_fingerprint(MESSY))
    assert not report["passed"] and len(report["missed_problems"]) == 2


def test_claim_without_any_tool_call_is_unsupported():
    a = answer([{"check": "duplicates", "column": None, "rows": [1, 20]}])
    report = verify(MESSY, a, [], file_fingerprint(MESSY))
    assert not report["passed"] and len(report["unsupported_claims"]) == 2


def test_modified_file_fails():
    a = answer([])
    report = verify(MESSY, a, [], "not-the-real-hash")
    assert not report["dataset_unchanged"] and not report["passed"]
