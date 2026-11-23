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


ALL_CALLS = [
    {"name": "check_missing_values", "args": {}, "result": {"ok": True}},
    *[{"name": "check_unusual_values", "args": {"column": c}, "result": {"ok": True}}
      for c in ["age", "gpa", "credits_completed", "enrollment_year"]],
    {"name": "check_duplicates", "args": {}, "result": {"ok": True}},
    {"name": "check_duplicates", "args": {"columns": ["student_id"]}, "result": {"ok": True}},
]
ALL_FINDINGS = [
    {"check": "missing_values", "column": "age", "rows": [2]},
    {"check": "missing_values", "column": "gpa", "rows": [6]},
    {"check": "missing_values", "column": "major", "rows": [10]},
    {"check": "unusual_values", "column": "age", "rows": [4]},
    {"check": "unusual_values", "column": "gpa", "rows": [13]},
    {"check": "duplicates", "column": None, "rows": [1, 20]},
    {"check": "duplicates", "column": "student_id", "rows": [1, 20, 8, 21]},
]


def test_full_scan_flags_skipped_checks():
    calls = [ALL_CALLS[0], ALL_CALLS[5]]           # only missing values + identical rows
    a = answer([f for f in ALL_FINDINGS if f["check"] == "missing_values"]
               + [ALL_FINDINGS[5]])
    report = verify(MESSY, a, calls, file_fingerprint(MESSY), full_scan=True)
    assert not report["passed"]
    assert ("unusual_values", "age") in report["not_checked"]
    assert ("duplicates", "student_id") in report["not_checked"]
    assert ("unusual_values", "age", 4) in report["missed_problems"]


def test_full_scan_passes_when_everything_checked():
    report = verify(MESSY, answer(ALL_FINDINGS), ALL_CALLS, file_fingerprint(MESSY), full_scan=True)
    assert report["passed"] and report["not_checked"] == [] and len(report["confirmed"]) == 11
