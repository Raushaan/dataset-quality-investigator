"""Tools must find exactly the problems planted in data/students_messy.csv
(listed in data/KNOWN_PROBLEMS.md) and nothing in the clean dataset."""

from dqi.tools import run_tool
from dqi.verify import file_fingerprint

MESSY = "data/students_messy.csv"
CLEAN = "data/students_clean.csv"
NUMERIC = ["age", "gpa", "credits_completed", "enrollment_year"]


def result(name, args, path=MESSY):
    out = run_tool(name, args, path)
    assert out["ok"], out
    return out["result"]


def test_missing_values_found_in_messy():
    r = result("check_missing_values", {})
    found = {col: info["row_index"] for col, info in r["columns_with_missing"].items()}
    assert found == {"age": [2], "gpa": [6], "major": [10]}


def test_unusual_values_found_in_messy():
    assert [u["row_index"] for u in result("check_unusual_values", {"column": "age"})["unusual"]] == [4]
    assert [u["row_index"] for u in result("check_unusual_values", {"column": "gpa"})["unusual"]] == [13]
    assert result("check_unusual_values", {"column": "credits_completed"})["unusual"] == []


def test_duplicates_found_in_messy():
    assert result("check_duplicates", {})["duplicate_groups"] == [[1, 20]]
    assert result("check_duplicates", {"columns": ["student_id"]})["duplicate_groups"] == [[1, 20], [8, 21]]


def test_clean_dataset_has_no_problems():
    assert result("check_missing_values", {}, CLEAN)["total_missing_cells"] == 0
    for col in NUMERIC:
        assert result("check_unusual_values", {"column": col}, CLEAN)["unusual_count"] == 0
    assert result("check_duplicates", {"columns": ["student_id"]}, CLEAN)["duplicate_groups"] == []


def test_unknown_column_returns_error_not_crash():
    out = run_tool("check_unusual_values", {"column": "salary"}, MESSY)
    assert not out["ok"] and "Available columns" in out["error"]


def test_text_column_rejected_for_outlier_check():
    out = run_tool("check_unusual_values", {"column": "name"}, MESSY)
    assert not out["ok"] and "Numeric columns" in out["error"]


def test_invalid_tool_and_arguments_rejected():
    assert not run_tool("delete_rows", {}, MESSY)["ok"]
    assert not run_tool("check_unusual_values", {}, MESSY)["ok"]          # missing column
    assert not run_tool("check_missing_values", {"rows": [1]}, MESSY)["ok"]  # unknown argument


def test_tools_never_modify_the_file():
    before = file_fingerprint(MESSY)
    for name, args in [("get_dataset_overview", {}), ("check_missing_values", {}),
                       ("check_unusual_values", {"column": "age"}), ("check_duplicates", {})]:
        run_tool(name, args, MESSY)
    assert file_fingerprint(MESSY) == before
