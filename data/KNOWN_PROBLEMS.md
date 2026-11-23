# Known problems planted in `students_messy.csv`

`students_messy.csv` is `students_clean.csv` (20 students) with these deliberate problems.
`row_index` is the 0-based pandas index; the CSV line number is `row_index + 2`.

| # | Check | Column | row_index | CSV line | What was changed |
|---|-------|--------|-----------|----------|------------------|
| 1 | missing_values | age | 2 | 4 | age left empty (S1003) |
| 2 | missing_values | gpa | 6 | 8 | gpa left empty (S1007) |
| 3 | missing_values | major | 10 | 12 | major left empty (S1011) |
| 4 | unusual_values | age | 4 | 6 | age typed as 210 instead of 21 (S1005) |
| 5 | unusual_values | gpa | 13 | 15 | gpa 7.8 on a 4.0 scale (S1014) |
| 6 | duplicates (whole row) | all columns | 1 and 20 | 3 and 22 | row 1 (S1002) copied to the end |
| 7 | duplicates (ID) | student_id | 8 and 21 | 10 and 23 | ID S1009 reused for a different student |

`check_duplicates()` with no arguments reports both kinds: identical rows [1, 20] and the repeated ID [8, 21].
Exact copies are counted once (as identical rows), so rows 1 and 20 are not reported again as a repeated ID.
A complete, correct answer therefore contains 9 (check, column, row_index) entries.

`students_clean.csv` has none of these problems: every check returns nothing.
Check this yourself without the agent: `python -m dqi.verify`
