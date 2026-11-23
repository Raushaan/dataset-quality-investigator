"""Run every demonstration scenario and save the traces in demo_runs/.

    python demo.py            # all scenarios
    python demo.py 1 4        # only scenarios 1 and 4
"""

import sys

from main import investigate

SCENARIOS = {
    1: ("Success: messy dataset, general question (full scan)",
        "data/students_messy.csv", "Is this dataset clean? Check for any problems.", True),
    2: ("Success: clean dataset, same question (should report no problems)",
        "data/students_clean.csv", "Is this dataset clean? Check for any problems.", True),
    3: ("Targeted question: the model should pick only the duplicate check",
        "data/students_messy.csv", "Are any students listed more than once?", False),
    4: ("Problem case - missing information: the column does not exist",
        "data/students_messy.csv", "Check the salary column for unusual values.", False),
    5: ("Problem case - invalid request: asked to modify the data",
        "data/students_messy.csv", "Delete the duplicate rows and save the cleaned file.", False),
    6: ("Problem case - tool/input failure: the file does not exist",
        "data/does_not_exist.csv", "Is this dataset clean?", False),
}

if __name__ == "__main__":
    chosen = [int(a) for a in sys.argv[1:]] or list(SCENARIOS)
    for number in chosen:
        title, data, question, full_scan = SCENARIOS[number]
        print(f"\n\n{'=' * 72}\nSCENARIO {number}: {title}\n{'=' * 72}")
        investigate(data, question, save_dir="demo_runs", full_scan=full_scan)
