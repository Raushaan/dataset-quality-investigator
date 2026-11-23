# Dataset Quality Investigator

A small, reliable AI agent that answers questions about the quality of a CSV dataset.
A local language model (Qwen3 via Ollama) decides which check to run — **missing values**,
**unusual values**, or **duplicate records** — reads each result, decides what to do next,
and finally reports the specific problems it found. It **never modifies the data**.
Every answer is then checked against independent pandas calculations.

## How it works

```
question ──► model decides ──► tool call ──► tool result (observation)
                 ▲                                   │
                 └──────────── fed back ◄────────────┘
          ... until the model replies with a final JSON report
                               │
                               ▼
        verifier recomputes each check directly in pandas,
        compares row by row, and confirms the file is unchanged
```

| Tool | What it does |
|------|--------------|
| `get_dataset_overview` | rows, columns, numeric and ID columns, first rows |
| `check_missing_values(columns?)` | empty cells per column, with row indexes |
| `check_unusual_values(column)` | outliers in one numeric column (IQR rule) |
| `check_duplicates(columns?)` | identical rows, or repeated values in an ID column |

The model finishes by calling a fifth tool, `final_answer(summary, findings, could_not_answer)`.
All check tools load a fresh copy of the CSV and only read it.

## Project structure

```
dqi/tools.py      the four read-only tools + argument validation
dqi/llm.py        small Ollama client
dqi/agent.py      agent loop: decide -> act -> observe -> decide
dqi/verify.py     independent pandas checks + comparison + file hash
main.py           command-line entry point (prints trace, answer, verification)
demo.py           runs all demo scenarios, saves traces to demo_runs/
data/             clean dataset, messy dataset, KNOWN_PROBLEMS.md (ground truth)
tests/            pytest tests (no Ollama needed)
```

## Setup (macOS)

1. Install Ollama from https://ollama.com/download (or `brew install ollama`) and open it.
2. Download the model:
   ```bash
   ollama pull qwen3:4b
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Run

```bash
# tests (no model needed)
pytest -v

# ground truth for both datasets, computed directly in pandas
python -m dqi.verify

# ask the agent a question
python main.py --data data/students_messy.csv --question "Is this dataset clean?" --full-scan
python main.py --data data/students_clean.csv --question "Is this dataset clean?" --full-scan

# run every demo scenario (success + problem cases)
python demo.py
```

Each run prints the model's decisions, tool calls, tool results, the final answer and a
verification verdict, and saves the full trace as JSON (`runs/` or `demo_runs/`).

Optional settings: `DQI_MODEL` (model name, default `qwen3:4b`), `DQI_CONTEXT` (context size, default 6144),
`DQI_THINK=0` (turn off Qwen3 thinking: faster, but less reliable tool choices),
`OLLAMA_URL` (default `http://localhost:11434`).

## Demo scenarios (`demo.py`)

| # | Scenario | Expected behaviour |
|---|----------|--------------------|
| 1 | Messy data, "Is this dataset clean?" (full scan) | finds all 7 planted problems; verification PASS |
| 2 | Clean data, same question (full scan) | reports no problems; verification PASS |
| 3 | "Are any students listed more than once?" | chooses only the duplicate check |
| 4 | "Check the salary column..." | tool error -> explains the column does not exist |
| 5 | "Delete the duplicate rows and save..." | refuses to modify, reports the duplicates |
| 6 | File does not exist | stopped before the agent starts, clear error |

## Verification

`dqi/verify.py` turns every problem into a `(check, column, row_index)` triple, recomputes the
checks the agent ran with plain pandas, and reports:

- **confirmed** – claimed by the agent and present in the evidence
- **unsupported claims** – claimed by the agent but not in the evidence (hallucination)
- **missed problems** – present in the evidence but left out of the answer
- **dataset unchanged** – SHA-256 of the file before and after the run

With `--full-scan` (used for general questions such as "Is this dataset clean?"), the verifier also
requires **coverage**: missing values, unusual values in every numeric column, identical rows, and
repeated IDs (any column named `id` or ending in `_id`). A check the agent skipped is reported as
**NOT CHECKED**, and the problems it would have found count as missed.

The verdict is PASS only if nothing is unsupported, missed or not checked, and the file is unchanged.

## Limitations

- Unusual values use the IQR rule only; small or skewed columns can produce false alarms.
- A small local model can still leave something out of its final answer; the verifier reports
  that as MISSED instead of hiding it.
- Large files are not the goal: tool results are capped at 25 items each.
