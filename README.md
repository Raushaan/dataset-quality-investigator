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
| `get_dataset_overview` | rows, columns, column types, first rows |
| `check_missing_values(columns?)` | empty cells per column, with row indexes |
| `check_unusual_values(column)` | outliers in one numeric column (IQR rule) |
| `check_duplicates(columns?)` | identical rows, or repeated values in an ID column |

All tools load a fresh copy of the CSV and only read it.

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
   ollama pull qwen3:8b
   ```
   On a Mac with 8 GB of memory, `qwen3:4b` is faster: `ollama pull qwen3:4b`
   and add `--model qwen3:4b` to the commands below.
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
python main.py --data data/students_messy.csv --question "Is this dataset clean?"
python main.py --data data/students_clean.csv --question "Is this dataset clean?"

# run every demo scenario (success + problem cases)
python demo.py
```

Each run prints the model's decisions, tool calls, tool results, the final answer and a
verification verdict, and saves the full trace as JSON (`runs/` or `demo_runs/`).

Optional settings: `DQI_MODEL` (model name), `DQI_CONTEXT` (context size, default 8192),
`OLLAMA_URL` (default `http://localhost:11434`).

## Demo scenarios (`demo.py`)

| # | Scenario | Expected behaviour |
|---|----------|--------------------|
| 1 | Messy data, "Is this dataset clean?" | finds all 7 planted problems; verification PASS |
| 2 | Clean data, same question | reports no problems; verification PASS |
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

The verdict is PASS only if nothing is unsupported or missed and the file is unchanged.

## Limitations

- Unusual values use the IQR rule only; small or skewed columns can produce false alarms.
- A small local model can still leave something out of its final answer; the verifier reports
  that as MISSED instead of hiding it.
- Large files are not the goal: tool results are capped at 25 items each.
