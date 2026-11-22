"""Read-only data-quality tools.

Every tool receives a pandas DataFrame that was freshly loaded from the CSV
file, inspects it, and returns a JSON-friendly dict. No tool ever writes to
disk, so the agent can only REPORT problems, never change the data.

Rows are identified by `row_index`: the 0-based pandas index.
(row_index 0 is the first data row, which is line 2 of the CSV file.)
"""

from __future__ import annotations

import json

import pandas as pd

MAX_ITEMS = 25  # cap long lists so tool results stay small for the model


class ToolError(Exception):
    """A problem the model should read and react to (e.g. unknown column)."""


# ---------------------------------------------------------------- helpers
def load_csv(path: str) -> pd.DataFrame:
    """Load a fresh, read-only copy of the dataset."""
    return pd.read_csv(path)


def _row_list(index) -> list[int]:
    return [int(i) for i in index]


def _plain(value):
    """Turn numpy/pandas scalars into plain Python values for JSON."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    unknown = [c for c in columns if c not in df.columns]
    if unknown:
        raise ToolError(
            f"Unknown column(s): {unknown}. Available columns: {list(df.columns)}"
        )


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------- tools
def get_dataset_overview(df: pd.DataFrame) -> dict:
    """Shape, column names, types and a small preview of the data."""
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "numeric_columns": _numeric_columns(df),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "first_rows": json.loads(df.head(3).to_json(orient="records")),
    }


def check_missing_values(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Find empty cells (NaN or blank text) in the given columns (default: all)."""
    cols = columns or list(df.columns)
    _require_columns(df, cols)

    per_column = {}
    for col in cols:
        blank_text = df[col].astype(str).str.strip() == ""
        mask = df[col].isna() | blank_text
        if mask.any():
            rows = _row_list(df.index[mask])
            per_column[col] = {"count": len(rows), "row_index": rows[:MAX_ITEMS]}

    return {
        "check": "missing_values",
        "columns_checked": cols,
        "total_missing_cells": sum(v["count"] for v in per_column.values()),
        "columns_with_missing": per_column,
    }


def check_unusual_values(df: pd.DataFrame, column: str) -> dict:
    """Flag outliers in ONE numeric column using the IQR rule (1.5 x IQR)."""
    _require_columns(df, [column])
    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        values = series
        non_numeric = pd.Series(False, index=series.index)
    else:
        values = pd.to_numeric(series, errors="coerce")
        non_numeric = series.notna() & values.isna()
        # A real text column (names, majors) is not a numeric column with typos.
        if values.notna().sum() < 0.8 * series.notna().sum():
            raise ToolError(
                f"Column '{column}' contains text, so the unusual-values check "
                f"cannot run on it. Numeric columns: {_numeric_columns(df)}"
            )

    present = values.dropna()
    if len(present) < 4:
        raise ToolError(f"Column '{column}' has fewer than 4 numeric values.")

    q1, q3 = present.quantile(0.25), present.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (values < low) | (values > high)

    unusual = [
        {"row_index": int(i), "value": _plain(values[i])} for i in df.index[mask]
    ]
    text_entries = [
        {"row_index": int(i), "value": str(series[i])} for i in df.index[non_numeric]
    ]
    return {
        "check": "unusual_values",
        "column": column,
        "method": "IQR rule: value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR",
        "q1": round(float(q1), 3),
        "q3": round(float(q3), 3),
        "lower_bound": round(float(low), 3),
        "upper_bound": round(float(high), 3),
        "unusual_count": len(unusual),
        "unusual": unusual[:MAX_ITEMS],
        "non_numeric_entries": text_entries[:MAX_ITEMS],
    }


def check_duplicates(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Find repeated rows. With `columns` (e.g. an ID), compare only those columns."""
    if columns:
        _require_columns(df, columns)
    key_cols = columns or list(df.columns)

    mask = df.duplicated(subset=key_cols, keep=False)
    groups = [
        _row_list(group.index)
        for _, group in df[mask].groupby(key_cols, dropna=False, sort=False)
    ]
    return {
        "check": "duplicates",
        "compared_columns": "all columns" if not columns else key_cols,
        "duplicate_group_count": len(groups),
        "extra_copies": int(df.duplicated(subset=key_cols, keep="first").sum()),
        "duplicate_groups": groups[:MAX_ITEMS],
    }


# ---------------------------------------------------------------- registry
TOOL_FUNCTIONS = {
    "get_dataset_overview": get_dataset_overview,
    "check_missing_values": check_missing_values,
    "check_unusual_values": check_unusual_values,
    "check_duplicates": check_duplicates,
}

# Allowed arguments per tool: name -> (type, required)
TOOL_ARGS = {
    "get_dataset_overview": {},
    "check_missing_values": {"columns": (list, False)},
    "check_unusual_values": {"column": (str, True)},
    "check_duplicates": {"columns": (list, False)},
}

# Tool descriptions the language model sees (Ollama / OpenAI function format)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_dataset_overview",
            "description": "Get the row count, column names, column types and first rows. "
            "Use this first when you do not yet know the columns.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_missing_values",
            "description": "Find empty cells. Omit 'columns' to check every column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to check (optional).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_unusual_values",
            "description": "Find outliers in ONE numeric column using the IQR rule. "
            "Call it once per numeric column you want to check.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "A numeric column."}
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_duplicates",
            "description": "Find duplicate records. Omit 'columns' to find fully identical "
            "rows; pass an ID column (e.g. ['student_id']) to find repeated IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns that should be unique (optional).",
                    }
                },
                "required": [],
            },
        },
    },
]


def validate_args(name: str, args) -> dict:
    """Check a tool call from the model before running it. Raises ToolError."""
    if name not in TOOL_FUNCTIONS:
        raise ToolError(f"Unknown tool '{name}'. Available tools: {list(TOOL_FUNCTIONS)}")
    if args is None:
        args = {}
    if isinstance(args, str):  # some models send arguments as a JSON string
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as exc:
            raise ToolError(f"Arguments for '{name}' are not valid JSON: {exc}") from exc
    if not isinstance(args, dict):
        raise ToolError(f"Arguments for '{name}' must be an object.")

    allowed = TOOL_ARGS[name]
    clean = {}
    for key, value in args.items():
        if key not in allowed:
            raise ToolError(f"'{name}' does not accept '{key}'. Allowed: {list(allowed) or 'none'}")
        expected, _ = allowed[key]
        if value is None:
            continue
        if expected is list and isinstance(value, str):
            value = [value]  # forgive "age" instead of ["age"]
        if not isinstance(value, expected):
            raise ToolError(f"'{key}' for '{name}' must be a {expected.__name__}.")
        if expected is list and not all(isinstance(v, str) for v in value):
            raise ToolError(f"'{key}' for '{name}' must be a list of column names.")
        clean[key] = value
    for key, (_, required) in allowed.items():
        if required and key not in clean:
            raise ToolError(f"'{name}' needs the argument '{key}'.")
    return clean


def run_tool(name: str, args, csv_path: str) -> dict:
    """Validate, run one tool on a fresh copy of the CSV, and wrap the result."""
    try:
        clean_args = validate_args(name, args)
        df = load_csv(csv_path)
        return {"ok": True, "result": TOOL_FUNCTIONS[name](df, **clean_args)}
    except ToolError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # unexpected failure: report it, don't crash the agent
        return {"ok": False, "error": f"Tool '{name}' failed: {type(exc).__name__}: {exc}"}
