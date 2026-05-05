"""Long ↔ wide parquet helpers for motif sweeps.

Canonical long-format schema (PROJECT_BRIEF.md §6):

    size: str       e.g. "70m", "160m", "410m"
    step: int       training step
    layer: int
    head: int
    motif: str      e.g. "induction_prefix_match", "copy_suppression_qk", "copy_suppression_ov"
    score: float

The same parquet can hold rows for any number of motifs simultaneously, which
is what Phase 2's `motif_sweep.parquet` will look like. Wide-format views are
derivable on demand via `to_wide`; persisting both forms is fine for
preview-stage convenience but the long-format file is the source of truth.

Helpers:
    write_long(rows, path)           append-or-replace long-format parquet
    read_long(path)                  read long-format parquet
    to_wide(long_df, *, index_cols)  pivot long → wide DataFrame
    write_wide(long_df, path)        derive + write wide parquet from long
    read_wide(path, *, index_cols)   read long, return wide pivot
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import pandas as pd

LONG_COLUMNS: tuple[str, ...] = ("size", "step", "layer", "head", "motif", "score")
DEFAULT_INDEX_COLS: tuple[str, ...] = ("size", "step", "layer", "head")


@dataclass(frozen=True)
class SweepRow:
    size: str
    step: int
    layer: int
    head: int
    motif: str
    score: float


def _coerce_rows(rows: Iterable) -> pd.DataFrame:
    materialized = list(rows)
    if not materialized:
        return pd.DataFrame(columns=list(LONG_COLUMNS))
    sample = materialized[0]
    if isinstance(sample, dict):
        df = pd.DataFrame(materialized)
    elif is_dataclass(sample):
        df = pd.DataFrame([asdict(r) for r in materialized])
    elif isinstance(sample, pd.DataFrame):
        df = pd.concat(materialized, ignore_index=True)
    else:
        # assume tuple-like in canonical column order
        df = pd.DataFrame(materialized, columns=list(LONG_COLUMNS))
    missing = [c for c in LONG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"rows missing required columns: {missing}")
    return df[list(LONG_COLUMNS)]


def write_long(rows: Iterable, path: Path | str) -> Path:
    """Write rows to a long-format parquet.

    `rows` may be an iterable of dicts, dataclasses, tuples in canonical column
    order, or DataFrames. Existing files are overwritten.
    """
    df = _coerce_rows(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def read_long(path: Path | str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in LONG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"parquet at {path} missing columns: {missing}")
    return df[list(LONG_COLUMNS)]


def to_wide(
    long_df: pd.DataFrame,
    *,
    index_cols: tuple[str, ...] = DEFAULT_INDEX_COLS,
) -> pd.DataFrame:
    """Pivot a long-format frame so each motif becomes its own column.

    The returned frame has `index_cols` as flat columns and one column per motif.
    """
    pivoted = (
        long_df.pivot_table(index=list(index_cols), columns="motif", values="score")
        .reset_index()
    )
    pivoted.columns.name = None
    return pivoted


def write_wide(
    long_df: pd.DataFrame,
    path: Path | str,
    *,
    index_cols: tuple[str, ...] = DEFAULT_INDEX_COLS,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    to_wide(long_df, index_cols=index_cols).to_parquet(out, index=False)
    return out


def read_wide(
    path: Path | str,
    *,
    index_cols: tuple[str, ...] = DEFAULT_INDEX_COLS,
) -> pd.DataFrame:
    return to_wide(read_long(path), index_cols=index_cols)
