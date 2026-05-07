"""Schema-invariant + duplication-regression tests for `read_long`.

Targets the loader-bug class that allowed Phase 2's gate verdict to be
silently broken: `pd.concat([read_long(sweep_path(motif, s)) for s in SIZES])`
triple-loaded the same parquet because `sweep_path` returned the same file
for every size. The tests below pin down:

  1. read_long preserves the canonical long-format schema (column dtypes).
  2. Reading a single parquet once and filtering by size produces unique
     `(size, motif, step, layer, head)` rows — i.e., no silent triplication
     when the loader is used correctly.
  3. The duplication-detection invariant the gate runner *should* assert
     after concat: `(size, motif, step, layer, head)` must be unique per
     parquet. Concating the same parquet twice produces duplicates that
     this invariant would catch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from notebooks._lib.sweep_io import (
    DEFAULT_INDEX_COLS,
    LONG_COLUMNS,
    SweepRow,
    read_long,
    write_long,
)

INDEX_COLS_FULL: tuple[str, ...] = DEFAULT_INDEX_COLS + ("motif",)


def _synthetic_rows(
    sizes: tuple[str, ...] = ("70m", "160m", "410m"),
    motifs: tuple[str, ...] = ("induction", "successor"),
    steps: tuple[int, ...] = (0, 1000, 8000, 143000),
    n_layers: int = 2,
    n_heads: int = 2,
    seed: int = 0,
) -> list[SweepRow]:
    rng = np.random.default_rng(seed)
    rows: list[SweepRow] = []
    for size in sizes:
        for motif in motifs:
            for step in steps:
                for layer in range(n_layers):
                    for head in range(n_heads):
                        rows.append(
                            SweepRow(
                                size=size, step=step, layer=layer,
                                head=head, motif=motif,
                                score=float(rng.random()),
                            )
                        )
    return rows


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


def test_read_long_preserves_canonical_columns(tmp_path: Path):
    rows = _synthetic_rows()
    path = tmp_path / "sweep.parquet"
    write_long(rows, path)

    df = read_long(path)
    assert tuple(df.columns) == LONG_COLUMNS


def test_read_long_preserves_dtypes(tmp_path: Path):
    """Column dtypes must survive the parquet round-trip.

    `step`, `layer`, `head` are integer columns; `score` is float; `size` and
    `motif` are object/string columns.
    """
    rows = _synthetic_rows()
    path = tmp_path / "sweep.parquet"
    write_long(rows, path)

    df = read_long(path)
    assert pd.api.types.is_integer_dtype(df["step"]), df["step"].dtype
    assert pd.api.types.is_integer_dtype(df["layer"]), df["layer"].dtype
    assert pd.api.types.is_integer_dtype(df["head"]), df["head"].dtype
    assert pd.api.types.is_float_dtype(df["score"]), df["score"].dtype
    # `size` and `motif` are string-like — pandas may use either `object` or
    # the newer `string`/`StringDtype` depending on pyarrow version.
    assert df["size"].dtype == object or pd.api.types.is_string_dtype(df["size"])
    assert df["motif"].dtype == object or pd.api.types.is_string_dtype(df["motif"])


def test_read_long_rejects_parquet_missing_columns(tmp_path: Path):
    """A parquet missing any canonical long-format column must raise."""
    bad = pd.DataFrame({
        "size": ["70m"], "step": [0], "layer": [0], "head": [0],
        # 'motif' deliberately missing
        "score": [0.5],
    })
    path = tmp_path / "bad.parquet"
    bad.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        read_long(path)


# ---------------------------------------------------------------------------
# Index uniqueness — per (size, parquet) pair
# ---------------------------------------------------------------------------


def test_read_long_index_unique_per_parquet(tmp_path: Path):
    """`(size, motif, step, layer, head)` is unique within a single parquet."""
    rows = _synthetic_rows()
    path = tmp_path / "sweep.parquet"
    write_long(rows, path)

    df = read_long(path)
    duplicate_mask = df.duplicated(subset=list(INDEX_COLS_FULL))
    assert duplicate_mask.sum() == 0, df[duplicate_mask]


def test_read_long_filtered_by_size_is_unique_per_size(tmp_path: Path):
    """Filter-by-size is unique per (motif, step, layer, head) within size."""
    rows = _synthetic_rows()
    path = tmp_path / "sweep.parquet"
    write_long(rows, path)
    df = read_long(path)

    for size, sub in df.groupby("size"):
        dup = sub.duplicated(subset=["motif", "step", "layer", "head"])
        assert dup.sum() == 0, f"duplicates in size={size}: {sub[dup]}"


# ---------------------------------------------------------------------------
# Loader-bug regression class — silent triplication
# ---------------------------------------------------------------------------


def test_concat_of_same_parquet_per_size_triplicates_and_is_detectable(
    tmp_path: Path,
):
    """Regression test for the Phase 2 loader bug class.

    The bug: `pd.concat([read_long(sweep_path(motif, s)) for s in SIZES])`
    triple-loaded the same parquet because `sweep_path` returned the same
    path for every size. This test reproduces the failure mode and confirms
    that a uniqueness check on `(size, motif, step, layer, head)` *would*
    have caught it.
    """
    rows = _synthetic_rows(sizes=("70m", "160m", "410m"))
    path = tmp_path / "phase2_motif.parquet"
    write_long(rows, path)

    # Buggy pattern: concat the same parquet 3 times.
    combined_buggy = pd.concat(
        [read_long(path) for _ in ("70m", "160m", "410m")],
        ignore_index=True,
    )
    # Correct pattern: read once.
    combined_correct = read_long(path)

    # The bug triplicates row count.
    assert len(combined_buggy) == 3 * len(combined_correct)

    # The uniqueness invariant catches the bug.
    dup_buggy = combined_buggy.duplicated(subset=list(INDEX_COLS_FULL)).sum()
    dup_correct = combined_correct.duplicated(subset=list(INDEX_COLS_FULL)).sum()
    assert dup_correct == 0
    assert dup_buggy > 0
    # Specifically: each correct row appears 3x, so n_duplicates = 2 * n_correct.
    assert dup_buggy == 2 * len(combined_correct)


def test_filter_by_size_after_buggy_concat_triplicates_per_size(tmp_path: Path):
    """After the buggy concat, filtering by a single size returns 3× rows.

    This is the exact failure mode that broke the Phase 2 gate verdict — a
    per-size filter on the buggy concat returns triplicated rows for that
    size, silently inflating bootstrap counts.
    """
    rows = _synthetic_rows(sizes=("70m", "160m", "410m"))
    path = tmp_path / "phase2_motif.parquet"
    write_long(rows, path)

    correct = read_long(path)
    n_70m_correct = (correct["size"] == "70m").sum()

    buggy = pd.concat(
        [read_long(path) for _ in range(3)], ignore_index=True
    )
    n_70m_buggy = (buggy["size"] == "70m").sum()

    assert n_70m_buggy == 3 * n_70m_correct


def test_concat_of_distinct_per_size_parquets_is_not_triplication(
    tmp_path: Path,
):
    """The correct loader pattern: per-size parquets have non-overlapping `size`.

    When `sweep_path(motif, size)` returns truly per-size files, concating
    them yields the union — and `(size, motif, step, layer, head)` remains
    unique.
    """
    rows_70m = _synthetic_rows(sizes=("70m",))
    rows_160m = _synthetic_rows(sizes=("160m",), seed=1)
    rows_410m = _synthetic_rows(sizes=("410m",), seed=2)

    p70 = tmp_path / "phase2_70m.parquet"
    p160 = tmp_path / "phase2_160m.parquet"
    p410 = tmp_path / "phase2_410m.parquet"
    write_long(rows_70m, p70)
    write_long(rows_160m, p160)
    write_long(rows_410m, p410)

    combined = pd.concat(
        [read_long(p) for p in (p70, p160, p410)], ignore_index=True
    )
    dup = combined.duplicated(subset=list(INDEX_COLS_FULL)).sum()
    assert dup == 0
    assert set(combined["size"].unique()) == {"70m", "160m", "410m"}
