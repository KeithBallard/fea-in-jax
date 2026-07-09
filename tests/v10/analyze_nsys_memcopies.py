"""Attribute Nsight Systems CUDA memcpy events to NVTX ranges.

Run after Nsight has exported a SQLite file, for example:

    nsys export --type sqlite --force true -o petsc_jax_mat_func_profile.sqlite petsc_jax_mat_func_profile.nsys-rep
    /home/alberto/venvs/mpi-gpu/bin/python v10/analyze_nsys_memcopies.py petsc_jax_mat_func_profile.sqlite

This script is intentionally schema-tolerant because Nsight's SQLite column
names vary a little across releases.
"""

from __future__ import annotations

from collections import defaultdict
import sqlite3
import sys
from pathlib import Path


RANGE_MARKERS = ("snes_", "profile_")


def _tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {row[0] for row in rows}


def _columns(conn, table):
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _first_existing(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _string_map(conn, tables):
    if "StringIds" not in tables:
        return {}
    cols = _columns(conn, "StringIds")
    id_col = _first_existing(cols, ("id", "Id"))
    value_col = _first_existing(cols, ("value", "Value", "string", "String"))
    if id_col is None or value_col is None:
        return {}
    return dict(conn.execute(f"SELECT {id_col}, {value_col} FROM StringIds").fetchall())


def _enum_map(conn, tables):
    for table in ("ENUM_CUDA_MEMCPY_OPER", "ENUM_CUDA_MEMCPY_KIND"):
        if table not in tables:
            continue
        cols = _columns(conn, table)
        id_col = _first_existing(cols, ("id", "Id", "value", "Value"))
        name_col = _first_existing(cols, ("label", "Label", "name", "Name"))
        if id_col is not None and name_col is not None:
            return dict(conn.execute(f"SELECT {id_col}, {name_col} FROM {table}").fetchall())
    return {}


def _load_nvtx_ranges(conn, tables, strings):
    if "NVTX_EVENTS" not in tables:
        return []

    cols = _columns(conn, "NVTX_EVENTS")
    start_col = _first_existing(cols, ("start", "startTime"))
    end_col = _first_existing(cols, ("end", "endTime"))
    text_col = _first_existing(cols, ("text", "Text"))
    text_id_col = _first_existing(cols, ("textId", "textID", "messageId", "messageID"))

    if start_col is None or end_col is None:
        return []

    select_cols = [start_col, end_col]
    if text_col:
        select_cols.append(text_col)
    if text_id_col:
        select_cols.append(text_id_col)

    ranges = []
    query = f"SELECT {', '.join(select_cols)} FROM NVTX_EVENTS WHERE {start_col} IS NOT NULL AND {end_col} IS NOT NULL"
    for row in conn.execute(query):
        start = row[0]
        end = row[1]
        idx = 2
        name = None
        if text_col:
            name = row[idx]
            idx += 1
        if not name and text_id_col:
            name = strings.get(row[idx])
        if not name:
            continue
        name = str(name)
        if any(marker in name for marker in RANGE_MARKERS):
            ranges.append((int(start), int(end), name))
    ranges.sort(key=lambda item: (item[0], item[1] - item[0]))
    return ranges


def _load_memcopies(conn, tables, enum_labels):
    table = "CUPTI_ACTIVITY_KIND_MEMCPY"
    if table not in tables:
        memcpy_tables = sorted(name for name in tables if "MEMCPY" in name.upper())
        if memcpy_tables:
            table = memcpy_tables[0]
        else:
            return [], None

    if table not in tables:
        raise RuntimeError(f"{table} not found in SQLite export")

    cols = _columns(conn, table)
    start_col = _first_existing(cols, ("start", "startTime"))
    end_col = _first_existing(cols, ("end", "endTime"))
    bytes_col = _first_existing(cols, ("bytes", "Bytes", "size", "Size"))
    kind_col = _first_existing(cols, ("copyKind", "copyKindId", "memcpyKind"))

    missing = [
        name
        for name, col in (
            ("start", start_col),
            ("end", end_col),
            ("bytes", bytes_col),
            ("copy kind", kind_col),
        )
        if col is None
    ]
    if missing:
        raise RuntimeError(f"Could not identify memcpy columns: {missing}. Columns are: {cols}")

    query = f"SELECT {start_col}, {end_col}, {bytes_col}, {kind_col} FROM {table}"
    memcopies = []
    for start, end, nbytes, kind in conn.execute(query):
        label = enum_labels.get(kind, str(kind))
        memcopies.append((int(start), int(end), int(nbytes), label))
    return memcopies, table


def _innermost_range_for_event(start, end, ranges):
    midpoint = (start + end) // 2
    containing = [
        (range_end - range_start, name)
        for range_start, range_end, name in ranges
        if range_start <= midpoint <= range_end
    ]
    if not containing:
        return "NO_MATCHING_NVTX_RANGE"
    containing.sort()
    return containing[0][1]


def _fmt_mb(nbytes):
    return nbytes / 1_000_000.0


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python v10/analyze_nsys_memcopies.py path/to/profile.sqlite")

    sqlite_path = Path(sys.argv[1])
    conn = sqlite3.connect(sqlite_path)
    tables = _tables(conn)
    strings = _string_map(conn, tables)
    enum_labels = _enum_map(conn, tables)
    ranges = _load_nvtx_ranges(conn, tables, strings)
    memcopies, memcpy_table = _load_memcopies(conn, tables, enum_labels)

    if memcpy_table is None:
        print("NVTX ranges matched:", len(ranges))
        print("Memcpy events: 0")
        print()
        print("No CUPTI memcpy table was found in this SQLite export.")
        print("This usually means this profile recorded no CUDA memcpy events,")
        print("or the report was captured without CUDA memcpy activity enabled.")
        print()
        print("CUDA/CUPTI-like tables found:")
        for table in sorted(name for name in tables if "CUDA" in name.upper() or "CUPTI" in name.upper()):
            print(" ", table)
        return

    by_kind_and_range = defaultdict(lambda: {"count": 0, "bytes": 0, "time": 0})
    top_events = []

    for start, end, nbytes, kind in memcopies:
        range_name = _innermost_range_for_event(start, end, ranges)
        key = (kind, range_name)
        by_kind_and_range[key]["count"] += 1
        by_kind_and_range[key]["bytes"] += nbytes
        by_kind_and_range[key]["time"] += end - start
        top_events.append((nbytes, end - start, kind, range_name, start, end))

    print("NVTX ranges matched:", len(ranges))
    print("Memcpy table:", memcpy_table)
    print("Memcpy events:", len(memcopies))
    print()
    print("=== Memcpy Summary By Kind And Innermost NVTX Range ===")
    print(f"{'Kind':<28} {'Count':>8} {'MB':>12} {'Time ms':>12}  Range")
    print("-" * 100)
    for (kind, range_name), stats in sorted(
        by_kind_and_range.items(),
        key=lambda item: (item[0][0], -item[1]["bytes"], -item[1]["time"]),
    ):
        print(
            f"{kind:<28} {stats['count']:>8} "
            f"{_fmt_mb(stats['bytes']):>12.6f} {stats['time'] / 1e6:>12.6f}  {range_name}"
        )

    print()
    print("=== Largest Individual Memcpy Events ===")
    print(f"{'Kind':<28} {'MB':>12} {'Time us':>12}  Range")
    print("-" * 90)
    for nbytes, duration, kind, range_name, _start, _end in sorted(top_events, reverse=True)[:30]:
        print(f"{kind:<28} {_fmt_mb(nbytes):>12.6f} {duration / 1e3:>12.3f}  {range_name}")


if __name__ == "__main__":
    main()
