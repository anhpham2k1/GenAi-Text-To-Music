"""CSV logging utilities for training + quality metrics."""

from __future__ import annotations

import csv
import os
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


_lock = threading.Lock()


def append_csv_row(path: str, row: Dict[str, Any], fieldnames: Optional[List[str]] = None) -> None:
    """Append one dict row to CSV (create with header if missing)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = dict(row)
    if "timestamp" not in row:
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")

    with _lock:
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        if fieldnames is None:
            if exists:
                with open(path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    fieldnames = list(reader.fieldnames or [])
                # merge new keys
                for k in row.keys():
                    if k not in fieldnames:
                        fieldnames.append(k)
            else:
                fieldnames = list(row.keys())

        # rewrite header if new columns appeared
        if exists:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                old_fields = list(reader.fieldnames or [])
                old_rows = list(reader)
            if old_fields != fieldnames:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    for r in old_rows:
                        writer.writerow(r)
                    writer.writerow(row)
                return

        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def write_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    """Overwrite CSV with full list of rows."""
    rows = list(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


class CSVLogger:
    """Simple stateful CSV logger for training loops."""

    def __init__(self, path: str, fieldnames: Optional[List[str]] = None):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, **kwargs) -> None:
        append_csv_row(self.path, kwargs, fieldnames=self.fieldnames)
