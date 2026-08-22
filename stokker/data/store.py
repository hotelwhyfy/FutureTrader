"""Parquet-backed cache with a DuckDB query layer.

Parquet files on disk are the source of truth; DuckDB is attached over them for
ad-hoc SQL.  Deliberately boring -- the point is that re-running a fetch is
cheap and that raw vendor data is never mutated in place.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from stokker.config import CACHE_DIR, DB_PATH

log = logging.getLogger(__name__)


class Store:
    """Namespaced parquet cache.  A dataset is `<namespace>/<key>.parquet`."""

    def __init__(self, root: Path = CACHE_DIR) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, namespace: str, key: str) -> Path:
        d = self.root / namespace
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.parquet"

    def exists(self, namespace: str, key: str) -> bool:
        return self.path(namespace, key).exists()

    def put(self, namespace: str, key: str, df: pd.DataFrame) -> Path:
        p = self.path(namespace, key)
        # Write to a sibling temp file then rename, so an interrupted write
        # cannot leave a half-parquet that later reads treat as real data.
        tmp = p.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=True)
        tmp.replace(p)
        log.debug("wrote %s rows=%d", p, len(df))
        return p

    def get(self, namespace: str, key: str) -> pd.DataFrame:
        p = self.path(namespace, key)
        if not p.exists():
            raise FileNotFoundError(f"no cached dataset at {p}")
        return pd.read_parquet(p)

    def keys(self, namespace: str) -> list[str]:
        d = self.root / namespace
        if not d.exists():
            return []
        return sorted(f.stem for f in d.glob("*.parquet"))

    def drop(self, namespace: str, key: str) -> None:
        self.path(namespace, key).unlink(missing_ok=True)

    def connect(self) -> duckdb.DuckDBPyConnection:
        """DuckDB connection with each namespace exposed as a view."""
        con = duckdb.connect(str(DB_PATH))
        for ns_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if not any(ns_dir.glob("*.parquet")):
                continue
            view = ns_dir.name.replace("-", "_")
            con.execute(
                f"CREATE OR REPLACE VIEW {view} AS "
                f"SELECT * FROM read_parquet('{ns_dir}/*.parquet', union_by_name=true, "
                f"filename=true)"
            )
        return con


STORE = Store()
