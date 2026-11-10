"""Engine and session management. The schema is created on first use and version-checked."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from factorymind.db.models import APPEND_ONLY_TABLES, SCHEMA_VERSION, Base, SchemaMeta


class SchemaMismatchError(RuntimeError):
    """The database was created by an incompatible schema version."""


def _sqlite_pragmas(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _install_append_only_triggers(engine: Engine) -> None:
    with engine.begin() as conn:
        for table in APPEND_ONLY_TABLES:
            for verb in ("UPDATE", "DELETE"):
                conn.execute(
                    text(
                        f"CREATE TRIGGER IF NOT EXISTS {table}_no_{verb.lower()} "
                        f"BEFORE {verb} ON {table} "
                        f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END"
                    )
                )


class Database:
    def __init__(self, url: str) -> None:
        parsed = make_url(url)
        self._sqlite = parsed.get_backend_name() == "sqlite"
        if self._sqlite and parsed.database and parsed.database != ":memory:":
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {"future": True}
        if self._sqlite:
            kwargs["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(url, **kwargs)
        if self._sqlite:
            _sqlite_pragmas(self.engine)
        self._ready = False
        self._lock = threading.Lock()

    def ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            Base.metadata.create_all(self.engine)
            with Session(self.engine) as session:
                row = session.execute(
                    select(SchemaMeta).where(SchemaMeta.key == "schema_version")
                ).scalar_one_or_none()
                if row is None:
                    session.add(SchemaMeta(key="schema_version", value=SCHEMA_VERSION))
                    session.commit()
                elif row.value != SCHEMA_VERSION:
                    raise SchemaMismatchError(
                        f"database schema {row.value} != expected {SCHEMA_VERSION}"
                    )
            if self._sqlite:
                _install_append_only_triggers(self.engine)
            self._ready = True

    @contextmanager
    def session(self) -> Iterator[Session]:
        """One transaction: committed on success, rolled back on any exception."""
        self.ensure_ready()
        with Session(self.engine, expire_on_commit=False) as session:
            try:
                yield session
                session.commit()
            except BaseException:
                session.rollback()
                raise

    def dispose(self) -> None:
        self.engine.dispose()
