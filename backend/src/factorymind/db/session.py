"""Engine and session management. The schema is created on first use and version-checked."""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from sqlalchemy import Engine, create_engine, event, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from factorymind.db.audit import AuditContentionError, AuditUnavailableError
from factorymind.db.models import APPEND_ONLY_TABLES, SCHEMA_VERSION, Base, SchemaMeta

log = logging.getLogger("factorymind.db")
T = TypeVar("T")
RETRY_ATTEMPTS = 8


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
        if not self._sqlite:
            log.warning(
                "append-only triggers are installed for SQLite only; on %s, enforce immutability "
                "of evidence, decision and audit_event with database privileges (no UPDATE/DELETE "
                "for the application role). The audit hash chain still detects tampering.",
                parsed.get_backend_name(),
            )
        if self._sqlite:
            _sqlite_pragmas(self.engine)
        self._ready = False
        self._lock = threading.Lock()
        self._failure_lock = threading.Lock()
        self.audit_failures = 0  # events that could not be recorded; never reset

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

    def run_with_retry(self, work: Callable[[Session], T], *, attempts: int = RETRY_ATTEMPTS) -> T:
        """Run ``work`` in a fresh transaction, retrying when writers race for the audit chain.

        Each attempt starts a new session, so the audit sequence is recomputed from committed
        state and a failed attempt leaves nothing behind. Only contention (a sequence collision, or
        a lock/serialization conflict) is retried; any other exception, including HTTP errors
        raised by ``work``, propagates at once. After the last attempt ``AuditUnavailableError`` is
        raised: the work was rolled back, it is never half-applied.
        """
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                with self.session() as session:
                    return work(session)
            except (AuditContentionError, OperationalError) as exc:
                last = exc
                # exponential backoff with jitter so racing writers stop colliding in lockstep
                delay = min(0.002 * 2**attempt, 0.1) * (0.5 + secrets.randbelow(1000) / 1000)
                time.sleep(delay)
        raise AuditUnavailableError("audit log unavailable after retries") from last

    def note_audit_failure(self) -> None:
        with self._failure_lock:
            self.audit_failures += 1

    def dispose(self) -> None:
        self.engine.dispose()
