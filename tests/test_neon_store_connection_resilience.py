from __future__ import annotations

from types import SimpleNamespace

from src.data_pipeline.neon_store import NeonStore


def test_fetchall_reconnects_after_transient_operational_error(monkeypatch):
    class FakeCursor:
        def __init__(self, owner):
            self.owner = owner
            self.description = [("ok",)]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params=()):
            self.owner.calls += 1
            if self.owner.calls == 1:
                raise RuntimeError("consuming input failed: SSL connection has been closed unexpectedly")

        def fetchall(self):
            return [(1,)]

    class FakeConnection:
        def __init__(self):
            self.calls = 0
            self.closed = False

        def cursor(self):
            return FakeCursor(self)

        def commit(self):
            pass

        def close(self):
            self.closed = True

    store = object.__new__(NeonStore)
    store.dsn = "postgresql://user:pass@example.com/db"
    store.connection = FakeConnection()
    store._owns_connection = True
    store.batch_retries = 1
    replacement = FakeConnection()
    replacement.calls = 1

    monkeypatch.setattr(store, "_reconnect", lambda: setattr(store, "connection", replacement))
    result = store._fetchall("select 1")

    assert result == [{"ok": 1}]
    assert store.connection is replacement


def test_transient_ssl_error_is_recognized(monkeypatch):
    store = object.__new__(NeonStore)
    assert store._is_connection_error(RuntimeError("SSL connection has been closed unexpectedly"))
    assert store._is_connection_error(RuntimeError("consuming input failed"))
    assert not store._is_connection_error(RuntimeError("syntax error"))
