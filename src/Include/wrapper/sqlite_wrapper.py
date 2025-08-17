import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

class SQLiteWrapper:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()

        self._initialize_db()

    @contextmanager
    def _get_conn(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def _initialize_db(self):
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")

    def execute_file(self, file_path: str):
        with self._get_conn() as conn:
            with open(file_path, "r") as f:
                conn.executescript(f.read())

    def execute(self, query: str, params: tuple = ()) -> None:
        with self._get_conn() as conn:
            conn.execute(query, params)

    def execute_many(self, query: str, params: list[tuple] = []) -> None:
        with self._get_conn() as conn:
            conn.executemany(query, params)

    def fetchall(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._get_conn() as conn:
            return conn.execute(query, params).fetchall()

    def fetchone(self, query: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._get_conn() as conn:
            return conn.execute(query, params).fetchone()