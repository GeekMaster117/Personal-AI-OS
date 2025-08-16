from Include.wrapper.sqlite_wrapper import SQLiteWrapper

import settings

class UsagedataService:
    def __init__(self, usagedata_dir: str):
        self._db = SQLiteWrapper(usagedata_dir)

    def create_if_not_exists_schema(self) -> None:
        self._db.execute_file(settings.schema_dir)

    def add_day_log(self, time_anchor: str, monotonic_start: float, monotonic_last_updated: float) -> None:
        query = "INSERT INTO day_log (time_anchor, monotonic_start, monotonic_last_updated) VALUES (?, ?, ?)"
        self._db.execute(query, (time_anchor, monotonic_start, monotonic_last_updated))

    def get_latest_day_log(self) -> dict | None:
        query = "SELECT * FROM day_log ORDER BY id DESC LIMIT 1"
        result = self._db.fetchone(query)

        return dict(result) if result else None

    def get_day_log_row_count(self) -> int:
        query = "SELECT COUNT(*) FROM day_log"
        result = self._db.fetchone(query)

        return result[0] if result else 0

    def remove_oldest_day_log(self) -> None:
        query = "DELETE FROM day_log WHERE id = (SELECT MIN(id) FROM day_log)"
        self._db.execute(query)