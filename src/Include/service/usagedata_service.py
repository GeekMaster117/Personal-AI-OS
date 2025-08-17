from Include.wrapper.sqlite_wrapper import SQLiteWrapper

import settings

class UsagedataService:
    _day_log_columns: tuple[str] = (
        "time_anchor",
        "monotonic_start",
        "monotonic_last_updated",
        "total_downtime_duration",
        "total_anomalies"
    )

    def __init__(self, usagedata_dir: str):
        self._db = SQLiteWrapper(usagedata_dir)

    def create_if_not_exists_schema(self) -> None:
        self._db.execute_file(settings.schema_dir)

    def add_day_log(self, time_anchor: str, monotonic_anchor: float) -> None:
        query = "INSERT INTO day_log (time_anchor, monotonic_start, monotonic_last_updated) VALUES (?, ?, ?)"
        self._db.execute(query, (time_anchor, monotonic_anchor, monotonic_anchor))

    def get_latest_day_log(self, columns: tuple[str] | None = None) -> dict | None:
        if columns:
            for column in columns:
                if column not in UsagedataService._day_log_columns:
                    raise ValueError(f"Invalid column name: {column}")
        else:
            columns = UsagedataService._day_log_columns

        query = f"SELECT {', '.join(columns)} FROM day_log ORDER BY id DESC LIMIT 1"
        result = self._db.fetchone(query)

        return dict(result) if result else None

    def get_day_log_row_count(self) -> int:
        query = "SELECT COUNT(*) FROM day_log"
        result = self._db.fetchone(query)

        return result[0] if result else 0

    def update_latest_day_log(self, column_values: dict[str, float | int]) -> None:
        if not column_values:
            return

        columns = []
        values = []
        for column, value in column_values.items():
            if column not in UsagedataService._day_log_columns:
                raise ValueError(f"Invalid column name: {column}")
            columns.append(column + " = ?")
            values.append(value)

        query = f"UPDATE day_log SET {', '.join(columns)} ORDER BY id DESC LIMIT 1"
        self._db.execute(query, values)

    def remove_oldest_day_log(self) -> None:
        query = "DELETE FROM day_log ORDER BY id ASC LIMIT 1"
        self._db.execute(query)