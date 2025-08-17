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
    _focus_period_columns: tuple[str] = (
        "day_hour",
        "focus_duration",
        "focus_count"
    )

    def __init__(self, usagedata_dir: str):
        self._db = SQLiteWrapper(usagedata_dir)

    def create_if_not_exists_schema(self) -> None:
        with open(settings.schema_dir, "r") as file:
            self._db.execute(file.read())

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

    def get_latest_day_log_id(self) -> int | None:
        query = "SELECT id FROM day_log ORDER BY id DESC LIMIT 1"
        result = self._db.fetchone(query)

        return result[0] if result else None

    def get_day_log_row_count(self) -> int:
        query = "SELECT COUNT(*) FROM day_log"
        result = self._db.fetchone(query)

        return result[0] if result else 0

    def get_latest_day_log_app_log_title_log(self) -> dict[str, str | int | float | dict[str, str | int | float]] | None:
        result = None
        with open(settings.fetch_all_apps_titles, "r") as file:
            result = self._db.fetchall(file.read())
        
        if not result:
            return None

        apps_titles = dict()
        for row in result:
            app_name = row['app_name']
            if app_name not in apps_titles:
                apps_titles[app_name] = {
                    'total_duration': row['app_total_duration'],
                    'total_focus_duration': row['app_total_focus_duration'],
                    'total_focus_count': row['app_total_focus_count'],
                    'titles': {}
                }
            apps_titles[app_name]['titles'][row['title_name']] = {
                'total_duration': row['title_total_duration'],
                'total_focus_duration': row['title_total_focus_duration'],
                'total_focus_count': row['title_total_focus_count']
            }

        return apps_titles

    def latest_day_log_app_log_name_exists(self, app_name: str) -> bool:
        latest_day_log_id = self.get_latest_day_log_id()
        if not latest_day_log_id:
            return False

        query = """
            SELECT 1 FROM app_log
            WHERE day_log_id = ? AND app_name = ?
        """
        result = self._db.fetchone(query, (latest_day_log_id, app_name))

        return bool(result)

    def get_latest_day_log_downtime_period(self) -> dict[int, float] | None:
        latest_day_log_id = self.get_latest_day_log_id()
        if not latest_day_log_id:
            return None

        query = """
            SELECT day_hour, downtime_duration FROM downtime_period
            WHERE day_log_id = ?
        """
        result = self._db.fetchall(query, (latest_day_log_id,))

        return {row[0]: row[1] for row in result} if result else None
    
    def get_latest_day_log_app_focus_period(self, app_name: str) -> dict[int, dict[str, int | float]] | None:
        latest_day_log_id = self.get_latest_day_log_id()
        if not latest_day_log_id:
            return None

        query = """
            SELECT day_hour, focus_duration, focus_count FROM app_focus_period
            WHERE day_log_id = ? AND app_name = ?
        """
        result = self._db.fetchall(query, (latest_day_log_id, app_name))

        return {row[0]: {'focus_duration': row[1], 'focus_count': row[2]} for row in result} if result else None

    def get_latest_day_log_title_focus_period(self, app_name: str, title_name: str) -> dict[int, dict[str, int | float]] | None:
        latest_day_log_id = self.get_latest_day_log_id()
        if not latest_day_log_id:
            return None

        query = """
            SELECT day_hour, focus_duration, focus_count FROM title_focus_period
            WHERE day_log_id = ? AND app_name = ? AND title_name = ?
        """
        result = self._db.fetchall(query, (latest_day_log_id, app_name, title_name))

        return {row[0]: {'focus_duration': row[1], 'focus_count': row[2]} for row in result} if result else None

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

    def upsert_latest_day_log_downtime_period(self, hour_durations: dict[int, float]) -> None:
        if not hour_durations:
            return

        for hour, duration in hour_durations.items():
            if hour < 0 or hour > 23:
                raise ValueError(f"Invalid hour: {hour}")
            if duration < 0 or duration > 3600:
                raise ValueError(f"Invalid duration: {duration}")

        latest_day_log_id = self.get_latest_day_log_id()
        if not latest_day_log_id:
            raise ValueError("No latest day log found.")
        
        query = """
            INSERT INTO downtime_period (day_log_id, day_hour, downtime_duration)
            VALUES (?, ?, ?)
            ON CONFLICT(day_log_id, day_hour) DO UPDATE SET downtime_duration = excluded.downtime_duration
        """
        values = []

        for hour, duration in hour_durations.items():
            values.append((latest_day_log_id, hour, duration))

        self._db.execute_many(query, values)

    def upsert_latest_day_log_app_focus_period(self, app_name: str, app_focus_periods: dict[int, dict[str, int | float]]) -> None:
        if not app_focus_periods:
            return
        
        for hour, focus_data in app_focus_periods.items():
            if hour < 0 or hour > 23:
                raise ValueError(f"Invalid hour: {hour}")
            for focus_column, data in focus_data.items():
                if focus_column not in UsagedataService._focus_period_columns:
                    raise ValueError(f"Invalid column name: {focus_column}")
                if focus_column == "focus_duration" and (data < 0 or data > 3600):
                    raise ValueError(f"Invalid duration: {data}")
                if focus_column == "focus_count" and data < 0:
                    raise ValueError(f"Invalid focus count: {data}")

        latest_day_log_id = self.get_latest_day_log_id()
        if not latest_day_log_id:
            raise ValueError("No latest day log found.")

        if not self.latest_day_log_app_log_name_exists(app_name):
            raise ValueError(f"App name '{app_name}' not found in the latest day log app logs.")

        query = """
            INSERT INTO app_focus_period (day_log_id, app_name, day_hour, focus_duration, focus_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(day_log_id, app_name, day_hour) DO UPDATE SET
                focus_duration = excluded.focus_duration,
                focus_count = excluded.focus_count
        """
        values = []

        for hour, focus_data in app_focus_periods.items():
            values.append((latest_day_log_id, app_name, hour, focus_data["focus_duration"], focus_data["focus_count"]))

        self._db.execute_many(query, values)

    def remove_oldest_day_log(self) -> None:
        query = "DELETE FROM day_log ORDER BY id ASC LIMIT 1"
        self._db.execute(query)