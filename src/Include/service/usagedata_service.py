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
    _app_title_log_columns: tuple[str] = (
        "total_duration",
        "total_focus_duration",
        "total_focus_count",
    )
    _focus_period_columns: tuple[str] = (
        "day_hour",
        "focus_duration",
        "focus_count"
    )

    def __init__(self, usagedata_dir: str):
        self._db = SQLiteWrapper(usagedata_dir)

    def create_if_not_exists_schema(self) -> None:
        self._db.execute_script(settings.schema_dir)

    def add_day_log(self, time_anchor: str, monotonic_anchor: float) -> None:
        query = "INSERT INTO day_log (time_anchor, monotonic_start, monotonic_last_updated) VALUES (?, ?, ?)"
        self._db.execute(query, (time_anchor, monotonic_anchor, monotonic_anchor))

    def get_latest_day_log(self, columns: tuple[str] | None = None) -> dict[str, str | int | float]:
        if columns:
            for column in columns:
                if column not in UsagedataService._day_log_columns:
                    raise ValueError(f"Invalid column name: {column}")
        else:
            columns = UsagedataService._day_log_columns

        query = f"SELECT {', '.join(columns)} FROM day_log ORDER BY id DESC LIMIT 1"
        result = self._db.fetchone(query)

        return dict(result) if result else dict()
    
    def get_day_log(self, id: int, columns: tuple[str] | None = None) -> dict[str, str | int | float]:
        if columns:
            for column in columns:
                if column not in UsagedataService._day_log_columns:
                    raise ValueError(f"Invalid column name: {column}")
        else:
            columns = UsagedataService._day_log_columns

        query = f"SELECT {', '.join(columns)} FROM day_log WHERE id = ?"
        result = self._db.fetchone(query, (id,))

        return dict(result) if result else dict()

    def get_latest_day_log_id(self) -> int | None:
        query = "SELECT id FROM day_log ORDER BY id DESC LIMIT 1"
        result = self._db.fetchone(query)

        return result[0] if result else None

    def get_day_log_ids(self) -> list[int]:
        query = "SELECT id FROM day_log ORDER BY id ASC"
        result = self._db.fetchall(query)

        return [row[0] for row in result] if result else []

    def get_day_log_row_count(self) -> int:
        query = "SELECT COUNT(*) FROM day_log"
        result = self._db.fetchone(query)

        return result[0] if result else 0
    
    def get_app_log_title_log(self, day_log_id: int) -> dict[str, dict[str, int | float | dict[str, str | int | float]]]:
        query = """
            SELECT 
                app_log.app_name,
                app_log.total_duration AS app_total_duration,
                app_log.total_focus_duration AS app_total_focus_duration,
                app_log.total_focus_count AS app_total_focus_count,
                title_log.title_name,
                title_log.total_duration AS title_total_duration,
                title_log.total_focus_duration AS title_total_focus_duration,
                title_log.total_focus_count AS title_total_focus_count
            FROM app_log
            LEFT JOIN title_log 
                ON app_log.day_log_id = title_log.day_log_id 
            AND app_log.app_name = title_log.app_name
            WHERE app_log.day_log_id = ?
        """

        result = self._db.fetchall(query, (day_log_id,))
        
        if not result:
            return dict()

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

    def get_latest_day_log_app_log_title_log(self) -> dict[str, dict[str, int | float | dict[str, str | int | float]]]:
        latest_day_log_id = self.get_latest_day_log_id()
        if not latest_day_log_id:
            return dict()
        
        return self.get_app_log_title_log(latest_day_log_id)

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

    def latest_day_log_title_log_name_exists(self, app_name: str, title_name: str) -> bool:
        latest_day_log_id = self.get_latest_day_log_id()
        if not latest_day_log_id:
            return False

        query = """
            SELECT 1 FROM title_log
            WHERE day_log_id = ? AND app_name = ? AND title_name = ?
        """
        result = self._db.fetchone(query, (latest_day_log_id, app_name, title_name))

        return bool(result)

    def get_latest_day_log_downtime_period(self) -> dict[int, float]:
        latest_day_log_id = self.get_latest_day_log_id()
        if not latest_day_log_id:
            return dict()

        query = """
            SELECT day_hour, downtime_duration FROM downtime_period
            WHERE day_log_id = ?
        """
        result = self._db.fetchall(query, (latest_day_log_id,))

        return {row[0]: row[1] for row in result} if result else dict()

    def get_app_focus_period(self, day_log_id: int, app_name: str) -> dict[int, dict[str, int | float]]:
        query = """
            SELECT day_hour, focus_duration, focus_count FROM app_focus_period
            WHERE day_log_id = ? AND app_name = ?
        """
        result = self._db.fetchall(query, (day_log_id, app_name))

        return {row[0]: {'focus_duration': row[1], 'focus_count': row[2]} for row in result} if result else dict()

    def get_latest_day_log_app_focus_period(self, app_name: str) -> dict[int, dict[str, int | float]]:
        latest_day_log_id = self.get_latest_day_log_id()
        if not latest_day_log_id:
            return dict()

        return self.get_app_focus_period(latest_day_log_id, app_name)
    
    def get_title_focus_period(self, day_log_id: int, app_name: str, title_name: str) -> dict[int, dict[str, int | float]]:
        query = """
            SELECT day_hour, focus_duration, focus_count FROM title_focus_period
            WHERE day_log_id = ? AND app_name = ? AND title_name = ?
        """
        result = self._db.fetchall(query, (day_log_id, app_name, title_name))

        return {row[0]: {'focus_duration': row[1], 'focus_count': row[2]} for row in result} if result else dict()

    def get_latest_day_log_title_focus_period(self, app_name: str, title_name: str) -> dict[int, dict[str, int | float]]:
        latest_day_log_id = self.get_latest_day_log_id()
        if not latest_day_log_id:
            return dict()

        return self.get_title_focus_period(latest_day_log_id, app_name, title_name)

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

        query = f"""
            UPDATE day_log SET {', '.join(columns)}
            WHERE id = (SELECT id FROM day_log ORDER BY id DESC LIMIT 1);
        """
        self._db.execute(query, values)

    def upsert_latest_day_log_app_log_title_log(self, apps_titles: dict[str, dict[str, int | float | dict[str, str | int | float]]]) -> None:
        if not apps_titles:
            return

        latest_day_log_id = self.get_latest_day_log_id()
        if not latest_day_log_id:
            raise ValueError("No latest day log found.")

        app_log_query = """
            INSERT INTO app_log (day_log_id, app_name, total_duration, total_focus_duration, total_focus_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(day_log_id, app_name) DO UPDATE SET
                total_duration = excluded.total_duration,
                total_focus_duration = excluded.total_focus_duration,
                total_focus_count = excluded.total_focus_count
        """

        title_log_query = """
            INSERT INTO title_log (day_log_id, app_name, title_name, total_duration, total_focus_duration, total_focus_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(day_log_id, app_name, title_name) DO UPDATE SET
                total_duration = excluded.total_duration,
                total_focus_duration = excluded.total_focus_duration,
                total_focus_count = excluded.total_focus_count
        """

        app_values = []
        title_values = []

        for app_name, app_data in apps_titles.items():
            titles = dict()

            for app_column in app_data:
                if app_column == 'titles':
                    titles = app_data.get('titles', {})
                    continue

                if app_column not in UsagedataService._app_title_log_columns:
                    raise ValueError(f"Invalid column name: {app_column}")

            total_duration = app_data.get('total_duration', 0)
            if total_duration < 0:
                raise ValueError(f"Invalid duration: {total_duration}")

            total_focus_duration = app_data.get('total_focus_duration', 0)
            if total_focus_duration < 0:
                raise ValueError(f"Invalid duration: {total_focus_duration}")

            total_focus_count = app_data.get('total_focus_count', 0)
            if total_focus_count < 0:
                raise ValueError(f"Invalid focus count: {total_focus_count}")
            
            app_values.append((latest_day_log_id, app_name, total_duration, total_focus_duration, total_focus_count))

            for title_name, title_data in titles.items():
                for title_column in title_data:
                    if title_column not in UsagedataService._app_title_log_columns:
                        raise ValueError(f"Invalid column name: {title_column}")

                total_duration = title_data.get('total_duration', 0)
                if total_duration < 0:
                    raise ValueError(f"Invalid duration: {total_duration}")

                total_focus_duration = title_data.get('total_focus_duration', 0)
                if total_focus_duration < 0:
                    raise ValueError(f"Invalid duration: {total_focus_duration}")

                total_focus_count = title_data.get('total_focus_count', 0)
                if total_focus_count < 0:
                    raise ValueError(f"Invalid focus count: {total_focus_count}")
                
                title_values.append((latest_day_log_id, app_name, title_name, total_duration, total_focus_duration, total_focus_count))

        with self._db.transaction() as tx:
            tx.execute_many(app_log_query, app_values)
            tx.execute_many(title_log_query, title_values)

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
            
            for focus_column in focus_data:
                if focus_column not in UsagedataService._focus_period_columns:
                    raise ValueError(f"Invalid column name: {focus_column}")

            focus_duration = focus_data.get("focus_duration", 0)
            if focus_duration < 0 or focus_duration > 3600:
                raise ValueError(f"Invalid duration: {focus_duration}")

            focus_count = focus_data.get("focus_count", 0)
            if focus_count < 0:
                raise ValueError(f"Invalid focus count: {focus_count}")

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

    def upsert_latest_day_log_title_focus_period(self, app_name: str, title_name: str, title_focus_periods: dict[int, dict[str, int | float]]) -> None:
        if not title_focus_periods:
            return
        
        for hour, focus_data in title_focus_periods.items():
            if hour < 0 or hour > 23:
                raise ValueError(f"Invalid hour: {hour}")
            
            for focus_column in focus_data:
                if focus_column not in UsagedataService._focus_period_columns:
                    raise ValueError(f"Invalid column name: {focus_column}")

            focus_duration = focus_data.get("focus_duration", 0)
            if focus_duration < 0 or focus_duration > 3600:
                raise ValueError(f"Invalid duration: {focus_duration}")

            focus_count = focus_data.get("focus_count", 0)
            if focus_count < 0:
                raise ValueError(f"Invalid focus count: {focus_count}")

        latest_day_log_id = self.get_latest_day_log_id()
        if not latest_day_log_id:
            raise ValueError("No latest day log found.")

        if not self.latest_day_log_app_log_name_exists(app_name):
            raise ValueError(f"App name '{app_name}' not found in the latest day log app logs.")
        
        if not self.latest_day_log_title_log_name_exists(app_name, title_name):
            raise ValueError(f"Title name '{title_name}' for app '{app_name}' not found in the latest day log title logs.")

        query = """
            INSERT INTO title_focus_period (day_log_id, app_name, title_name, day_hour, focus_duration, focus_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(day_log_id, app_name, title_name, day_hour) DO UPDATE SET
                focus_duration = excluded.focus_duration,
                focus_count = excluded.focus_count
        """
        values = []

        for hour, focus_data in title_focus_periods.items():
            values.append((latest_day_log_id, app_name, title_name, hour, focus_data["focus_duration"], focus_data["focus_count"]))

        self._db.execute_many(query, values)

    def remove_oldest_day_log(self) -> None:
        query = """
            DELETE FROM day_log 
            WHERE id = (SELECT id FROM day_log ORDER BY id ASC LIMIT 1);
        """
        self._db.execute(query)