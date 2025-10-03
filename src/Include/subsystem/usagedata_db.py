import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from Include.service.usagedata_service import UsagedataService

import settings

class UsagedataDB:
    def __init__(self, usagedata_dir: str):
        usagedata_dir: Path = Path(usagedata_dir)
        usagedata_dir.mkdir(parents=True, exist_ok=True)

        self.db_path: Path = usagedata_dir / "usagedata.db"
        self._service: UsagedataService = UsagedataService(self.db_path)

        self.apps_open: dict[str, set[str]] = dict()
        self.active_app: str | None = None
        self.active_title: str | None = None

        self._service.create_if_not_exists_schema()

        self._ensure_log_integrity()
    def _ensure_log_integrity(self) -> None:
        self._ensure_today_log()
        self._ensure_max_logs()

    def _ensure_today_log(self) -> None:
        datetime_today: datetime = datetime.today()
        current_date: datetime.date = datetime_today.date()
        today: str = datetime_today.isoformat()

        latest_day: dict | None = self._service.get_latest_daylog(("time_anchor",))
        if latest_day and datetime.fromisoformat(latest_day["time_anchor"]).date() == current_date:
            return

        now_monotonic: float = time.monotonic()
        self._service.add_daylog(today, now_monotonic)

    def _ensure_max_logs(self) -> None:
        while self._service.get_daylog_rowcount() > settings.max_logs:
            self._service.remove_oldest_daylog()

    def _convert_mono_to_time(self, monotonic_anchor: float, datetime_compare: str, monotonic_time: float) -> Any:
        elapsed_mono: float = monotonic_time - monotonic_anchor
        return (datetime.fromisoformat(datetime_compare) + timedelta(seconds=elapsed_mono)).time()

    def update_apps(self, app_title_map: dict[str, set[str]], app_executable_path: dict[str, str], active_app: str | None = None, active_title: str | None = None) -> None:
        self._ensure_log_integrity()

        today_log: dict[str, float | int] = self._service.get_latest_daylog()

        now: float = time.monotonic()
        now_datetime: datetime = datetime.today()

        datetime_shift: timedelta = now_datetime - datetime.fromisoformat(today_log["time_anchor"])
        monotime_shift: float = now - today_log["monotonic_start"]
        if abs(datetime_shift.total_seconds() - monotime_shift) > settings.time_threshold.total_seconds():
            today_log["time_anchor"] = now_datetime.isoformat()
            today_log["monotonic_start"] = now
            today_log["monotonic_last_updated"] = now
            today_log["total_anomalies"] += 1

            self.apps_open.clear()
            self.apps_open.update(app_title_map)

            if active_app and active_title:
                self.active_app: str = active_app
                self.active_title: str = active_title

            self._service.update_latest_daylog(today_log)

            return
        
        current_hour: int = self._convert_mono_to_time(today_log["monotonic_start"], today_log["time_anchor"], now).hour

        downtime: float = now - today_log["monotonic_last_updated"]
        if downtime > settings.time_threshold.total_seconds():
            today_log["total_downtime_duration"] += downtime

            last_update_timestamp: Any = self._convert_mono_to_time(today_log["monotonic_start"], today_log["time_anchor"], today_log["monotonic_last_updated"])
            last_update_hour: int = last_update_timestamp.hour

            downtime_period: dict[int, float] = self._service.get_latest_downtimeperiod()

            blackout_hours: int = (current_hour - last_update_hour) % 24
            while blackout_hours > 1:
                blackout_hour: int = (last_update_hour + blackout_hours - 1) % 24
                if blackout_hour not in downtime_period:
                    downtime_period[blackout_hour] = 0
                downtime_period[blackout_hour] = 3600

                downtime -= 3600
                blackout_hours -= 1

            downtime: float = max(0, downtime)

            if blackout_hours == 1:
                if last_update_hour not in downtime_period:
                    downtime_period[last_update_hour] = 0

                last_update_hour_downtime: int = 3600 - (last_update_timestamp.minute * 60 + last_update_timestamp.second)
                downtime_period[last_update_hour] = min(3600, downtime_period[last_update_hour] + last_update_hour_downtime)

                downtime -= last_update_hour_downtime

            downtime: float = max(0, downtime)

            if current_hour not in downtime_period:
                downtime_period[current_hour] = 0
            downtime_period[current_hour] = min(3600, downtime_period[current_hour] + downtime)

            self.apps_open.clear()
            self.apps_open.update(app_title_map)

            if active_app and active_title:
                self.active_app: str = active_app
                self.active_title: str = active_title

            today_log["monotonic_last_updated"] = now

            self._service.upsert_latest_downtimeperiod(downtime_period)
            self._service.update_latest_daylog(today_log)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

            return
        
        elapsed_time: float = now - today_log["monotonic_last_updated"]

        apps_titles: dict[str, dict[str, int | float | dict[str, str | int | float]]] = self._service.get_latest_applog_titlelog()

        # Update focus time and count for active app and active title
        if active_app and active_title and active_app in apps_titles:
            active_app_focus_period: dict[str, dict[int, float]] = self._service.get_latest_appfocusperiod(active_app)
            if current_hour not in active_app_focus_period:
                active_app_focus_period[current_hour] = {
                    "focus_duration": 0,
                    "focus_count": 0
                }

            if active_app in self.apps_open:
                active_app_focus_period[current_hour]["focus_duration"] = min(3600, active_app_focus_period[current_hour]["focus_duration"] + elapsed_time)
                apps_titles[active_app]["total_focus_duration"] += elapsed_time

            if not self.active_app or active_app != self.active_app:
                active_app_focus_period[current_hour]["focus_count"] += 1
                apps_titles[active_app]["total_focus_count"] += 1

            self._service.upsert_latest_appfocusperiod(active_app, active_app_focus_period)

            if active_title in apps_titles[active_app]["titles"]:
                active_title_focus_period: dict[str, dict[int, float]] = self._service.get_latest_titlefocusperiod(active_app, active_title)
                if current_hour not in active_title_focus_period:
                    active_title_focus_period[current_hour] = {
                        "focus_duration": 0,
                        "focus_count": 0
                    }

                if active_title in self.apps_open.get(active_app, {}):
                    active_title_focus_period[current_hour]["focus_duration"] = min(3600, active_title_focus_period[current_hour]["focus_duration"] + elapsed_time)
                    apps_titles[active_app]["titles"][active_title]["total_focus_duration"] += elapsed_time

                if not self.active_title or active_title != self.active_title:
                    active_title_focus_period[current_hour]["focus_count"] += 1
                    apps_titles[active_app]["titles"][active_title]["total_focus_count"] += 1

                self._service.upsert_latest_titlefocusperiod(active_app, active_title, active_title_focus_period)

        # Ensure all apps and titles are present in the database
        for app in app_title_map:
            if app not in apps_titles:
                apps_titles[app] = {
                    "executable_path": app_executable_path[app],
                    "total_duration": 0,
                    "total_focus_duration": 0,
                    "total_focus_count": 0,
                    "titles": {}
                }
            for title in app_title_map[app]:
                if title not in apps_titles[app]["titles"]:
                    apps_titles[app]["titles"][title] = {
                        "total_duration": 0,
                        "total_focus_duration": 0,
                        "total_focus_count": 0
                    }

        # Update executable and durations for all apps and titles
        for app in app_title_map:
            if app in self.apps_open:
                apps_titles[app]["total_duration"] += elapsed_time

            for title in app_title_map[app]:
                if title in self.apps_open.get(app, {}):
                    apps_titles[app]["titles"][title]["total_duration"] += elapsed_time

        # Update apps_open with current state
        self.apps_open.clear()
        self.apps_open.update(app_title_map)

        # Update active app and title
        if active_app and active_title:
            self.active_app: str = active_app
            self.active_title: str = active_title

        today_log["monotonic_last_updated"] = now

        self._service.upsert_latest_applog_titlelog(apps_titles)
        self._service.update_latest_daylog(today_log)
    
    def get_daylog_ids(self) -> list[int]:
        self._ensure_log_integrity()

        return self._service.get_daylog_ids()

    def get_recent_daylog(self, columns: tuple[str] | None = None) -> dict[str, float | int]:
        self._ensure_log_integrity()

        return self._service.get_latest_applog_titlelog(columns)

    def get_daylog(self, day_log_id: int, columns: tuple[str] | None = None) -> dict[str, float | int]:
        self._ensure_log_integrity()

        return self._service.get_daylog(day_log_id, columns)

    def get_applog_titlelog(self, day_log_id: int) -> dict[str, dict[str, int | float | dict[str, str | int | float]]]:
        self._ensure_log_integrity()

        return self._service.get_applog_titlelog(day_log_id)
    
    def get_appfocusperiod(self, day_log_id: int, app_name: str) -> dict[int, dict[str, float]]:
        self._ensure_log_integrity()

        return self._service.get_appfocusperiod(day_log_id, app_name)

    def get_titlefocusperiod(self, day_log_id: int, app_name: str, title_name: str) -> dict[int, dict[str, float]]:
        self._ensure_log_integrity()

        return self._service.get_titlefocusperiod(day_log_id, app_name, title_name)