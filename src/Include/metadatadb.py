from datetime import datetime, timedelta
from pathlib import Path
import time
from tinydb import TinyDB

import settings

class MetadataDB:
    def __init__(self, metadata_dir: str):
        metadata_dir = Path(metadata_dir)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        self.db_path: Path = metadata_dir / "metadata.json"
        self.db: TinyDB = TinyDB(self.db_path)

        self.apps_open: dict[str, set[str]] = dict()
        self.active_app: str | None = None
        self.active_title: str | None = None

        self._ensure_log_integrity()

    def _ensure_log_integrity(self) -> None:
        self._ensure_today_log()
        self._ensure_max_logs()

    def _ensure_today_log(self) -> None:
        datetime_today: datetime = datetime.today()
        current_date: datetime.date = datetime_today.date()
        today: str = datetime_today.isoformat()

        doc_id: int = max([doc.doc_id for doc in self.db.all()], default=0)
        latest_day: dict = self.db.get(doc_id=doc_id)
        if latest_day and datetime.fromisoformat(latest_day["time_anchor"]).date() == current_date:
            return

        # Append new day entry
        now_monotonic: float = time.monotonic()
        new_day: dict = {
            "time_anchor": today,
            "monotonic_anchor": now_monotonic,
            "monotonic_last_updated": now_monotonic,
            "apps": {},
            "web": {},
            "github": {},
            "input": {},
            "downtime_periods": {},
            "total_downtime_duration": 0,
            "total_anomalies": 0
        }

        self.db.insert(new_day)

    def _ensure_max_logs(self) -> None:
        doc_ids: list[int] = sorted([doc.doc_id for doc in self.db.all()])
        if len(doc_ids) > settings.max_logs:
            excess_count: int = len(doc_ids) - settings.max_logs
            self.db.remove(doc_ids=doc_ids[:excess_count])

    def _convert_mono_to_time(self, monotonic_anchor: float, datetime_compare: str, monotonic_time: float) -> datetime.time:
        elapsed_mono: float = monotonic_time - monotonic_anchor
        return (datetime.fromisoformat(datetime_compare) + timedelta(seconds=elapsed_mono)).time()

    def update_apps(self, all_apps: dict[str, set[str]], active_app: str | None = None, active_title: str | None = None) -> None:
        self._ensure_log_integrity()

        doc_id: int = max(doc.doc_id for doc in self.db.all())
        today_log: dict = self.db.get(doc_id=doc_id)

        apps_log: dict = today_log["apps"]

        for app in all_apps:
            if app not in apps_log:
                apps_log[app] = {
                    "titles": {},
                    "total_duration": 0,
                    "focus_periods": {},
                    "focus_time": 0,
                    "focus_count": 0
                }
            
            for title in all_apps[app]:
                if title not in apps_log[app]["titles"]:
                    apps_log[app]["titles"][title] = {
                        "total_duration": 0,
                        "focus_periods": {},
                        "focus_time": 0,
                        "focus_count": 0
                    }
        
        if active_app and active_title:
            if active_app not in apps_log:
                apps_log[active_app] = {
                    "titles": {},
                    "total_duration": 0,
                    "focus_periods": {},
                    "focus_time": 0,
                    "focus_count": 0
                }
            if active_title not in apps_log[active_app]["titles"]:
                apps_log[active_app]["titles"][active_title] = {
                    "total_duration": 0,
                    "focus_periods": {},
                    "focus_time": 0,
                    "focus_count": 0
                }

        now: float = time.monotonic()
        now_datetime: datetime = datetime.today()
        current_hour: str = str(self._convert_mono_to_time(today_log["monotonic_anchor"], today_log["time_anchor"], now).hour)

        datetime_shift: timedelta = now_datetime - datetime.fromisoformat(today_log["time_anchor"])
        monotime_shift: float = now - today_log["monotonic_anchor"]
        if abs(datetime_shift.total_seconds() - monotime_shift) > settings.time_threshold.total_seconds():
            today_log["time_anchor"] = now_datetime.isoformat()
            today_log["monotonic_anchor"] = now
            today_log["monotonic_last_updated"] = now

            today_log["total_anomalies"] += 1

            self.apps_open.clear()
            self.apps_open.update(all_apps)

            if active_app and active_title:
                self.active_app: str = active_app
                self.active_title: str = active_title

            return

        downtime: float = now - today_log["monotonic_last_updated"]
        if downtime > settings.time_threshold.total_seconds():
            today_log["total_downtime_duration"] += downtime

            last_update_timestamp: datetime.time = self._convert_mono_to_time(today_log["monotonic_anchor"], today_log["time_anchor"], today_log["monotonic_last_updated"])
            last_update_hour: int = last_update_timestamp.hour

            blackout_hours: int = (int(current_hour) - last_update_hour) % 24
            while blackout_hours > 1:
                blackout_hour: str = str((last_update_hour + blackout_hours - 1) % 24)
                if blackout_hour not in today_log["downtime_periods"]:
                    today_log["downtime_periods"][blackout_hour] = {
                        "duration": 0
                    }
                today_log["downtime_periods"][blackout_hour]["duration"] = 3600

                downtime -= 3600
                blackout_hours -= 1

            downtime: float = max(0, downtime)

            if blackout_hours == 1:
                last_update_hour: str = str(last_update_hour)

                if last_update_hour not in today_log["downtime_periods"]:
                    today_log["downtime_periods"][last_update_hour] = {
                        "duration": 0
                    }

                last_update_hour_downtime: int = 3600 - (last_update_timestamp.minute * 60 + last_update_timestamp.second)
                today_log["downtime_periods"][last_update_hour]["duration"] += last_update_hour_downtime

                downtime -= last_update_hour_downtime

            downtime: float = max(0, downtime)

            if current_hour not in today_log["downtime_periods"]:
                today_log["downtime_periods"][current_hour] = {
                    "duration": 0
                }
            today_log["downtime_periods"][current_hour]["duration"] += downtime

            self.apps_open.clear()
            self.apps_open.update(all_apps)

            if active_app and active_title:
                self.active_app: str = active_app
                self.active_title: str = active_title

            today_log["monotonic_last_updated"] = now

            self.db.update(today_log, doc_ids=[doc_id])

            return
        
        elapsed_time: float = now - today_log["monotonic_last_updated"]
        
        # Update focus time and count for active app and title
        if active_app and active_title:
            if active_app in apps_log and current_hour not in apps_log[active_app]["focus_periods"]:
                apps_log[active_app]["focus_periods"][current_hour] = {
                    "focus_time": 0,
                    "focus_count": 0
                }
            if current_hour not in apps_log[active_app]["titles"][active_title]["focus_periods"]:
                apps_log[active_app]["titles"][active_title]["focus_periods"][current_hour] = {
                    "focus_time": 0,
                    "focus_count": 0
                }

            if active_app in self.apps_open:
                apps_log[active_app]["focus_periods"][current_hour]["focus_time"] += elapsed_time
                apps_log[active_app]["focus_time"] += elapsed_time

            if not self.active_app or active_app != self.active_app:
                apps_log[active_app]["focus_periods"][current_hour]["focus_count"] += 1
                apps_log[active_app]["focus_count"] += 1

            if active_title in self.apps_open.get(active_app, {}):
                apps_log[active_app]["titles"][active_title]["focus_periods"][current_hour]["focus_time"] += elapsed_time
                apps_log[active_app]["titles"][active_title]["focus_time"] += elapsed_time

            if not self.active_title or active_title != self.active_title:
                apps_log[active_app]["titles"][active_title]["focus_periods"][current_hour]["focus_count"] += 1
                apps_log[active_app]["titles"][active_title]["focus_count"] += 1

        # Update durations for all apps and titles
        for app in all_apps:
            if app in self.apps_open:
                apps_log[app]["total_duration"] += elapsed_time

            for title in all_apps[app]:
                if title in self.apps_open.get(app, {}):
                    apps_log[app]["titles"][title]["total_duration"] += elapsed_time

        # Update apps_open with current state
        self.apps_open.clear()
        self.apps_open.update(all_apps)

        # Update active app and title
        if active_app and active_title:
            self.active_app: str = active_app
            self.active_title: str = active_title

        # Update log and monotonic_last_updated
        today_log["apps"] = apps_log
        today_log["monotonic_last_updated"] = now

        self.db.update(today_log, doc_ids=[doc_id])

    def get_log_count(self) -> int:
        self._ensure_log_integrity()

        return len(self.db.all())
    
    def get_document_ids(self) -> list[int]:
        self._ensure_log_integrity()

        return [doc.doc_id for doc in self.db.all()]

    def get_recent_log(self) -> dict:
        self._ensure_log_integrity()

        doc_ids = max([doc.doc_id for doc in self.db.all()])
        return self.db.get(doc_id=doc_ids[-1])

    def get_log(self, doc_id: int) -> dict:
        self._ensure_log_integrity()

        if doc_id < 0 or doc_id > max([doc.doc_id for doc in self.db.all()]):
            raise ValueError("doc_id is out of range")

        return self.db.get(doc_id=doc_id)

    def close(self):
        self._ensure_log_integrity()

        self.db.close()