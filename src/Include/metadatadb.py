from datetime import datetime, timedelta
from pathlib import Path
import time
from tinydb import TinyDB, Query

import Include.settings as settings

metadata_dir = Path(Path(__file__) / "../../metadata/").resolve()

class MetadataDB:
    def __init__(self):
        metadata_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = metadata_dir / "metadata.json"
        self.db = TinyDB(self.db_path)

        self.apps_open = dict()
        self.active_app = None
        self.active_title = None

        self._ensure_today_log()

    def _ensure_today_log(self):
        datetime_today = datetime.today()
        current_date = datetime_today.date()
        today = datetime_today.isoformat()

        doc_id = max([doc.doc_id for doc in self.db.all()], default=0)
        latest_day = self.db.get(doc_id=doc_id)
        if latest_day and datetime.fromisoformat(latest_day["time_created"]).date() == current_date:
            return

        # Append new day entry
        now_monotonic = time.monotonic()
        new_day = {
            "time_created": today,
            "monotonic_start": now_monotonic,
            "monotonic_last_updated": now_monotonic,
            "apps": {},
            "web": {},
            "github": {},
            "input": {},
            "downtime_periods": {},
            "total_downtime_duration": 0
        }

        self.db.insert(new_day)

    def _convert_mono_to_time(self, monotonic_start, datetime_compare, monotonic_time):
        elapsed_mono = monotonic_time - monotonic_start
        return (datetime.fromisoformat(datetime_compare) + timedelta(seconds=elapsed_mono)).time()

    def update_apps(self, all_apps, active_app=None, active_title=None):
        self._ensure_today_log()  # Re-check in case day rolled over or time zone changed

        doc_id = max(doc.doc_id for doc in self.db.all())
        today_log = self.db.get(doc_id=doc_id)

        apps_log = today_log["apps"]

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

        now = time.monotonic()
        current_hour = str(self._convert_mono_to_time(today_log["monotonic_start"], today_log["time_created"], now).hour)

        if now - today_log["monotonic_last_updated"] > settings.downtime_buffer.total_seconds():
            if current_hour not in today_log["downtime_periods"]:
                today_log["downtime_periods"][current_hour] = {
                    "duration": 0
                }

            today_log["downtime_periods"][current_hour]["duration"] += now - today_log["monotonic_last_updated"]
            today_log["total_downtime_duration"] += now - today_log["monotonic_last_updated"]

            self.apps_open.clear()
            self.apps_open.update(all_apps)

            if active_app and active_title:
                self.active_app = active_app
                self.active_title = active_title

            today_log["monotonic_last_updated"] = now

            self.db.update(today_log, doc_ids=[doc_id])

            return
        
        # Update focus time and count for active app and title
        if active_app and active_title:
            if current_hour not in apps_log[active_app]["focus_periods"]:
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
                apps_log[active_app]["focus_periods"][current_hour]["focus_time"] += now - today_log["monotonic_last_updated"]
                apps_log[active_app]["focus_time"] += now - today_log["monotonic_last_updated"]
        
            if not self.active_app or active_app != self.active_app:
                apps_log[active_app]["focus_periods"][current_hour]["focus_count"] += 1
                apps_log[active_app]["focus_count"] += 1

            if active_title in self.apps_open.get(active_app, {}):
                apps_log[active_app]["titles"][active_title]["focus_periods"][current_hour]["focus_time"] += now - today_log["monotonic_last_updated"]
                apps_log[active_app]["titles"][active_title]["focus_time"] += now - today_log["monotonic_last_updated"]

            if not self.active_title or active_title != self.active_title:
                apps_log[active_app]["titles"][active_title]["focus_periods"][current_hour]["focus_count"] += 1
                apps_log[active_app]["titles"][active_title]["focus_count"] += 1

        # Update durations for all apps and titles
        for app in all_apps:
            if app in self.apps_open:
                apps_log[app]["total_duration"] += now - today_log["monotonic_last_updated"]

            for title in all_apps[app]:
                if title in self.apps_open.get(app, {}):
                    apps_log[app]["titles"][title]["total_duration"] += now - today_log["monotonic_last_updated"]

        # Update apps_open with current state
        self.apps_open.clear()
        self.apps_open.update(all_apps)

        # Update active app and title
        if active_app and active_title:
            self.active_app = active_app
            self.active_title = active_title

        # Update log and monotonic_last_updated
        today_log["apps"] = apps_log
        today_log["monotonic_last_updated"] = now

        self.db.update(today_log, doc_ids=[doc_id])

    def get_today_log(self):
        self._ensure_today_log()

        doc_id = max([doc.doc_id for doc in self.db.all()], default=0)
        return self.db.get(doc_id=doc_id)

    def close(self):
        self.db.close()