import time
from pathlib import Path

import Include.app_monitor as app_monitor
import settings as settings

from Include.usagedata_db import UsagedataDB

metadata = UsagedataDB(settings.metadata_dir)

def handle_app_data() -> None:
    active_app_title: tuple[str, str] | None = app_monitor.get_active_app_title()
    active_app, active_title = active_app_title
    app_data: dict[str, set[str]] = app_monitor.get_all_apps()

    metadata.update_apps(app_data, active_app, active_title)

while True:
    handle_app_data()

    time.sleep(settings.tick.total_seconds())