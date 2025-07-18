import time

import Include.core.app_monitor as app_monitor
import Include.core.metadatadb as metadata_db
import Include.core.settings as settings

metadata = metadata_db.MetadataDB()

def handle_app_data() -> None:
    active_app_title: tuple[str, str] | None = app_monitor.get_active_app_title()
    active_app, active_title = active_app_title
    app_data: dict[str, set[str]] = app_monitor.get_all_apps()

    metadata.update_apps(app_data, active_app, active_title)

while True:
    handle_app_data()

    time.sleep(settings.tick.total_seconds())