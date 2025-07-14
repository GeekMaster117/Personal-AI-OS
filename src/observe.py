import time

import Include.app_monitor as app_monitor
import Include.metadatadb as metadata_db
import Include.settings as settings

metadata = metadata_db.MetadataDB()

def handle_app_data():
    active_app, active_title = app_monitor.get_active_app_title()
    app_data = app_monitor.get_all_apps()

    metadata.update_apps(app_data, active_app, active_title)

while True:
    handle_app_data()

    time.sleep(settings.tick.total_seconds())