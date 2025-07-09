import app_monitor
import Include.metadatadb as metadata_db

app_data = app_monitor.get_all_apps()
metadata = metadata_db.MetadataDB()

metadata.load_app_data()
for app_name, titles in app_data.items():
    for title in titles:
        metadata.update_app(app_name, title, 0, 0, 0)
metadata.save_app_data()