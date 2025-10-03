import time
import signal
import textwrap

from Include.app_monitor import AppMonitor
import settings
from Include.subsystem.usagedata_db import UsagedataDB

shutdown_request: bool = False
app_monitor = AppMonitor()

def shutdown_handler(signum, frame) -> None:
    global shutdown_request
    print("Shutting down...", flush=True)
    shutdown_request = True

def handle_app_data() -> None:
    active_app, active_title = app_monitor.get_active_app_title()

    app_title_map, app_executablepath_map = app_monitor.get_all_apps_titles_executablepaths()

    usagedataDB.update_apps(app_title_map, app_executablepath_map, active_app, active_title)

prototype_message = textwrap.dedent("""
=================== Personal AI OS Prototype =======================
This is an early release. Solid, but still evolving. Explore freely!
====================================================================
""")
print(prototype_message)

usagedataDB = UsagedataDB(settings.usagedata_dir)
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

sleep_interval = 1

print("Press Ctrl+C to stop")

while not shutdown_request:
    handle_app_data()

    elapsed_time = 0
    while elapsed_time < settings.tick.total_seconds() and not shutdown_request:
        time.sleep(sleep_interval)
        elapsed_time += sleep_interval

input("\nPress any key to exit...")