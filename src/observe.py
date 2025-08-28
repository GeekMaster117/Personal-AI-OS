import time
import signal

import Include.app_monitor as app_monitor
import settings
from Include.usagedata_db import UsagedataDB
from Include.verify_install import verify_installation

try:
    verify_installation()
except Exception as e:
    print(f"Installation verification failed: {e}. Please run install.exe")

    input("\nPress any key to exit...")
    exit(1)

shutdown_request: bool = False

def shutdown_handler(signum, frame) -> None:
    global shutdown_request
    print("Shutting down...", flush=True)
    shutdown_request = True

def handle_app_data() -> None:
    active_app_title: tuple[str, str] | None = app_monitor.get_active_app_title()
    active_app, active_title = active_app_title
    app_data: dict[str, set[str]] = app_monitor.get_all_apps()

    usagedataDB.update_apps(app_data, active_app, active_title)

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