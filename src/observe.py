import signal
import platform

import time

import textwrap

from Include.app_monitor import AppMonitor
import settings
from Include.subsystem.usagedata_db import UsagedataDB

shutdown_request: bool = False

def shutdown_handler(signum, frame) -> None:
    global shutdown_request
    print("Shutting down...", flush=True)
    shutdown_request = True

def handle_app_data(app_monitor: AppMonitor) -> None:
    active_app, active_title = app_monitor.get_active_app_title()

    app_title_map, app_executablepath_map = app_monitor.get_all_apps_titles_executablepaths()

    usagedataDB.update_apps(app_title_map, app_executablepath_map, active_app, active_title)

prototype_message = textwrap.dedent("""
=================== Personal AI OS Prototype =======================
This is an early release. Solid, but still evolving. Explore freely!
====================================================================
""")

if __name__ == "__main__":
    print(prototype_message)

    os_name = platform.system()
    if os_name not in settings.SupportedOS:
        raise NotImplementedError(f"Unsupported operating system: {os_name}")
    os_name = settings.SupportedOS(os_name)

    app_monitor = AppMonitor(os_name)

    usagedataDB = UsagedataDB(settings.usagedata_dir)
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    sleep_interval = 1

    print("Press Ctrl+C to stop")

    while not shutdown_request:
        handle_app_data(app_monitor)

        elapsed_time = 0
        while elapsed_time < settings.tick.total_seconds() and not shutdown_request:
            time.sleep(sleep_interval)
            elapsed_time += sleep_interval

    input("\nPress any key to exit...")