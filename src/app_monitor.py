import pywinctl
import psutil

import Include.app_window_blacklist as blacklist

def get_all_apps():
    result = {}
    
    for win in pywinctl.getAllWindows():
        try:
            pid = win.getPID()
            proc = psutil.Process(pid)
            name = proc.name()

            # Skip blacklisted apps
            if name.lower() in blacklist.app_blacklist:
                continue

            title = win.title.strip()
            
            # Skip hidden or empty
            if not title or not win.isVisible:
                continue

            # Skip blacklisted windows
            if title.lower() in blacklist.window_blacklist or title.lower() in blacklist.specific_window_blacklist.get(name.lower(), set()):
                continue

            # Group all window titles under one app
            if name not in result:
                result[name] = set()
            result[name].add(title)

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return result