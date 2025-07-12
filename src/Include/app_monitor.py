import pywinctl
import psutil

import Include.app_title_blacklist as blacklist

def is_app_blacklisted(app_name):
    return app_name.lower() in blacklist.app_blacklist

def is_title_blacklisted(window_title, app_name):
    # Check if the window title is blacklisted
    if window_title.lower() in blacklist.title_blacklist:
        return True
    
    # Check if the app has specific blacklisted windows
    specific_blacklist = blacklist.specific_title_blacklist.get(app_name.lower(), set())
    if window_title.lower() in specific_blacklist:
        return True
    
    return False

def get_all_apps():
    result = {}
    
    for win in pywinctl.getAllWindows():
        try:
            pid = win.getPID()
            proc = psutil.Process(pid)
            name = proc.name()

            # Skip blacklisted apps
            if is_app_blacklisted(name):
                continue

            title = win.title.strip()
            
            # Skip hidden or empty
            if not title or not win.isVisible:
                continue

            # Skip blacklisted windows
            if is_title_blacklisted(title, name):
                continue

            # Group all window titles under one app
            if name not in result:
                result[name] = set()
            result[name].add(title)

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return result

def get_active_app_title():
    try:
        active_window = pywinctl.getActiveWindow()
        if not active_window:
            return None, None

        pid = active_window.getPID()
        proc = psutil.Process(pid)
        name = proc.name()
        title = active_window.title.strip()

        # Skip blacklisted apps
        if is_app_blacklisted(name) or is_title_blacklisted(title, name):
            return None, None

        return name, title

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None, None