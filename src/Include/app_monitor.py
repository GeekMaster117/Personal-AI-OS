import pywinctl
import psutil
import win32api
import os

import Include.filter.app_title_blacklist as blacklist
import Include.map.system_executable_map as system_executable_map

class AppMonitor:
    def __init__(self) -> None:
        self._app_cache: dict[str, str] = dict()

    def _is_executable_blacklisted(self, executable: str) -> bool:
        # Check if executable is blacklisted

        return executable.lower() in blacklist.app_blacklist

    def _is_title_blacklisted(self, title: str, executable: str) -> bool:
        # Check if title is blacklisted in global and specific scope

        if title.lower() in blacklist.title_blacklist:
            return True
        
        specific_blacklist: set[str] = blacklist.specific_title_blacklist.get(executable.lower(), set())
        if title.lower() in specific_blacklist:
            return True
        
        return False
    
    def _get_app(self, executable: str, pid: int | None) -> str:
        if executable in self._app_cache:
            return self._app_cache[executable]
        if executable.lower() in system_executable_map.system_exe_map:
            return system_executable_map.system_exe_map[executable]
        
        app: str | None = None
        if pid is not None:
            try:
                app = win32api.GetFileVersionInfo(psutil.Process(pid).exe(), "\\StringFileInfo\\040904b0\\ProductName")
            except Exception:
                pass
        if not app:
            app = os.path.splitext(executable)[0]

        self._app_cache[executable] = app
        return app

    def get_all_apps_titles(self) -> tuple[dict[str, str], dict[str, set]]:
        # Fetches all apps and their titles, with executable names
        # Filters hidden and blacklisted apps and titles

        executable_app_map: dict[str, str] = dict()
        executable_title_map: dict[str, set[str]] = dict()

        for window in pywinctl.getAllWindows():
            executable: str = window.getAppName()
            if self._is_executable_blacklisted(executable):
                continue

            title: str = window.title.strip()
            if not title or not window.isVisible:
                continue
            if self._is_title_blacklisted(title, executable):
                continue

            if executable in self._app_cache:
                app = self._app_cache[executable]
            else:
                app = self._get_app(executable, window.getPID())

            if executable not in executable_app_map:
                executable_app_map[executable] = app
                executable_title_map[executable] = set()

            executable_title_map[executable].add(title)

        return executable_app_map, executable_title_map

    def get_active_app_title(self) -> tuple[tuple[str, str], str] | tuple[None, None]:
        # Fetches active app and title, with executable name

        active_window = pywinctl.getActiveWindow()
        if not active_window:
            return None, None

        executable: str = active_window.getAppName()
        title: str = active_window.title.strip()

        if self._is_executable_blacklisted(executable) or self._is_title_blacklisted(title, executable):
            return None, None
        
        app: str | None = self._fetch_app(executable)
        if not app:
            return None, None

        return (executable, app), title