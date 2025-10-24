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
    
    def _save_app_cache(self, executable: str, app) -> None:
        self._app_cache[executable] = app
    
    def _get_app_cache(self, executable: str) -> str | None:
        return self._app_cache.get(executable)
    
    def _get_app_system(self, executable: str) -> str | None:
        return system_executable_map.system_exe_map.get(executable.lower())
    
    def _get_app_api(self, executable_path: str) -> str | None:
        try:
            return win32api.GetFileVersionInfo(executable_path, "\\StringFileInfo\\040904b0\\ProductName").lower()
        except Exception:
            return None
        
    def _get_app_default(self, executable: str) -> str:
        return os.path.splitext(executable)[0].lower()

    def _get_executable_path(self, pid: int) -> str:
        return psutil.Process(pid).exe()
    
    def _get_app(self, executable: str, executable_path: str) -> str:
        if not executable_path.endswith(executable):
            raise ValueError(f'{executable} does not represent the path {executable_path}')
        
        app: str | None = None
        app = self._get_app_cache(executable)
        if not app:
            app = self._get_app_system(executable)

        if not app:
            app = self._get_app_api(executable_path)
            if not app:
                app = self._get_app_default(executable)

            self._save_app_cache(executable, app)

        return app

    def get_all_apps_titles_executablepaths(self) -> tuple[dict[str, set], dict[str, str]]:
        # Fetches all apps and their titles, with executable names
        # Filters hidden and blacklisted apps and titles

        app_executablepath_map: dict[str, str] = dict()
        app_title_map: dict[str, set[str]] = dict()

        for window in pywinctl.getAllWindows():
            executable_path: str = self._get_executable_path(window.getPID())
            if not executable_path:
                continue

            executable: str = os.path.basename(executable_path)
            if not executable or self._is_executable_blacklisted(executable):
                continue

            title: str = window.title.strip()
            if not title or self._is_title_blacklisted(title, executable):
                continue

            app = self._get_app(executable, executable_path)
            if app not in app_executablepath_map:
                app_executablepath_map[app] = executable_path
                app_title_map[app] = set()

            app_title_map[app].add(title)

        return app_title_map, app_executablepath_map

    def get_active_app_title(self) -> tuple[str, str, str] | tuple[None, None, None]:
        # Fetches active app and title, with executable name

        active_window = pywinctl.getActiveWindow()
        if not active_window:
            return None, None, None
        
        executable_path: str = self._get_executable_path(active_window.getPID())
        if not executable_path:
            return None, None, None

        executable: str = os.path.basename(executable_path)
        if not executable or self._is_executable_blacklisted(executable):
            return None, None, None
        
        title: str = active_window.title.strip()
        if not title or self._is_title_blacklisted(title, executable):
            return None, None, None
        
        app = self._get_app(executable, executable_path)

        return app, title