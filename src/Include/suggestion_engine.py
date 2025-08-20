import threading
import heapq
import queue
from datetime import datetime
import textwrap

from Include.usagedata_db import UsagedataDB
import settings

class SuggestionEngine:
    def __init__(self, db_handler: UsagedataDB):
        self.db_handler = db_handler
        self.preprocessed_logs = queue.Queue()

        day_log_ids = self.db_handler.get_day_log_ids()
        threads = []

        for i in range(len(day_log_ids)):
            t = threading.Thread(target=self._preprocess_log, args=(day_log_ids[i],))
            t.start()
            threads.append(t)

    def _score(self, app_or_title: dict) -> float:
        weight1 = 0.2
        weight2 = 15

        return app_or_title["total_focus_duration"] + (weight1 * app_or_title["total_duration"]) + (weight2 * app_or_title["total_focus_count"])

    def _twelvehour_format(self, hour: int) -> str:
        if hour < 0 or hour > 23:
            raise ValueError("Hour must be between 0 and 23")
        
        if hour == 0:
            return "12 AM"
        elif hour < 12:
            return f"{hour} AM"
        elif hour == 12:
            return "12 PM"
        else:
            return f"{hour - 12} PM"
        
    def _round_off(self, seconds: float) -> int:
        if seconds >= 3600:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
        elif seconds >= 60:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            return f"{round(seconds)} seconds"

    def _top_apps_titles(self, day_log_id: int) -> dict[str, dict[str, int | float | dict[str, str | int | float]]]:
        apps_titles = self.db_handler.get_app_log_title_log(day_log_id)

        apps_titles = heapq.nlargest(settings.data_limit, apps_titles.items(), key=lambda x: self._score(x[1]))

        #dict gets converted to tuple by heapq.nlargest, so we convert it back to dict
        apps_titles = dict(apps_titles)

        for app_data in apps_titles.values():
            app_data["titles"] = heapq.nlargest(settings.data_limit, app_data["titles"].items(), key=lambda x: self._score(x[1]))

            #dict gets converted to tuple by heapq.nlargest, so we convert it back to dict
            app_data["titles"] = dict(app_data["titles"])

        return apps_titles

    def _preprocess_log(self, day_log_id: int) -> None:
        day_log = self.db_handler.get_day_log(day_log_id, ('time_anchor',))
        apps_titles = self._top_apps_titles(day_log_id)

        summary = textwrap.dedent(f"""
        Date Created: {datetime.fromisoformat(day_log['time_anchor']).date().isoformat()}
        Top {settings.data_limit} Apps and their Top {settings.data_limit} Titles:""")

        for i, (app_name, app_data) in enumerate(apps_titles.items()):
            app_focus_period = self.db_handler.get_app_focus_period(day_log_id, app_name)
            summary += textwrap.dedent(f""" 
            {i + 1}. {app_name}:
            - Total Focus Duration: {self._round_off(app_data['total_focus_duration'])}
            - Total Duration: {self._round_off(app_data['total_duration'])}
            - Hourly Focus Duration: [{', '.join(f"{self._twelvehour_format(int(hour))}: {self._round_off(attributes['focus_duration'])}" for hour, attributes in app_focus_period.items())}]
            """)

            for j, (title_name, title_data) in enumerate(app_data["titles"].items()):
                title_focus_period = self.db_handler.get_title_focus_period(day_log_id, app_name, title_name)
                summary += textwrap.dedent(f"""
                - {i + 1}.{j + 1}. {title_name}:
                -- Total Focus Duration: {self._round_off(title_data['total_focus_duration'])}
                -- Total Duration: {self._round_off(title_data['total_duration'])}
                -- Hourly Focus Duration: [{', '.join(f"{self._twelvehour_format(int(hour))}: {self._round_off(attributes['focus_duration'])}" for hour, attributes in title_focus_period.items())}]
                """)

        self.preprocessed_logs.put(summary)