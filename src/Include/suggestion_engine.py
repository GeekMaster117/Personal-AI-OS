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
        self.processed_logs = queue.Queue()

        if self.db_handler.get_log_count() > 1:
            doc_ids = sorted(self.db_handler.get_document_ids())

            for i in range(len(doc_ids) - 1):
                self._process_log(doc_ids[i])

    def _score(self, app_or_title: dict) -> float:
        weight1 = 0.2
        weight2 = 15

        return app_or_title["focus_time"] + (weight1 * app_or_title["total_duration"]) + (weight2 * app_or_title["focus_count"])
    
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

    def _preprocess_log(self, log_id: int) -> None:
        log = self.db_handler.get_log(log_id)

        top_apps = heapq.nlargest(settings.data_limit, log["apps"].items(), key=lambda x: self._score(x[1]))
        for app in top_apps:
            app[1]["titles"] = heapq.nlargest(settings.data_limit, app[1]["titles"].items(), key=lambda x: self._score(x[1]))

        log["apps"] = top_apps

        return log

    def _process_log(self, log_id: int) -> None:
        log = self._preprocess_log(log_id)

        summary = textwrap.dedent(f"""
        Date Created: {datetime.fromisoformat(log['time_anchor']).date().isoformat()}
        Top {settings.data_limit} Apps and their Top {settings.data_limit} Titles:""")

        for i, app in enumerate(log["apps"]):
            summary += textwrap.dedent(f""" 
            {i + 1}. {app[0]}:
            - Total Focus Duration: {self._round_off(app[1]['focus_time'])}
            - Total Duration: {self._round_off(app[1]['total_duration'])}
            - Hourly Focus Duration: [{', '.join(f"{self._twelvehour_format(int(hour))}: {self._round_off(attributes['focus_time'])}" for hour, attributes in app[1]['focus_periods'].items())}]
            """)

            for j, title in enumerate(app[1]["titles"]):
                summary += textwrap.dedent(f"""
                - {i + 1}.{j + 1}. {title[0]}:
                -- Total Focus Duration: {self._round_off(title[1]['focus_time'])}
                -- Total Duration: {self._round_off(title[1]['total_duration'])}
                -- Hourly Focus Duration: [{', '.join(f"{self._twelvehour_format(int(hour))}: {self._round_off(attributes['focus_time'])}" for hour, attributes in title[1]['focus_periods'].items())}]
                """)

        self.processed_logs.put(summary)