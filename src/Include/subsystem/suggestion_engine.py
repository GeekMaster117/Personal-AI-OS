import threading
import heapq
import textwrap
from datetime import datetime

from Include.subsystem.usagedata_db import UsagedataDB
from Include.service.suggestion_engine_service import SuggestionEngineService
from Include.service.suggestion_engine_service import SuggestionType
from Include.loading_spinner import loading_spinner
import settings

class SuggestionEngine:
    def __init__(self, db_handler: UsagedataDB):
        try:
            self._service = SuggestionEngineService()
        except Exception as e:
            raise RuntimeError(f"Error initializing SuggestionEngineService: {e}")

        self._db_handler: UsagedataDB = db_handler
        self._day_log_ids: list[int] = self._db_handler.get_day_log_ids() # Assumes that day logs are sorted in ascending order
        if len(self._day_log_ids) == 0:
            raise RuntimeError("No day logs found in the database.")

        self.preprocessed_logs: dict[int, str] = dict()
        self.preprocess_threads: list[threading.Thread] = []

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
        
    def _aggregate_focus_hours(self, focus_period: dict[int, dict[str, float]]) -> list[str]:
        hours = sorted(focus_period.keys()) + [float("inf")]
        aggregated_hours = []
        start = prev = hours[0]

        for hour in hours:
            if hour - prev > 1:
                if start != prev:
                    aggregated_hours.append(f"{self._twelvehour_format(start)} - {self._twelvehour_format(prev)}")
                else:
                    aggregated_hours.append(f"{self._twelvehour_format(start)}")

                start = hour
            prev = hour
        
        return aggregated_hours

    # top data dict structure
    # {
    #   "app_name": {
    #       "total_duration": float,
    #       "total_focus_duration": float,
    #       "hourly_focus_data" (Present only if aggregate set to False): {
    #           "hour": {
    #               "focus_duration": float,
    #               "focus_count": integer
    #           }
    #       },
    #       "aggregated_focus_duration" (Present only if aggregate set to True): list[str],
    #       "titles": {
    #           "title_name": {
    #               "total_duration": float,
    #               "total_focus_duration": float,
    #               "hourly_focus_data" (Present only if aggregate set to False): {
    #                   "hour": {
    #                       "focus_duration": float,
    #                       "focus_count": integer
    #                   }
    #               },
    #               "aggregated_focus_duration" (Present only if aggregate set to True): list[str]
    #           }
    #       }
    #   }
    # }
    def _top_data(self, day_log_id: int, only_apps: bool = False, aggregate: bool = False) -> dict:
        apps_titles = self._db_handler.get_app_log_title_log(day_log_id)

        apps_titles = heapq.nlargest(settings.data_limit, apps_titles.items(), key=lambda x: self._score(x[1]))

        #dict gets converted to tuple by heapq.nlargest, so we convert it back to dict
        apps_titles = dict(apps_titles)

        for app_name, app_data in apps_titles.items():
            if aggregate:
                app_data["aggregated_focus_duration"] = self._aggregate_focus_hours(self._db_handler.get_app_focus_period(day_log_id, app_name))
            else:
                app_data["hourly_focus_data"] = self._db_handler.get_app_focus_period(day_log_id, app_name)

            if only_apps:
                continue

            app_data["titles"] = heapq.nlargest(settings.data_limit, app_data["titles"].items(), key=lambda x: self._score(x[1]))

            #dict gets converted to tuple by heapq.nlargest, so we convert it back to dict
            app_data["titles"] = dict(app_data["titles"])

            for title_name, title_data in app_data["titles"].items():
                if aggregate:
                    title_data["aggregated_focus_duration"] = self._aggregate_focus_hours(self._db_handler.get_title_focus_period(day_log_id, app_name, title_name))
                else:
                    title_data["hourly_focus_data"] = self._db_handler.get_title_focus_period(day_log_id, app_name, title_name)

        return apps_titles

    def _preprocess_log_detailed(self, day_log_id: int) -> None:
        day_log = self._db_handler.get_day_log(day_log_id, ('time_anchor',))
        apps_titles = self._top_data(day_log_id)

        summary = textwrap.dedent(f"""
        Date Created: {datetime.fromisoformat(day_log['time_anchor']).date().isoformat()}""")

        if len(apps_titles) == 0:
            summary += "\nNo app data available.\n"
            self.preprocessed_logs[day_log_id] = summary
            return

        summary += textwrap.dedent(f"""
        Top {len(apps_titles)} Apps and their Top Titles:
        """)

        for i, (app_name, app_data) in enumerate(apps_titles.items()):
            summary += textwrap.dedent(f""" 
            {i + 1}. {app_name}:
            - Total Duration: {self._round_off(app_data['total_duration'])}
            - Total Focus Duration: {self._round_off(app_data['total_focus_duration'])}""")
            if app_data['hourly_focus_data']:
                summary += textwrap.dedent(f"""
                - Hourly Focus Data: [{', '.join(f"{self._twelvehour_format(int(hour))}: {self._round_off(attributes['focus_duration'])}" for hour, attributes in app_data['hourly_focus_data'].items())}]
                """)
            else:
                summary += textwrap.dedent(f"""
                - Hourly Focus Data: No data available
                """)

            for j, (title_name, title_data) in enumerate(app_data["titles"].items()):
                summary += textwrap.dedent(f"""
                {i + 1}.{j + 1}. {title_name}:
                - Total Duration: {self._round_off(title_data['total_duration'])}
                - Total Focus Duration: {self._round_off(title_data['total_focus_duration'])}""")
                if title_data['hourly_focus_data']:
                    summary += textwrap.dedent(f"""
                    - Hourly Focus Data: [{', '.join(f"{self._twelvehour_format(int(hour))}: {self._round_off(attributes['focus_duration'])}" for hour, attributes in title_data['hourly_focus_data'].items())}]
                    """)
                else:
                    summary += textwrap.dedent(f"""
                    - Hourly Focus Data: No data available
                    """)

        self.preprocessed_logs[day_log_id] = summary

    def _preprocess_log_condensed(self, day_log_id: int) -> None:
        day_log = self._db_handler.get_day_log(day_log_id, ('time_anchor',))
        apps = self._top_data(day_log_id, only_apps=True, aggregate=True)

        summary = textwrap.dedent(f"""
        Date Created: {datetime.fromisoformat(day_log['time_anchor']).date().isoformat()}""")

        if len(apps) == 0:
            summary += "\nNo app data available.\n"
            self.preprocessed_logs[day_log_id] = summary
            return

        summary += textwrap.dedent(f"""
        Top {len(apps)} Apps
        """)

        for i, (app_name, app_data) in enumerate(apps.items()):
            if app_data["aggregated_focus_duration"]:
                summary += textwrap.dedent(f"""
                {i + 1}. {app_name}: {app_data['aggregated_focus_duration']}""")
            else:
                summary += textwrap.dedent(f"""
                {i + 1}. {app_name}: No data available""")
        summary += "\n"

        self.preprocessed_logs[day_log_id] = summary

    def close(self):
        self._service.close()

    def preprocess_logs(self) -> None:
        if self.preprocess_threads:
            for thread in self.preprocess_threads:
                thread.join()
            self.preprocess_threads.clear()

        thread: threading.Thread = threading.Thread(target=self._preprocess_log_detailed, args=(self._day_log_ids[-1],), daemon=True)
        thread.start()
        self.preprocess_threads.append(thread)

        for i in range(len(self._day_log_ids) - 1):
            thread = threading.Thread(target=self._preprocess_log_condensed, args=(self._day_log_ids[i],), daemon=True)
            thread.start()
            self.preprocess_threads.append(thread)

    def wait_until_preprocessed_logs(self) -> None:
        if not self.preprocess_threads:
            raise RuntimeError("No preprocessing threads found.")

        spinner_flag = {"running": True}
        spinner_thread = threading.Thread(
            target = loading_spinner, 
            args = ("Preprocessing data", spinner_flag), 
            daemon = True
        )
        spinner_thread.start()

        for thread in self.preprocess_threads:
            thread.join()

        spinner_flag["running"] = False
        spinner_thread.join()

    def generate_suggestions(self, suggestion_type: SuggestionType) -> None:
        for day_log_id in self._day_log_ids:
            if day_log_id not in self.preprocessed_logs:
                raise RuntimeError(f"Day log {day_log_id} not preprocessed yet.")

        user_prompt = textwrap.dedent(f"""
        Give me {suggestion_type.value} suggestions based on this app data:

        Current Day App Data:""")
        user_prompt += self.preprocessed_logs[self._day_log_ids[-1]]

        removable_suffixes = [self.preprocessed_logs[self._day_log_ids[i]] for i in range(len(self._day_log_ids) - 2, -1, -1)]

        self._service.chat(
            user_prompt, 
            suggestion_type, 
            removable_suffixes,
            no_suffix_attached_message = "\nNo historical data available.",
            any_suffix_attached_message = "\nDay(s) Historical Day(s) App Data Summary:"
        )