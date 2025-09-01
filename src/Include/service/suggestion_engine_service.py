import os
import json
import threading
import textwrap
from enum import Enum

from Include.wrapper.llama_wrapper import LlamaCPP
from Include.loading_spinner import loading_spinner

class SuggestionType(Enum):
    ROUTINE = "routine"
    PRODUCTIVITY = "productivity"
    PERSONAL = "personal"

class SuggestionEngineService:
    def __init__(self):
        if not os.path.exists('device_config.json'):
            raise FileNotFoundError("Configuration file 'device_config.json' not found. Please run the benchmark first.")

        cpu_optimal_batchsize = None
        gpu_optimal_batchsize = None
        with open('device_config.json', 'r') as file:
            device_config = json.load(file)
            try:
                cpu_optimal_batchsize: int = device_config["cpu_optimal_batchsize"]
                gpu_optimal_batchsize: int = device_config["gpu_optimal_batchsize"]

                if not isinstance(cpu_optimal_batchsize, int):
                    raise ValueError("cpu_optimal_batchsize must be an integer.")
                if not isinstance(gpu_optimal_batchsize, int):
                    raise ValueError("gpu_optimal_batchsize must be an integer.")
            except:
                raise ValueError("cpu_optimal_batchsize and gpu_optimal_batchsize not found in the configuration file. Please run the benchmark.")

        self._llama = self._initialize_llama(cpu_optimal_batchsize, gpu_optimal_batchsize)

    def _initialize_llama(self, cpu_optimal_batchsize: int, gpu_optimal_batchsize: int) -> LlamaCPP:
        spinner_flag = {"running": True}
        spinner_thread = threading.Thread(
            target = loading_spinner, 
            args = ("Initializing Model", spinner_flag), 
            daemon = True
        )
        spinner_thread.start()

        try:
            llama = LlamaCPP(cpu_optimal_batchsize, gpu_optimal_batchsize)
        except Exception as e:
            raise RuntimeError(f"Error initializing LlamaCPP: {e}")
        finally:
            spinner_flag["running"] = False
            spinner_thread.join()

        return llama
    
    def _get_system_prompt_prefix(self) -> str:
        return textwrap.dedent(f"""
        You are a Personal AI Meta Operating System that gives user suggestions based on their app data. You speak warmly and professionally.
        """)
    
    def _get_system_prompt_suffix(self) -> str:
        return textwrap.dedent(f"""
        You will be given Current Day App Data in the following format:
            Date created: 2025-07-23T19:22:45.123456 in the format of YYYY-MM-DDTHH:MM:SS.ffffff
            Top Apps and their Top Titles:

            (Top nth App). App name:
            - Total Focus Duration: Total time spent actively on app. Example - 1.5 hours
            - Total Duration: Total time spent on app actively and passively. Example - 2 hours
            - Hourly Focus Duration: Time spent actively on each hour, formatted as [Hour: Duration]. Example - [2PM: 30.7 minutes, 4PM: 30 seconds]

            (Top nth App).(Top nth Title). Title name:
            - Total Focus Duration: Total time spent actively on title. Example - 30.6 minutes
            - Total Duration: Total time spent on title actively and passively. Example - 49.3 minutes
            - Hourly Focus Duration: Time spent actively on each hour, formatted as [Hour: Duration]. Example - [2PM: 20.4 minutes]

        You will be given Historical Day(s) App Data Summary in the following format:
            Date created: 2025-07-23T19:22:45.123456 in the format of YYYY-MM-DDTHH:MM:SS.ffffff
            Top Apps

            (Top nth App). App name: Aggregated Time spent actively. Example - [11 AM, 2 PM - 4 PM, 5 PM - 7 PM]

        A title is a specific window or file, e.g., "YouTube - Firefox" or "main.py - VSCode".
        An app is a general application (e.g., Firefox, VSCode).
        Hourly Focus Duration are grouped by hour. Each hour contains how much time spent actively in that hour (Hour is represented in 12-hour format)

        General Instructions:
        - Only use the data present in the app. Never assume, hallucinate, or infer missing data.
        - If the app data is insufficient, respond exactly: "Not enough data to make suggestions."
        - Do not provide general tips, opinions, or information outside the app data.
        - Provide up to 5 suggestions only if fully justified by the app data.
        - Each suggestion can be up to 2 sentences.
        - If unsure about correctness, do not provide a suggestion.
        - Stop after providing the suggestions.""")
    
    def _get_routine_suggestion_system_prompt(self) -> str:
        prompt = textwrap.dedent(f"""
            You will be asked to provide Routine Suggestions, with Current Day App Data and Historical Day(s) App Data Summary:
            - Focus on usage patterns and time distribution.
            - Describe when and how consistently apps/titles are used, or when idle recovery occurs.
            - Keep this observational.
        """)

        return self._get_system_prompt_prefix() + prompt + self._get_system_prompt_suffix()

    def _get_personal_suggestion_system_prompt(self) -> str:
        prompt = textwrap.dedent(f"""
            You will be asked to provide Personal Suggestions, with Current Day App Data and Historical Day(s) App Data Summary:
            - Focus on frequent distractions, late-night sessions, and work-life balance.
            - Suggest small lifestyle adjustments, like taking breaks or avoiding late-night sessions.
            - Keep it constructive.
        """)

        return self._get_system_prompt_prefix() + prompt + self._get_system_prompt_suffix()

    def _get_productivity_suggestion_system_prompt(self) -> str:
        prompt = textwrap.dedent(f"""
            You will be asked to provide Productivity Suggestions, with Current Day App Data and Historical Day(s) App Data Summary:
            - Focus on active hours, interruptions, and context-switching patterns.
            - Suggest workflow adjustments, like longer uninterrupted blocks or protecting peak focus hours for deep work.
            - Keep it practical.
        """)

        return self._get_system_prompt_prefix() + prompt + self._get_system_prompt_suffix()
    
    def _get_system_prompt(self, suggestion_type: SuggestionType) -> str:
        if suggestion_type == SuggestionType.ROUTINE:
            return self._get_routine_suggestion_system_prompt()
        elif suggestion_type == SuggestionType.PRODUCTIVITY:
            return self._get_productivity_suggestion_system_prompt()
        elif suggestion_type == SuggestionType.PERSONAL:
            return self._get_personal_suggestion_system_prompt()
        else:
            raise ValueError("Invalid SuggestionType provided.")

    def close(self):
        del self._llama

    def chat(self, user_prompt: str, suggestion_type: SuggestionType) -> None:
        max_tokens = 350
        stop = ["<|end|>"]
        self._llama.chat(self._get_system_prompt(suggestion_type), user_prompt, max_tokens = max_tokens, stop = stop)