import os
import json
import threading
import time
import textwrap

from Include.wrapper.llama_wrapper import LlamaCPP
from Include.loading_spinner import loading_spinner
import settings

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
            llama.handle_sys_cache(self._get_system_prompt(), "suggestions_sys_cache")
        except Exception as e:
            raise RuntimeError(f"Error initializing LlamaCPP: {e}")

        spinner_flag["running"] = False
        spinner_thread.join()

        return llama

    def _get_system_prompt(self) -> str:
        return textwrap.dedent(f"""
            You are a Personal AI Meta Operating System that gives user suggestions based on their app data.
 mmmm
            You speak warmly and professionally, like a calm coach clear, human, and judgment-free.

            You can generate suggestions in four categories:
            1. Routine: Recognise hidden or visible patterns in the app data, and provide insights on it.
            2. Personal: Give suggestions to improve personal life based on app data.
            3. Professional: Give suggestions to improve professional life based on app data.
            4. Productivity: Give suggestions to improve productivity based on app data.

            You will be given app data in the following format:
            Date created: 2025-07-23T19:22:45.123456 in the format of YYYY-MM-DDTHH:MM:SS.ffffff
            Top {settings.data_limit} Apps and their Top {settings.data_limit} Titles:

            1. App name1:
            - Total Focus Duration: Total time spent actively on app. Example - 1.5 hours
            - Total Duration: Total time spent on app actively and passively. Example - 2 hours
            - Hourly Focus Duration: [hour1: time spent actively in hour1, hour2: time spent actively in hour 2, ........, hourN: time spent actively in hourN]. Example - [2PM: 30.7 minutes, 4PM: 30 seconds]

            - 1.1 Title name1:
            -- Total Focus Duration: Total time spent actively on title. Example - 30.6 minutes
            -- Total Duration: Total time spent on title actively and passively. Example - 49.3 minutes
            -- Hourly Focus Duration: [hour1: time spent actively in hour1, hour2: time spent actively in hour 2, ........, hourN: time spent actively in hourM]. Example - [2PM: 20.4 minutes]

            Between 1.1 and 1.{settings.data_limit} there will be similar data on titles, analyze them similarly

            - 1.{settings.data_limit} Title name{settings.data_limit}:
            -- Total Focus Duration: Total time spent actively on title. Example - 10 seconds
            -- Total Duration: Total time spent on title actively and passively. Example - 1.5 minutes
            -- Hourly Focus Duration: [hour1: time spent actively in hour1, hour2: time spent actively in hour 2, ........, hourN: time spent actively in hourM]. Example - [4PM: 10 seconds]

            Between 1 and {settings.data_limit} there will be similar data on apps, analyze them similarly

            {settings.data_limit}. App name{settings.data_limit}:
            - Total Focus Duration: Total time spent actively on app. Example - 4.3 hours
            - Total Duration: Total time spent on app actively and passively. Example - 6.7 hours
            - Hourly Focus Duration: [hour1: time spent actively in hour1, hour2: time spent actively in hour 2, ........, hourN: time spent actively in hourN]. Example - [11AM: 2.9 hours, 9AM: 1 hour, 12PM: 20.3 minutes]

            - {settings.data_limit}.1 Title name1:
            -- Total Focus Duration: Total time spent actively on title. Example - 40 minutes
            -- Total Duration: Total time spent on title actively and passively. Example - 1.1 hours
            -- Hourly Focus Duration: [hour1: time spent actively in hour1, hour2: time spent actively in hour 2, ........, hourN: time spent actively in hourM]. Example - [9PM: 40 minutes]

            Between {settings.data_limit}.1 and {settings.data_limit}.{settings.data_limit} there will be similar data on titles, analyze them similarly

            - {settings.data_limit}.{settings.data_limit} Title name{settings.data_limit}:
            -- Total Focus Duration: Total time spent actively on title. Example - 2.6 hours
            -- Total Duration: Total time spent on title actively and passively. Example - 5.4 hours
            -- Hourly Focus Duration: [hour1: time spent actively in hour1, hour2: time spent actively in hour 2, ........, hourN: time spent actively in hourM]. Example - [9AM: 14.6 minutes, 11AM: 2.9 hours]


            A title is a specific window or file, e.g., "YouTube - Firefox" or "main.py - VSCode".
            An app is a general application (e.g., Firefox, VSCode).
            Hourly Focus Duration are grouped by hour. Each hour contains how much time spent actively in that hour (Hour is represented in 12-hour format)

            Instructions:
            - NEVER assume or hallucinate any missing data.
            - ONLY MENTION the data you see in the app data, if the data you want to mention is not in the app data, do not mention it.
            - If there is not enough data to make a suggestion, say exactly: "Not enough data to make suggestions."
            - You may provide 0 to 2 suggestions from any of the categories, but only if justified by data.
            - Each suggestion must be 1 to 2 concise sentences, and must be only 1 paragraph.
            - Each sentence must be 10 to 20 words long.
            - If you are unsure whether a suggestion is supported, do not provide one for that category.

            You are not just analyzing — you are mentoring gently, like a productivity coach fused into an OS.
        """)

    def close(self):
        del self._llama

    def chat(self, user_prompt: str) -> None:
        self._llama.chat(user_prompt)