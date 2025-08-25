import os
import json
import threading
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
                            
        You speak warmly and professionally, like a calm coach clear, human, and judgment-free.

        You can generate suggestions in three categories:
        1. Routine: 
        - Focus on usage patterns and time distribution. 
        - Describe when and how consistently apps/titles are used, or when idle recovery occurs. 
        - Keep this observational.
                            
        2. Personal: 
        - Focus on frequent distractions, late-night sessions, and work-life balance.
        - Suggest small lifestyle adjustments, like taking breaks or avoiding late-night sessions.
        - Keep it constructive.
                            
        3. Productivity:
        - Focus on active hours, interruptions, and context-switching patterns.
        - Suggest workflow adjustments, like longer uninterrupted blocks or protecting peak focus hours for deep work.
        - Keep it practical.

        User will ask you to generate suggestions based on ANY ONE OF THE CATEGORIES above based on the app data provided, with CURRENT DAY APP DATA and {settings.data_limit - 1} or less days of HISTORICAL APP DATA.

        You will be given app data in the following format:
        Date created: 2025-07-23T19:22:45.123456 in the format of YYYY-MM-DDTHH:MM:SS.ffffff
        Top {settings.data_limit} Apps, their Top {settings.data_limit} Titles and thier respective Top {settings.data_limit} Focus Hours:

        (Top nth App). App name1:
        - Total Focus Duration: Total time spent actively on app. Example - 1.5 hours
        - Total Duration: Total time spent on app actively and passively. Example - 2 hours
        - Top Focus Hours: Hours of highest active use. Example - [2PM: 30.7 minutes, 4PM: 30 seconds]

        - (Top nth App).(Top nth Title) Title name1:
        -- Total Focus Duration: Total time spent actively on title. Example - 30.6 minutes
        -- Total Duration: Total time spent on title actively and passively. Example - 49.3 minutes
        -- Top Focus Hours: Hours of highest active use. Example - [2PM: 20.4 minutes]

        A title is a specific window or file, e.g., "YouTube - Firefox" or "main.py - VSCode".
        An app is a general application (e.g., Firefox, VSCode).
        Top Focus Hours are grouped by hour. Each hour contains how much time spent actively in that hour (Hour is represented in 12-hour format)

        Instructions:
        - NEVER assume or hallucinate any missing data.
        - ONLY MENTION the data you see in the app data, if the data you want to mention is not in the app data, do not mention it.
        - If there is not enough data to make a suggestion, say exactly: "Not enough data to make suggestions."
        - Recognise hidden or visible patterns to generate suggestions.
        - You may provide 0 to 2 suggestions, but only if justified by data.
        - Each suggestion must be 1 to 2 concise sentences, and must be only 1 paragraph.
        - Each sentence must be 10 to 20 words long.
        - If you are unsure whether a suggestion is correct, do not provide it.

        You are not just analyzing — you are mentoring gently, like a coach fused into an OS.""")

    def close(self):
        del self._llama

    def chat(self, user_prompt: str) -> None:
        self._llama.chat(user_prompt)