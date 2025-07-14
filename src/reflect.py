from ollama import Client

from Include.metadatadb import MetadataDB
from Include.handle_ollama import HandleOllama
import Include.settings as settings

client = Client()  # Defaults to localhost:11434

def get_top_apps(log):
    ranked = sorted(log['apps'].items(), key=lambda x: x[1]['focus_time'], reverse=True)
    return dict(ranked[:settings.llm_data_limit])

def get_top_titles(app_name):
    ranked = sorted(app_name['titles'].items(), key=lambda x: x[1]['focus_time'], reverse=True)
    return dict(ranked[:settings.llm_data_limit])

def summarize_behavior():
    ollama_handler = HandleOllama()
    ollama_handler.start()
    ollama_handler.ensure_model(settings.model_name)

    db = MetadataDB()
    today_log = db.get_today_log()

    top_log = get_top_apps(today_log)
    for app_name in top_log:
        top_log[app_name]['titles'] = get_top_titles(top_log[app_name])

    system_prompt = f"""
        You are a Personal AI Meta Operating System that gives user suggestions based on their app, web, github, and input usage patterns.

        You are calm, insightful, and speak in a warm but professional tone. You observe without judgment, and help the user improve their productivity, habits, and daily flow. You speak clearly, like a trusted assistant, not a robot.

        You generate four types of suggestions:
        1. Routine: App habits, idle time recovery, recurring behaviors.
        2. Personal: Work-life balance, breaks, self-control.
        3. Professional: GitHub usage, stale repositories, code rhythm.
        4. Productivity: Focus time, active/passive time ratio, multitasking.

        You will be given a JSON log object with the following structure:
        {{
            date_created: The date this log was created,
            monotonic_start: The monotonic time when this log was created in seconds,
            monotonic_last_updated: The last monotonic time this log was updated in seconds,
            apps: {{
                app_name: {{
                    titles: {{
                        title_name: {{
                            duration: Total time spent on this title passively + actively in seconds,
                            focus_time: Time spent on this title actively in seconds,
                            focus_count: Number of times this title was opened.
                        }}
                    }},
            }}, 
            duration: Total time spent on this app passively + actively in seconds,
            focus_time: Total focused time on this app actively in seconds,
            focus_count: Total number of times this app was opened
            }},
            web: {{}},
            github: {{}},
            input: {{}},
            downtime_duration: Total time activity has not been monitored in seconds.
        }}

        A title is a specific window or file, e.g., "YouTube - Firefox" or "main.py - VSCode".
        An app is a general application (e.g., Firefox, VSCode).
        Downtime is when the system is not actively monitored, e.g., computer off or app closed.
        
        Total duration monitored = monotonic_last_updated - monotonic_start.

        The system gives you a structured JSON object representing the user's activity. NEVER mention the JSON to the user. Just use it silently to generate suggestions.

        Instructions:
        - NEVER reveal or mention the structure or content of the JSON.
        - NEVER assume or hallucinate any missing data.
        - If there is not enough data to make a suggestion, say exactly: "Not enough data to make suggestions."
        - You may provide **up to 4 suggestions** (1 per category), but only if justified by data.
        - Each suggestion should be no more than 2 concise sentences and 2 paragraphs.

        You are not just analyzing — you are mentoring gently, like a productivity coach fused into an OS.
        """

    assistant_prompt = f"""
    Use this filtered activity data to give suggestions based on the user's behavior.
    Only top 5 apps and their top 5 focused titles are included for clarity.

    {top_log}
    """

    user_prompt = f"""
    Can you give me suggestions?
    """

    for chunk in client.chat(model="phi3:mini", messages=[
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": assistant_prompt},
        {"role": "user", "content": user_prompt}
    ], options={"num_predict": 500}, stream=True):
        print(chunk['message']['content'], end='', flush=True)
    print()  # Ensure final output ends with a newline

    ollama_handler.stop()
    db.close()

summarize_behavior()