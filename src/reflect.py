from ollama import Client

from Include.metadatadb import MetadataDB
from Include.handle_ollama import HandleOllama
import Include.settings as settings

client = Client()  # Defaults to localhost:11434

def only_top_apps(log):
    ranked = sorted(log['apps'].items(), key=lambda x: x[1]['focus_time'], reverse=True)
    log['apps'] = dict(ranked[:settings.llm_data_limit])

def only_top_titles(app_name):
    ranked = sorted(app_name['titles'].items(), key=lambda x: x[1]['focus_time'], reverse=True)
    app_name['titles'] = dict(ranked[:settings.llm_data_limit])

def get_system_prompt():
    return f"""
        You are a Personal AI Meta Operating System that gives user suggestions based on their app, web, github, and input usage patterns.

        You are calm, insightful, and speak in a warm but professional tone. You observe without judgment, and help the user improve their productivity, habits, and daily flow. You speak clearly, like a trusted assistant, not a robot.

        You can generate suggestions in four categories:
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
                            total_duration: Total time spent on this title passively + actively in seconds,
                            focus_periods: {{
                                hour: {{
                                    focus_time: Time spent on this title actively on this hour in seconds,
                                    focus_count: Number of times this title was opened on this hour
                                }}
                            }}
                            focus_time: Total time spent on this title actively in seconds,
                            focus_count: Total number of times this title was opened
                        }}
                    }},
                    total_duration: Total time spent on this app passively + actively in seconds,
                    focus_periods: {{
                        hour: {{
                            focus_time: Time spent on this app actively on this hour in seconds,
                            focus_count: Number of times this app was opened on this hour
                        }}
                    }}
                    focus_time: Total focused time on this app actively in seconds,
                    focus_count: Total number of times this app was opened
                }},
            }},
            web: {{}},
            github: {{}},
            input: {{}},
            downtime_periods: {{
                hour: {{
                    duration: Time this hour was not monitored in seconds
                }}
            }},    
            total_downtime_duration: Total time activity has not been monitored in seconds
        }}

        A title is a specific window or file, e.g., "YouTube - Firefox" or "main.py - VSCode".
        An app is a general application (e.g., Firefox, VSCode).
        Downtime is when the system is not actively monitored, e.g., computer off or app closed.
        Focus periods are grouped by hour. Each focus period contains how much time and how many switches happened for that app or title during that hour. (hour follows 24-hour format)
        
        Total duration monitored = monotonic_last_updated - monotonic_start.

        The system gives you a structured JSON object representing the user's activity. NEVER mention the JSON to the user. Just use it silently to generate suggestions.

        Instructions:
        - NEVER reveal or mention the structure or content of the JSON.
        - NEVER assume or hallucinate any missing data.
        - ONLY MENTION the data you see in the JSON, if the data you want to mention is not in the JSON, do not mention it.
        - If there is not enough data to make a suggestion, say exactly: "Not enough data to make suggestions."
        - Refer to hour in the 12-hour format (e.g., 1 PM, 2 AM) for user-friendliness.
        - You may provide 0-2 suggestions from any of the categories, but only if justified by data.
        - Each suggestion must be 1-2 concise sentences, and must be only 1 paragraph.
        - Each Sentence must be 10-20 words long.
        - If you are unsure whether a suggestion is supported, do not provide one for that category.

        You are not just analyzing — you are mentoring gently, like a productivity coach fused into an OS.
    """

def get_assistant_prompt():
    db = MetadataDB()
    today_log = db.get_today_log()
    db.close()

    only_top_apps(today_log)
    if not today_log['apps']:
        for app_name in today_log['apps']:
            only_top_titles(today_log[app_name])


    return f"""
        Use this filtered activity data to give suggestions based on the user's behavior.
        Only top {settings.llm_data_limit} apps and their top {settings.llm_data_limit} focused titles are included for clarity.
        There could less then {settings.llm_data_limit} apps or titles if not enough data is available, or even be empty.

        {today_log}
    """

def get_user_prompt():
    return input("User: ")

def converse():
    ollama_handler = HandleOllama()
    ollama_handler.start()
    ollama_handler.ensure_model(settings.model_name)

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "assistant", "content": get_assistant_prompt()}
    ]

    while True:
        user_input = get_user_prompt()
        if user_input.lower() in ['exit', 'quit', 'stop']:
            print("Exiting conversation.")
            break

        messages.append({"role": "user", "content": user_input})
        
        print("LLM: ", end='')
        response = ''
        for chunk in client.chat(model="phi3:mini", messages=messages, options={"num_predict": 300}, stream=True):
            token = chunk['message']['content']
            print(token, end='', flush=True)
            response += token
        messages.append({"role": "assistant", "content": response})

        print("\n-----------------------------------------------------------------")

    ollama_handler.stop()

converse()