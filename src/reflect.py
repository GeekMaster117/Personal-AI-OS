from ollama import Client
from Include.metadatadb import MetadataDB  # your existing module

client = Client()  # Defaults to localhost:11434

def summarize_behavior():
    db = MetadataDB()
    today_log = db.get_today_log()

    system_prompt = f"""
    You are a Personal AI Meta Operating System that gives user suggestions based on their app, web, github, and input usage patterns.

    You will give 4 types of suggestions:
    - Routine suggestions: App Habits, Idle Recovery, etc.
    - Personal suggestions: Work-Life balance, Self Control, etc.
    - Professional suggestions: Github Commits, Stale Repos, Coding Rhytm, etc.
    - Productivity suggestions: Focus Time, Active-Passive Work Ratio, etc.

    Activity will be fed in as a JSON object with the following structure:
    {{
        date_created: The date this log was created,
        monotonic_start: The monotonic time when this log started,
        monotonic_last_updated: The monotonic time when this log was last updated,
        apps: {{
            app_name: {{
                titles: {{
                    title_name: {{
                        duration: Total time spent on this title passively + actively,
                        focus_time: Time spent on this title actively,
                        focus_count: Number of times this title was opened.
                    }}
                }},
                duration: Total time spent on this app passively + actively,
                focus_time: Total focused time on this app actively,
                focus_count: Total number of times this app was opened
            }}
        }},
        web: {{ No data yet }},
        github: {{ No data yet }},
        input: {{ No data yet }},
        downtime_duration: Total time you were down or have not monitored user activity.
    }}
    
    A Title is a specific window or document within an app, like a browser tab or a text editor file.
    A App is a general application like a web browser, text editor, or IDE.
    Duration will always be greater than or equal to focus_time, because it counts passive time as well.


    THE JSON OBJECT WILL BE PROVIDED TO YOU BY THE ASSISTANT, FOLLOW THE INSTRUCTIONS BELOW TO GENERATE SUGGESTIONS BASED ON IT.

    DO NOT REVEAL OR MENTION THE JSON TO THE USER, ONLY USE IT TO GENERATE SUGGESTIONS.
    IF YOU DO NOT HAVE ENOUGH DATA TO MAKE A SUGGESTION, SAY SO. DO NOT GIVE REASONS OR EXPLANATIONS, JUST SAY YOU DO NOT HAVE ENOUGH DATA.

    TRY TO KEEP SUGGESTIONS SHORT AND CONCISE. 

    DO NOT HALLUCINATE OR MAKE UP DATA. ONLY USE THE DATA PROVIDED IN THE JSON OBJECT.
    DO NOT MAKE UP SUGGESTIONS. ONLY SUGGEST BASED ON THE DATA PROVIDED.
    DO NOT SUGGEST ANYTHING THAT IS NOT SUPPORTED BY THE DATA.
    """

    assistant_prompt = f"""
    Use this activity to give suggestions based on the user's behavior.
    {today_log}
    """

    user_prompt = f"""
    Can you give me suggestions?
    """

    for chunk in client.chat(model="phi3:mini", messages=[
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": assistant_prompt},
        {"role": "user", "content": user_prompt}
    ], stream=True):
        print(chunk['message']['content'], end='', flush=True)

summarize_behavior()