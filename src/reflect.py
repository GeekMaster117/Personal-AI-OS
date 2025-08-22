from enum import Enum

from Include.usagedata_db import UsagedataDB
from Include.suggestion_engine import SuggestionEngine
import settings

class ExitCodes(Enum):
    EXIT = -1
    CONTINUE = 0
    BACK = 1

def exit_program() -> ExitCodes:
    return ExitCodes.EXIT

def back_program() -> ExitCodes:
    return ExitCodes.BACK

def handle_options(options: list[str]) -> int:
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option[0]}")

    choice = input(f"Select an option (1-{len(options)}): ")
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        return int(choice) - 1
    return -1

def wait_until_preprocessed_logs() -> None:
    try:
        suggestion_engine.wait_until_preprocessed_logs()
    except Exception as e:
        raise RuntimeError(f"Error waiting for preprocessed logs: {e}")

# Suggestion Categories Handlers
def handle_routine_suggestions() -> ExitCodes:
    try:
        wait_until_preprocessed_logs()
    except Exception as e:
        print(f"Error handling routine suggestions: {e}")
        return ExitCodes.EXIT

    suggestion_engine.generate_suggestions(SuggestionEngine.SuggestionType.ROUTINE)

    print("-----------------------------")

    return ExitCodes.CONTINUE

def handle_productivity_suggestions() -> ExitCodes:
    try:
        wait_until_preprocessed_logs()
    except Exception as e:
        print(f"Error handling productivity suggestions: {e}")
        return ExitCodes.EXIT

    suggestion_engine.generate_suggestions(SuggestionEngine.SuggestionType.PRODUCTIVITY)

    print("-----------------------------")

    return ExitCodes.CONTINUE

def handle_personal_suggestions() -> ExitCodes:
    try:
        wait_until_preprocessed_logs()
    except Exception as e:
        print(f"Error handling personal suggestions: {e}")
        return ExitCodes.EXIT

    suggestion_engine.generate_suggestions(SuggestionEngine.SuggestionType.PERSONAL)

    print("-----------------------------")

    return ExitCodes.CONTINUE

def handle_professional_suggestions() -> ExitCodes:
    try:
        wait_until_preprocessed_logs()
    except Exception as e:
        print(f"Error handling professional suggestions: {e}")
        return ExitCodes.EXIT

    suggestion_engine.generate_suggestions(SuggestionEngine.SuggestionType.PROFESSIONAL)

    print("-----------------------------")

    return ExitCodes.CONTINUE

# Suggestions handler
def handle_suggestions() -> ExitCodes:
    options = [
        ("Routine Suggestions", handle_routine_suggestions), 
        ("Productivity Suggestions", handle_productivity_suggestions), 
        ("Personal Suggestions", handle_personal_suggestions), 
        ("Professional Suggestions", handle_professional_suggestions), 
        ("Back to Main Menu", back_program), 
        ("Exit", exit_program)
    ]

    while True:
        choice = handle_options(options)

        print("-----------------------------")
        if choice != -1:
            result = options[choice][1]()
            if result == ExitCodes.EXIT:
                return ExitCodes.EXIT
            elif result == ExitCodes.BACK:
                break
        else:
            print("Invalid option. Please try again.")

    return ExitCodes.CONTINUE

# Automation handler
def handle_automation() -> ExitCodes:
    print("Automation handling is not yet implemented.")
    return ExitCodes.CONTINUE

# Main menu handler
def handle_menu() -> None:
    options = [
        ("Get Suggestions", handle_suggestions), 
        ("Build Automations", handle_automation), 
        ("Exit", exit_program)
    ]

    while True:
        choice = handle_options(options)

        print("-----------------------------")
        if choice != -1:
            result = options[choice][1]()
            if result == ExitCodes.EXIT:
                break
        else:
            print("Invalid option. Please try again.")

usagedataDB = UsagedataDB(settings.usagedata_dir)
try:
    suggestion_engine = SuggestionEngine(usagedataDB)
except Exception as e:
    print(f"\nError initialising Suggestion Engine: {e}")
    exit(1)

suggestion_engine.preprocess_logs()

print("What would you like to do?")
handle_menu()

suggestion_engine.close()