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

# Suggestion Categories Handlers
def handle_routine_suggestions() -> ExitCodes:
    print("Routine suggestions are not yet implemented.")
    return ExitCodes.CONTINUE

def handle_productivity_suggestions() -> ExitCodes:
    print("Productivity suggestions are not yet implemented.")
    return ExitCodes.CONTINUE

def handle_personal_suggestions() -> ExitCodes:
    print("Personal suggestions are not yet implemented.")
    return ExitCodes.CONTINUE

def handle_professional_suggestions() -> ExitCodes:
    print("Professional suggestions are not yet implemented.")
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
suggestion_engine = SuggestionEngine(usagedataDB)

print("What would you like to do?")
handle_menu()