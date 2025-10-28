from enum import Enum

import textwrap

from typing import Any

import settings
from Include.subsystem.usagedata_db import UsagedataDB
from Include.subsystem.suggestion_engine import SuggestionEngine
from Include.service.suggestion_engine_service import SuggestionType
from Include.verify_install import verify_installation

class ExitCodes(Enum):
    EXIT = -1
    CONTINUE = 0

def exit_program() -> ExitCodes:
    return ExitCodes.EXIT

def handle_options(options: list[Any]) -> int:
    while True:
        for i, option in enumerate(options, start=1):
            print(f"{i}. {option[0]}")

        choice = input(f"Select an option (1-{len(options)}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice) - 1
        print("-----------------------------")
        
        print(f"Invalid option. Please enter a valid option between 1-{len(options)}.")
        print("-----------------------------")

def wait_until_preprocessed_logs() -> None:
    try:
        suggestion_engine.wait_until_preprocessed_logs()
    except Exception as e:
        raise RuntimeError(f"Error waiting for preprocessed logs: {e}")

# Suggestion Categories Handlers
def handle_routine_suggestions() -> ExitCodes:
    try:
        wait_until_preprocessed_logs()
        suggestion_engine.generate_suggestions(SuggestionType.ROUTINE)
    except Exception as e:
        raise RuntimeError(f"Error handling routine suggestions: {e}")

    print("-----------------------------")

    return ExitCodes.CONTINUE

def handle_productivity_suggestions() -> ExitCodes:
    try:
        wait_until_preprocessed_logs()
        suggestion_engine.generate_suggestions(SuggestionType.PRODUCTIVITY)
    except Exception as e:
        raise RuntimeError(f"Error handling productivity suggestions: {e}")

    print("-----------------------------")

    return ExitCodes.CONTINUE

def handle_personal_suggestions() -> ExitCodes:
    try:
        wait_until_preprocessed_logs()
        suggestion_engine.generate_suggestions(SuggestionType.PERSONAL)
    except Exception as e:
        raise RuntimeError(f"Error handling personal suggestions: {e}")

    print("-----------------------------")

    return ExitCodes.CONTINUE

# Suggestions handler
def handle_menu() -> None:
    options = [
        ("Routine Suggestions", handle_routine_suggestions), 
        ("Productivity Suggestions", handle_productivity_suggestions), 
        ("Personal Suggestions", handle_personal_suggestions),
        ("Exit", exit_program)
    ]

    while True:
        choice = handle_options(options)
        print("-----------------------------")

        result = options[choice][1]()
        if result == ExitCodes.EXIT:
            break

prototype_message = textwrap.dedent("""
=================== Personal AI OS Prototype =======================
This is an early release. Solid, but still evolving. Explore freely!
====================================================================
""")

if __name__ == "__main__":
    print(prototype_message)

    try:
        verify_installation()
    except Exception as e:
        print(f"Installation verification failed: {e}\nPlease run install.exe")

        input("\nPress any key to exit...")
        exit(1)

    usagedataDB = UsagedataDB(settings.usagedata_dir)
    try:
        suggestion_engine = SuggestionEngine(usagedataDB)
    except Exception as e:
        print(f"\nError initialising SuggestionEngine: {e}")

        input("\nPress any key to exit...")
        exit(1)

    suggestion_engine.preprocess_logs()

    print("What would you like to get?")

    try:
        handle_menu()
    except Exception as e:
        print(f"Error handling reflect: {e}")

    suggestion_engine.close()

    input("\nPress any key to exit...")