import sys

import textwrap

import settings
from Include.subsystem.parser import ExitCodes
from Include.subsystem.parser import Parser

prototype_message = textwrap.dedent("""
=================== Personal AI OS Prototype =======================
This is an early release. Solid, but still evolving. Explore freely!
====================================================================
""")

if __name__ == "__main__":
    print(prototype_message)

    environment = sys.argv[1] if len(sys.argv) > 1 else settings.Environment.PROD
    if environment not in settings.Environment:
        print(f"Invalid environment: '{environment}'. Valid options are: {[env.value for env in settings.Environment]}")
        exit(1)
    environment = settings.Environment(environment)

    try:
        parser = Parser(environment)
    except Exception as e:
        print(f"Error initialising parser: {e}")
        print("-----------------------------")

        exit(1)

    while True:
        query: str = input("Enter request: ")
        print("-----------------------------")

        try:
            action, arguments = parser.extract_action_arguments(query)

            if not action:
                print("Skipped request")
                print("-----------------------------")

                continue
        except Exception as e:
            print(f"Error parsing query: {e}")
            print("-----------------------------")

            continue

        try:
            if parser.execute_action(action, arguments) == ExitCodes.EXIT:
                break
        except Exception as e:
            print(f"Error executing action: {e}")
            print("-----------------------------")

    del parser