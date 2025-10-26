import sys

import settings
from Include.subsystem.parser import Parser

if __name__ == "__main__":
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

    query: str = input("Enter request: ")
    print("-----------------------------")

    action: str | None = None
    while True:
        try:
            action, arguments = parser.extract_action_arguments(query)

            if action:
                break

            print("Skipped request")
            print("-----------------------------")
        except Exception as e:
            print(f"Error parsing query: {e}")
            print("-----------------------------")

        query = input(f"Enter request: ")
        print("-----------------------------")

    try:
        parser.execute_action(action, arguments)
    except Exception as e:
        print(f"Error executing action: {e}")
        print("-----------------------------")

    del parser