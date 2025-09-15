import subprocess

from Include.service.parser_service import ParserService

class Parser:
    def __init__(self):
        try:
            self._service = ParserService()
        except Exception as e:
            raise RuntimeError(f"Error initialising parser service: {e}")
        
    def extract_action_arguments(self, query: str, probability_cutoff: float = 0.85) -> tuple[str | list[str]]:
        try:
            tokens: list[tuple[str | bool]] = self._service.extract_tokens(query)
        except Exception as e:
            raise SyntaxError(f"Syntax Error: {e}")

        try:
            keywords, non_keywords = self._service.extract_keywords_nonkeywords(tokens, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error extracting keywords: {e}")
        
        if not keywords:
            raise SyntaxError("No keywords found")

        actions_normalised: dict = self._service.extract_actions_normalised(keywords)

        try:
            action = self._service.extract_action_frequency(actions_normalised, probability_cutoff)
            if not action:
                action = self._service.extract_action_classification(keywords, 5)
        except Exception as e:
            raise RuntimeError(f"Error extracting action: {e}")
        
        if not action:
            return "", []

        classified_non_keywords, classified_priority_non_keywords = self._service.extract_classified_non_keywords(non_keywords)

        try:
            arguments: list[str] = self._service.extract_arguments(action, classified_non_keywords, classified_priority_non_keywords)
        except Exception as e:
            raise SyntaxError(f"Error extracting arguments: {e}")
        
        return action, arguments
    
    def execute_action(self, action : str, arguments: list[str]) -> None:
        if not self._service.canRunAction(action):
            return
            
        command = " ".join([action] + arguments)
            
        subprocess.run(command, shell=True)
        print("Command Executed: " + command)
        print("-----------------------------")

try:
    parser = Parser()
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

if action == "exit":
    print("Exiting application...")
    print("-----------------------------")
    exit(0)

parser.execute_action(action, arguments)