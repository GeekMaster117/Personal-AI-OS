import subprocess

from Include.service.parser_service import ParserService

class Parser:
    def __init__(self):
        try:
            self._service = ParserService()
        except Exception as e:
            raise RuntimeError(f"Error initialising parser service: {e}")
        
    def extract_action_arguments(self, query: str, probability_cutoff: float = 0.85) -> tuple[str | list[str]]:
        # Extract tokens from query
        try:
            tokens: list[tuple[str | bool]] = self._service.extract_tokens(query)
        except Exception as e:
            raise SyntaxError(f"Syntax Error: {e}")

        # Extract action keywords and action groups(non keywords divided by action keywords) from tokens
        try:
            action_keywords, action_groups = self._service.extract_action_keywords_groups(tokens, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error extracting action keywords: {e}")
        
        # If no action keywords found, raise syntax error
        if not action_keywords:
            raise SyntaxError("No keywords found")
        
        # Predict action using frequency method first, then classification method if frequency method fails
        try:
            action = self._service.predict_action_frequency(action_keywords, probability_cutoff)
            if not action:
                action = self._service.predict_action_classification(action_keywords, 5)
        except Exception as e:
            raise RuntimeError(f"Error extracting action: {e}")
        
        # If action is None, user has requested to skip the request
        if not action:
            return "", []
        
        # Extract non keywords and argument groups(a group contains argument keywords and their respective non keywords) from action groups
        try:
            non_keywords, argument_groups = self._service.extract_nonkeywords_argument_groups(action, action_groups, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error extracting argument keywords: {e}")
        
        # Fetch required and optional argument indices for the action
        try:
            required_indices, required_needed = self._service.get_required_arguments(action)
            optional_indices = self._service.get_optional_arguments(action)
        except Exception as e:
            raise RuntimeError(f"Error fetching arguments for action '{action}': {e}")
        
        # Extract classified non keywords and classified priority non keywords(priority non keywords are quoted non keywords) from non keywords. Classified into variable types
        classified_non_keywords, classified_priority_non_keywords = self._service.extract_classified_non_keywords(non_keywords)

        # Check if all required arguments are available, else throw error
        self._service.check_argument_availability_else_throw(required_needed, classified_non_keywords, classified_priority_non_keywords)

        arguments, unassigned_required_indices, unassigned_optional_indices = self._service.extract_arguments_type_mapping(action, classified_non_keywords, classified_priority_non_keywords, required_indices, optional_indices)
        
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