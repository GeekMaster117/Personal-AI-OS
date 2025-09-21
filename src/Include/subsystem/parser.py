import heapq

import subprocess

from Include.service.parser_service import ParserService

class Parser:
    def __init__(self):
        try:
            self._service = ParserService()
        except Exception as e:
            raise RuntimeError(f"Error initialising parser service: {e}")
        
    def extract_action_arguments(self, query: str, probability_cutoff: float = 0.85) -> tuple[str, list[str]] | None:
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
                action = self._service.predict_action_classification(action_keywords, 5, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error predicting action: {e}")
        
        # If action is None, user has requested to skip the request
        if not action:
            return None
        
        # Extract argument groups(a group contains argument keywords and their respective non keywords) and blind non keyword(non keywords with no argument keywords associated) from action groups
        try:
            argument_groups, blind_non_keywords = self._service.extract_argument_groups(action, action_groups, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error extracting argument groups: {e}")
        
        arguments = [None] * self._service.get_arguments_count(action)
        
        while argument_groups[-1][0]:
            group = argument_groups[-1]

            # Predict argument index using frequency method first, then classification method if frequency method fails
            argument_index, non_keyword = None, None
            try:
                argument_index, non_keyword = self._service.predict_argument_nonkeyword_frequency(action, group, probability_cutoff)
                if not argument_index:
                    argument_index, non_keyword = self._service.predict_argument_nonkeyword_classification(action, group, 5, probability_cutoff)
                if not argument_index:
                    return None

                arguments[argument_index] = non_keyword
            except SyntaxError:
                print("Warning: Valid values for some arguments may not have been found")
                blind_non_keywords.extend(group[1])
            except Exception as e:
                raise RuntimeError(f"Error predicting argument: {e}")
            
            argument_groups.pop()

        classified_nonkeywords, classified_priority_nonkeywords = self._service.extract_classified_nonkeywords(blind_non_keywords)

        try:
            required_arguments, optional_arguments, unassigned_required_indices, unassigned_optional_indices = self._service.extract_argument_indices_information(action, arguments)
        except Exception as e:
            raise RuntimeError(f"Error extracting unassigned arguments: {e}")

        try:
            required_arguments_typemapping, optional_arguments_typemapping, unassigned_required_indices, unassigned_optional_indices = self._service.extract_arguments_typemapping(action, classified_nonkeywords, classified_priority_nonkeywords, unassigned_required_indices, unassigned_optional_indices)
        except Exception as e:
            raise RuntimeError(f"Error extracting arguments using type mapping: {e}")
        
        required_arguments = heapq.merge(required_arguments, required_arguments_typemapping, key = lambda argument: argument[0])
        optional_arguments = heapq.merge(optional_arguments, optional_arguments_typemapping, key = lambda argument: argument[0])
        
        if unassigned_required_indices:
            required_arguments_questions = self._service.extract_arguments_questions(action, unassigned_required_indices, classified_nonkeywords, classified_priority_nonkeywords)
            if not required_arguments_questions:
                return None

            required_arguments = heapq.merge(required_arguments, required_arguments_questions, key = lambda argument: argument[0])

        for argument in heapq.merge(required_arguments, optional_arguments):
            if argument:
                arguments[argument[0]] = argument[1]

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