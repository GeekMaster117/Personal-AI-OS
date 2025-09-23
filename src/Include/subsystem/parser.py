import heapq

import subprocess

from collections import defaultdict

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
            action_keywords, action_groups = self._service.extract_action_groups(tokens, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error extracting action keywords: {e}")
        
        # If no action keywords found, raise syntax error
        if not action_keywords:
            raise SyntaxError("No keywords found")
        
        # Predict action using frequency method first, then classification method if frequency method fails. 
        # If skip = true, user has asked to skip request.
        try:
            action = self._service.predict_action_frequency(action_keywords, probability_cutoff)

            if not action:
                action, skip = self._service.predict_action_classification(action_keywords, 5, probability_cutoff)
            if skip:
                return None, []
        except Exception as e:
            raise RuntimeError(f"Error predicting action: {e}")
        
        del action_keywords
        
        # Extract argument groups and blind non keywords from action groups
        # Argument group: a group contains argument keywords and their respective non keywords
        # blind non keywords: non keywords with no argument keywords associated
        try:
            argument_groups, blind_non_keywords = self._service.extract_argument_groups(action, action_groups, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error extracting argument groups: {e}")

        del action_groups
        
        arguments: list[str | None] = [None] * self._service.get_arguments_count(action)

        argument_ambigous_groups = []
        predicted_argument_group = defaultdict(list)

        argument_keywords, non_keywords = [], set()
        for group in argument_groups:
            try:
                argument_index = self._service.predict_argument_frequency(action, group[0], probability_cutoff)
            except Exception as e:
                raise RuntimeError(f"Error predicting argument: {e}")

            if argument_index is None:
                argument_keywords.extend(group[0])
                non_keywords.update(group[1])
            else:
                if argument_keywords:
                    argument_ambigous_groups((argument_keywords, non_keywords))

                predicted_argument_group[argument_index].extend(group[1])
        if argument_keywords:
            argument_ambigous_groups((argument_keywords, non_keywords))

        del argument_groups, argument_keywords, non_keywords
        
        for group in argument_ambigous_groups:
            # Predict argument index using frequency method first, then classification method if frequency method fails. 
            # If skip = true, user has asked to skip request.
            try:
                argument_index = self._service.predict_argument_frequency(action, group[0], probability_cutoff)
                if argument_index is None:
                    argument_index, skip = self._service.predict_argument_classification(action, group[0], 5, probability_cutoff)
                if skip:
                    return None, []

                predicted_argument_group[argument_index].extend(group[1])
            except Exception as e:
                raise RuntimeError(f"Error predicting argument: {e}")
            
        del argument_ambigous_groups

        for argument_index, non_keywords in predicted_argument_group.items():
            non_keyword = self._service.extract_nonkeyword_typemapping(action, argument_index, non_keywords)
            if non_keyword:
                arguments[argument_index] = non_keyword
            else:
                blind_non_keywords.extend(non_keywords)

        del predicted_argument_group

        # Classify non keywords with their type, priority non keywords are quoted tokens
        classified_nonkeywords, classified_priority_nonkeywords = self._service.extract_classified_nonkeywords(blind_non_keywords)

        # Extract arguments that have been assigned, and not been assigned.
        try:
            required_arguments, optional_arguments, unassigned_required_indices, unassigned_optional_indices = self._service.extract_argument_indices_information(action, arguments)
        except Exception as e:
            raise RuntimeError(f"Error extracting unassigned arguments: {e}")

        # If any arguments haven't been assigned, assign using type mapping. 
        # Merge newly mapped arguments with existing arguments
        if unassigned_required_indices or unassigned_optional_indices:
            try:
                required_arguments_typemapping, optional_arguments_typemapping, unassigned_required_indices, unassigned_optional_indices = self._service.extract_arguments_typemapping(action, unassigned_required_indices, unassigned_optional_indices, classified_nonkeywords, classified_priority_nonkeywords)
            except Exception as e:
                raise RuntimeError(f"Error extracting arguments using type mapping: {e}")

            required_arguments = heapq.merge(required_arguments, required_arguments_typemapping, key = lambda argument: argument[0])
            optional_arguments = heapq.merge(optional_arguments, optional_arguments_typemapping, key = lambda argument: argument[0])
        
        # If required arguments haven't been assigned, assign by asking questions to user. 
        # If required arguments questions is None, then user has asked to skip request. 
        # Merge newly mapped arguments with existing arguments
        if unassigned_required_indices:
            required_arguments_questions = self._service.extract_arguments_questions(action, unassigned_required_indices, classified_nonkeywords, classified_priority_nonkeywords)
            if required_arguments_questions is None:
                return None

            required_arguments = heapq.merge(required_arguments, required_arguments_questions, key = lambda argument: argument[0])

        # Combine required arguments and optional arguments.
        for argument in heapq.merge(required_arguments, optional_arguments):
            if argument:
                arguments[argument[0]] = argument[1]

        return action, arguments
    
    def execute_action(self, action : str, arguments: list[str]) -> None:
        # Check if action can be executed
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