import sys
import subprocess

from collections import defaultdict

from typing import Any

import settings
from Include.service.parser_service import ParserService

class Parser:
    def __init__(self, environment: settings.Environment):
        self._service: ParserService | None = None
        try:
            self._service = ParserService(environment)
        except Exception as e:
            raise RuntimeError(f"Error initialising parser service: {e}")
        
    def __del__(self):
        if self._service is not None:
            del self._service
        
    def _merge_sort(self, list1: list, list2: list) -> list:
        result = []
        ptr1, ptr2 = 0, 0

        while ptr1 < len(list1) and ptr2 < len(list2):
            if list1[ptr1] < list2[ptr2]:
                result.append(list1[ptr1])
                ptr1 += 1
            else:
                result.append(list2[ptr2])
                ptr2 += 1

        # Add leftovers
        if ptr1 < len(list1):
            result.extend(list1[ptr1:])
        if ptr2 < len(list2):
            result.extend(list2[ptr2:])

        return result
    
    def _handle_parseraction(self, action: str, arguments: list[str]) -> bool:
        # Handle special parser actions, and return if action is handled

        return_value = True

        if action == "exit":
            print("Exiting application...")
            print("-----------------------------")
            exit(0)
        else:
            return_value = False

        return return_value
    
    def _preprocess_arguments(self, action: str, arguments: list[str]) -> Any:
        # Preprocess arguments based on action type, and returns any data needed for later

        preprocess_data: Any = None

        if action == "start":
            preprocess_data = self._service.extract_app(arguments[0])
            if preprocess_data is None:
                raise ValueError(f"Could not find app/nickname/class {arguments[0]}. Try having the app open, running in background.")
            
            executable_path = self._service.get_executablepath(preprocess_data)
            arguments[0] = executable_path

        for idx in range(len(arguments)):
            if arguments[idx] is None:
                arguments[idx] = ''
            else:
                arguments[idx] = self._service.get_argument_format(action, idx) + '"' + arguments[idx] + '"'

        return preprocess_data
    
    def _postprocess(self, action: str, data: Any) -> None:
        # Postprocess after action execution based on action type and custom data

        if action == "start":
            self._service.handle_nickname_class(data)
        
    def extract_action_arguments(self, query: str, probability_cutoff: float = 0.85) -> tuple[str, list[str]] | tuple[None, None]:
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
                    return None, None
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
        
        # Argument list to which non keywords are mapped
        arguments: list[str | None] = [None] * self._service.get_arguments_count(action)

        # If any group can be predicted successfully, add the non keywords to predicted arguments
        # Merge the rest of groups flushed by predicted groups.
        merged_argument_groups = []
        predicted_arguments = defaultdict(list)

        # Predict argument index using frequency method first, then classification method if frequency method fails.
        argument_keywords, non_keywords = [], set()
        for group in argument_groups:
            try:
                argument_index = self._service.predict_argument_frequency(action, group[0], probability_cutoff)
                if argument_index is None:
                    argument_index = self._service.predict_argument_classification(action, group[0], probability_cutoff)
            except Exception as e:
                raise RuntimeError(f"Error predicting argument: {e}")

            if argument_index is None:
                argument_keywords.extend(group[0])
                non_keywords.update(group[1])
            else:
                if argument_keywords:
                    merged_argument_groups((argument_keywords, non_keywords))

                predicted_arguments[argument_index].extend(group[1])
        if argument_keywords:
            merged_argument_groups((argument_keywords, non_keywords))

        del argument_groups, argument_keywords, non_keywords
        
        # If any argument can be predicted successfully, add the non keywords to predicted arguments
        # Else asks user for correct possible option. Options - (argument, non keyword)
        # Predict argument index using frequency method first, then classification method if frequency method fails.
        for group in merged_argument_groups:
            try:
                argument_index = self._service.predict_argument_frequency(action, group[0], probability_cutoff)
                if argument_index is None:
                    argument_index, non_keyword, skip = self._service.predict_argument_nonkeyword_classification(action, group, 5, probability_cutoff)
                    if skip:
                        return None, None
                    
                    if non_keyword:
                        arguments[argument_index] = non_keyword
                    else:
                        predicted_arguments[argument_index].extend(group[1])
            except Exception as e:
                raise RuntimeError(f"Error predicting argument: {e}")
        
        del merged_argument_groups

        # Tries to predict non keyword using type mapping.
        # Else asks user for correct non keyword.
        for argument_index, non_keywords in predicted_arguments.items():
            non_keyword = self._service.extract_nonkeyword_typemapping(action, argument_index, non_keywords)

            if not non_keyword:
                argument = self._service.extract_arguments_questions_nonkeywords(action, [argument_index], non_keywords)
                if not argument:
                    return None, None
                
                arguments[argument_index] = argument[0][1]
            else:
                arguments[argument_index] = non_keyword

        del predicted_arguments

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

            required_arguments = self._merge_sort(required_arguments, required_arguments_typemapping)
            optional_arguments = self._merge_sort(optional_arguments, optional_arguments_typemapping)
        
        # If required arguments haven't been assigned, assign by asking questions to user. 
        # If required arguments questions is None, then user has asked to skip request. 
        # Merge newly mapped arguments with existing arguments
        if unassigned_required_indices:
            required_arguments_questions = self._service.extract_arguments_questions_classified_nonkeywords(action, unassigned_required_indices, classified_nonkeywords, classified_priority_nonkeywords, True)
            if required_arguments_questions is None:
                return None, None

            required_arguments = self._merge_sort(required_arguments, required_arguments_questions)

        # Combine required arguments and optional arguments.
        for argument in self._merge_sort(required_arguments, optional_arguments):
            if argument:
                arguments[argument[0]] = argument[1]

        return action, arguments
    
    def execute_action(self, action : str, arguments: list[str]) -> None:
        try:
            arguments_count = self._service.get_arguments_count(action)
        except Exception as e:
            raise ValueError(f"Error getting arguments count: {e}")
        
        if len(arguments) != arguments_count:
            raise ValueError(f"Invalid arguments count. Expected {arguments_count}, got {len(arguments)}")
        
        # Handle special parser actions
        if self._handle_parseraction(action, arguments):
            return

        # Check if action can be executed
        if not self._service.canrun_action(action):
            return
        
        # Preprocess action and arguments before execution
        preprocess_data = self._preprocess_arguments(action, arguments)
            
        command = " ".join([action] + arguments)

        print("Executing Command: " + command)
        subprocess.run(command, shell=True)
        print("-----------------------------")

        print("Command Executed")
        print("-----------------------------")

        # Postprocess after action execution
        self._postprocess(action, preprocess_data)

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