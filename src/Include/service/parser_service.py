import json
import hashlib

from collections import defaultdict, Counter

import shlex
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from Include.wrapper.parser_wrapper import ParserWrapper

class ParserService:
    def __init__(self):
        try:
            self._wrapper = ParserWrapper()
        except Exception as e:
            raise RuntimeError(f"Error initialising parser wrapper: {e}")
    
    def _check_argument_availability_else_throw(self, required_needed: Counter, classified_non_keywords: dict, classified_priority_non_keywords: dict) -> bool:
        def get_type_count(type: str) -> int:
            return len(classified_non_keywords.get(type, [])) + len(classified_priority_non_keywords.get(type, []))
        
        def get_distinct_keys() -> set[str]:
            return classified_non_keywords.keys() | classified_priority_non_keywords.keys()
        
        def get_distinct_non_any_keys() -> set[str]:
            distinct_keys = classified_non_keywords.keys() | classified_priority_non_keywords.keys()
            distinct_keys.discard("any")

            return distinct_keys
        
        def raise_arguments_not_found_error() -> None:
            arguments_required = ""
            for type, required in required_needed.items():
                arguments_required += f"{type}: {required}\n"

            arguments_found = ""
            for type in get_distinct_keys():
                arguments_found += f"{type}: {get_type_count(type)}\n"
            if not arguments_found:
                arguments_found = "No arguments found"

            error_message = (
                "Found less arguments then required",
                "Arguments required:",
                arguments_required,
                "Arguments found:",
                arguments_found
                )
            raise SyntaxError('\n'.join(error_message))
        
        any_type_count = get_type_count("any")
        non_any_type_count = sum([get_type_count(type) for type in get_distinct_non_any_keys()])
        for type, required in required_needed.items():
            type_count = get_type_count(type)

            if type == "any":
                if type_count < required:
                    if non_any_type_count < required - type_count:
                        raise_arguments_not_found_error()
                    
                    any_type_count = 0
                    non_any_type_count -= required - type_count
                else:
                    any_type_count -= required
            else:
                if type_count < required:
                    raise_arguments_not_found_error()
                else:
                    non_any_type_count -= required

    def _pop_non_keyword(self, type: str, description: str, classified_non_keywords: dict, classified_priority_non_keywords: dict) -> str | None:
        def pop(non_keywords: list[str], borrowed_dict: dict, borrowed_type: str, index: int = -1) -> str:
            non_keyword: str = non_keywords.pop(index)
            del borrowed_dict[borrowed_type]

            return non_keyword

        non_keywords: list[list[str]] | None = None
        borrowed_dict: dict | None = None
        borrowed_types: dict | None = None

        if type == "any":
            if classified_priority_non_keywords:
                non_keywords, borrowed_types = [], dict()
                for t, nk in classified_priority_non_keywords.items():
                    borrowed_types[t] = (len(non_keywords), len(non_keywords) + len(nk) - 1)
                    non_keywords.extend(nk)
                borrowed_dict = classified_priority_non_keywords
            elif classified_non_keywords:
                non_keywords, borrowed_types = [], dict()
                for t, nk in classified_non_keywords.items():
                    borrowed_types[t] = (len(non_keywords), len(non_keywords) + len(nk) - 1)
                    non_keywords.extend(nk)
                borrowed_dict = classified_non_keywords
        else:
            if type in classified_priority_non_keywords:
                non_keywords = classified_priority_non_keywords[type]
                borrowed_dict = classified_priority_non_keywords
                borrowed_types = {"type": (0, len(non_keywords) - 1)}
            elif type in classified_non_keywords:
                non_keywords = [classified_non_keywords[type]]
                borrowed_dict = classified_non_keywords
                borrowed_types = {"type": (0, len(non_keywords) - 1)}

        if not non_keywords:
            raise ValueError("Non keywords not found")
        
        if len(non_keywords) == 1:
            return pop(non_keywords, borrowed_dict, borrowed_types.popitem()[0])
        
        answer = self.handle_options(non_keywords, options_message = f"What is, {description}")
        if answer == -1:
            return None
        
        for t, range in borrowed_types.items():
            if range[0] <= answer <= range[1]:
                return pop(non_keywords, borrowed_dict, t, answer)
            
        raise RuntimeError("Unable to map non keyword to type")
    
    def _handle_options(self, options: list[str], options_message = "Select an option:", select_message = "Enter an answer", key = lambda x: x) -> int:
        print(options_message)
        for i, option in enumerate(options, start=1):
            print(f"{i}. {key(option)}")
        print(f"{len(options) + 1}. Skip request")

        choice = input(f"{select_message} (1-{len(options) + 1}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice) - 1
        
        return -1
    
    def _extract_required_arguments(self, action: str, required_indices: list[int], classified_non_keywords: dict, classified_priority_non_keywords: dict) -> list[str]:
        required_arguments = []
        any_type_argument_indices = []
        for idx in required_indices:
            type = self._wrapper.get_action_args_type(action, idx)
            description = self._wrapper.get_action_args_description(action, idx)

            if type == "any":
                any_type_argument_indices.append(len(required_arguments))
                required_arguments.append(None)
            else:
                try:
                    non_keyword = self._pop_non_keyword(type, description, classified_non_keywords, classified_priority_non_keywords)
                except Exception as e:
                    raise RuntimeError(f"Error mapping non keywords: {e}")
                
                if not non_keyword:
                    raise RuntimeError("Could not map non keywords, even if they are available")
                
                required_arguments.append(non_keyword)

        for idx in any_type_argument_indices:
            description = self._wrapper.get_action_args_description(action, idx)

            try:
                non_keyword = self._pop_non_keyword(type, description, classified_non_keywords, classified_priority_non_keywords)
            except Exception as e:
                raise RuntimeError(f"Error fetching non keywords: {e}")
            
            if not non_keyword:
                raise RuntimeError("Could not fetch non keywords, even if they are available")

            required_arguments[idx] = non_keyword

        return required_arguments

    def _extract_optional_arguments(self, action: str, optional_indices: list[int], classified_non_keywords: dict, classified_priority_non_keywords: dict) -> list[str]:
        optional_arguments = []
        any_type_argumenet_indices = []
        for idx in optional_indices:
            type = self._wrapper.get_action_args_type(action, idx)
            description = self._wrapper.get_action_args_description(action, idx)

            if type == "any":
                any_type_argumenet_indices.append(len(optional_arguments))
                optional_arguments.append(None)
            else:
                try:
                    non_keyword = self._pop_non_keyword(type, description, classified_non_keywords, classified_priority_non_keywords)
                except Exception as e:
                    raise RuntimeError(f"Error mapping non keywords: {e}")
                
                if not non_keyword:
                    break
                
                optional_arguments.append(non_keyword)

        for idx in any_type_argumenet_indices:
            description = self._wrapper.get_action_args_description(action, idx)

            try:
                non_keyword = self._pop_non_keyword(type, description, classified_non_keywords, classified_priority_non_keywords)
            except Exception as e:
                raise RuntimeError(f"Error fetching non keywords: {e}")
            
            if not non_keyword:
                break

            optional_arguments[idx] = non_keyword

        return optional_arguments
        
    def canRunAction(self, action: str) -> bool:
        if self._wrapper.has_action_warning(action):
            answer = input(f"Do you want to, {self._wrapper.get_action_description(action)} (Y/N): ").lower()
            if answer != 'y':
                print("Skipping request...")
                return False
        return True

    def extract_tokens(self, query: str) -> list[tuple[str, bool]]:
        lexer = shlex.shlex(query)
        lexer.whitespace_split = True

        tokens = []
        quotes = {'"', "'"}
        for token in lexer:
            quoted = token[0] in quotes and token[-1] in quotes
            if quoted:
                if len(token) < 3:
                    continue
                tokens.append((token[1:len(token) - 1], quoted))
            else:
                tokens.append((token, quoted))
        
        return tokens
    
    def extract_keywords_nonkeywords(self, tokens: list[tuple[str, bool]], probability_cutoff: float) -> tuple[list[str], list[tuple[str, bool]]]:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        keywords = []
        non_keywords = []
        for token, quoted in tokens:
            if quoted:
                non_keywords.append((token, quoted))
                continue

            keyword = process.extractOne(token, self._wrapper.get_all_keywords(), scorer=fuzz.ratio, score_cutoff=probability_cutoff * 100)
            if keyword:
                keywords.append(keyword[0])
            else:
                non_keywords.append((token, quoted))
        
        return keywords, non_keywords
    
    def extract_classified_non_keywords(self, non_keywords: list[tuple[str, bool]]) -> tuple[dict, dict]:
        classified_non_keywords, classified_priority_non_keywords = defaultdict(list), defaultdict(list)
        for token, quoted in non_keywords:
            if not quoted and token in ENGLISH_STOP_WORDS:
                continue

            type = "any"
            if token.isdecimal():
                type = "int"
            elif token.isalpha():
                type = "str"
            
            if quoted:
                classified_priority_non_keywords[type].append(token)
            else:
                classified_non_keywords[type].append(token)

        return classified_non_keywords, classified_priority_non_keywords

    def extract_actions_normalised(self, keywords: list[str]) -> dict:
        keywords_counter = Counter(keywords)

        action_counter = Counter()
        for keyword, count in keywords_counter.items():
            actions = self._wrapper.get_actions_for_keyword(keyword)
            for action in actions:
                action_counter[action] += count
        
        actions_normalised = dict()
        for action, keyword_count in action_counter.items():
            actions_normalised[action] = keyword_count / action_counter.total()
        
        return actions_normalised

    def extract_action_frequency(self, actions_normalised: dict, probability_cutoff: float = 0.85) -> str | None:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        if not actions_normalised:
            return None

        action_normalised = max(actions_normalised.items(), key = lambda action_normalised: action_normalised[1])
        if action_normalised[1] < probability_cutoff:
            return None
        
        return action_normalised[0]
    
    def extract_action_classification(self, keywords: list[str], top_actions_count: int) -> str | None:
        try:
            actions = self._wrapper.predict_top_actions(keywords, top_actions_count)
        except Exception as e:
            raise RuntimeError(f"Error extracting top actions: {e}")
        
        if actions[0][1] >= 0.85:
            return actions[0][0]
        else:
            try:
                answer = self._handle_options(actions, options_message = "What do you want to do?", key = lambda action: self._wrapper.get_action_description(action[0]))
                print("-----------------------------")
            except Exception as e:
                raise RuntimeError(f"Error fetching answer: {e}")
            
            if answer == -1:
                return None

            self._wrapper.train(keywords, actions[answer][0])
            return actions[answer][0]

    def extract_arguments(self, action: str, classified_non_keywords: dict, classified_priority_non_keywords: dict) -> list[str]:
        if not self._wrapper.has_args(action):
            return []

        try:
            required_indices, required_needed = self._wrapper.get_required_arguments(action)
            optional_indices = self._wrapper.get_optional_arguments(action)
        except Exception as e:
            raise RuntimeError(f"Error fetching arguments for action '{action}': {e}")
        
        if not required_indices and not optional_indices:
            return []
        
        self._check_argument_availability_else_throw(required_needed, classified_non_keywords, classified_priority_non_keywords)

        required_arguments: list[str] = self._extract_required_arguments(action, required_indices, classified_non_keywords, classified_priority_non_keywords)
        optional_arguments: list[str] = self._extract_optional_arguments(action, optional_indices, classified_non_keywords, classified_priority_non_keywords)

        #merge sort required and optional arguments
        arguments = []

        ptr1, ptr2 = 0, 0
        while ptr1 < len(required_indices) or ptr2 < len(optional_indices):
            if ptr2 >= len(optional_indices) or required_indices[ptr1] < optional_indices[ptr2]:
                arguments.append(required_arguments[ptr1])
                ptr1 += 1
            elif ptr2 < len(optional_arguments):
                arguments.append(optional_arguments[ptr2])
                ptr2 += 1
            else:
                arguments.append(None)
                ptr2 += 1

        return arguments