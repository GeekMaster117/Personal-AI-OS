from collections import defaultdict, Counter

import shlex

from Include.wrapper.parser_wrapper import ParserWrapper

class ParserService:
    def __init__(self):
        try:
            self._wrapper = ParserWrapper()
        except Exception as e:
            raise RuntimeError(f"Error initialising parser wrapper: {e}")

    def _pop_non_keyword_type_matching(self, type: str, classified_non_keywords: dict, classified_priority_non_keywords: dict) -> str | None:
        non_keyword: str | None = None

        if type == "any":
            if classified_priority_non_keywords and len(classified_priority_non_keywords) == 1 and len(next(iter(classified_priority_non_keywords.values()))) == 1:
                non_keyword = classified_priority_non_keywords.popitem()[1].pop()
            elif classified_non_keywords and len(classified_non_keywords) == 1 and len(next(iter(classified_non_keywords.values()))) == 1:
                non_keyword = classified_non_keywords.popitem()[1].pop()
        else:
            if type in classified_priority_non_keywords and len(classified_priority_non_keywords[type]) == 1:
                non_keyword = classified_priority_non_keywords[type].pop()

                if not classified_priority_non_keywords[type]:
                    del classified_priority_non_keywords[type]
            elif type in classified_non_keywords and len(classified_non_keywords[type]) == 1:
                non_keyword = classified_non_keywords[type].pop()

                if not classified_non_keywords[type]:
                    del classified_non_keywords[type]

        return non_keyword
    
    def _handle_options(self, options: list[str], options_message = "Select an option:", select_message = "Enter an answer", key = lambda x: x) -> int:
        print(options_message)
        for i, option in enumerate(options, start=1):
            print(f"{i}. {key(option)}")
        print(f"{len(options) + 1}. Skip request")

        choice = input(f"{select_message} (1-{len(options) + 1}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice) - 1
        
        return -1
    
    def _extract_arguments_type_matching(self, action: str, indices: list[int], classified_non_keywords: dict, classified_priority_non_keywords: dict) -> tuple[list[str], list[int]]:
        arguments = []
        any_type_indices = []
        for idx in indices:
            type = self._wrapper.get_action_args_type(action, idx)

            if type == "any":
                any_type_indices.append(len(arguments))
                arguments.append(None)
            else:
                non_keyword = self._pop_non_keyword_type_matching(type, classified_non_keywords, classified_priority_non_keywords)
                arguments.append(non_keyword)

        for idx in any_type_indices:
            non_keyword = self._pop_non_keyword_type_matching(type, classified_non_keywords, classified_priority_non_keywords)
            arguments[idx] = non_keyword

        unassigned_indices = [i for i, arg in enumerate(arguments) if arg is None]

        return arguments, unassigned_indices
    
    def check_argument_availability_else_throw(self, required_needed: Counter, classified_non_keywords: dict, classified_priority_non_keywords: dict) -> bool:
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
    
    def extract_action_keywords_groups(self, tokens: list[tuple[str, bool]], probability_cutoff: float) -> tuple[list[str], list[list[tuple]]]:
        action_keywords = []

        action_groups = []
        non_keywords = []

        for token, quoted in tokens:
            if quoted:
                non_keywords.append((token, quoted))
                continue

            action_keyword = self._wrapper.match_action_keyword(token, probability_cutoff)
            if action_keyword:
                action_keywords.append(action_keyword)

                if non_keywords:
                    action_groups.append(non_keywords)
                    non_keywords = []

                continue
            
            if not self._wrapper.is_stop_word(token, probability_cutoff):
                non_keywords.append((token, quoted))

        if non_keywords:
            action_groups.append(non_keywords)

        return action_keywords, action_groups
    
    def extract_nonkeywords_argument_groups(self, action: str, action_groups: list[list[tuple[str, bool]]], probability_cutoff: float) -> tuple[list[str], list[tuple[str, bool]]]:
        argument_groups = []
        non_keywords_flat = []

        for group in action_groups:
            argument_keywords = []
            non_keywords = []

            for token, quoted in group:
                if quoted:
                    non_keywords.append((token, quoted))
                    continue

                argument_keyword = self._wrapper.match_argument_keyword(action, token, probability_cutoff)
                if argument_keyword:
                    argument_keywords.append(argument_keyword)
                else:
                    non_keywords.append((token, quoted))
                
            argument_groups.append((argument_keywords, non_keywords))
            non_keywords_flat.extend(non_keywords)
            
        return non_keywords_flat, argument_groups

    def extract_classified_non_keywords(self, non_keywords: list[tuple[str, bool]]) -> tuple[dict, dict]:
        classified_non_keywords, classified_priority_non_keywords = defaultdict(list), defaultdict(list)
        for token, quoted in non_keywords:
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

    def predict_action_frequency(self, action_keywords: list[str], probability_cutoff: float = 0.85) -> str | None:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        if not action_keywords:
            return None

        keywords_counter = Counter(action_keywords)

        action_counter = Counter()
        for keyword, count in keywords_counter.items():
            actions = self._wrapper.get_actions_for_keyword(keyword)
            for action in actions:
                action_counter[action] += count

        max_frequency_action = max(action_counter.items(), key = lambda action_counter: action_counter[1])
        if max_frequency_action[1] / action_counter.total() < probability_cutoff:
            return None
        
        return max_frequency_action[0]
    
    def predict_action_classification(self, keywords: list[str], top_actions_count: int) -> str | None:
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

    def predict_argument_frequency(self, action: str, argument_keywords: list[str], probability_cutoff: float = 0.85) -> str | None:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        if not argument_keywords:
            return None

        argument_keywords_counter = Counter(argument_keywords)

        argument_counter = Counter()
        for keyword, count in argument_keywords_counter.items():
            arguments = self._wrapper.get_arguments_for_keyword(action, keyword)
            for argument in arguments:
                argument_counter[argument] += count

        max_frequency_argument = max(argument_counter.items(), key = lambda argument_counter: argument_counter[1])
        if max_frequency_argument[1] / argument_counter.total() < probability_cutoff:
            return None

        return max_frequency_argument[0]

    def extract_arguments_type_mapping(self, action: str, classified_non_keywords: dict, classified_priority_non_keywords: dict, required_indices: list[int], optional_indices: list[int]) -> tuple[list[str], list[int], list[int]]:
        required_arguments, unassigned_required_indices = self._extract_arguments_type_matching(action, required_indices, classified_non_keywords, classified_priority_non_keywords)
        optional_arguments, unassigned_optional_indices = self._extract_arguments_type_matching(action, optional_indices, classified_non_keywords, classified_priority_non_keywords)

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

        return arguments, unassigned_required_indices, unassigned_optional_indices
    
    def get_required_arguments(self, action: str) -> tuple[list[int], Counter]:
        return self._wrapper.get_required_arguments(action)
        
    def get_optional_arguments(self, action: str) -> tuple[list[int]]:
        return self._wrapper.get_optional_arguments(action)