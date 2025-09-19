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
            try:
                type = self._wrapper.get_action_args_type(action, idx)
            except Exception as e:
                raise RuntimeError(f"Error fetching argument type for action '{action}' and index '{idx}': {e}")

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
    
    def _extract_argumentgroup_options(self, action: str, argument_indices: list[int], non_keywords: list[str], throw_if_exceed_count: int = float('inf')) -> list[tuple[int, str]]:
        classified_non_keywords, classified_priority_non_keywords = self.extract_classified_non_keywords(non_keywords)

        options = []
        for idx in argument_indices:
            type = self._wrapper.get_argument_type(action, idx)

            if type == "any":
                if classified_priority_non_keywords:
                    for non_keywords in classified_priority_non_keywords.values():
                        for non_keyword in non_keywords:
                            options.append((idx, non_keyword))
                else:
                    for non_keywords in classified_non_keywords.values():
                        for non_keyword in non_keywords:
                            options.append((idx, non_keyword))
            else:
                if type in classified_priority_non_keywords:
                    for non_keyword in classified_priority_non_keywords[type]:
                        options.append((idx, non_keyword))
                elif type in classified_non_keywords:
                    for non_keyword in classified_non_keywords[type]:
                        options.append((idx, non_keyword))

            if len(options) > throw_if_exceed_count:
                raise RuntimeError(f"Too many possibilities")
        
        return options
    
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
        try:
            if self._wrapper.has_action_warning(action):
                answer = input(f"Do you want to, {self._wrapper.get_action_description(action)} (Y/N): ").lower()
                if answer != 'y':
                    print("Skipping request...")
                    return False
            return True
        except Exception as e:
            raise RuntimeError(f"Error checking if action can run: {e}")

    def predict_action_frequency(self, action_keywords: list[str], probability_cutoff: float = 0.85) -> str | None:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        if not action_keywords:
            return None

        keywords_counter = Counter(action_keywords)

        action_counter = Counter()
        for keyword, count in keywords_counter.items():
            try:
                actions = self._wrapper.get_actions_for_keyword(keyword)
            except Exception as e:
                raise RuntimeError(f"Error fetching actions for keyword '{keyword}': {e}")
            
            for action in actions:
                action_counter[action] += count

        max_frequency_action = max(action_counter.items(), key = lambda action_counter: action_counter[1])
        if max_frequency_action[1] / action_counter.total() < probability_cutoff:
            return None
        
        return max_frequency_action[0]

    def predict_action_classification(self, keywords: list[str], max_possibilities: int, probability_cutoff: float = 0.85) -> str | None:
        try:
            actions = self._wrapper.predict_top_actions(keywords, max_possibilities)
        except Exception as e:
            raise RuntimeError(f"Error extracting top actions: {e}")
        
        if actions[0][1] >= probability_cutoff:
            return actions[0][0]
        else:
            try:
                answer = self._handle_options(actions, options_message = "What do you want to do?", key = lambda action: self._wrapper.get_action_description(action[0]))
                print("-----------------------------")
            except Exception as e:
                raise RuntimeError(f"Error fetching answer: {e}")
            
            if answer == -1:
                return None

            try:
                self._wrapper.train_action_pipeline(keywords, actions[answer][0])
            except Exception as e:
                print("Warning: Unable to train parser:", e)
            
            return actions[answer][0]

    def predict_argument_nonkeyword_frequency(self, action: str, argument_group: tuple[list[str], list[str]], probability_cutoff: float = 0.85) -> str | None:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")
        
        argument_keywords, non_keywords = argument_group
        if not argument_keywords:
            return None

        argument_keywords_counter = Counter(argument_keywords)

        argument_counter = Counter()
        for keyword, count in argument_keywords_counter.items():
            try:
                argument_indices = self._wrapper.get_argument_indices_for_keyword(action, keyword)
            except Exception as e:
                raise RuntimeError(f"Error fetching argument indices for keyword '{keyword}' and action '{action}': {e}")
            
            for idx in argument_indices:
                argument_counter[idx] += count

        max_frequency_argument = max(argument_counter.items(), key = lambda argument_counter: argument_counter[1])
        if max_frequency_argument[1] / argument_counter.total() < probability_cutoff:
            return None

    def predict_argument_nonkeyword_classification(self, action: str, argument_group: tuple[list[str], list[str]], max_possibilites: int, probability_cutoff: float = 0.85) -> tuple[int, str] | tuple[None, None]:
        def handle_argument_group_options(indices: list[int]) -> tuple[int, str | None] | None:
            try:
                options = self._extract_argumentgroup_options(action, indices, non_keywords, 1)
            except Exception as e:
                print("Warning: Too many possibilities, please refine your query")
                return None
            
            if not options:
                raise SyntaxError("No valid arguments found")

            if len(options) == 1:
                return options[0]
            
            try:
                answer = self._handle_options(options, options_message = "Which of these is correct?", key = lambda option: f"{self._wrapper.get_argument_description(action, option[0])}: {option[1]}")
                print("-----------------------------")
            except Exception as e:
                raise RuntimeError(f"Error fetching answer: {e}")
            
            if answer == -1:
                return None
            
            try:
                self._wrapper.train_argument_pipeline(action, argument_keywords, options[answer][0])
            except Exception as e:
                print("Warning: Unable to train parser:", e)
            
            return options[answer]

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        argument_keywords, non_keywords = argument_group

        try:
            arguments = self._wrapper.predict_top_arguments_indices(action, argument_keywords, max_possibilites)
        except Exception as e:
            raise RuntimeError(f"Error extracting top arguments: {e}")
        
        if arguments[0][1] >= probability_cutoff:
            return handle_argument_group_options([arguments[0][0]])
        else:
            indices = [arg[0] for arg in arguments]
            return handle_argument_group_options(indices)
            

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
            
            try:
                action_keyword = self._wrapper.match_action_keyword(token, probability_cutoff)
            except Exception as e:
                raise RuntimeError(f"Error matching action keyword: {e}")

            if action_keyword:
                action_keywords.append(action_keyword)

                if non_keywords:
                    action_groups.append(non_keywords)
                    non_keywords = []

                continue
            
            try:
                if not self._wrapper.is_stop_word(token, probability_cutoff):
                    non_keywords.append((token, quoted))
            except Exception as e:
                raise RuntimeError(f"Error checking stop word: {e}")

        if non_keywords:
            action_groups.append(non_keywords)

        return action_keywords, action_groups
    
    def extract_nonkeywords_argument_groups(self, action: str, action_groups: list[list[tuple[str, bool]]], probability_cutoff: float) -> tuple[list[str], list[tuple[list, list]]]:
        priority_groups = []
        blind_groups = []

        non_keywords_flat = []

        for group in action_groups:
            argument_keywords = []
            non_keywords = []

            for token, quoted in group:
                if quoted:
                    non_keywords.append((token, quoted))
                    continue
                
                try:
                    argument_keyword = self._wrapper.match_argument_keyword(action, token, probability_cutoff)
                except Exception as e:
                    raise RuntimeError(f"Error matching argument keyword: {e}")

                if argument_keyword:
                    argument_keywords.append(argument_keyword)
                else:
                    non_keywords.append((token, quoted))

            if argument_keywords:
                priority_groups.append((argument_keywords, non_keywords))
            else:
                blind_groups.append((argument_keywords, non_keywords))
            
            non_keywords_flat.extend(non_keywords)
            
        return non_keywords_flat, blind_groups + priority_groups

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
        try:
            return self._wrapper.get_required_arguments(action)
        except Exception as e:
            raise RuntimeError(f"Error fetching required arguments for action '{action}': {e}")
        
    def get_optional_arguments(self, action: str) -> tuple[list[int]]:
        try:
            return self._wrapper.get_optional_arguments(action)
        except Exception as e:
            raise RuntimeError(f"Error fetching optional arguments for action '{action}': {e}")
        
    def get_arguments_count(self, action: str) -> int:
        try:
            return self._wrapper.get_arguments_count(action)
        except Exception as e:
            raise RuntimeError(f"Error fetching arguments count for action '{action}': {e}")