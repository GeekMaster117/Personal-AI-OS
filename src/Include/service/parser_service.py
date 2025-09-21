from collections import defaultdict, Counter

import shlex

from Include.wrapper.parser_wrapper import ParserWrapper

class ParserService:
    def __init__(self):
        try:
            self._wrapper = ParserWrapper()
        except Exception as e:
            raise RuntimeError(f"Error initialising parser wrapper: {e}")
        
    def _handle_options(self, options: list[str], options_message = "Select an option:", select_message = "Enter an answer", key = lambda x: x) -> int:
        print(options_message)
        for i, option in enumerate(options, start=1):
            print(f"{i}. {key(option)}")
        print(f"{len(options) + 1}. Skip request")

        choice = input(f"{select_message} (1-{len(options) + 1}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice) - 1
        
        return -1
    
    def _handle_argument_group_options(self, action: str, argument_indices: list[int], argument_group: tuple[list[str], set[tuple]], max_possibilities: int) -> tuple[int, str] | tuple[None, None]:
        argument_keywords, non_keywords = argument_group

        try:
            options = self._extract_argumentgroup_options(action, argument_indices, non_keywords, max_possibilities)
        except SyntaxError as e:
            print("Warning: Too many possibilities")
            raise SyntaxError("Too many possibilities")
        
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
            return None, None
        
        try:
            self._wrapper.train_argument_pipeline(action, argument_keywords, options[answer][0])
        except Exception as e:
            print("Warning: Unable to train parser:", e)
        
        return options[answer]

    def _pop_nonkeyword(self, type: str, classified_nonkeywords: dict, classified_priority_nonkeywords: dict, throw_if_not_found: bool = False) -> str | None:
        non_keyword: str | None = None

        if type == "any":
            if throw_if_not_found and not classified_nonkeywords and not classified_priority_nonkeywords:
                raise SyntaxError(f"Could not find valid value for type '{type}'")

            if classified_priority_nonkeywords and len(classified_priority_nonkeywords) == 1 and len(next(iter(classified_priority_nonkeywords.values()))) == 1:
                non_keyword = classified_priority_nonkeywords.popitem()[1].pop()
            elif classified_nonkeywords and len(classified_nonkeywords) == 1 and len(next(iter(classified_nonkeywords.values()))) == 1:
                non_keyword = classified_nonkeywords.popitem()[1].pop()
        else:
            if throw_if_not_found and type not in classified_nonkeywords and type not in classified_priority_nonkeywords:
                raise SyntaxError(f"Could not find valid value for type '{type}'")

            if type in classified_priority_nonkeywords and len(classified_priority_nonkeywords[type]) == 1:
                non_keyword = classified_priority_nonkeywords[type].pop()

                if not classified_priority_nonkeywords[type]:
                    del classified_priority_nonkeywords[type]
            elif type in classified_nonkeywords and len(classified_nonkeywords[type]) == 1:
                non_keyword = classified_nonkeywords[type].pop()

                if not classified_nonkeywords[type]:
                    del classified_nonkeywords[type]

        return non_keyword
    
    def _pop_nonkeyword_question(self, type: str, description: str, classified_nonkeywords: dict, classified_priority_nonkeywords: dict) -> str | None:
        options: list[str] = []
        borrowed_dict: dict | None = None
        borrowed_types: dict | None = None

        if type == "any":
            if not classified_nonkeywords and not classified_priority_nonkeywords:
                raise SyntaxError(f"Could not find '{description}'")

            if classified_priority_nonkeywords:
                borrowed_dict = classified_priority_nonkeywords
            elif classified_nonkeywords:
                borrowed_dict = classified_nonkeywords

            for borrowed_type, non_keywords in borrowed_dict.items():
                borrowed_types[borrowed_type] = (len(options) + 1, len(options) + len(non_keywords))
                options.extend(non_keywords)
        else:
            if type not in classified_nonkeywords and type not in classified_priority_nonkeywords:
                raise SyntaxError(f"Could not find '{description}'")

            if type in classified_priority_nonkeywords:
                borrowed_dict = classified_priority_nonkeywords
            elif type in classified_nonkeywords:
                borrowed_dict = classified_nonkeywords

            borrowed_types[borrowed_type] = (1, borrowed_dict[type])
            options.extend(borrowed_dict[type])

        answer = self._handle_options(options, options_message = f"What is, {description}")
        if answer == -1:
            return None
        
        for type, interval in borrowed_types.items():
            if interval[0] <= answer <= interval[1]:
                borrowed_dict[type].remove(options[answer])
                break
        else:
            raise RuntimeError("Could not parse the answer")

        return options[answer]
    
    def _extract_arguments_typemapping(self, action: str, argument_indices: list[int], classified_nonkeywords: dict, classified_priority_nonkeywords: dict, throw_if_not_found: bool = False) -> tuple[list[tuple], list[int]]:
        arguments = []
        any_type_indices = []
        for idx in argument_indices:
            try:
                type = self._wrapper.get_argument_type(action, idx)
            except Exception as e:
                raise RuntimeError(f"Error fetching argument type for action '{action}' and index '{idx}': {e}")

            if type == "any":
                any_type_indices.append(len(arguments))
                arguments.append(None)
            else:
                try:
                    non_keyword = self._pop_nonkeyword(type, classified_nonkeywords, classified_priority_nonkeywords, throw_if_not_found)
                except Exception:
                    raise SyntaxError(f"Could not find valid value for argument '{self._wrapper.get_argument_description(action, idx)}'")

                arguments.append((idx, non_keyword))

        for idx in any_type_indices:
            try:
                non_keyword = self._pop_nonkeyword(type, classified_nonkeywords, classified_priority_nonkeywords, throw_if_not_found)
            except Exception:
                raise SyntaxError(f"Could not find valid value for argument '{idx}'")
            
            arguments[idx] = (idx, non_keyword)

        unassigned_indices = [i for i, arg in enumerate(arguments) if arg is None]

        return arguments, unassigned_indices
    
    def _extract_argumentgroup_options(self, action: str, argument_indices: list[int], non_keywords: set[tuple[str, bool]], throw_if_exceed_count: int = float('inf')) -> list[tuple[int, str]]:
        classified_non_keywords, classified_priority_non_keywords = self.extract_classified_nonkeywords(non_keywords)

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
                raise SyntaxError(f"Too many possibilities")
        
        return options
        
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

    def predict_action_classification(self, action_keywords: list[str], max_possibilities: int, probability_cutoff: float = 0.85) -> str | None:
        try:
            actions = self._wrapper.predict_top_actions(action_keywords, max_possibilities, probability_cutoff)
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
                self._wrapper.train_action_pipeline(action_keywords, actions[answer][0])
            except Exception as e:
                print("Warning: Unable to train parser:", e)
            
            return actions[answer][0]

    def predict_argument_nonkeyword_frequency(self, action: str, argument_group: tuple[list[str], set[tuple]], max_possibilities: int, probability_cutoff: float = 0.85) -> tuple[int, str] | tuple[None, None]:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")
        
        argument_keywords = argument_group[0]
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
        
        return self._handle_argument_group_options(action, [max_frequency_argument[0]], argument_group, max_possibilities)

    def predict_argument_nonkeyword_classification(self, action: str, argument_group: tuple[list[str], set[tuple]], max_possibilites: int, probability_cutoff: float = 0.85) -> tuple[int, str] | tuple[None, None]:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        argument_keywords = argument_group[0]

        try:
            arguments = self._wrapper.predict_top_arguments_indices(action, argument_keywords, max_possibilites, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error extracting top arguments: {e}")
        
        if arguments[0][1] >= probability_cutoff:
            return self._handle_argument_group_options(action, [arguments[0][0]], argument_group, max_possibilites)
        else:
            indices = [arg[0] for arg in arguments]
            return self._handle_argument_group_options(action, indices, argument_group, max_possibilites)
            
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
    
    def extract_argument_groups(self, action: str, action_groups: list[list[tuple]], probability_cutoff: float) -> tuple[list[tuple], list[str]]:
        argument_groups = []
        blind_non_keywords = []

        for group in action_groups:
            argument_keywords = []
            non_keywords = set()

            for token, quoted in group:
                if quoted:
                    blind_non_keywords.append((token, quoted))
                    continue
                
                try:
                    argument_keyword = self._wrapper.match_argument_keyword(action, token, probability_cutoff)
                except Exception as e:
                    raise RuntimeError(f"Error matching argument keyword: {e}")

                if argument_keyword:
                    argument_keywords.append(argument_keyword)
                else:
                    non_keywords.add((token, quoted))

            if argument_keywords:
                argument_groups.append((argument_keywords, non_keywords))
            else:
                blind_non_keywords.extend(non_keywords)
            
        return argument_groups, blind_non_keywords

    def extract_classified_nonkeywords(self, non_keywords: list[tuple[str, bool]]) -> tuple[dict, dict]:
        classified_nonkeywords, classified_priority_nonkeywords = defaultdict(list), defaultdict(list)
        for token, quoted in non_keywords:
            type = "any"
            if token.isdecimal():
                type = "int"
            elif token.isalpha():
                type = "str"
            
            if quoted:
                classified_priority_nonkeywords[type].append(token)
            else:
                classified_nonkeywords[type].append(token)

        return classified_nonkeywords, classified_priority_nonkeywords
    
    def extract_argument_indices_information(self, action: str, arguments: list[str]) -> tuple[list[tuple], list[tuple], list[int], list[int]]:
        try:
            required_indices = self._wrapper.get_required_arguments(action)
            optional_indices = self._wrapper.get_optional_arguments(action)
        except Exception as e:
            raise RuntimeError(f"Error fetching arguments for action '{action}': {e}")
        
        assigned_required_arguments = []
        assigned_optional_arguments = []
        unassigned_required_indices = []
        unassigned_optional_indices = []
        
        for idx, non_keyword in enumerate(arguments):
            if idx in required_indices:
                if non_keyword:
                    assigned_required_arguments.append((idx, non_keyword))
                else:
                    unassigned_required_indices.append(idx)
            elif idx in optional_indices:
                if non_keyword:
                    assigned_optional_arguments.append((idx, non_keyword))
                else:
                    unassigned_optional_indices.append(idx)

        return assigned_required_arguments, assigned_optional_arguments, unassigned_required_indices, unassigned_optional_indices
    
    def extract_arguments_typemapping(self, action: str, required_indices: list[int], optional_indices: list[int], classified_non_keywords: dict, classified_priority_non_keywords: dict) -> tuple[list[tuple], list[tuple], list[int], list[int]]:
        required_arguments, unassigned_required_indices = self._extract_arguments_typemapping(action, required_indices, classified_non_keywords, classified_priority_non_keywords, throw_if_not_found = True)
        optional_arguments, unassigned_optional_indices = self._extract_arguments_typemapping(action, optional_indices, classified_non_keywords, classified_priority_non_keywords)

        return required_arguments, optional_arguments, unassigned_required_indices, unassigned_optional_indices
    
    def extract_arguments_questions(self, action: str, argument_indices: list[int], classified_non_keywords: dict, classified_priority_non_keywords: dict) -> list[tuple] | None:
        arguments = []

        for idx in argument_indices:
            try:
                type = self._wrapper.get_argument_type(action, idx)
                description = self._wrapper.get_action_description(action, idx)
            except Exception as e:
                raise RuntimeError(f"Error fetching argument data: {e}")
            
            non_keyword = self._pop_nonkeyword_question(type, description, classified_non_keywords, classified_priority_non_keywords)
            if not non_keyword:
                return None
            
            arguments.append((idx, non_keyword))

        return arguments

    def get_arguments_count(self, action: str) -> int:
        return self._wrapper.get_arguments_count(action)