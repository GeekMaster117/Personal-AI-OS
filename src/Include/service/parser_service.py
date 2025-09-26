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
        # An extra option(skip request) is provided to user.
        # If user enters skip request or any number outside given options, -1 is returned.

        print(options_message)
        for i, option in enumerate(options, start=1):
            print(f"{i}. {key(option)}")
        print(f"{len(options) + 1}. Skip request")

        choice = input(f"{select_message} (1-{len(options) + 1}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice) - 1
        
        return -1
    
    def _handle_argument_group_options(self, action: str, argument_indices: list[int], argument_group: tuple[list[str], set[tuple]], max_possibilities: int, train: bool) -> tuple[int, str] | tuple[None, None]:
        # Generates all possibilities of arguments and non keywords and asks user for correct option.
        # Either returns (argument keyword, non keyword) or (None, None)

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
        
        if train:
            try:
                self._wrapper.train_argument_pipeline(action, argument_keywords, options[answer][0])
            except Exception as e:
                print("Warning: Unable to train parser:", e)
        
        return options[answer]

    def _pop_nonkeyword(self, type: str, classified_nonkeywords: dict, classified_priority_nonkeywords: dict, throw_if_not_found: bool = False) -> str | None:
        # Will try to pop from priority non keywords then non keywords.
        # If type is 'any' will try to pop from every type available.

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
        # If multiple non keywords are available with the same type and priority, then will ask user for choice.

        options: list[str] = []
        borrowed_dict: dict = dict()
        borrowed_types: dict = dict()

        if type == "any":
            if not classified_nonkeywords and not classified_priority_nonkeywords:
                raise SyntaxError(f"Could not find '{description}'")

            if classified_priority_nonkeywords:
                borrowed_dict = classified_priority_nonkeywords
            elif classified_nonkeywords:
                borrowed_dict = classified_nonkeywords

            for borrowed_type, non_keywords in borrowed_dict.items():
                borrowed_types[borrowed_type] = (len(options), len(options) + len(non_keywords) - 1)
                options.extend(non_keywords)
        else:
            if type not in classified_nonkeywords and type not in classified_priority_nonkeywords:
                raise SyntaxError(f"Could not find '{description}'")

            if type in classified_priority_nonkeywords:
                borrowed_dict = classified_priority_nonkeywords
            elif type in classified_nonkeywords:
                borrowed_dict = classified_nonkeywords

            borrowed_types[type] = (0, len(borrowed_dict[type] - 1))
            options.extend(borrowed_dict[type])

        if len(borrowed_types) == 1 and len(next(iter(borrowed_dict.values()))) == 1:
            type, non_keywords = borrowed_dict.popitem()

            return non_keywords[0]

        answer = self._handle_options(options, options_message = f"What is, {description}")
        print("-----------------------------")
        if answer == -1:
            return None

        for type, interval in borrowed_types.items():
            if interval[0] <= answer <= interval[1]:
                borrowed_dict[type].remove(options[answer])
                if not borrowed_dict[type]:
                    del borrowed_dict[type]
                
                break
        else:
            raise RuntimeError("Could not parse the answer")

        return options[answer]
    
    def _extract_arguments_typemapping(self, action: str, argument_indices: list[int], classified_nonkeywords: dict, classified_priority_nonkeywords: dict, throw_if_not_found: bool = False) -> tuple[list[tuple], list[int]]:
        # Tries to map non-any types first, then maps any type

        arguments = []
        any_type_indices = []
        for argument_index in argument_indices:
            try:
                type = self._wrapper.get_argument_type(action, argument_index)
            except Exception as e:
                raise RuntimeError(f"Error fetching argument type for action '{action}' and index '{argument_index}': {e}")

            if type == "any":
                any_type_indices.append(len(arguments))
                arguments.append(None)
            else:
                try:
                    argument = self._pop_nonkeyword(type, classified_nonkeywords, classified_priority_nonkeywords, throw_if_not_found)
                except Exception:
                    raise SyntaxError(f"Could not find valid value for argument '{self._wrapper.get_argument_description(action, argument_index)}'")

                arguments.append((argument_index, argument))

        for argument_index in any_type_indices:
            try:
                argument = self._pop_nonkeyword(type, classified_nonkeywords, classified_priority_nonkeywords, throw_if_not_found)
            except Exception:
                raise SyntaxError(f"Could not find valid value for argument '{argument_index}'")
            
            if argument:
                arguments[argument_index] = (argument_index, argument)

        unassigned_indices = []
        assigned_indicies = []
        for argument_index, argument in enumerate(arguments):
            if argument:
                assigned_indicies.append(argument)
            else:
                unassigned_indices.append(argument_index)

        return assigned_indicies, unassigned_indices
    
    def _extract_argumentgroup_options(self, action: str, argument_indices: list[int], non_keywords: set[tuple[str, bool]], throw_if_exceed_count: int = float('inf')) -> list[tuple[int, str]]:
        # Returns all possibilites between each argument index and non keyword.
        # If possibilities exceed by 'throw_if_exceed_count' will throw an error.

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
        # Checks if action has warning flag set to true, and asks user for permission to run.

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
        # Predicts action using frequency of action keywords.
        # If confidence is less then probability cutoff, will return None.

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

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

    def predict_action_classification(self, action_keywords: list[str], max_possibilities: int, probability_cutoff: float = 0.85) -> tuple[str | None, bool]:
        # Predicts action with classification using action keywords.
        # If confidence is less then probability cutoff, asks user for clarification.
        # Trains classifier using user input.

        try:
            actions = self._wrapper.predict_top_actions(action_keywords, max_possibilities, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error extracting top actions: {e}")
        
        if actions[0][1] >= probability_cutoff:
            return actions[0][0], False
        
        try:
            answer = self._handle_options(actions, options_message = "What do you want to do?", key = lambda action: self._wrapper.get_action_description(action[0]))
            print("-----------------------------")
        except Exception as e:
            raise RuntimeError(f"Error fetching answer: {e}")
        
        if answer == -1:
            return None, True

        try:
            self._wrapper.train_action_pipeline(action_keywords, actions[answer][0])
        except Exception as e:
            print("Warning: Unable to train parser:", e)
        
        return actions[answer][0], False
    
    def predict_argument_frequency(self, action: str, argument_keywords: list[str], probability_cutoff: float = 0.85) -> int | None:
        # Predicts argument using frequency of argument keywords.
        # If confidence is less then probability cutoff, will return None.

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")
        
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
        
        return max_frequency_argument[0]
    
    def predict_argument_classification(self, action: str, argument_keywords: list[str], probability_cutoff: float = 0.85) -> str | None:
        # Predicts argument with classification using argument keywords.
        # If confidence is less then probability cutoff, returns None.

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")
        
        try:
            argument_index = self._wrapper.predict_argument_index(action, argument_keywords, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error extracting top arguments: {e}")
        
        return argument_index

    def predict_argument_nonkeyword_classification(self, action: str, argument_group: tuple[list[str], set[tuple]], max_possibilites: int, probability_cutoff: float = 0.85) -> tuple[int, str, bool] | tuple[int, None, bool] | tuple[None, None, bool]:
        # Predicts argument and non keyword with classification using argument group.
        # If confidence of argument is less then probability cutoff, asks user for clarification and trains the classifier.
        # Asks user for clarification is multiple non keywords are suitable candidates for the argument.

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        argument_keywords = argument_group[0]

        try:
            arguments = self._wrapper.predict_top_arguments_indices(action, argument_keywords, max_possibilites, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error extracting top arguments: {e}")
        
        if arguments[0][1] >= probability_cutoff:
            return argument_index, None, False
        else:
            indices = [arg[0] for arg in arguments]
            argument_index, non_keyword = self._handle_argument_group_options(action, indices, argument_group, max_possibilites, True)
        
        if argument_index is None:
            return None, None, True
        return argument_index, non_keyword, False
            
    def extract_tokens(self, query: str) -> list[tuple[str, bool]]:
        # Extracts tokens from query.
        # Marks each token if it is quoted or not.

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
    
    def extract_action_groups(self, tokens: list[tuple[str, bool]], probability_cutoff: float) -> tuple[list[str], list[list[tuple]]]:
        # Extracts a list of action keywords and, tokens divided by each action keyword encountered.
        # If a token is quoted then is added to action group.
        # If a token is an action keyword (predicted using fuzzy matching) is added to action keywords.
        # If a token belongs is a stop word and is not quoted, it is discarded.

        action_keywords = []

        action_groups = []
        non_keywords = []

        for token, quoted in tokens:
            if quoted:
                non_keywords.append((token, quoted))
                continue
            
            try:
                action_keyword = self._wrapper.match_action_keyword(token.lower(), probability_cutoff)
            except Exception as e:
                raise RuntimeError(f"Error matching action keyword: {e}")

            if action_keyword:
                action_keywords.append(action_keyword)

                if non_keywords:
                    action_groups.append(non_keywords)
                    non_keywords = []
            else:
                non_keywords.append((token, quoted))

        if non_keywords:
            action_groups.append(non_keywords)

        return action_keywords, action_groups
    
    def extract_argument_groups(self, action: str, action_groups: list[list[tuple]], probability_cutoff: float) -> tuple[list[tuple], list[str]]:
        # Extracts a list of argument keywords and their related non keywords, and a list of unrelated non keywords.
        # If a token is an argument keyword (predicted using fuzzy matching) add it to argument keywords, else to non keywords.
        # If a group has no argument keywords then add the non keywords to blind non keywords.

        argument_groups = []
        blind_non_keywords = []

        for group in action_groups:
            argument_keyword: str | None = None
            non_keywords = set()

            for token, quoted in group:
                if quoted:
                    blind_non_keywords.append((token, quoted))
                    continue
                
                try:
                    keyword = self._wrapper.match_argument_keyword(action, token.lower(), probability_cutoff)
                except Exception as e:
                    raise RuntimeError(f"Error matching argument keyword: {e}")

                if keyword:
                    if argument_keyword:
                        argument_groups.append(([argument_keyword], non_keywords))
                    else:
                        blind_non_keywords.extend(non_keywords)

                    argument_keyword = keyword
                    non_keywords = set()

                    continue

                try:
                    if not self._wrapper.is_stop_word(token.lower(), probability_cutoff):
                        non_keywords.add((token, quoted))
                except Exception as e:
                    raise RuntimeError(f"Error checking stop word: {e}")

            if argument_keyword:
                argument_groups.append(([argument_keyword], non_keywords))
            else:
                blind_non_keywords.extend(non_keywords)
            
        return argument_groups, blind_non_keywords

    def extract_classified_nonkeywords(self, non_keywords: list[tuple[str, bool]]) -> tuple[dict, dict]:
        # Classifies non keywords using types.
        # If a non keyword is quoted gets added to classified priority non keywords, else to classified non keywords.

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
        # Extracts information which required and optional arguments have been assigned, which required and optional arguments haven't been assigned.

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
    
    def extract_arguments_typemapping(self, action: str, required_indices: list[int], optional_indices: list[int], classified_nonkeywords: dict, classified_priority_nonkeywords: dict) -> tuple[list[tuple], list[tuple], list[int], list[int]]:
        # Extracts arguments using type mapping (assigning non keyword to arguments based on type)
        # If non keywords are not sufficient for required arguments, will throw error.
        # If non keywords are not available or not sufficient for optional arguments, will ignore.
        # Returns assigned required and optional arguments, unassigned required and optional arguments.

        required_arguments, unassigned_required_indices = self._extract_arguments_typemapping(action, required_indices, classified_nonkeywords, classified_priority_nonkeywords, throw_if_not_found = True)
        optional_arguments, unassigned_optional_indices = self._extract_arguments_typemapping(action, optional_indices, classified_nonkeywords, classified_priority_nonkeywords)

        return required_arguments, optional_arguments, unassigned_required_indices, unassigned_optional_indices
    
    def extract_arguments_questions_nonkeywords(self, action: str, argument_indices: list[int], non_keywords: list[tuple[str, bool]]) -> list[tuple] | None:
        classified_nonkeywords, classified_priority_nonkeywords = self.extract_classified_nonkeywords(non_keywords)

        return self.extract_arguments_questions_classified_nonkeywords(action, argument_indices, classified_nonkeywords, classified_priority_nonkeywords)
    
    def extract_arguments_questions_classified_nonkeywords(self, action: str, argument_indices: list[int], classified_nonkeywords: dict, classified_priority_nonkeywords: dict) -> list[tuple] | None:
        # Extracts arguments by asking questions to user.

        arguments = []

        for idx in argument_indices:
            try:
                type = self._wrapper.get_argument_type(action, idx)
                description = self._wrapper.get_argument_description(action, idx)
            except Exception as e:
                raise RuntimeError(f"Error fetching argument data: {e}")
            
            non_keyword = self._pop_nonkeyword_question(type, description, classified_nonkeywords, classified_priority_nonkeywords)
            if not non_keyword:
                return None
            
            arguments.append((idx, non_keyword))

        return arguments
    
    def extract_nonkeyword_typemapping(self, action: str, argument_index: int, non_keywords: list[tuple[str, bool]]) -> str | None:
        classified_nonkeyword, classified_priority_nonkeywords = self.extract_classified_nonkeywords(non_keywords)

        assigned_arguments, _ = self._extract_arguments_typemapping(action, [argument_index], classified_nonkeyword, classified_priority_nonkeywords)
        
        if assigned_arguments:
            return assigned_arguments[0][1]
        return None

    def get_arguments_count(self, action: str) -> int:
        # Fetched no.of arguments available for an action.

        return self._wrapper.get_arguments_count(action)