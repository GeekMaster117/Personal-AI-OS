import os
import subprocess

import json
import joblib
import hashlib

from numpy import ndarray
from collections import defaultdict, Counter

import shlex
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, make_pipeline

class Parser:
    _commands_path = "commands.json"
    _parser_state_path = "parser_state.bin"

    def __init__(self):
        try:
            with open(Parser._commands_path, 'r') as file:
                self._commands: dict = json.load(file)
        except Exception as e:
            raise RuntimeError(f"Error fetching from commands: {e}")
        try:
            self._commands_hash: str = self._get_commands_hash(self._commands)
        except Exception as e:
            raise RuntimeError(f"Error fetching commands hash: {e}")

        try:
            self.keyword_action_map, self._pipeline, self._vectorizer, self._classifier = self._load_parser_state(self._commands_hash)
        except Exception as e:
            raise RuntimeError(f"Error fetching parser state: {e}")
        
    def _save_parser_state(self, current_commands_hash: str, keyword_action_map: dict, pipeline: Pipeline, vectorizer: CountVectorizer, classifier: SGDClassifier) -> None:
        state = {
            "commands_hash": current_commands_hash,
            "keyword_action_map": keyword_action_map,
            "pipeline": pipeline,
            "vectorizer": vectorizer,
            "classifier": classifier
        }
        with open(Parser._parser_state_path, "wb") as file:
            joblib.dump(state, file)
    
    def _load_parser_state(self, current_commands_hash: str) -> tuple[dict, Pipeline, CountVectorizer, SGDClassifier]:
        try:
            if not os.path.exists(Parser._parser_state_path):
                raise FileNotFoundError("Parser state file not found")

            with open(Parser._parser_state_path, "rb") as file:
                state: dict = joblib.load(file)

            load_commands_hash: str = state["commands_hash"]
            keyword_action_map: dict = state["keyword_action_map"]
            pipeline: Pipeline = state["pipeline"]
            vectorizer: CountVectorizer = state["vectorizer"]
            classifier: SGDClassifier = state["classifier"]
        except Exception as e:
            load_commands_hash = None
            keyword_action_map = None
            pipeline = None
            vectorizer = None
            classifier = None

        if any(v is None for v in [load_commands_hash, keyword_action_map, pipeline, vectorizer, classifier]) or current_commands_hash != load_commands_hash:
            try:
                vectorizer = CountVectorizer()
                classifier = SGDClassifier(loss="log_loss")
                pipeline = make_pipeline(vectorizer, classifier)
            except Exception as e:
                raise RuntimeError(f"Error initialising pipeline: {e}")
            
            try:
                keyword_action_map: dict = defaultdict(set)
                keywords, actions = [], []

                for action, structure in self._commands.items():
                    keywords.append(" ".join(structure["keywords"]))
                    actions.append(action)

                    for keyword in structure["keywords"]:
                        keyword_action_map[keyword].add(action)

                pipeline.fit(keywords, actions)
            except Exception as e:
                raise RuntimeError(f"Error mapping commands keywords to actions: {e}")

            self._save_parser_state(current_commands_hash, keyword_action_map, pipeline, vectorizer, classifier)

        return keyword_action_map, pipeline, vectorizer, classifier
    
    def _train(self, keywords: list[str], action: str) -> None:
        X = self._vectorizer.transform([" ".join(keywords)])
        self._classifier.partial_fit(X, [action], classes = list(self._commands.keys()))

        self._save_parser_state(self._commands_hash, self.keyword_action_map, self._pipeline, self._vectorizer, self._classifier)

    def _handle_options(self, options: list[str], options_message = "Select an option:", select_message = "Enter an answer", key = lambda x: x) -> int:
        print(options_message)
        for i, option in enumerate(options, start=1):
            print(f"{i}. {key(option)}")
        print(f"{len(options) + 1}. Skip request")

        choice = input(f"{select_message} (1-{len(options) + 1}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice) - 1
        
        return -1
    
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
        
        answer = self._handle_options(non_keywords, options_message = f"What is, {description}")
        if answer == -1:
            return None
        
        for t, range in borrowed_types.items():
            if range[0] <= answer <= range[1]:
                return pop(non_keywords, borrowed_dict, t, answer)
            
        raise RuntimeError("Unable to map non keyword to type")

    def _get_commands_hash(self, commands: dict) -> str:
        data_str = json.dumps(commands, sort_keys=True)

        return hashlib.md5(data_str.encode('utf-8')).hexdigest()
    
    def _get_required_arguments(self, action: str) -> tuple[list[int], Counter]:
        if "args" not in self._commands[action]:
            return dict(), Counter()
        
        all_arguments: list[dict] = self._commands[action]["args"]
        
        indices, needed = [], Counter()
        for i in range(len(all_arguments)):
            if all_arguments[i]["required"]:
                indices.append(i)
                needed[all_arguments[i]["type"]] += 1
        
        return indices, needed
    
    def _get_optional_arguments(self, action: str) -> tuple[list[int]]:
        if "args" not in self._commands[action]:
            return dict(), Counter()
        
        all_arguments: list[dict] = self._commands[action]["args"]
        
        indices = []
        for i in range(len(all_arguments)):
            if not all_arguments[i]["required"]:
                indices.append(i)
        
        return indices
    
    def _extract_tokens(self, query: str) -> list[tuple[str, bool]]:
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
    
    def _extract_keywords_nonkeywords(self, tokens: list[tuple[str, bool]], probability_cutoff: float) -> tuple[list[str], list[tuple[str, bool]]]:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        keywords = []
        non_keywords = []
        for token, quoted in tokens:
            if quoted:
                non_keywords.append((token, quoted))
                continue

            keyword = process.extractOne(token, self.keyword_action_map.keys(), scorer=fuzz.ratio, score_cutoff=probability_cutoff * 100)
            if keyword:
                keywords.append(keyword[0])
            else:
                non_keywords.append((token, quoted))
        
        return keywords, non_keywords
    
    def _extract_classified_non_keywords(self, non_keywords: list[tuple[str, bool]]) -> tuple[dict, dict]:
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

    def _extract_actions_normalised(self, keywords: list[str]) -> dict:
        keywords_counter = Counter(keywords)

        action_counter = Counter()
        for keyword, count in keywords_counter.items():
            actions = self.keyword_action_map[keyword]
            for action in actions:
                action_counter[action] += count
        
        actions_normalised = dict()
        for action, keyword_count in action_counter.items():
            actions_normalised[action] = keyword_count / action_counter.total()
        
        return actions_normalised

    def _extract_action_frequency(self, actions_normalised: dict, probability_cutoff: float = 0.85) -> str | None:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        if not actions_normalised:
            return None

        action_normalised = max(actions_normalised.items(), key = lambda action_normalised: action_normalised[1])
        if action_normalised[1] < probability_cutoff:
            return None
        
        return action_normalised[0]

    def _extract_actions_classification(self, keywords: list[str], top_actions_count: int) -> list[tuple]:
        probabilities: ndarray = self._pipeline.predict_proba([" ".join(keywords)])[0]

        classes = [(str(self._pipeline.classes_[idx]), float(probability)) for idx, probability in enumerate(probabilities)]
        top_actions = sorted(classes, reverse=True, key = lambda x: x[1])[:min(len(probabilities), top_actions_count)]

        return top_actions
    
    def _extract_required_arguments(self, action: str, required_indices: list[int], classified_non_keywords: dict, classified_priority_non_keywords: dict) -> list[str]:
        required_arguments = []
        any_type_argument_indices = []
        for idx in required_indices:
            type = self._commands[action]["args"][idx]["type"]
            description = self._commands[action]["args"][idx]["description"]

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
            description = self._commands[action]["args"][idx]["description"]

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
            type = self._commands[action]["args"][idx]["type"]
            description = self._commands[action]["args"][idx]["description"]

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
            description = self._commands[action]["args"][idx]["description"]

            try:
                non_keyword = self._pop_non_keyword(type, description, classified_non_keywords, classified_priority_non_keywords)
            except Exception as e:
                raise RuntimeError(f"Error fetching non keywords: {e}")
            
            if not non_keyword:
                break

            optional_arguments[idx] = non_keyword

        return optional_arguments
    
    def _extract_arguments(self, action: str, classified_non_keywords: dict, classified_priority_non_keywords: dict) -> list[str]:
        if 'args' not in self._commands[action]:
            return []

        try:
            required_indices, required_needed = self._get_required_arguments(action)
            optional_indices = self._get_optional_arguments(action)
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

    def extract_action_arguments(self, query: str, probability_cutoff: float = 0.85) -> tuple[str | list[str]]:
        try:
            tokens: list[tuple[str | bool]] = self._extract_tokens(query)
        except Exception as e:
            raise SyntaxError(f"Syntax Error: {e}")

        try:
            keywords, non_keywords = self._extract_keywords_nonkeywords(tokens, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error extracting keywords: {e}")
        
        if not keywords:
            raise SyntaxError("No keywords found")
        
        actions_normalised: dict = self._extract_actions_normalised(keywords)
        
        try:
            action = self._extract_action_frequency(actions_normalised, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error extracting action: {e}")
        
        if not action:
            try:
                actions = self._extract_actions_classification(keywords, 5)
            except Exception as e:
                raise RuntimeError(f"Error extracting top actions: {e}")
            
            if actions[0][1] >= 0.85:
                action = actions[0][0]
            else:
                try:
                    answer = self._handle_options(actions, options_message = "What do you want to do?", key = lambda action: self._commands[action[0]]["description"])
                    print("-----------------------------")
                except Exception as e:
                    raise RuntimeError(f"Error fetching answer: {e}")
                
                if answer == -1:
                    return "", []

                self._train(keywords, actions[answer][0])
                action = actions[answer][0]

        classified_non_keywords, classified_priority_non_keywords = self._extract_classified_non_keywords(non_keywords)

        try:
            arguments: list[str] = self._extract_arguments(action, classified_non_keywords, classified_priority_non_keywords)
        except Exception as e:
            raise SyntaxError(f"Error extracting arguments: {e}")
        
        return action, arguments
    
    def execute_action(self, action : str, arguments: list[str]) -> None:
        if self._commands[action]["warning"] == True:
            answer = input(f"Do you want to, {self._commands[action]["description"]} (Y/N): ").lower()
            if answer != 'y':
                print("Skipping request...")
                return
            
        command = " ".join([action] + arguments)
            
        subprocess.run(command, shell=True)
        print("Command Executed: " + command)

try:
    parser: Parser = Parser()
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