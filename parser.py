import os

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
    commands_path: str = "commands.json"
    commands_hash_path: str = "commands_hash.txt"
    vectorizer_path: str = "vectorizer.bin"
    classifier_path: str = "classifier.bin"

    def __init__(self):
        try:
            with open(Parser.commands_path, 'r') as file:
                self.commands: dict = json.load(file)
        except Exception as e:
            raise RuntimeError(f"Error fetching from commands: {e}")

        try:
            self.keyword_action_map: dict = defaultdict(set)
            for action, structure in self.commands.items():
                for keyword in structure["keywords"]:
                    self.keyword_action_map[keyword].add(action)
        except Exception as e:
            raise RuntimeError(f"Error mapping commands keywords to actions: {e}")

        try:
            model = self._load_model()
            self._pipeline: Pipeline = model[0]
            self._vectorizer: CountVectorizer = model[1]
            self._classifier: SGDClassifier = model[2]
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def _get_commands_hash(self) -> str:
        with open(Parser.commands_path, 'r') as file:
            data = json.load(file)
        data_str = json.dumps(data, sort_keys=True)
        
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()

    def _save_commands_hash(self) -> None:
        with open(Parser.commands_hash_path, "w") as file:
            file.write(self._get_commands_hash())

    def _load_commands_hash(self) -> str | None:
        if not os.path.exists(Parser.commands_hash_path):
            return None
        with open(Parser.commands_hash_path, "r") as file:
            return file.read().strip()

    def _check_commands_hash(self) -> bool:
        try:
            saved_hash = self._load_commands_hash()
            current_hash = self._get_commands_hash()
        except:
            return False
        
        if not saved_hash or current_hash != saved_hash:
            return False
        return True
    
    def _save_vectorizer(self, vectorizer: CountVectorizer) -> None:
        joblib.dump(vectorizer, Parser.vectorizer_path)

    def _save_classifier(self, classifier: SGDClassifier) -> None:
        joblib.dump(classifier, Parser.classifier_path)
    
    def _save_model(self, vectorizer: CountVectorizer, classifier: SGDClassifier) -> None:
        self._save_vectorizer(vectorizer)
        self._save_classifier(classifier)

    def _load_vectorizer(self) -> CountVectorizer | None:
        if not os.path.exists(Parser.vectorizer_path):
            return None
        return joblib.load(Parser.vectorizer_path)
    
    def _load_classifier(self) -> SGDClassifier | None:
        if not os.path.exists(Parser.classifier_path):
            return None
        return joblib.load(Parser.classifier_path)
    
    def _load_model(self) -> tuple:
        try:
            vectorizer = self._load_vectorizer()
            classifier = self._load_classifier()
        except:
            vectorizer, classifier = None, None

        if not vectorizer or not classifier or not self._check_commands_hash():
            try:
                vectorizer = CountVectorizer()
                classifier = SGDClassifier(loss="log_loss")
                pipeline = self._init_train(vectorizer, classifier)
            except Exception as e:
                raise RuntimeError(f"Error initialising pipeline: {e}")
            
            try:
                self._save_model(vectorizer, classifier)
                self._save_commands_hash()
            except Exception as e:
                raise RuntimeError(f"Error saving: {e}")

            return pipeline, vectorizer, classifier
        
        try:
            pipeline = make_pipeline(vectorizer, classifier)
        except Exception as e:
            raise RuntimeError(f"Error making pipeline: {e}")
        
        return pipeline, vectorizer, classifier

    def _init_train(self, vectorizer: CountVectorizer, classifier: SGDClassifier) -> Pipeline:
        pipeline = make_pipeline(vectorizer, classifier)

        keywords, labels = [], []
        for action, structure in self.commands.items():
            keywords.append(" ".join(structure["keywords"]))
            labels.append(action)

        pipeline.fit(keywords, labels)

        return pipeline
    
    def extract_tokens(self, query: str) -> list[tuple[str | bool]]:
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
    
    def extract_keywords_nkeywords(self, tokens: list[tuple[str | bool]], probability_cutoff: float) -> tuple[list[str]]:
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
    
    def extract_args(self, tokens: list[tuple[str | bool]]) -> dict:
        args_types_map = defaultdict(set)
        for token, quoted in tokens:
            if not quoted and token in ENGLISH_STOP_WORDS:
                continue

            if token.isdecimal():
                args_types_map["int"].add(token)
            elif token.isalpha():
                args_types_map["str"].add(token)
            else:
                args_types_map["any"].add(token)

        return args_types_map

    def extract_actions_normalised(self, keywords: list[str]) -> dict:
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

    def extract_action_frequency(self, actions_normalised: dict, probability_cutoff: float = 0.85) -> str | None:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        if not actions_normalised:
            return None

        action_normalised = max(actions_normalised.items(), key = lambda action_normalised: action_normalised[1])
        if action_normalised[1] < probability_cutoff:
            return None
        
        return action_normalised[0]

    def extract_actions_classification(self, keywords: list[str], top_actions_count: int) -> list[tuple]:
        probabilities: ndarray = self._pipeline.predict_proba([" ".join(keywords)])[0]

        classes = [(str(self._pipeline.classes_[idx]), float(probability)) for idx, probability in enumerate(probabilities)]
        top_actions = sorted(classes, reverse=True, key = lambda x: x[1])[:min(len(probabilities), top_actions_count)]

        return top_actions
    
    def train(self, keywords: list[str], action: str) -> None:
        X = self._vectorizer.transform([" ".join(keywords)])
        self._classifier.partial_fit(X, [action], classes = list(self.commands.keys()))

        self._save_model(self._vectorizer, self._classifier)

    def handle_options(self, options: list[str], options_message = "Select an option:", select_message = "Enter an answer", key = lambda x: x) -> int:
        while True:
            print(options_message)
            for i, option in enumerate(options, start=1):
                print(f"{i}. {key(option)}")

            choice = input(f"{select_message} (1-{len(options)}): ")
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return int(choice) - 1
            print("-----------------------------")
            
            print(f"Invalid option. Please enter a valid option between 1-{len(options)}.")
            print("-----------------------------")

    def get_args_needed(self, action: str) -> Counter:
        if "args" not in self.commands[action]:
            return Counter()
        
        args_needed = Counter()
        for i in range(len(self.commands[action]["args"])):
            args_needed[self.commands[action]["args"][i]["type"]] += 1

        return args_needed
    
    def get_args_required(self, action: str) -> Counter:
        if "args" not in self.commands[action]:
            return Counter()
        
        args_needed = Counter()
        for i in range(len(self.commands[action]["args"])):
            if self.commands[action]["args"][i]["required"]:
                args_needed[self.commands[action]["args"][i]["type"]] += 1

        return args_needed

    def get_action_args(self, query: str, probability_cutoff: float = 0.85) -> tuple[str | list[str]]:
        try:
            tokens: list[tuple[str | bool]] = self.extract_tokens(query)
        except Exception as e:
            raise SyntaxError(f"Syntax Error: {e}")

        try:
            keywords, non_keywords = self.extract_keywords_nkeywords(tokens, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error extracting keywords: {e}")
        
        if not keywords:
            raise SyntaxError("No keywords found")
        
        actions_normalised: dict = self.extract_actions_normalised(keywords)
        
        try:
            action = self.extract_action_frequency(actions_normalised, probability_cutoff)
        except Exception as e:
            raise RuntimeError(f"Error extracting action: {e}")
        
        if not action:
            try:
                actions = self.extract_actions_classification(keywords, 5)
            except Exception as e:
                raise RuntimeError(f"Error extracting top actions: {e}")
            
            if actions[0][1] >= 0.85:
                action = actions[0][0]
            else:
                try:
                    answer = self.handle_options(actions, options_message = "What do you want to do?", key = lambda action: self.commands[action[0]]["description"])
                    print("-----------------------------")
                except Exception as e:
                    raise RuntimeError(f"Error fetching answer: {e}")

                self.train(keywords, actions[answer][0])
                action = actions[answer][0]

        try:
            args_needed: Counter = self.get_args_needed(action)
        except Exception as e:
            raise RuntimeError(f"Error fetching arguments needed for action '{action}': {e}")
        
        try:
            args_required: Counter = self.get_args_required(action)
        except Exception as e:
            raise RuntimeError(f"Error fetching arguments required for action '{action}': {e}")
        
        if not args_needed:
            return action, []
        
        args: dict = self.extract_args(non_keywords)
        args_count = sum(len(args_set) for args_set in args.values())
        
        if args_count < args_required.total():
            arguments_required = ""
            for type, needed in args_required.items():
                arguments_required += f"{type}: {needed}\n"
            arguments_found = ""
            for type, args_set in args.items():
                arguments_found += f"{type}: {len(args_set)}\n"
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
        
        return action, args
    
    def execute_action(self, action : str, args) -> None:
        if self.commands[action]["warning"] == True:
            answer = input(f"Do you want to, {self.commands[action]["description"]} (Y/N): ").lower()
            if answer != 'y':
                print("Skipping request...")
                return
            
        #Todo: Execute Command

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
        action, args = parser.get_action_args(query)
    except Exception as e:
        print(f"Error parsing query: {e}")
        print("-----------------------------")

        query = input(f"Enter request: ")
        print("-----------------------------")

        continue
    
    break

parser.execute_action(action, args)