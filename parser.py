import os

import json
import joblib
import hashlib

from numpy import ndarray
from collections import defaultdict, Counter

import shlex
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, make_pipeline

class Parser:
    commands_path: str = "commands.json"
    commands_hash_path: str = "commands_hash.txt"
    vectorizer_path: str = "vectorizer.bin"
    classifier_path: str = "classifier.bin"

    def __init__(self):
        with open(Parser.commands_path, 'r') as file:
            self.commands: dict = json.load(file)

        self.keyword_action_map: dict = defaultdict(set)
        for action, structure in self.commands.items():
            for keyword in structure["keywords"]:
                self.keyword_action_map[keyword].add(action)

        model = self._load_model()
        self._pipeline: Pipeline = model[0]
        self._vectorizer: CountVectorizer = model[1]
        self._classifier: SGDClassifier = model[2]

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
        saved_hash = self._load_commands_hash()
        if not saved_hash or self._get_commands_hash() != saved_hash:
            return False
        return True

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
            vectorizer = CountVectorizer()
            classifier = SGDClassifier(loss="log_loss")
            model = self._init_train(vectorizer, classifier), vectorizer, classifier
            
            self._save_model(vectorizer, classifier)
            self._save_commands_hash()

            return model
        return make_pipeline(vectorizer, classifier), vectorizer, classifier
    
    def _save_vectorizer(self, vectorizer: CountVectorizer) -> None:
        joblib.dump(vectorizer, Parser.vectorizer_path)

    def _save_classifier(self, classifier: SGDClassifier) -> None:
        joblib.dump(classifier, Parser.classifier_path)
    
    def _save_model(self, vectorizer: CountVectorizer, classifier: SGDClassifier) -> None:
        self._save_vectorizer(vectorizer)
        self._save_classifier(classifier)

    def _init_train(self, vectorizer: CountVectorizer, classifier: SGDClassifier) -> Pipeline:
        pipeline = make_pipeline(vectorizer, classifier)

        keywords, labels = [], []
        for action, structure in self.commands.items():
            keywords.append(" ".join(structure["keywords"]))
            labels.append(action)

        pipeline.fit(keywords, labels)

        return pipeline
    
    def extract_keywords_nkeywords(self, tokens: set[str], probability_cutoff: float) -> tuple[set[str]]:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        keywords = set()
        non_keywords = set()
        for token in tokens:
            keyword = process.extractOne(token, self.keyword_action_map.keys(), scorer=fuzz.ratio, score_cutoff=probability_cutoff * 100)
            if keyword:
                keywords.add(keyword[0])
            else:
                non_keywords.add(token)
        
        return keywords, non_keywords
    
    def extract_args_types(self, tokens: set[str]) -> dict:
        args_types_map = defaultdict(set)
        for arg in tokens:
            if arg.isdecimal():
                args_types_map["int"].add(arg)
            elif arg.isalpha():
                args_types_map["str"].add(arg)
            else:
                args_types_map["any"].add(arg)

        return args_types_map
    
    def extract_args_needed(self, action: str) -> Counter:
        if "args" not in self.commands[action]:
            return Counter()
        
        args_needed = Counter()
        for arg in self.commands[action]["args"]:
            args_needed[arg["type"]] += 1

        return args_needed
    
    def extract_args_required(self, action: str) -> Counter:
        if "args" not in self.commands[action]:
            return Counter()
        
        args_needed = Counter()
        for arg in self.commands[action]["args"]:
            if self.commands[action]["args"]["required"]:
                args_needed[arg["type"]] += 1

        return args_needed

    def extract_actions_normalised(self, keywords: set[str]) -> dict:
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

    def extract_actions_classification(self, keywords: set[str], top_actions_count: int) -> list[tuple]:
        probabilities: ndarray = self._pipeline.predict_proba([" ".join(keywords)])[0]

        classes = [(str(self._pipeline.classes_[idx]), float(probability)) for idx, probability in enumerate(probabilities)]
        top_actions = sorted(classes, reverse=True, key = lambda x: x[1])[:min(len(probabilities), top_actions_count)]

        return top_actions
    
    def train(self, keywords: set[str], action: str) -> None:
        print(" ".join(keywords))
        X = self._vectorizer.transform([" ".join(keywords)])
        self._classifier.partial_fit(X, [action], classes = list(self.commands.keys()))

        self._save_model(self._vectorizer, self._classifier)

    def get_action(self, query: str, probability_cutoff: float = 0.85) -> tuple[str | list[str]]:
        tokens: set[str] = set(shlex.split(query.lower()))
        keywords, non_keywords = self.extract_keywords_nkeywords(tokens, probability_cutoff)
        if not keywords:
            raise SyntaxError("Unable to parse the query")
        
        actions_normalised: dict = self.extract_actions_normalised(keywords)

        action = self.extract_action_frequency(actions_normalised)
        if not action:
            actions = self.extract_actions_classification(keywords, 5)
            print(actions)
            if actions[0][1] >= 0.85:
                action = actions[0][0]
            else:
                print("What do you want to do?")
                for i, action in enumerate(actions):
                    print(f"{i + 1}. {self.commands[action[0]]["description"]}")
                answer = int(input(f"Enter answer ({1}-{len(actions)}): "))

                self.train(keywords, actions[answer - 1][0])
                action = actions[answer - 1][0]

        args_needed: Counter = self.extract_args_needed(action)
        args_required: Counter = self.extract_args_required(action)
        if not args_needed:
            return action, []
        
        args_types: dict = self.extract_args_types(non_keywords)
        args_available = sum(len(args_datatype) for args_datatype in args_types.values())
        
        if args_available < args_needed.total():
            raise SyntaxError("Unable to parse the query")
    
    def execute_action(self, action : str, args: list[str]) -> None:
        if self.commands[action]["warning"] == True:
            answer = input(f"Do you want to, {self.commands[action]["description"]} (Y/N): ").lower()
            if answer != 'y':
                print("Skipping request...")
                return
            
        #Todo: Execute Command

parser: Parser = Parser()

query: str = input("Enter request: ")
action: str | None = None
while True:
    try:
        action, args = parser.get_action(query)
    except:
        query = input("Unable to parse. Enter again: ")
        continue
    
    break

parser.execute_action(action, args)