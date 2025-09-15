import os

import json
import joblib
import hashlib

from collections import defaultdict, Counter
from collections.abc import KeysView

from numpy import ndarray
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, make_pipeline

import settings

class ParserWrapper:
    def __init__(self):
        try:
            with open(settings.commands_dir, 'r') as file:
                self._commands: dict = json.load(file)
        except Exception as e:
            raise RuntimeError(f"Error fetching from commands: {e}")
        try:
            self._commands_hash: str = self._get_commands_hash(self._commands)
        except Exception as e:
            raise RuntimeError(f"Error fetching commands hash: {e}")

        try:
            self._keyword_action_map, self._pipeline, self._vectorizer, self._classifier = self._load_parser_state(self._commands_hash)
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
        with open(settings.parser_state_dir, "wb") as file:
            joblib.dump(state, file)
    
    def _load_parser_state(self, current_commands_hash: str) -> tuple[dict, Pipeline, CountVectorizer, SGDClassifier]:
        try:
            if not os.path.exists(settings.parser_state_dir):
                raise FileNotFoundError("Parser state file not found")

            with open(settings.parser_state_dir, "rb") as file:
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

        self._save_parser_state(self._commands_hash, self._keyword_action_map, self._pipeline, self._vectorizer, self._classifier)

    def _get_commands_hash(self, commands: dict) -> str:
        data_str = json.dumps(commands, sort_keys=True)

        return hashlib.md5(data_str.encode('utf-8')).hexdigest()
    
    def predict_top_actions(self, keywords: list[str], top_actions_count: int) -> list[tuple]:
        probabilities: ndarray = self._pipeline.predict_proba([" ".join(keywords)])[0]

        classes = [(str(self._pipeline.classes_[idx]), float(probability)) for idx, probability in enumerate(probabilities)]
        top_actions = sorted(classes, reverse=True, key = lambda x: x[1])[:min(len(probabilities), top_actions_count)]

        return top_actions
    
    def has_args(self, action: str) -> bool:
        return "args" in self._commands[action] and len(self._commands[action]["args"]) > 0
    
    def has_action_warning(self, action: str) -> bool:
        return "warning" in self._commands[action] and self._commands[action]["warning"]
    
    def get_action_description(self, action: str) -> str:
        if "description" not in self._commands[action]:
            raise ValueError(f"Action '{action}' has no description")

        return self._commands[action]["description"]

    def get_all_keywords(self) -> KeysView[str]:
        return self._keyword_action_map.keys()
    
    def get_actions_for_keyword(self, keyword: str) -> set[str]:
        return self._keyword_action_map.get(keyword, set())

    def get_required_arguments(self, action: str) -> tuple[list[int], Counter]:
        if "args" not in self._commands[action]:
            return [], Counter()
        
        all_arguments: list[dict] = self._commands[action]["args"]
        
        indices, needed = [], Counter()
        for i in range(len(all_arguments)):
            if all_arguments[i]["required"]:
                indices.append(i)
                needed[all_arguments[i]["type"]] += 1
        
        return indices, needed
    
    def get_optional_arguments(self, action: str) -> tuple[list[int]]:
        if "args" not in self._commands[action]:
            return []
        
        all_arguments: list[dict] = self._commands[action]["args"]
        
        indices = []
        for i in range(len(all_arguments)):
            if not all_arguments[i]["required"]:
                indices.append(i)
        
        return indices
    
    def get_action_args_type(self, action: str, idx: int) -> str:
        if "args" not in self._commands[action]:
            raise ValueError(f"Action '{action}' has no arguments")
        
        all_arguments: list[dict] = self._commands[action]["args"]
        
        if idx < 0 or idx >= len(all_arguments):
            raise IndexError(f"Argument index {idx} out of range for action '{action}'")

        return all_arguments[idx]["type"]
    
    def get_action_args_description(self, action: str, idx: int) -> str:
        if "args" not in self._commands[action]:
            raise ValueError(f"Action '{action}' has no arguments")
        
        all_arguments: list[dict] = self._commands[action]["args"]
        
        if idx < 0 or idx >= len(all_arguments):
            raise IndexError(f"Argument index {idx} out of range for action '{action}'")

        return all_arguments[idx]["description"]