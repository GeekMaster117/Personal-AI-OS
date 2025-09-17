import os

import joblib

from collections import Counter
from collections.abc import KeysView

from rapidfuzz import process, fuzz

from numpy import ndarray
from typing import Any

from Include.filter.stop_words import ENGLISH_STOP_WORDS
import settings

class ParserWrapper:
    def __init__(self):
        try:
            self._commands, self._keyword_action_map, self._pipeline, self._vectorizer, self._classifier = self._load_parser_model()
        except Exception as e:
            raise RuntimeError(f"Error fetching parser model: {e}")

    def _save_parser_model(self, commands: dict, keyword_action_map: dict, pipeline: Any, vectorizer: Any, classifier: Any) -> None:
        model = {
            "commands": commands,
            "keyword_action_map": keyword_action_map,
            "action_pipeline": pipeline,
            "action_vectorizer": vectorizer,
            "action_classifier": classifier
        }
        with open(settings.parser_model_dir, "wb") as file:
            joblib.dump(model, file)
    
    def _load_parser_model(self) -> tuple[dict, dict, Any, Any, Any]:
        if not os.path.exists(settings.parser_model_dir):
            raise FileNotFoundError("Parser model file not found")

        try:
            with open(settings.parser_model_dir, "rb") as file:
                model: dict = joblib.load(file)
        except Exception as e:
            raise RuntimeError(f"Error loading parser model file: {e}")

        try:
            commands: dict = model["commands"]
            keyword_action_map: dict = model["keyword_action_map"]
            pipeline: Any = model["action_pipeline"]
            vectorizer: Any = model["action_vectorizer"]
            classifier: Any = model["action_classifier"]
        except Exception as e:
            raise RuntimeError(f"Error extracting parser model components: {e}")

        return commands, keyword_action_map, pipeline, vectorizer, classifier
    
    def train(self, keywords: list[str], action: str) -> None:
        X = self._vectorizer.transform([" ".join(keywords)])
        self._classifier.partial_fit(X, [action], classes = list(self._commands.keys()))

        self._save_parser_model(self._commands, self._keyword_action_map, self._pipeline, self._vectorizer, self._classifier)
    
    def predict_top_actions(self, keywords: list[str], top_actions_count: int) -> list[tuple]:
        probabilities: ndarray = self._pipeline.predict_proba([" ".join(keywords)])[0]

        classes = [(str(self._pipeline.classes_[idx]), float(probability)) for idx, probability in enumerate(probabilities)]
        top_actions = sorted(classes, reverse=True, key = lambda x: x[1])[:min(len(probabilities), top_actions_count)]

        return top_actions
    
    def match_action_keyword(self, token: str, probability_cutoff: float) -> str | None:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")
        
        keyword = process.extractOne(token, self.get_action_keywords(), scorer=fuzz.ratio, score_cutoff=probability_cutoff * 100)
        if keyword:
            return keyword[0]
        return None

    def match_argument_keyword(self, action: str, token: str, probability_cutoff: float) -> str | None:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        keyword = process.extractOne(token, self.get_argument_keywords(action), scorer=fuzz.ratio, score_cutoff=probability_cutoff * 100)
        if keyword:
            return keyword[0]
        return None

    def is_stop_word(self, token: str, probability_cutoff: float) -> bool:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")
        
        return process.extractOne(token, ENGLISH_STOP_WORDS, scorer=fuzz.ratio, score_cutoff=probability_cutoff * 100) is not None

    def has_args(self, action: str) -> bool:
        return "args" in self._commands[action] and len(self._commands[action]["args"]) > 0
    
    def has_action_warning(self, action: str) -> bool:
        return "warning" in self._commands[action] and self._commands[action]["warning"]
    
    def get_action_description(self, action: str) -> str:
        if "description" not in self._commands[action]:
            raise ValueError(f"Action '{action}' has no description")

        return self._commands[action]["description"]

    def get_action_keywords(self) -> KeysView[str]:
        return self._keyword_action_map.keys()
    
    def get_argument_keywords(self, action: str) -> KeysView[str]:
        return self._commands[action]["keyword_argument_map"].keys()
    
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