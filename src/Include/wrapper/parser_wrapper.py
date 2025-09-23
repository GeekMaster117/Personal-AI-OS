import os

import joblib

from collections.abc import KeysView

from rapidfuzz import process, fuzz

from numpy import ndarray
from typing import Any

from Include.filter.stop_words import ENGLISH_STOP_WORDS
import settings

class ParserWrapper:
    def __init__(self):
        try:
            self._commands, self._keyword_action_map, self._action_pipeline = self._load_parser_model()
        except Exception as e:
            raise RuntimeError(f"Error fetching parser model: {e}")

    def _save_parser_model(self, commands: dict, keyword_action_map: dict, pipeline: Any) -> None:
        # joblibs parser into a file

        model = {
            "commands": commands,
            "keyword_action_map": keyword_action_map,
            "action_pipeline": pipeline,
        }
        with open(settings.parser_model_dir, "wb") as file:
            joblib.dump(model, file)
    
    def _load_parser_model(self) -> tuple[dict, dict, Any]:
        # Loads parser from a file

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
            action_pipeline: Any = model["action_pipeline"]
        except Exception as e:
            raise RuntimeError(f"Error extracting parser model components: {e}")

        return commands, keyword_action_map, action_pipeline
    
    def train_action_pipeline(self, action_keywords: list[str], action: str) -> None:
        # Trains action pipeline
        # If length of commands is less then 1, avoids training

        if len(self._commands) <= 1:
            return

        X = self._action_pipeline.named_steps["countvectorizer"].transform([" ".join(action_keywords)])
        self._action_pipeline.named_steps["sgdclassifier"].partial_fit(X, [action])

        self._save_parser_model(self._commands, self._keyword_action_map, self._action_pipeline)

    def train_argument_pipeline(self, action: str, argument_keywords: list[str], argument_index: int) -> None:
        # Trains argument pipeline
        # If length of arguments for an action is less then 1, avoids training

        if action not in self._commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "argument_pipeline" not in self._commands[action]:
            raise ValueError(f"Argument '{argument_index}' has no argument pipeline")
        
        if len(self._commands[action]['args']) <= 1:
            return

        argument_pipeline = self._commands[action]["argument_pipeline"]
        X = argument_pipeline.named_steps["countvectorizer"].transform([" ".join(argument_keywords)])
        argument_pipeline.named_steps["sgdclassifier"].partial_fit(X, [argument_index])

        self._save_parser_model(self._commands, self._keyword_action_map, self._action_pipeline)
    
    def predict_top_actions(self, action_keywords: list[str], max_possibilities: int, probability_cutoff: float) -> list[tuple]:
        # Predicts top most possible actions for action keywords.
        # Keeps adding actions to list, until total probability exceeds probability cutoff.
        # Returns a list of actions with their probabilities in descending order.

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        probabilities: ndarray = self._action_pipeline.predict_proba([" ".join(action_keywords)])[0]

        classes = [(str(self._action_pipeline.classes_[idx]), float(probability)) for idx, probability in enumerate(probabilities)]
        top_actions_max = sorted(classes, reverse=True, key = lambda x: x[1])[:min(len(probabilities), max_possibilities)]
        
        top_actions = []
        total_probability = 0
        for action in top_actions_max:
            top_actions.append(action)
            total_probability += action[1]

            if total_probability >= probability_cutoff:
                break

        return top_actions
    
    def predict_top_arguments_indices(self, action: str, keywords: list[str], max_possibilities: int, probability_cutoff: float) -> list[tuple]:
        # Predicts top most possible arguments for the action and argument keywords.
        # Keeps adding arguments to list, until total probability exceeds probability cutoff.
        # Returns a list of arguments with their probabilities in descending order.

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        if action not in self._commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "argument_pipeline" not in self._commands[action]:
            raise ValueError(f"Action '{action}' has no argument pipeline")
        
        argument_pipeline = self._commands[action]["argument_pipeline"]
        probabilities: ndarray = argument_pipeline.predict_proba([" ".join(keywords)])[0]

        classes = [(int(argument_pipeline.classes_[idx]), float(probability)) for idx, probability in enumerate(probabilities)]
        top_arguments_indices_max = sorted(classes, reverse=True, key = lambda x: x[1])[:min(len(probabilities), max_possibilities)]

        top_arguments_indices = []
        total_probability = 0
        for argument in top_arguments_indices_max:
            top_arguments_indices.append(argument)
            total_probability += argument[1]

            if total_probability >= probability_cutoff:
                break

        return top_arguments_indices
    
    def match_action_keyword(self, token: str, probability_cutoff: float) -> str | None:
        # Matches token with action keywords using fuzzy matching.
        # If confidence is less then probability cutoff, returns None

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")
        
        keyword = process.extractOne(token, self.get_action_keywords(), scorer=fuzz.ratio, score_cutoff=probability_cutoff * 100)
        if keyword:
            return keyword[0]
        return None

    def match_argument_keyword(self, action: str, token: str, probability_cutoff: float) -> str | None:
        # Matches token with argument keywords for action.
        # If confidence is less then probability cutoff, returns None.

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        keyword = process.extractOne(token, self.get_argument_keywords(action), scorer=fuzz.ratio, score_cutoff=probability_cutoff * 100)
        if keyword:
            return keyword[0]
        return None

    def is_stop_word(self, token: str, probability_cutoff: float) -> bool:
        # Matches token with stop words.
        # If confidence is less then probability cutoff returns False, else True

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")
        
        return process.extractOne(token, ENGLISH_STOP_WORDS, scorer=fuzz.ratio, score_cutoff=probability_cutoff * 100) is not None
    
    def has_action_warning(self, action: str) -> bool:
        # Checks and returns if an action has warning set to True.

        if action not in self._commands:
            raise ValueError(f"Action '{action}' not found in commands")

        return self._commands[action]["warning"]
    
    def get_action_description(self, action: str) -> str:
        # Fetches description of an action.

        if action not in self._commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "description" not in self._commands[action]:
            raise ValueError(f"Action '{action}' has no description")

        return self._commands[action]["description"]

    def get_action_keywords(self) -> KeysView[str]:
        # Fetches all action keywords.

        return self._keyword_action_map.keys()
    
    def get_argument_keywords(self, action: str) -> KeysView[str]:
        # Fetches all argument keywords for an action

        if action not in self._commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "keyword_argument_map" not in self._commands[action]:
            raise ValueError(f"Action '{action}' has no argument keywords")

        return self._commands[action]["keyword_argument_map"].keys()
    
    def get_actions_for_keyword(self, action_keyword: str) -> set[str]:
        # Fetches all action for an action_keyword

        return self._keyword_action_map.get(action_keyword, set())
    
    def get_argument_indices_for_keyword(self, action: str, argument_keyword: str) -> set[int]:
        # Fetches all arguments for an argument keyword

        if action not in self._commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "keyword_argument_map" not in self._commands[action]:
            raise ValueError(f"Action '{action}' has no argument keywords")

        return self._commands[action]["keyword_argument_map"].get(argument_keyword, set())

    def get_required_arguments(self, action: str) -> list[int]:
        # Fetches required arguments for an action

        if action not in self._commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "args" not in self._commands[action]:
            raise ValueError(f"Action '{action}' has no arguments")
        
        all_arguments: list[dict] = self._commands[action]["args"]
        
        indices = []
        for i in range(len(all_arguments)):
            if all_arguments[i]["required"]:
                indices.append(i)
        
        return indices
    
    def get_optional_arguments(self, action: str) -> list[int]:
        # Fetches optional arguments for an action

        if action not in self._commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "args" not in self._commands[action]:
            return []
        
        all_arguments: list[dict] = self._commands[action]["args"]
        
        indices = []
        for i in range(len(all_arguments)):
            if not all_arguments[i]["required"]:
                indices.append(i)
        
        return indices
    
    def get_argument_type(self, action: str, idx: int) -> str:
        # Fetches type of an argument

        if action not in self._commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "args" not in self._commands[action]:
            raise ValueError(f"Action '{action}' has no arguments")
        
        all_arguments: list[dict] = self._commands[action]["args"]
        
        if idx < 0 or idx >= len(all_arguments):
            raise IndexError(f"Argument index {idx} out of range for action '{action}'")

        return all_arguments[idx]["type"]
    
    def get_argument_description(self, action: str, idx: int) -> str:
        # Fetches type of an argument

        if action not in self._commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "args" not in self._commands[action]:
            raise ValueError(f"Action '{action}' has no arguments")
        
        all_arguments: list[dict] = self._commands[action]["args"]
        
        if idx < 0 or idx >= len(all_arguments):
            raise IndexError(f"Argument index {idx} out of range for action '{action}'")

        return all_arguments[idx]["description"]
    
    def get_arguments_count(self, action: str) -> int:
        # Fetched no.of arguments available for an action.

        if action not in self._commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "args" not in self._commands[action]:
            return 0
        
        return len(self._commands[action]["args"])