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
        self._commands: dict | None = None
        self._keyword_action_map: dict | None = None
        self._action_pipeline: Any | None = None
        self._keyword_argument_maps: dict = dict()
        self._argument_pipelines: dict[dict] = dict()
        
    def _load_commands(self) -> dict:
        # Loads commands from file

        if not os.path.exists(settings.commands_dir):
            raise FileNotFoundError("Commands file not found")
        
        try:
            return joblib.load(settings.commands_dir)
        except Exception as e:
            raise RuntimeError(f"Error loading commands: {e}")
        
    def _load_keywordmap(self, action: str | None = None) -> dict:
        # Loads keyword map for either action or argument from file

        if action is not None:
            if not os.path.exists(os.path.join(settings.keyword_argument_maps_dir, action)):
                raise ValueError(f"No argument keyword map found for action: {action}")

        map_dir = settings.keyword_argument_map_dir(action) if action else settings.keyword_action_map_dir
        if not os.path.exists(map_dir):
            raise FileNotFoundError(f"Keyword argument map file not found for action: {action}") if action else FileNotFoundError("Keyword action map file not found")
        
        try:
            return joblib.load(map_dir)
        except Exception as e:
            raise RuntimeError(f"Error loading keyword argument map for action '{action}': {e}") if action else RuntimeError(f"Error loading keyword action map: {e}")
        
    def _load_pipeline(self, action: str | None = None) -> dict:
        # Loads pipeline for either action or argument from file

        if action is not None:
            if not os.path.exists(os.path.join(settings.argument_pipelines_dir, action)):
                raise ValueError(f"No argument pipeline found for action: {action}")

        pipeline_dir = settings.argument_pipeline_dir(action) if action else settings.action_pipeline_dir
        if not os.path.exists(pipeline_dir):
            raise FileNotFoundError(f"Argument pipeline file not found for action: {action}") if action else FileNotFoundError("Action pipeline file not found")
        
        try:
            return joblib.load(pipeline_dir)
        except Exception as e:
            raise RuntimeError(f"Error loading argument pipeline for action '{action}': {e}") if action else RuntimeError(f"Error loading action pipeline: {e}")
        
    def _save_pipeline(self, pipeline: Any, action: str | None = None) -> None:
        # Saves pipeline for either action or argument from file

        if action is not None:
            if not os.path.exists(os.path.join(settings.argument_pipelines_dir, action)):
                raise ValueError(f"No argument pipeline found for action: {action}")
            
        pipeline_dir = settings.argument_pipeline_dir(action) if action else settings.action_pipeline_dir
        if not os.path.exists(pipeline_dir):
            raise FileNotFoundError(f"Argument pipeline file not found for action: {action}") if action else FileNotFoundError("Action pipeline file not found")
        
        try:
            joblib.dump(pipeline, pipeline_dir)
        except Exception as e:
            raise RuntimeError(f"Error saving argument pipeline for action '{action}': {e}") if action else RuntimeError(f"Error saving action pipeline: {e}")
        
    def _save_action_pipeline(self) -> None:
        # Checks if action pipeline has been loaded and saves it

        if self._action_pipeline is None:
            raise RuntimeError("Action pipeline has not been loaded")

        self._save_pipeline(self._action_pipeline)

    def _save_argument_pipeline(self, action: str) -> None:
        # Checks if argument pipeline has been loaded and saves it

        if action not in self._argument_pipelines:
            raise RuntimeError(f"Argument pipeline for action '{action}' has not been loaded")

        self._save_pipeline(self._argument_pipelines[action], action)
        
    def _get_commands(self) -> dict:
        # Checks if commands is available in memory else loads from file

        if self._commands is None:
            self._commands: dict = self._load_commands()
        
        return self._commands
    
    def _get_keyword_action_map(self) -> dict:
        # Checks if keyword action map is available in memory else loads from file

        if self._keyword_action_map is None:
            self._keyword_action_map: dict = self._load_keywordmap()

        return self._keyword_action_map
    
    def _get_keyword_argument_map(self, action: str) -> dict:
        # Checks if keyword argument map is available in memory else loads from file

        if action not in self._keyword_argument_maps:
            self._keyword_argument_maps[action] = self._load_keywordmap(action)

        return self._keyword_argument_maps[action]
    
    def _get_action_pipeline(self) -> Any:
        # Checks if action pipeline is available in memory else loads from file

        if self._action_pipeline is None:
            self._action_pipeline = self._load_pipeline()

        return self._action_pipeline
    
    def _get_argument_pipeline(self, action: str) -> Any:
        # Checks if argument pipeline is available in memory else loads from file
        
        if action not in self._argument_pipelines:
            self._argument_pipelines[action] = self._load_pipeline(action)

        return self._argument_pipelines[action]
    
    def train_action_pipeline(self, action_keywords: list[str], action: str) -> None:
        # Trains action pipeline

        if len(self._commands) <= 1:
            raise RuntimeError("Action pipeline cannot be trained on less then 2 actions")

        X = self._get_action_pipeline().named_steps["countvectorizer"].transform([" ".join(action_keywords)])
        self._get_action_pipeline().named_steps["sgdclassifier"].partial_fit(X, [action])

        self._save_action_pipeline()

    def train_argument_pipeline(self, action: str, argument_keywords: list[str], argument_index: int) -> None:
        # Trains argument pipeline
        
        if len(self._commands[action]['args']) <= 1:
            raise RuntimeError("Argument pipeline cannot be trained on less then 2 arguments")

        X = self._get_argument_pipeline(action).named_steps["countvectorizer"].transform([" ".join(argument_keywords)])
        self._get_argument_pipeline(action).named_steps["sgdclassifier"].partial_fit(X, [argument_index])

        self._save_argument_pipeline(action)
    
    def predict_top_actions(self, action_keywords: list[str], max_possibilities: int, probability_cutoff: float) -> list[tuple]:
        # Predicts top most possible actions for action keywords.
        # Keeps adding actions to list, until total probability exceeds probability cutoff.
        # Returns a list of actions with their probabilities in descending order.

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        probabilities: ndarray = self._get_action_pipeline().predict_proba([" ".join(action_keywords)])[0]

        classes = [(str(self._get_action_pipeline().classes_[idx]), float(probability)) for idx, probability in enumerate(probabilities)]
        top_actions_max = sorted(classes, reverse=True, key = lambda x: x[1])[:min(len(probabilities), max_possibilities)]
        
        top_actions = []
        total_probability = 0
        for action in top_actions_max:
            top_actions.append(action)
            total_probability += action[1]

            if total_probability >= probability_cutoff:
                break

        return top_actions
    
    def predict_argument_index(self, action: str, argument_keywords: list[str], probability_cutoff: float) -> int | None:
        # Predicts argument using argument keywords.
        # If confidence is less then probability cutoff, returns None.

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")
        
        probabilities: ndarray = self._get_argument_pipeline(action).predict_proba([" ".join(argument_keywords)])[0]

        argument_index = max([(int(self._get_argument_pipeline(action).classes_[idx]), float(probability)) for idx, probability in enumerate(probabilities)])

        if argument_index[1] < probability_cutoff:
            return None
        return argument_index
    
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

        if action not in self._get_commands():
            raise ValueError(f"Action '{action}' not found in commands")
        if "warning" not in self._get_commands()[action]:
            raise ValueError(f"Action '{action}' has no warning")

        return self._get_commands()[action]["warning"]
    
    def get_action_description(self, action: str) -> str:
        # Fetches description of an action.

        if action not in self._get_commands():
            raise ValueError(f"Action '{action}' not found in commands")
        if "description" not in self._get_commands()[action]:
            raise ValueError(f"Action '{action}' has no description")

        return self._get_commands()[action]["description"]

    def get_action_keywords(self) -> KeysView[str]:
        # Fetches all action keywords.

        return self._get_keyword_action_map().keys()
    
    def get_argument_keywords(self, action: str) -> KeysView[str]:
        # Fetches all argument keywords for an action

        return self._get_keyword_argument_map(action).keys()
    
    def get_actions_for_keyword(self, action_keyword: str) -> set[str]:
        # Fetches all action for an action_keyword

        return self._get_keyword_action_map().get(action_keyword, set())
    
    def get_argument_indices_for_keyword(self, action: str, argument_keyword: str) -> set[int]:
        # Fetches all arguments for an argument keyword

        return self._get_keyword_argument_map().get(argument_keyword, set())

    def get_required_arguments(self, action: str) -> list[int]:
        # Fetches required arguments for an action

        if action not in self._get_commands():
            raise ValueError(f"Action '{action}' not found in commands")
        if "args" not in self._get_commands()[action]:
            raise ValueError(f"Action '{action}' has no arguments")
        
        all_arguments: list[dict] = self._get_commands()[action]["args"]
        
        indices = []
        for i in range(len(all_arguments)):
            if "required" not in all_arguments[i]:
                raise ValueError(f"Argument: '{i}' for Action '{action}' has no required")
            
            if all_arguments[i]["required"]:
                indices.append(i)
        
        return indices
    
    def get_optional_arguments(self, action: str) -> list[int]:
        # Fetches optional arguments for an action

        if action not in self._get_commands():
            raise ValueError(f"Action '{action}' not found in commands")
        if "args" not in self._get_commands()[action]:
            raise ValueError(f"Action '{action}' has no arguments")
        
        all_arguments: list[dict] = self._get_commands()[action]["args"]
        
        indices = []
        for i in range(len(all_arguments)):
            if "required" not in all_arguments[i]:
                raise ValueError(f"Argument: '{i}' for Action '{action}' has no required")
            
            if not all_arguments[i]["required"]:
                indices.append(i)
        
        return indices
    
    def get_argument_type(self, action: str, idx: int) -> str:
        # Fetches type of an argument

        if idx < 0 or idx >= self.get_arguments_count(action):
            raise ValueError(f"Index out of bounds, index given: {idx} arguments available: {self.get_arguments_count()}")
        if "type" not in self._get_commands()[action]["args"][idx]:
            raise ValueError(f"Argument: '{idx}' for Action '{action}' has no type")

        return self._get_commands()[action]["args"][idx]["type"]
    
    def get_argument_format(self, action: str, idx: int) -> str:
        # Fetches format of an argument.

        if idx < 0 or idx >= self.get_arguments_count(action):
            raise ValueError(f"Index out of bounds, index given: {idx} arguments available: {self.get_arguments_count()}")
        if "format" not in self._get_commands()[action]["args"][idx]:
            raise ValueError(f"Argument: '{idx}' for Action '{action}' has no format")
        
        return self._get_commands()[action]["args"][idx]["format"]
    
    def get_argument_description(self, action: str, idx: int) -> str:
        # Fetches type of an argument

        if idx < 0 or idx >= self.get_arguments_count(action):
            raise ValueError(f"Index out of bounds, index given: {idx} arguments available: {self.get_arguments_count()}")
        if "description" not in self._get_commands()[action]["args"][idx]:
            raise ValueError(f"Argument: '{idx}' for Action '{action}' has no description")
        
        return self._get_commands()[action]["args"][idx]["description"]
    
    def get_arguments_count(self, action: str) -> int:
        # Fetched no.of arguments available for an action.

        if action not in self._get_commands():
            raise ValueError(f"Action '{action}' not found in commands")
        if "args" not in self._get_commands()[action]:
            raise ValueError(f"Action '{action}' has no arguments")
        
        return len(self._get_commands()[action]["args"])