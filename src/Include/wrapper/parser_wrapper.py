import os
import sys
import subprocess
import joblib

from collections.abc import KeysView

from rapidfuzz import process, fuzz

from numpy import ndarray
from typing import Any

from Include.filter.stop_words import ENGLISH_STOP_WORDS
from Include.subsystem.usagedata_db import UsagedataDB

import settings

class ParserWrapper:
    def __init__(self, environment: settings.Environment):
        self._commands: dict | None = None

        self._keyword_action_map: dict[str, set[str]] | None = None
        self._action_pipeline: Any | None = None
        self._keyword_argument_maps: dict[str, dict[str, set[int]]] = dict()
        self._argument_pipelines: dict[str, Any] = dict()

        self._app_executablepath_map: dict[str, str] | None = None
        self._class_app_map: dict[str, set[str]] | None = None
        self._nickname_app_map: dict[str, str] | None = None

        self._apps_with_nicknames: set | None = None
        self._apps_in_class: set | None = None

        self._usagedata_db: UsagedataDB = UsagedataDB(settings.usagedata_dir)

        self._observe: subprocess.Popen | None = None
        if environment == settings.Environment.PROD:
            self._observe = subprocess.Popen(["observe.exe"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif environment == settings.Environment.DEV:
            self._observe = subprocess.Popen([sys.executable, "src/observe.py"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            raise ValueError(f"Invalid environment: '{environment}'. Valid options are: {[env.value for env in settings.Environment]}")
        
    def _load_commands(self) -> dict:
        # Loads commands from file

        if not os.path.exists(settings.commands_dir):
            raise FileNotFoundError("Commands file not found")
        
        try:
            return joblib.load(settings.commands_dir, mmap_mode = 'r')
        except Exception as e:
            raise RuntimeError(f"Error loading commands: {e}")
        
    def _load_keyword_map(self, action: str | None = None) -> dict:
        # Loads keyword map for either action or argument from file

        if action is not None:
            if not os.path.exists(os.path.join(settings.parser_dir, action)):
                raise ValueError(f"No argument keyword map found for action: {action}")

        map_dir = settings.keyword_argument_map_dir(action) if action else settings.keyword_action_map_dir
        if not os.path.exists(map_dir):
            raise FileNotFoundError(f"Keyword argument map file not found for action: {action}") if action else FileNotFoundError("Keyword action map file not found")
        
        try:
            return joblib.load(map_dir, mmap_mode = 'r')
        except Exception as e:
            raise RuntimeError(f"Error loading keyword argument map for action '{action}': {e}") if action else RuntimeError(f"Error loading keyword action map: {e}")
        
    def _load_pipeline(self, action: str | None = None) -> dict:
        # Loads pipeline for either action or argument from file

        if action is not None:
            if not os.path.exists(os.path.join(settings.parser_dir, action)):
                raise ValueError(f"No argument pipeline found for action: {action}")

        pipeline_dir = settings.argument_pipeline_dir(action) if action else settings.action_pipeline_dir
        if not os.path.exists(pipeline_dir):
            raise FileNotFoundError(f"Argument pipeline file not found for action: {action}") if action else FileNotFoundError("Action pipeline file not found")
        
        try:
            return joblib.load(pipeline_dir)
        except Exception as e:
            raise RuntimeError(f"Error loading argument pipeline for action '{action}': {e}") if action else RuntimeError(f"Error loading action pipeline: {e}")
        
    def _load_app_executablepath_map(self) -> dict:
        # Loads app executable path map from file

        if not os.path.exists(settings.app_executablepath_map_dir):
            raise FileNotFoundError("App executable path map file not found")
        
        try:
            return joblib.load(settings.app_executablepath_map_dir, mmap_mode = 'r')
        except Exception as e:
            raise RuntimeError(f"Error loading app executable path map: {e}")
        
    def _load_nickname_app_map(self) -> dict:
        # Loads nickname app map from file

        if not os.path.exists(settings.nickname_app_map_dir):
            raise FileNotFoundError("Nickname app map file not found")
        
        try:
            return joblib.load(settings.nickname_app_map_dir, mmap_mode = 'r')
        except Exception as e:
            raise RuntimeError(f"Error loading nickname app path map: {e}")
        
    def _load_class_app_map(self) -> dict:
        # Loads class app map from file

        if not os.path.exists(settings.class_app_map_dir):
            raise FileNotFoundError("Class app map file not found")
        
        try:
            return joblib.load(settings.class_app_map_dir, mmap_mode = 'r')
        except Exception as e:
            raise RuntimeError(f"Error loading class app path map: {e}")
        
    def _save_pipeline(self, pipeline: Any, action: str | None = None) -> None:
        # Saves pipeline for either action or argument from file

        if action is not None:
            if not os.path.exists(os.path.join(settings.parser_dir, action)):
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

    def _save_app_executablepath_map(self) -> None:
        # Checks if app executable path map has been loaded and saves it

        if self._app_executablepath_map is None:
            raise RuntimeError("App executable path map has not been loaded")
        
        try:
            joblib.dump(self._app_executablepath_map, settings.app_executablepath_map_dir)
        except Exception as e:
            raise RuntimeError(f"Error loading app executable path map: {e}")
        
    def _save_nickname_app_map(self) -> None:
        # Checks if nickname app map has been loaded and saves it

        if self._nickname_app_map is None:
            raise RuntimeError("Nickname app map has not been loaded")
        
        try:
            joblib.dump(self._nickname_app_map, settings.nickname_app_map_dir)
        except Exception as e:
            raise RuntimeError(f"Error loading nickname app map: {e}")
        
    def _save_class_app_map(self) -> None:
        # Checks if class app map has been loaded and saves it

        try:
            joblib.dump(self._class_app_map, settings.class_app_map_dir)
        except Exception as e:
            raise RuntimeError(f"Error loading class app map: {e}")
        
    def _get_commands(self) -> dict:
        # Checks if commands is available in memory else loads from file

        if self._commands is None:
            self._commands = self._load_commands()
        
        return self._commands
    
    def _get_keyword_action_map(self) -> dict:
        # Checks if keyword action map is available in memory else loads from file

        if self._keyword_action_map is None:
            self._keyword_action_map = self._load_keyword_map()

        return self._keyword_action_map
    
    def _get_keyword_argument_map(self, action: str) -> dict:
        # Checks if keyword argument map is available in memory else loads from file

        if action not in self._keyword_argument_maps:
            self._keyword_argument_maps[action] = self._load_keyword_map(action)

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
    
    def _get_app_executablepath_map(self) -> dict:
        # Checks if app executable path map is available in memory else loads from file

        if self._app_executablepath_map is None:
            self._app_executablepath_map = self._load_app_executablepath_map()

        return self._app_executablepath_map
    
    def _get_nickname_app_map(self) -> dict:
        # Checks if nickname app map is available in memory else loads from file

        if self._nickname_app_map is None:
            self._nickname_app_map = self._load_nickname_app_map()

        return self._nickname_app_map
    
    def _get_class_app_map(self) -> dict:
        # Checks if class app map is available in memory else loads from file

        if self._class_app_map is None:
            self._class_app_map = self._load_class_app_map()

        return self._class_app_map
    
    def _get_apps_with_nicknames(self) -> set:
        # Checks if apps with nicknames is available in memory else creates it

        if self._apps_with_nicknames is None:
            self._apps_with_nicknames = set(self._get_nickname_app_map().values())

        return self._apps_with_nicknames
    
    def _get_apps_in_class(self) -> set:
        # Checks if apps in class is available in memory else creates it

        if self._apps_in_class is None:
            all_apps = set()
            for apps in self._get_class_app_map().values():
                for app in apps:
                    all_apps.add(app)
            self._apps_in_class = all_apps

        return self._apps_in_class
    
    def close(self):
        if self._observe is not None:
            self._observe.terminate()
            self._observe.wait()
    
    def has_nicknames(self, app: str) -> bool:
        # Returns whether an app has nicknames

        return app in self._get_apps_with_nicknames()
    
    def in_class(self, app: str) -> bool:
        # Returns whether an app is in a class

        return app in self._get_apps_in_class()
    
    def set_nickname(self, nickname: str, app: str) -> None:
        # Adds a nickname for an app

        self._get_nickname_app_map()[nickname] = app

        try:
            self._save_nickname_app_map()
        except Exception as e:
            print("Warning: Unable to save nickname:", e)

    def add_to_class(self, class_name: str, app: str) -> None:
        # Adds an app to a class

        class_app_map = self._get_class_app_map()
        if class_name not in class_app_map:
            class_app_map[class_name] = []

        if app not in class_app_map[class_name]:
            class_app_map[class_name].append(app)

            try:
                self._save_class_app_map()
            except Exception as e:
                print("Warning: Unable to save to class:", e)

    def train_action_pipeline(self, action_keywords: list[str], action: str) -> None:
        # Trains action pipeline

        if len(self._get_commands()) <= 1:
            raise RuntimeError("Action pipeline cannot be trained on less then 2 actions")
        
        action_pipeline = self._get_action_pipeline()

        X = action_pipeline.named_steps["countvectorizer"].transform([" ".join(action_keywords)])
        action_pipeline.named_steps["sgdclassifier"].partial_fit(X, [action])

        self._save_action_pipeline()

    def train_argument_pipeline(self, action: str, argument_keywords: list[str], argument_index: int) -> None:
        # Trains argument pipeline
        
        if len(self._get_commands()[action]['args']) <= 1:
            raise RuntimeError("Argument pipeline cannot be trained on less then 2 arguments")
        
        argument_pipeline = self._get_argument_pipeline(action)

        X = argument_pipeline.named_steps["countvectorizer"].transform([" ".join(argument_keywords)])
        argument_pipeline.named_steps["sgdclassifier"].partial_fit(X, [argument_index])

        self._save_argument_pipeline(action)
    
    def predict_top_actions(self, action_keywords: list[str], max_possibilities: int, probability_cutoff: float) -> list[tuple]:
        # Predicts top most possible actions for action keywords.
        # Keeps adding actions to list, until total probability exceeds probability cutoff.
        # Returns a list of actions with their probabilities in descending order.

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")
        
        action_pipeline = self._get_action_pipeline()

        probabilities: ndarray = action_pipeline.predict_proba([" ".join(action_keywords)])[0]

        classes = [(str(action_pipeline.classes_[idx]), float(probability)) for idx, probability in enumerate(probabilities)]
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
        
        argument_pipeline = self._get_argument_pipeline(action)
        
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
    
    def predict_argument_index(self, action: str, argument_keywords: list[str], probability_cutoff: float) -> int | None:
        # Predicts argument using argument keywords.
        # If confidence is less then probability cutoff, returns None.

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")
        
        argument_pipeline = self._get_argument_pipeline(action)
        
        probabilities: ndarray = argument_pipeline.predict_proba([" ".join(argument_keywords)])[0]

        argument_index = max([(int(argument_pipeline.classes_[idx]), float(probability)) for idx, probability in enumerate(probabilities)], key = lambda x: x[1])

        if argument_index[1] < probability_cutoff:
            return None
        return argument_index[0]
    
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
    
    def match_existing_app(self, token: str, probability_cutoff: float) -> str | None:
        # Matches token with apps.
        # If confidence is less then probability cutoff returns None, else app name

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        app = process.extractOne(token, self.get_existing_apps(), scorer=fuzz.partial_token_set_ratio, score_cutoff=probability_cutoff * 100)
        if app:
            return app[0]
        return None
    
    def match_monitored_app(self, token: str, probability_cutoff: float) -> str | None:
        # Matches token with monitored apps.
        # If confidence is less then probability cutoff returns None, else app name

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")
        
        monitored_app_executablepath_map = self.get_monitored_apps_executablepaths()

        app = process.extractOne(token, monitored_app_executablepath_map.keys(), scorer=fuzz.partial_token_set_ratio, score_cutoff=probability_cutoff * 100)
        if app:
            self._get_app_executablepath_map()[app[0]] = monitored_app_executablepath_map[app[0]]
            self._save_app_executablepath_map()

            return app[0]
        return None
    
    def match_nickname(self, token: str, probability_cutoff: float) -> str | None:
        # Matches token with nicknames.
        # If confidence is less then probability cutoff returns None, else app name

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        nickname = process.extractOne(token, self._get_nickname_app_map().keys(), scorer=fuzz.partial_token_sort_ratio, score_cutoff=probability_cutoff * 100)
        if nickname:
            return nickname[0]
        return None
    
    def match_class(self, token: str, probability_cutoff: float) -> str | None:
        # Matches token with classes.
        # If confidence is less then probability cutoff returns None, else app name

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        class_name = process.extractOne(token, self._get_class_app_map().keys(), scorer=fuzz.partial_token_sort_ratio, score_cutoff=probability_cutoff * 100)
        if class_name:
            return class_name[0]
        return None

    def is_stop_word(self, token: str, probability_cutoff: float) -> bool:
        # Matches token with stop words.
        # If confidence is less then probability cutoff returns False, else True

        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")
        
        return process.extractOne(token, ENGLISH_STOP_WORDS, scorer=fuzz.ratio, score_cutoff=probability_cutoff * 100) is not None
    
    def has_action_warning(self, action: str) -> bool:
        # Checks and returns if an action has warning set to True.

        commands = self._get_commands()

        if action not in commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "warning" not in commands[action]:
            raise ValueError(f"Action '{action}' has no warning")

        return commands[action]["warning"]
    
    def get_action_description(self, action: str) -> str:
        # Fetches description of an action.

        commands = self._get_commands()

        if action not in commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "description" not in commands[action]:
            raise ValueError(f"Action '{action}' has no description")

        return commands[action]["description"]

    def get_action_keywords(self) -> KeysView[str]:
        # Fetches all action keywords.

        return self._get_keyword_action_map().keys()
    
    def get_argument_keywords(self, action: str) -> KeysView[str]:
        # Fetches all argument keywords for an action

        return self._get_keyword_argument_map(action).keys()
    
    def get_actions_for_keyword(self, action_keyword: str) -> set[str]:
        # Fetches all action for an action_keyword

        keyword_action_map = self._get_keyword_action_map()
        if action_keyword not in keyword_action_map:
            raise ValueError(f"Action keyword '{action_keyword}' not found in keyword action map")

        return keyword_action_map[action_keyword]
    
    def get_argument_indices_for_keyword(self, action: str, argument_keyword: str) -> set[int]:
        # Fetches all arguments for an argument keyword

        keyword_argument_map = self._get_keyword_argument_map(action)
        if argument_keyword not in keyword_argument_map:
            raise ValueError(f"Argument keyword '{argument_keyword}' not found in keyword argument map for action '{action}'")

        return keyword_argument_map.get(argument_keyword, set())

    def get_required_arguments(self, action: str) -> list[int]:
        # Fetches required arguments for an action

        commands = self._get_commands()

        if action not in commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "args" not in commands[action]:
            raise ValueError(f"Action '{action}' has no arguments")
        
        all_arguments: list[dict] = commands[action]["args"]
        
        indices = []
        for i in range(len(all_arguments)):
            if "required" not in all_arguments[i]:
                raise ValueError(f"Argument: '{i}' for Action '{action}' has no required")
            
            if all_arguments[i]["required"]:
                indices.append(i)
        
        return indices
    
    def get_optional_arguments(self, action: str) -> list[int]:
        # Fetches optional arguments for an action

        commands = self._get_commands()

        if action not in commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "args" not in commands[action]:
            raise ValueError(f"Action '{action}' has no arguments")
        
        all_arguments: list[dict] = commands[action]["args"]
        
        indices = []
        for i in range(len(all_arguments)):
            if "required" not in all_arguments[i]:
                raise ValueError(f"Argument: '{i}' for Action '{action}' has no required")
            
            if not all_arguments[i]["required"]:
                indices.append(i)
        
        return indices
    
    def get_argument_type(self, action: str, idx: int) -> str:
        # Fetches type of an argument

        if idx >= (arguments_count := self.get_arguments_count(action)) or idx < 0:
            raise ValueError(f"Index out of bounds, index given: {idx} arguments available: {arguments_count}")
        
        commands = self._get_commands()

        if "type" not in commands[action]["args"][idx]:
            raise ValueError(f"Argument: '{idx}' for Action '{action}' has no type")

        return commands[action]["args"][idx]["type"]
    
    def get_argument_format(self, action: str, idx: int) -> str:
        # Fetches format of an argument.

        if idx >= (arguments_count := self.get_arguments_count(action)) or idx < 0:
            raise ValueError(f"Index out of bounds, index given: {idx} arguments available: {arguments_count}")
        
        commands = self._get_commands()

        if "format" not in commands[action]["args"][idx]:
            raise ValueError(f"Argument: '{idx}' for Action '{action}' has no format")
        
        return commands[action]["args"][idx]["format"]
    
    def get_argument_description(self, action: str, idx: int) -> str:
        # Fetches type of an argument

        if idx >= (arguments_count := self.get_arguments_count(action)) or idx < 0:
            raise ValueError(f"Index out of bounds, index given: {idx} arguments available: {arguments_count}")
        
        commands = self._get_commands()

        if "description" not in commands[action]["args"][idx]:
            raise ValueError(f"Argument: '{idx}' for Action '{action}' has no description")
        
        return commands[action]["args"][idx]["description"]
    
    def get_arguments_count(self, action: str) -> int:
        # Fetched no.of arguments available for an action.

        commands = self._get_commands()

        if action not in commands:
            raise ValueError(f"Action '{action}' not found in commands")
        if "args" not in commands[action]:
            raise ValueError(f"Action '{action}' has no arguments")
        
        return len(commands[action]["args"])
    
    def get_existing_apps(self) -> KeysView[str]:
        # Fetches all apps that have been used before

        return self._get_app_executablepath_map().keys()
    
    def get_monitored_apps_executablepaths(self) -> dict[str, str]:
        # Fetches all apps that are monitored

        app_executablepath_map = dict()
        
        daylog_ids = self._usagedata_db.get_daylog_ids()
        for daylog_id in daylog_ids:
            applog_titlelog = self._usagedata_db.get_applog_titlelog(daylog_id)
            
            for app_name, app_data in applog_titlelog.items():
                if "executable_path" not in app_data:
                    raise ValueError(f"Some app log entries have no executable_path, daylog_id: {daylog_id}")

                app_executablepath_map[app_name] = app_data["executable_path"]

        return app_executablepath_map
    
    def get_app_for_nickname(self, nickname: str) -> str:
        # Fetches app for a nickname

        nickname_app_map = self._get_nickname_app_map()

        if nickname not in nickname_app_map:
            raise ValueError(f"Nickname {nickname} not found in nickname app map")

        return nickname_app_map[nickname]
    
    def get_executablepath(self, app: str) -> str:
        # Fetches executable path for app

        app_executablepath_map = self._get_app_executablepath_map()

        if app not in app_executablepath_map:
            raise ValueError(f"App {app} not found in app executable path map")
        
        return app_executablepath_map[app]
    
    def get_classes(self) -> KeysView[str]:
        # Fetches all classes

        return self._get_class_app_map().keys()
    
    def get_mostused_app_for_class(self, class_name: str) -> str | None:
        # Fetches app for a class

        class_app_map = self._get_class_app_map()

        if class_name not in class_app_map:
            raise ValueError(f"Class {class_name} not found in nickname app map")
        
        return self._usagedata_db.get_mostused_app(tuple(class_app_map[class_name]))