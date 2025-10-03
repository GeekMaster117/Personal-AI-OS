import json
from pathlib import Path
import shutil

import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline, Pipeline

import settings

action_keys = {'keywords', 'args', 'description', 'warning'}
argument_keys = {'keywords', 'type', 'format', 'required', 'description'}

def load_commands(path: str) -> dict:
    try:
        with open(path, "r") as file:
            commands: dict = json.load(file)
    except Exception as e:
        raise RuntimeError(f"Error loading commands: {e}")
    
    for action, structure in commands.items():
        for key in action_keys:
            if key not in structure:
                raise SyntaxError(f'{key} not found in action: {action}')
            
        for argument_index, argument_structure in enumerate(structure["args"]):
            for key in argument_keys:
                if key not in argument_structure:
                    raise SyntaxError(f'{key} not found in action: {action}, argument index: {argument_index}')
    
    return commands

def clear_directory(path: str) -> None:
    try:
        if Path(path).exists():
            shutil.rmtree(path)
    except Exception as e:
        raise RuntimeError(f"Error clearing path: {path}")

def ensure_parents(path: str) -> str:
    try:
        path: Path = Path(path)
        path.parent.mkdir(parents = True, exist_ok = True)
    except Exception as e:
        raise RuntimeError(f"Error creating parent directories: {e}")

    return str(path)

def dump(object, path: str) -> None:
    try:
        joblib.dump(object, path)
    except Exception as e:
        raise RuntimeError(f"Error dumping: {e}")

def throw_if_not_valid(structure: dict, keys: set, name = '') -> None:
    for key in keys:
        if key not in structure:
            raise SyntaxError(f'{key} not found in {name}')
        
def add_keywords(keyword_map: dict, keywords: list[str], value: str) -> None:
    for keyword in keywords:
        if keyword not in keyword_map:
            keyword_map[keyword] = set()

        keyword_map[keyword].add(value)

def make_keywordmaps_pipelines(commands: dict) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]], Pipeline, dict[Pipeline]]:
    keyword_action_map: dict[str, set[str]] = dict()
    keyword_argument_maps: dict[str, dict[str, set[str]]] = dict()
    action_pipeline: Pipeline = make_pipeline(CountVectorizer(), SGDClassifier(loss="log_loss"))
    argument_pipelines: dict[str, Pipeline] = dict()

    action_keywords, actions = [], []
    for action, structure in commands.items():
        throw_if_not_valid(structure, action_keys, f'action: {action}')

        add_keywords(keyword_action_map, structure["keywords"], action)

        action_keywords.append(" ".join(structure["keywords"]))
        actions.append(action)

        del structure["keywords"]

        keyword_argument_maps[action] = dict()

        argument_keywords, arguments = [], []
        for argument_index, argument_structure in enumerate(structure["args"]):
            throw_if_not_valid(argument_structure, argument_keys, f'action: {action}, argument index: {argument_index}')

            add_keywords(keyword_argument_maps[action], argument_structure["keywords"], argument_index)

            argument_keywords.append(" ".join(argument_structure["keywords"]))
            arguments.append(argument_index)

            del argument_structure["keywords"]

        argument_pipelines[action] = make_pipeline(CountVectorizer(), SGDClassifier(loss="log_loss"))
        if len(structure["args"]) <= 1:
            continue

        argument_pipelines[action].fit(argument_keywords, arguments)

    if len(commands) > 1:
        action_pipeline.fit(action_keywords, actions)

    return keyword_action_map, keyword_argument_maps, action_pipeline, argument_pipelines

commands = load_commands("dev/commands.json")
keyword_action_map, keyword_argument_maps, action_pipeline, argument_pipelines = make_keywordmaps_pipelines(commands)

clear_directory(settings.parser_dir)

dump(commands, ensure_parents(settings.commands_dir))
dump(keyword_action_map, ensure_parents(settings.keyword_action_map_dir))
dump(action_pipeline, ensure_parents(settings.action_pipeline_dir))

dump(dict(), ensure_parents(settings.app_executablepath_map_dir))

for action in commands:
    dump(keyword_argument_maps[action], ensure_parents(settings.keyword_argument_map_dir(action)))
    dump(argument_pipelines[action], ensure_parents(settings.argument_pipeline_dir(action)))