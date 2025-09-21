import json

import joblib

from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

import settings

try:
    with open("dev/commands.json", "r") as file:
        commands: dict = json.load(file)
except Exception as e:
    raise RuntimeError(f"Error loading commands: {e}")

try:
    action_pipeline = make_pipeline(CountVectorizer(), SGDClassifier(loss="log_loss"))

    for structure in commands.values():
        structure["argument_pipeline"] = make_pipeline(CountVectorizer(), SGDClassifier(loss="log_loss"))
except Exception as e:
    raise RuntimeError(f"Error initialising pipeline: {e}")

try:
    keyword_action_map: dict = defaultdict(set)
    action_keywords, actions = [], []

    for action, structure in commands.items():
        for keyword in structure["keywords"]:
            keyword_action_map[keyword].add(action)

        action_keywords.append(" ".join(structure["keywords"]))
        actions.append(action)

        del structure["keywords"]

        structure["keyword_argument_map"] = defaultdict(set)

        argument_keywords, arguments = [], []

        for idx, arg in enumerate(structure["args"]):
            for keyword in arg["keywords"]:
                structure["keyword_argument_map"][keyword].add(idx)

            argument_keywords.append(" ".join(arg["keywords"]))
            arguments.append(idx)

            del arg["keywords"]

        if len(structure["args"]) <= 1:
            continue

        structure["argument_pipeline"].fit(argument_keywords, arguments)

    if len(commands) > 1:
        action_pipeline.fit(action_keywords, actions)
except Exception as e:
    raise RuntimeError(f"Error mapping commands keywords to actions: {e}")

try:
    model = {
        "commands": commands,
        "keyword_action_map": keyword_action_map,
        "action_pipeline": action_pipeline
    }
    with open(settings.parser_model_dir, "wb") as file:
        joblib.dump(model, file)
except Exception as e:
    raise RuntimeError(f"Error saving parser model: {e}")