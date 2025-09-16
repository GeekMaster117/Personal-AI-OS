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
    vectorizer = CountVectorizer()
    classifier = SGDClassifier(loss="log_loss")
    pipeline = make_pipeline(vectorizer, classifier)
except Exception as e:
    raise RuntimeError(f"Error initialising pipeline: {e}")

try:
    keyword_action_map: dict = defaultdict(set)
    keywords, actions = [], []

    for action, structure in commands.items():
        keywords.append(" ".join(structure["keywords"]))
        actions.append(action)

        for keyword in structure["keywords"]:
            keyword_action_map[keyword].add(action)

    pipeline.fit(keywords, actions)
except Exception as e:
    raise RuntimeError(f"Error mapping commands keywords to actions: {e}")

try:
    model = {
        "commands": commands,
        "keyword_action_map": keyword_action_map,
        "pipeline": pipeline,
        "vectorizer": vectorizer,
        "classifier": classifier
    }
    with open(settings.parser_model_dir, "wb") as file:
        joblib.dump(model, file)
except Exception as e:
    raise RuntimeError(f"Error saving parser model: {e}")