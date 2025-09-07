import os

import json
import joblib

from numpy import ndarray
from collections import defaultdict, Counter

from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, make_pipeline

class Parser:
    commands_path: str = "commands.json"
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

    def _load_vectorizer(self) -> CountVectorizer | None:
        if not os.path.exists(Parser.vectorizer_path):
            return None
        return joblib.load(Parser.vectorizer_path)
    
    def _load_classifier(self) -> SGDClassifier | None:
        if not os.path.exists(Parser.classifier_path):
            return None
        return joblib.load(Parser.classifier_path)
    
    def _load_model(self) -> tuple:
        vectorizer = self._load_vectorizer()
        classifier = self._load_classifier()

        if not vectorizer or not classifier:
            vectorizer = CountVectorizer()
            classifier = SGDClassifier(loss="log_loss")
            model = self._init_train(vectorizer, classifier), vectorizer, classifier
            self._save_model(vectorizer, classifier)

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
    
    def extract_keywords(self, tokens: list[str], probability_cutoff: float) -> list[str]:
        if probability_cutoff < 0 or probability_cutoff > 1:
            raise ValueError(f"Probability cutoff must be in the interval [0, 1], value passed: {probability_cutoff}")

        keywords = []
        for token in tokens:
            keyword = process.extractOne(token, self.keyword_action_map.keys(), scorer=fuzz.ratio, score_cutoff=probability_cutoff * 100)
            if keyword:
                keywords.append(keyword[0])
        
        return keywords

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

    def extract_most_probable_action(self, actions_normalised: dict, probability_cutoff: float) -> str | None:
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
        self._classifier.partial_fit(X, [action], classes=list(self.commands.keys()))

        self._save_model(self._vectorizer, self._classifier)

parser: Parser = Parser()

query: str = input("Enter request: ")
tokens: list[str] = query.lower().split()

keywords: list[str] | None = None
while True:
    keywords = parser.extract_keywords(tokens, 0.85)
    if keywords:
        break
    
    query = input("Unable to parse. Enter again: ")
    tokens = query.lower().split()

actions_normalised: dict = parser.extract_actions_normalised(keywords)

possible_action = parser.extract_most_probable_action(actions_normalised, 0.85)
if not possible_action:
    actions = parser.extract_actions_classification(keywords, 5)
    if actions[0][1] >= 0.85:
        possible_action = actions[0][0]
    else:
        print("What do you want to do?")
        for i, action in enumerate(actions):
            print(f"{i + 1}. {parser.commands[action[0]]["description"]}")
        answer = int(input(f"Enter answer ({1}-{len(actions)}): "))

        parser.train(keywords, actions[answer - 1][0])
        possible_action = actions[answer - 1][0]

if parser.commands[possible_action]["warning"] == True:
    answer = input(f"Do you want to, {parser.commands[possible_action]["description"]} (Y/N): ").lower()
    if answer != 'y':
        print("Skipping request...")
        exit()

#Todo: Execute Command