import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
import numpy as np
from sklearn import svm


class BasicModel:
    def __init__(self, data='input/Weightless_dataset_train_A.csv'):
        data = pd.read_csv('input/Weightless_dataset_train_A.csv', sep=",")
        inference_data = [clean_sentence(s) for s in data.loc[:, "Text.used.to.make.inference"]]
        response_data = [clean_sentence(s) for s in data.loc[:, "Response"]]
        question_data = data.loc[:, "Question"]
        # print(inference_data, question_data, response_data)

        self.correct_answers = {}
        for q, r in zip(question_data, response_data):
            self.correct_answers[q] = r
        self.model = Word2Vec(inference_data, min_count=1)
        self.model.save("basic_model.model")

    def predict(self, question, response):
        v1 = avg_sentence_vector(self.correct_answers[question], self.model, 100)
        response = clean_sentence(response)
        v2 = avg_sentence_vector(response, self.model, 100)
        c = abs(cosine_similarity(v1, v2))
        if 0.37 < c < 0.63:
            return str(0.5)
        score = min([0, 1], key=lambda x: abs(x - c))
        return str(float(score))


class SVM:
    def __init__(self, data='input/Weightless_dataset_train.csv'):
        data = pd.read_csv('input/Weightless_dataset_train.csv', sep=",")
        self.model = Word2Vec.load("weightless_text.model")
        self.clf = svm.SVC(kernel="rbf", gamma=1e3, decision_function_shape="ovr", class_weight="balanced")

    def fit(self, X, y):
        vectors = [avg_sentence_vector(x, self.model, 100) for x in X]
        self.clf.fit(vectors, y)

    def predict(self, question, response):
        vectors = [avg_sentence_vector(r, self.model, 100) for r in response]
        return self.clf.predict(vectors)


class TestModel:
    def __init__(self, model_choice):
        # TODO handle different models
        model_choices = {
            "A": BasicModel,
            "B": SVM,
            "C": BasicModel
        }
        self.model = model_choices[model_choice]()

    def predict(self, question, response):
        return self.model.predict(question, response)


def avg_sentence_vector(words, model, num_features):
    feature_vec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in model.wv.index2word:
            nwords = nwords + 1
            feature_vec = np.add(feature_vec, model[word])

    if nwords > 0:
        feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


def clean_sentence(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    sentence = tokenizer.tokenize(sentence)
    return [w.lower() for w in sentence]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
