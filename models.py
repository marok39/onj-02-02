import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
import numpy as np
from sklearn import svm

from Cnn import *


class BasicModel:
    def __init__(self, data='input/Weightless_dataset_train_A.csv'):
        data = pd.read_csv('input/Weightless_dataset_train_A.csv', sep=",")
        inference_data = [clean_sentence(s) for s in data.loc[:, "Text.used.to.make.inference"]]
        response_data = [clean_sentence(s) for s in data.loc[:, "Response"]]
        question_data = [q.strip() for q in data.loc[:, "Question"]]
        # print(inference_data, question_data, response_data)

        self.correct_answers = {}
        for q, r in zip(question_data, response_data):
            self.correct_answers[q] = r
        self.model = Word2Vec.load("basic_model.model")

    def predict(self, question, response):
        v1 = avg_sentence_vector(self.correct_answers[question], self.model, 100)
        response = clean_sentence(response)
        v2 = avg_sentence_vector(response, self.model, 100)
        c = abs(cosine_similarity(v1, v2))
        if 0.12 < c < 0.45:
            return str(0.5)
        score = min([0, 1], key=lambda x: abs(x - c))
        return str(float(score))


class SVM:
    def __init__(self, data_path='input/Weightless_dataset_train.csv'):
        data = pd.read_csv(data_path, sep=",")
        self.model = Word2Vec.load("weightless_text.model")
        question_data = [q.strip() for q in data.loc[:, "Question"]]
        questions = {}
        idx = 0
        for q in question_data:
            if q not in questions:
                questions[q] = idx
                idx += 1
        self.questions = questions
        self.models = {}
        for q in self.questions:
            self.models[q] = svm.SVC(
                kernel="rbf", C=1e3, gamma=1e3, decision_function_shape="ovr", class_weight="balanced", probability=True
            )
        self.clf = svm.SVC(
            kernel="rbf", C=1e3, gamma=1e3, decision_function_shape="ovr", class_weight="balanced", probability=True
        )

    def fit(self, X, y, q=None):
        if q:
            question = {}
            for i in range(len(y)):
                if q[i] not in question:
                    question[q[i]] = {
                        'X': [X[i]],
                        'y': [y[i]]
                    }
                else:
                    question[q[i]]['X'].extend([X[i]])
                    question[q[i]]['y'].extend([y[i]])
            for q in self.questions:
                vs = [avg_sentence_vector(x, self.model, 100) for x in question[q]['X']]
                self.models[q].fit(vs, question[q]['y'])
        else:
            vectors = [avg_sentence_vector(x, self.model, 100) for x in X]
            self.clf.fit(vectors, y)

    def predict(self, question, response):
        vectors = [avg_sentence_vector(r, self.model, 100) for r in response]
        if question:
            return [self.models[q].predict([vectors[i]])[0] for i, q in enumerate(question)]
        else:
            return self.clf.predict(vectors)


class TestModel:
    def __init__(self):
        print("init models")
        self.cnn = Cnn()
        self.svm = SVM()
        self.svm.fit()
        self.basic_model = BasicModel()
        self.model = None

    def set_model(self, model_id):
        model_choices = {
            "A": self.basic_model,
            "B": self.svm,
            "C": self.cnn
        }
        self.model = model_choices[model_id]

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
