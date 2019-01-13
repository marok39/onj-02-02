from models import *
import requests
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def test_server():
    request_json = {
        "modelId": "A",
        "question": "How does Shiranna feel as the shuttle is taking off?",
        "questionResponse": "She is a little bit nervous."
    }

    r = requests.post("http://localhost:8000", json=request_json)
    print(r.text)


def test_modelA():
    data = pd.read_csv('input/Weightless_dataset_train.csv', sep=",")
    question_data = [q.strip() for q in data.loc[:, "Question"]]
    response_data = data.loc[:, "Response"]
    ground_truth = [str(r.replace(',', '.')) for r in data.loc[:, "Final.rating"]]

    model = BasicModel()
    predictions = [model.predict(q, r) for q, r in zip(question_data, response_data)]
    #print(ground_truth)
    #print(predictions)

    f_macro = f1_score(ground_truth, predictions, average='macro')
    f_micro = f1_score(ground_truth, predictions, average='micro')

    return f_macro, f_micro


def test_SVM():
    data = pd.read_csv('input/Weightless_dataset_train.csv', sep=",")
    question_data = [q.strip() for q in data.loc[:, "Question"]]
    response_data = data.loc[:, "Response"]
    ground_truth = [str(r.replace(',', '.')) for r in data.loc[:, "Final.rating"]]

    questions = set(question_data)

    X = [r for r in response_data]
    y = ground_truth

    X_train, X_test, y_train, y_test, q_train, q_test = train_test_split(X, y, question_data, test_size=0.2, random_state=23)

    model = SVM()
    model.fit(X_train, y_train, q_train)
    #model.fit(X, y)
    #model.fit(X, y, question_data)
    predictions = model.predict(q_test, X_test)
    #print(predictions)

    # for i, p in enumerate(predictions): print(q_test[i], y_test[i], p);

    f_macro = f1_score(y_test, predictions, average='macro')
    f_micro = f1_score(y_test, predictions, average='micro')
    f_classes = f1_score(y_test, predictions, average=None)

    return f_macro, f_micro, f_classes


#test_server()

res = test_SVM()
print("Macro:", res[0], "Micro:", res[1], "Per class:", res[2])