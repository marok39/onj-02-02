from models import *
import requests
import pandas as pd
from sklearn.metrics import f1_score

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
    print(ground_truth)
    print(predictions)

    f_macro = f1_score(ground_truth, predictions, average='macro')
    f_micro = f1_score(ground_truth, predictions, average='micro')

    return f_macro, f_micro


test_server()

res = test_modelA()
print(res)