from flask import Flask, url_for
from flask import json
from flask import request
from flask import Response, jsonify

from models import *

m = TestModel()
app = Flask(__name__)


@app.route('/predict', methods = ["POST"])
def api_articles():
    print("Request: " + json.dumps(request.json))

    data = request.get_json()
    question = data['question']
    response = data['questionResponse']
    model_id = data['modelId']

    m.set_model(model_id)
    if model_id is "C":
        [prediction, prob] = m.predict(question, response)
        response_data = {
            "score": prediction,
            "probability": str(prob)
        }
    else:
        prediction = float(m.predict(question, response))
        score = min([0, 0.5, 1], key=lambda x: abs(x - prediction))
        response_data = {
            "score": score,
            "probability": None
        }
    print(response_data)
    js = json.dumps(response_data)

    return Response(js, status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run(port=8080)