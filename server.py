from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
import json
import models


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    """Adapted from https://blog.anvileight.com/posts/simple-python-http-server/"""
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello, world!')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()

        data = json.loads(body) # save json data
        model = models.TestModel(data['modelId'])
        question = data['question']
        response = data['questionResponse']
        prediction = model.predict(question, response)

        score = min([0, 0.5, 1], key=lambda x: abs(x - prediction))
        response = {
            "score": score
        }
        self.wfile.write(json.dumps(response).encode())


httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()