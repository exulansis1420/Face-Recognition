from flask import Flask, request, jsonify
# import util

app = Flask(__name__)


@app.route('/hello')
def hello():
    return "hi"


if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    app.run(port=5000)