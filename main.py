from flask import Flask, request
from main_function import ssd_rtdb

app = Flask(__name__)


@app.route('/')
def index():
    user_id = "user2"
    ssd_rtdb(user_id)
    return "<h1>Welcome to CodingX</h1>"


@app.route('/rec_choice/<user_id>', methods=['POST'])
def generate(user_id):
    if request.method == 'POST':
        if user_id:
            ssd_rtdb(user_id)
            return "<h1>Success</h1>"
        else:
            return "<h1>Failed</h1>"
    return "<h1>Incorrect</h1>"
