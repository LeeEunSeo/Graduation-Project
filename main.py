from flask import Flask
from main_function import ssd_rtdb

app = Flask(__name__)


@app.route('/')
def index():
    user_id = "user2"
    ssd_rtdb(user_id)
    return "<h1>Welcome to CodingX</h1>"
