import pandas as pd
from flask import Flask, request, jsonify, abort
from functools import wraps
from time import time
import hashlib
import base64

app = Flask(__name__)

user_age_dict = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
user_job_dict = {
    0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin", 4: "college/grad student",
    5: "customer service", 6: "doctor/health care", 7: "executive/managerial", 8: "farmer", 9: "homemaker",
    10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist",
    16: "self-employed", 17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
}


# =========================== authentication ===========================
userList = {'steven': '111', "neo": '111', 'krist': '111', "tracy": '111', 'test': 'test', 'admin': 'admin'}
test = ""
user_auth = {}


def generate_token(username, secret_key, expires_in=600):
    timestamp = str(int(time()))
    data_to_sign = username + timestamp
    signature = hashlib.sha256(data_to_sign.encode() + secret_key.encode()).digest()
    token = base64.b64encode(data_to_sign.encode() + signature).decode()
    return token


def validate_token(token, secret_key, expires_in=600):
    try:
        decoded_token = base64.b64decode(token.encode())
        username = decoded_token[:-32].decode()
        signature = decoded_token[-32:]
        data_to_sign = username + str(int(time()))
        expected_signature = hashlib.sha256(data_to_sign.encode() + secret_key.encode()).digest()

        if not base64.b64encode(expected_signature).decode() == base64.b64encode(signature).decode():
            return None

        return username
    except Exception as e:
        return None


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        username = test
        if not username:
            abort(404, 'Authentication token is missing')
        if username not in userList:
            abort(401, "Authentication failed")

        username_from_token = validate_token(user_auth[username], 'simon2929')
        if username_from_token is None or username_from_token != username:
            abort(404, 'Token is invalid')

        return f(*args, **kwargs)

    return decorated


@app.route('/', methods=['GET'])
def index():
    return "<h1>This is the movie recommendation api</h1>"


@app.route('/add_user', methods=['GET'])
def add_user():
    username = request.args.get('account')
    pwd = request.args.get('pwd')
    userList[username] = pwd
    print(userList)
    return jsonify({"url": "login.html"})


@app.route('/token', methods=['GET'])
def get_token():
    global test
    test = request.args.get('username')
    password = request.args.get('password')
    if test in userList.keys() and userList[test] == password:
        user_auth[test] = generate_token(test, "simon5427")
        return jsonify({'code': '200'})

    return jsonify({'code': '400'})

# ======================================================================



# change users_info from [[], [],...] to [{}, {},...]
def process_users_info(users_info):
    processed_users_info = []
    for user in users_info:
        user_info_dict = {}
        user_info_dict['user_id'] = user[0]
        user_info_dict['gender'] = user[1]
        user_info_dict['age'] = user_age_dict[user[2]]
        user_info_dict['job'] = user_job_dict[user[3]]
        processed_users_info.append(user_info_dict)
    return processed_users_info


if __name__ == '__main__':
    app.run(port=9999, debug=True)
