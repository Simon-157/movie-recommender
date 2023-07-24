from functools import wraps
from time import time

import pandas as pd
from flask import Flask
from flask import request
from flask_restplus import Resource, Api, abort
from flask_restplus import fields
from flask_restplus import inputs
from flask_restplus import reqparse
from itsdangerous import SignatureExpired, JSONWebSignatureSerializer, BadSignature

test = 'neo'
l = {}
list = ["steven", "neo", "tracy", "chris"]



class AuthenticationToken:
    def __init__(self, secret_key, expires_in):
        self.secret_key = secret_key
        self.expires_in = expires_in
        self.serializer = JSONWebSignatureSerializer(secret_key)

    def generate_token(self, username):
        info = {
            'username': username,
            'creation_time': time()
        }

        token = self.serializer.dumps(info)
        return token.decode()

    def validate_token(self, token):
        info = self.serializer.loads(token.encode())

        if time() - info['creation_time'] > self.expires_in:
            raise SignatureExpired("The Token has been expired; get a new token")

        return info['username']


SECRET_KEY = "A SECRET KEY; USUALLY A VERY LONG RANDOM STRING"
expires_in = 600
auth = AuthenticationToken(SECRET_KEY, expires_in)

app = Flask(__name__)
api = Api(app, security='API-KEY',
          default="Books",  # Default namespace
          title="Book Dataset",  # Documentation Title
          description="This is just a simple example to show how publish data as a service.")  # Documentation Description


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # token = request.headers.get('AUTH-TOKEN')
        # print(token)
        # request.headers.insert("AUTH_TOKEN",)
        token = test

        print("aaa", token)
        if not token:
            abort(401, 'Authentication token is missing')
        if token not in l:
            abort(401, 'Authentication failed')

        try:
            user = auth.validate_token(l[token])
            print(token)
            print(request.headers)
        except SignatureExpired as e:
            print(token)
            abort(401, e.message)
        except BadSignature as e:
            print(token)
            abort(401, e.message)

        return f(*args, **kwargs)

    return decorated


parser = reqparse.RequestParser()
credential_model = api.model('credential', {
    'username': fields.String
})

credential_parser = reqparse.RequestParser()
credential_parser.add_argument('username', type=str)


@api.route('/token')
class Token(Resource):
    @api.response(200, 'Successful')
    @api.doc(description="Generates a authentication token")
    @api.expect(credential_parser, validate=True)
    def get(self):
        args = credential_parser.parse_args()

        username = args.get('username')
        global test
        test = username
        # list.append(username)
        # password = args.get('password')

        if username in list:

            a = auth.generate_token(username)
            l[username] = a
            print(l)
            print(request.headers)
            return {"token": auth.generate_token(username)},username

        return {"message": "authorization has been refused for those credentials."}, 401
