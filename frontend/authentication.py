from functools import wraps
from time import time

from flask import request
from flask_restplus import abort
from itsdangerous import SignatureExpired, JSONWebSignatureSerializer, BadSignature


default_user_name = 'admin'
default_user_password = 'admin'


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


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('AUTH-TOKEN')
        if not token:
            abort(404, 'Authentication token is missing')

        try:
            user = auth.validate_token(token)
        except SignatureExpired as e:
            abort(404, e.message)
        except BadSignature as e:
            abort(404, e.message)

        return f(*args, **kwargs)

    return decorated


def generate_token(username):
    return auth.generate_token(username)


def validate_user(username, password):
    if username == default_user_name and password == default_user_name:
        return True
    return False
