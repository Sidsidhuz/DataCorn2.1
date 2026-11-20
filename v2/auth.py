"""Admin-provisioned accounts for private deployments; no plaintext password storage."""
import base64
import hashlib
import hmac
import os
import re
import secrets
import time
import uuid
from .errors import ProductError
from .storage import utcnow

ITERATIONS = 600_000


def password_hash(password):
    if len(password) < 12 or len(password) > 1024:
        raise ProductError('Use a password between 12 and 1,024 characters.')
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, ITERATIONS)
    return f'pbkdf2_sha256${ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}'


def verify_password(password, encoded):
    try:
        algorithm, iterations, salt, expected = encoded.split('$')
        if algorithm != 'pbkdf2_sha256' or not 100_000 <= int(iterations) <= 2_000_000 or len(password) > 1024:
            return False
        digest = hashlib.pbkdf2_hmac('sha256', password.encode(), base64.b64decode(salt), int(iterations))
        return hmac.compare_digest(digest, base64.b64decode(expected))
    except (ValueError, TypeError):
        return False


def create_user(store, username, password):
    username = username.strip().lower()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]{2,63}', username):
        raise ProductError('Username must be 3–64 letters, numbers, dots, underscores, or hyphens.')
    encoded = password_hash(password)
    with store.connect() as db:
        if db.execute('SELECT 1 FROM users WHERE username=?', (username,)).fetchone():
            raise ProductError('This username already exists.')
        identifier = uuid.uuid4().hex
        db.execute('INSERT INTO users VALUES (?,?,?,?)', (identifier, username, encoded, utcnow()))
    return identifier


def bootstrap_from_environment(store):
    username = os.getenv('AUTOINSIGHT_ADMIN_USER')
    password = os.getenv('AUTOINSIGHT_ADMIN_PASSWORD')
    if username and password:
        with store.connect() as db:
            exists = db.execute('SELECT 1 FROM users WHERE username=?', (username.strip().lower(),)).fetchone()
        if not exists:
            create_user(store, username, password)


def authenticate(store, username, password, now=None):
    now = time.time() if now is None else now
    username = username.strip().lower()[:64]
    # Reserve an attempt atomically before expensive verification. Parallel
    # requests cannot repeatedly read the same old failure count.
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('DELETE FROM attempts WHERE last_attempt<?', (now-86400,))
        attempt = db.execute('SELECT * FROM attempts WHERE username=?', (username,)).fetchone()
        if attempt and attempt['count'] >= 5 and now - attempt['last_attempt'] < 60:
            raise ProductError('Too many attempts. Try again in a minute.', 'rate_limited')
        row = db.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
        count = attempt['count'] + 1 if attempt and now - attempt['last_attempt'] < 60 else 1
        db.execute('INSERT INTO attempts VALUES (?,?,?) ON CONFLICT(username) DO UPDATE SET count=excluded.count,last_attempt=excluded.last_attempt', (username, count, now))
    dummy = 'pbkdf2_sha256$600000$AAAAAAAAAAAAAAAAAAAAAA==$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA='
    valid = verify_password(password, row['password_hash'] if row else dummy)
    if valid and row:
        with store.connect() as db:
            db.execute('DELETE FROM attempts WHERE username=?', (username,))
        return {'id': row['id'], 'username': row['username'], 'expires_at': now + 8 * 3600}
    raise ProductError('Incorrect username or password.', 'authentication_failed')
