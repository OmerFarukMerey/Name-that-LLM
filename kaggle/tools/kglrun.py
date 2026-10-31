"""Run any command with a fresh in-memory Kaggle access token in KAGGLE_API_TOKEN (credentials.json untouched, token never printed).

    python kglrun.py <command> [args...]      e.g. python kglrun.py python script.py user/slug V28
"""
import os, subprocess, sys
from kagglesdk import KaggleClient
from kagglesdk.kaggle_creds import KaggleCredentials
with KaggleClient() as client:
    token = KaggleCredentials.load(client=client).generate_access_token().token
env = dict(os.environ, KAGGLE_API_TOKEN=token)
sys.exit(subprocess.run(sys.argv[1:], env=env).returncode)
