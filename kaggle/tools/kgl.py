"""Run a kaggle CLI command with a fresh in-memory access token (credentials.json untouched, token never printed)."""
import os, subprocess, sys
from kagglesdk import KaggleClient
from kagglesdk.kaggle_creds import KaggleCredentials
with KaggleClient() as client:
    token = KaggleCredentials.load(client=client).generate_access_token().token
env = dict(os.environ, KAGGLE_API_TOKEN=token)
sys.exit(subprocess.run(["kaggle", *sys.argv[1:]], env=env).returncode)
