"""Competition data loading and the fixed label set."""

import pandas as pd

from ntl.paths import DATA_DIR

# The 18 valid labels, in the order of the Evaluation page.
PROVIDERS = {
    "gpt-5.5": "OpenAI",
    "gpt-5.6-luna": "OpenAI",
    "gpt-5.6-sol": "OpenAI",
    "gpt-5.6-terra": "OpenAI",
    "gpt-6-astra": "OpenAI",
    "gemini-3.1-pro-preview": "Google",
    "gemini-3.6-flash": "Google",
    "gemini-3.7-flash": "Google",
    "gemini-3.8-flash": "Google",
    "claude-fable-5": "Anthropic",
    "claude-opus-4-6": "Anthropic",
    "claude-opus-4-8": "Anthropic",
    "claude-opus-5": "Anthropic",
    "claude-sonnet-5": "Anthropic",
    "qwen3.8-max": "Qwen",
    "glm-5.3": "ZAI",
    "glm-5.3-flash": "ZAI",
    "kimi-k3": "Moonshot AI",
}
LABELS = tuple(PROVIDERS)
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def _read(name):
    # dtype=str keeps hex ids such as "1e5..." from being parsed as numbers.
    return pd.read_csv(DATA_DIR / name, dtype=str, keep_default_na=False)


def load_train():
    """Columns: id, prompt_id, text, model."""
    return _read("train.csv")


def load_test():
    """Columns: id, text. Three rows locally; ~4000 in the hidden rerun."""
    return _read("test.csv")
