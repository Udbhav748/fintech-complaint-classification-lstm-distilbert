"""One-time-per-change differential test: does `src.keras_tokenizer.KerasTokenizer`
match the real `tf.keras.preprocessing.text.Tokenizer`?

Requires TensorFlow, which is intentionally NOT a normal project dependency (see
`src/keras_tokenizer.py` module docstring). Run this in a throwaway environment
that has `tensorflow` installed, e.g.:

    python -m venv /tmp/tf_check && /tmp/tf_check/bin/pip install tensorflow-cpu pandas numpy
    /tmp/tf_check/bin/python scripts/verify_keras_tokenizer_equivalence.py

Not part of `tests/` because the normal test suite must run without TensorFlow
installed. Re-run this whenever `src/keras_tokenizer.py`'s fit/transform logic
changes; the last recorded result is committed at
`artifacts/preprocessing/keras_tokenizer_equivalence.json`.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import tensorflow as tf
except ImportError:
    print("tensorflow is not installed in this environment - see module docstring.")
    sys.exit(2)

from src.keras_tokenizer import KerasTokenizer
from src.keras_tokenizer import pad_sequences as our_pad_sequences

MAX_FEATURES = 20000
OOV_TOKEN = "<OOV>"
FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

CASES = {
    "plain": [
        "the quick brown fox",
        "the lazy dog sleeps",
        "the fox and the dog",
    ],
    "ties_in_frequency": [
        "alpha beta gamma",
        "delta epsilon zeta",
        "eta theta iota",
    ],
    "punctuation_heavy": [
        "Debt-collector called re: acct #12345, said 'pay now!!' (urgent)",
        "I dispute this; it's not mine -- please investigate ASAP.",
    ],
    "xxxx_redaction": [
        "XXXX XXXX charged me on XX/XX/2024 for {$150.00}",
        "Called XXXX XXXX XXXX about my XXXX account XXXX times",
    ],
    "digits_and_amounts": [
        "I was charged $1,500.00 on account 4829103756",
        "Balance of 250.75 dollars overdue since 2023",
    ],
    "mixed_case": [
        "URGENT Complaint About My Credit Card ACCOUNT",
        "chime WONT let me Access my Money at ALL",
    ],
    "unicode": [
        "café résumé naïve über façade",
        "the café is closed — I paid ’ already",
    ],
    "empty_and_whitespace": [
        "",
        "   ",
        "single",
    ],
    "cfpb_realistic": [
        "I am disputing this account as inaccurate and request full validation",
        "This debt collector has engaged in conduct that violates the FDCPA",
        "My Chime account was compromised and multiple charges were made",
        "Navient failed to process my income driven repayment application",
    ],
}


def run_case(texts):
    ours = KerasTokenizer(num_words=MAX_FEATURES, oov_token=OOV_TOKEN, filters=FILTERS, lower=True)
    ours.fit_on_texts(texts)

    theirs = tf.keras.preprocessing.text.Tokenizer(
        num_words=MAX_FEATURES, oov_token=OOV_TOKEN, filters=FILTERS, lower=True
    )
    theirs.fit_on_texts(texts)

    issues = []
    if ours.word_index != theirs.word_index:
        issues.append("word_index differs")
    if ours.word_counts != dict(theirs.word_counts):
        issues.append("word_counts differs")

    our_seqs = ours.texts_to_sequences(texts)
    their_seqs = theirs.texts_to_sequences(texts)
    if our_seqs != their_seqs:
        issues.append(f"texts_to_sequences differs: ours={our_seqs} theirs={their_seqs}")

    holdout = ["completely novel words never seen before qwertyzxcv"]
    if ours.texts_to_sequences(holdout) != theirs.texts_to_sequences(holdout):
        issues.append("OOV texts_to_sequences differs")

    return issues


def run_padding_case():
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
    ours = our_pad_sequences(seqs, max_len=4, padding="pre", truncating="pre").tolist()
    theirs = tf.keras.preprocessing.sequence.pad_sequences(
        seqs, maxlen=4, padding="pre", truncating="pre", value=0
    ).tolist()
    return ours == theirs, ours, theirs


def main() -> int:
    results = {}
    all_pass = True
    for name, texts in CASES.items():
        issues = run_case(texts)
        results[name] = {"pass": not issues, "issues": issues}
        print(f"[{'PASS' if not issues else 'FAIL'}] {name}")
        all_pass = all_pass and not issues

    pad_ok, our_pad, their_pad = run_padding_case()
    results["pad_sequences_pre_pre_matches_keras"] = {"pass": pad_ok, "ours": our_pad, "theirs": their_pad}
    print(f"[{'PASS' if pad_ok else 'FAIL'}] pad_sequences (padding=pre, truncating=pre) matches tf.keras")
    all_pass = all_pass and pad_ok

    print(f"\ntensorflow version: {tf.__version__}")
    print("OVERALL:", "PASS" if all_pass else "FAIL")

    out_path = REPO_ROOT / "artifacts" / "preprocessing" / "keras_tokenizer_equivalence.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"tensorflow_version": tf.__version__, "overall_pass": all_pass, "cases": results}, indent=2),
        encoding="utf-8",
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
