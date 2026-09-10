"""Minimal demo UI for the CFPB complaint classifier (M4, best recurrent model).

Loads the M4 checkpoint (checkpoints/M4_seed42.weights.h5) and classifies a
complaint typed into the browser. Not part of the training/evaluation
pipeline - a standalone way to see the model work on new text.

Uses M4 rather than D0: D0's saved checkpoint bakes in AdamW optimizer state
that tf_keras's H5 loader cannot reconcile against a freshly built model in a
separate process (a confirmed, documented limitation - see scripts/run_d0.py,
validate_run()). M4's checkpoint has no such optimizer group and reloads
cleanly, the same way scripts/run_m4.py loads it.
"""

import numpy as np
import streamlit as st

from src.data import LABELS, SHORT_LABELS
from src.keras_tokenizer import KerasTokenizer, pad_sequences
from src.models import RecurrentModelConfig, build_recurrent_model
from src.preprocessing import normalize_text

WEIGHTS_PATH = "checkpoints/M4_seed42.weights.h5"
TOKENIZER_PATH = "artifacts/tokenizer/keras_tokenizer.json"
MAX_LEN = 256

EXAMPLES = {
    "Debt collection": (
        "A collection agency keeps calling me about a medical bill I already paid off last year. "
        "I sent them proof of payment twice and they still won't stop calling."
    ),
    "Student loan": (
        "My loan servicer moved my account without telling me and now my autopay discount is gone. "
        "I have called three times and no one can explain why my interest rate went up."
    ),
    "Money transfer": (
        "I sent money through the app to a friend and it has been stuck pending for a week. "
        "Support will not tell me where the funds actually are."
    ),
}

st.set_page_config(page_title="CFPB Complaint Classifier", page_icon=None, layout="centered")

st.markdown(
    """
    <style>
    .block-container { max-width: 760px; padding-top: 3rem; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = KerasTokenizer.load(TOKENIZER_PATH)
    config = RecurrentModelConfig.from_experiment("M4")
    embedding_matrix = np.zeros((config.vocab_size, config.embedding_dim), dtype=np.float32)
    model = build_recurrent_model(config, embedding_matrix=embedding_matrix)
    model.load_weights(WEIGHTS_PATH)
    return tokenizer, model


def predict(text, tokenizer, model):
    cleaned = normalize_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, max_len=MAX_LEN, padding="pre", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    return probs


st.title("CFPB Complaint Classifier")
st.caption("BiLSTM + GloVe, LR schedule, early stopping (M4) · macro-F1 0.871 on a frozen 10,181-complaint test set")

with st.spinner("Loading model..."):
    tokenizer, model = load_model()

st.write("Paste a consumer complaint below and classify it into one of 5 product categories.")

cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state["complaint_text"] = text

complaint = st.text_area(
    "Complaint text",
    key="complaint_text",
    height=140,
    placeholder="e.g. My bank closed my checking account without any notice and I still haven't gotten my funds back...",
    label_visibility="collapsed",
)

classify = st.button("Classify", type="primary", use_container_width=True)

if classify and complaint.strip():
    probs = predict(complaint, tokenizer, model)
    top_idx = int(np.argmax(probs))
    top_label = LABELS[top_idx]

    st.divider()
    st.metric("Predicted category", top_label, f"{probs[top_idx]:.1%} confidence")

    st.write("Class probabilities")
    order = np.argsort(probs)[::-1]
    for i in order:
        st.write(SHORT_LABELS[LABELS[i]])
        st.progress(float(probs[i]))
elif classify:
    st.warning("Enter a complaint first.")

st.divider()
st.caption(
    "Predictions come from a saved checkpoint on a frozen test split, "
    "part of a larger model comparison. See the README for the full results."
)
