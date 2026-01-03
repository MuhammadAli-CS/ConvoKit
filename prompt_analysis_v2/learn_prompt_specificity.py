# learn_prompt_specificity.py
import pandas as pd
import numpy as np
from pathlib import Path
from convokit import Corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import joblib

CORPUS_DIR = "prompt_only_corpus_annotated"
MODEL_OUT = Path("models/specificity_model.joblib")
DATA_OUT = Path("data/prompt_with_specificity.csv")

def build_training_df(corpus: Corpus) -> pd.DataFrame:
    rows = []
    for convo in corpus.iter_conversations():
        utts = list(convo.iter_utterances())
        utts.sort(key=lambda u: int(u.meta.get("action_index", 0) or 0))

        for t, u in enumerate(utts):
            text = (u.text or "").strip()
            rows.append({
                "conversation_id": convo.id,
                "timestep": t,
                "text": text,
                "prompt_length": int(u.meta.get("prompt_length", 0) or 0),
                "num_constraints_rule": int(u.meta.get("num_constraints_rule", 0) or 0),
                "descriptive_words": int(u.meta.get("descriptive_words", 0) or 0),
            })
    return pd.DataFrame(rows)

def make_weak_label(df: pd.DataFrame) -> np.ndarray:
    # weak target: mostly constraints, plus descriptiveness, plus a little length
    y = (
        df["num_constraints_rule"].values.astype(float)
        + 0.25 * df["descriptive_words"].values.astype(float)
        + 0.05 * df["prompt_length"].values.astype(float)
    )
    # normalize so values are comparable across sessions
    y = (y - y.mean()) / (y.std() + 1e-9)
    return y

def main():
    corpus = Corpus(filename=CORPUS_DIR)
    df = build_training_df(corpus)
    if len(df) < 10:
        raise RuntimeError("Not enough prompts to train a model (need at least ~10).")

    y = make_weak_label(df)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )),
        ("ridge", Ridge(alpha=1.0))
    ])

    pipe.fit(df["text"].values, y)
    preds = pipe.predict(df["text"].values)

    df["specificity_pred"] = preds

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, MODEL_OUT)
    df.to_csv(DATA_OUT, index=False)

    print(f"✅ Trained specificity model -> {MODEL_OUT.resolve()}")
    print(f"✅ Wrote prompt_with_specificity.csv -> {DATA_OUT.resolve()}")

if __name__ == "__main__":
    main()
