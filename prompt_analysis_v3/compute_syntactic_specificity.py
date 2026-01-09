import pandas as pd
import spacy
from pathlib import Path

IN = Path("data/prompt_trajectories.csv")
OUT = Path("data/prompt_with_specificity.csv")

nlp = spacy.load("en_core_web_sm")

def syntactic_features(text: str):
    doc = nlp(text or "")
    tokens = [t for t in doc if not t.is_space]
    n = max(len(tokens), 1)

    pos = {}
    dep = {}

    for t in tokens:
        pos[t.pos_] = pos.get(t.pos_, 0) + 1
        dep[t.dep_] = dep.get(t.dep_, 0) + 1

    adj_rate = pos.get("ADJ", 0) / n
    noun_rate = (pos.get("NOUN", 0) + pos.get("PROPN", 0)) / n
    num_rate = pos.get("NUM", 0) / n
    neg_rate = dep.get("neg", 0) / n

    specificity = (
        1.5 * adj_rate +
        1.0 * noun_rate +
        2.0 * num_rate +
        1.0 * neg_rate
    )

    return pd.Series({
        "adj_rate": adj_rate,
        "noun_rate": noun_rate,
        "num_rate": num_rate,
        "neg_rate": neg_rate,
        "syntactic_specificity": specificity
    })

def main():
    df = pd.read_csv(IN)
    feats = df["text"].apply(syntactic_features)
    out = pd.concat([df, feats], axis=1)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"✅ Wrote {OUT}")

if __name__ == "__main__":
    main()
