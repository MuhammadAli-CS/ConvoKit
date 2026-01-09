import pandas as pd
import numpy as np
from convokit import Corpus

CORPUS_DIR = r"C:\Users\Ali\.convokit\saved-corpora\prompt_only_corpus_annotated_v2"

def build_trajectory_dataframe(corpus: Corpus) -> pd.DataFrame:
    rows = []
    for convo in corpus.iter_conversations():
        utts = list(convo.iter_utterances())

        # Stable order: use action_index if you stored it, else keep insertion order
        utts.sort(key=lambda u: u.meta.get("action_index", u.meta.get("timestep", 0)))

        for t, u in enumerate(utts):
            rows.append({
                "conversation_id": convo.id,
                "timestep": t,
                "text": u.text,  #  REQUIRED for scikit-learn
                "prompt_intent": u.meta.get("prompt_intent_v2", "other"),
                "meta_cognitive": bool(u.meta.get("meta_cognitive", False)),
                "hedging": bool(u.meta.get("hedging", False)),
                "certainty": bool(u.meta.get("certainty", False)),
                "authority_transfer": bool(u.meta.get("authority_transfer", False)),
                "refinement": bool(u.meta.get("refinement", False)),
                "prompt_length": int(u.meta.get("prompt_length", 0) or len(u.text.split())),
                "num_constraints": int(u.meta.get("num_constraints", 0) or 0),  # weak label
                "descriptive_words": int(u.meta.get("descriptive_words", 0) or 0),
            })


    df = pd.DataFrame(rows)

    # add normalized time per conversation (0..1)
    df["t_norm"] = df.groupby("conversation_id")["timestep"].transform(
        lambda s: s / max(s.max(), 1)
    )

    return df

def summarize_sessions(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("conversation_id").agg(
        num_prompts=("timestep", "count"),
        refinement_rate=("refinement", "mean"),
        meta_cog_rate=("meta_cognitive", "mean"),
        hedging_rate=("hedging", "mean"),
        certainty_rate=("certainty", "mean"),
        authority_rate=("authority_transfer", "mean"),
        avg_prompt_length=("prompt_length", "mean"),
        avg_constraints=("num_constraints", "mean"),
        avg_descriptive_words=("descriptive_words", "mean"),
    ).reset_index()
    return summary

def main():
    corpus = Corpus(filename=CORPUS_DIR)
    df = build_trajectory_dataframe(corpus)
    summary = summarize_sessions(df)

    df.to_csv("prompt_trajectories_v2.csv", index=False)
    summary.to_csv("session_summary_v2.csv", index=False)

    print("✅ Wrote prompt_trajectories_v2.csv and session_summary_v2.csv")

if __name__ == "__main__":
    main()
