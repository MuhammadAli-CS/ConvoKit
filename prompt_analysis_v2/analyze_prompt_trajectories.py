# analyze_prompt_trajectories.py
import pandas as pd
import numpy as np
from pathlib import Path
from convokit import Corpus

CORPUS_DIR = "prompt_only_corpus_annotated"
OUT_TRAJ = Path("data/prompt_trajectories.csv")
OUT_SUMMARY = Path("data/session_summary.csv")

# optional: merge model preds if file exists
SPEC_FILE = Path("data/prompt_with_specificity.csv")

def main():
    corpus = Corpus(filename=CORPUS_DIR)

    rows = []
    for convo in corpus.iter_conversations():
        utts = list(convo.iter_utterances())
        utts.sort(key=lambda u: int(u.meta.get("action_index", 0) or 0))

        n = len(utts)
        for t, u in enumerate(utts):
            rows.append({
                "conversation_id": convo.id,
                "timestep": t,
                "prompt_intent": u.meta.get("prompt_intent_v2", "other"),
                "meta_cognitive": bool(u.meta.get("meta_cognitive", False)),
                "hedging": bool(u.meta.get("hedging", False)),
                "certainty": bool(u.meta.get("certainty", False)),
                "authority_transfer": bool(u.meta.get("authority_transfer", False)),
                "refinement": bool(u.meta.get("refinement", False)),
                "prompt_length": int(u.meta.get("prompt_length", 0) or 0),
                "num_constraints_rule": int(u.meta.get("num_constraints_rule", 0) or 0),
                "descriptive_words": int(u.meta.get("descriptive_words", 0) or 0),
                "t_norm": (t / (n - 1)) if n > 1 else 0.0
            })

    df = pd.DataFrame(rows)

    # merge specificity_pred if we have it
    if SPEC_FILE.exists():
        spec = pd.read_csv(SPEC_FILE)
        df = df.merge(
            spec[["conversation_id", "timestep", "specificity_pred"]],
            on=["conversation_id", "timestep"],
            how="left"
        )
    else:
        df["specificity_pred"] = np.nan

    OUT_TRAJ.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TRAJ, index=False)

    summary = df.groupby("conversation_id").agg(
        num_prompts=("timestep", "count"),
        refinement_rate=("refinement", "mean"),
        meta_cog_rate=("meta_cognitive", "mean"),
        hedging_rate=("hedging", "mean"),
        certainty_rate=("certainty", "mean"),
        authority_rate=("authority_transfer", "mean"),
        avg_prompt_length=("prompt_length", "mean"),
        avg_constraints=("num_constraints_rule", "mean"),
        avg_descriptive_words=("descriptive_words", "mean"),
        avg_specificity_pred=("specificity_pred", "mean"),
    ).reset_index()

    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"✅ Wrote {OUT_TRAJ.resolve()}")
    print(f"✅ Wrote {OUT_SUMMARY.resolve()}")

if __name__ == "__main__":
    main()
