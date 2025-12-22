import pandas as pd
from convokit import Corpus

CORPUS_DIR = "prompt_only_corpus_annotated"

corpus = Corpus(filename=CORPUS_DIR)

rows = []
for convo in corpus.iter_conversations():
    utts = list(convo.iter_utterances())
    # stable ordering: use action_index metadata if present
    utts.sort(key=lambda u: u.meta.get("action_index", 0))

    for t, u in enumerate(utts):
        rows.append({
            "conversation_id": convo.id,
            "timestep": t,
            "prompt_intent": u.meta.get("prompt_intent"),
            "hedging": u.meta.get("hedging"),
            "certainty": u.meta.get("certainty"),
            "authority_transfer": u.meta.get("authority_transfer"),
            "meta_cognitive": u.meta.get("meta_cognitive"),
            "prompt_length": u.meta.get("prompt_length"),
            "num_constraints": u.meta.get("num_constraints"),
        })

df = pd.DataFrame(rows)
df.to_csv("prompt_trajectories.csv", index=False)

summary = df.groupby("conversation_id").agg(
    num_prompts=("timestep", "count"),
    meta_cog_rate=("meta_cognitive", "mean"),
    hedging_rate=("hedging", "mean"),
    certainty_rate=("certainty", "mean"),
    avg_prompt_length=("prompt_length", "mean"),
    avg_constraints=("num_constraints", "mean"),
).reset_index()

summary.to_csv("session_summary.csv", index=False)

print("✅ Wrote prompt_trajectories.csv and session_summary.csv")
