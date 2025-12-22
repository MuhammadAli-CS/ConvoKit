from build_convokit_corpus import build_prompt_corpus
from annotate_prompts import annotate_corpus
from analyze_prompt_trajectories import (
    build_trajectory_dataframe,
    summarize_sessions,
)

# You already have this
from your_existing_pipeline import level_1_actions_per_session


# ---- build corpus ----
corpus = build_prompt_corpus(level_1_actions_per_session)

# ---- annotate ----
corpus = annotate_corpus(corpus)

# ---- analyze ----
df = build_trajectory_dataframe(corpus)
summary = summarize_sessions(df)

df.to_csv("prompt_trajectories.csv", index=False)
summary.to_csv("session_summary.csv", index=False)

print("✅ Pipeline complete")
