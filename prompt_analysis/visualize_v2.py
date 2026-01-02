import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRAJ = "prompt_trajectories_v2.csv"
SUMM = "session_summary_v2.csv"

df = pd.read_csv(TRAJ)
summ = pd.read_csv(SUMM)

# -------------------------
# A) Intent heatmap (clean)
# -------------------------
intent_order = [
    "meta_cognitive", "critique", "validation", "explain_or_clarify",
    "idea_generation", "delegation_generate", "delegation_modify",
    "other"
]

intent_to_code = {k: i for i, k in enumerate(intent_order)}

# build matrix: rows=session, cols=timestep (maxlen)
sessions = sorted(df["conversation_id"].unique())
max_len = df["timestep"].max() + 1

mat = np.full((len(sessions), max_len), fill_value=np.nan)
for r, sid in enumerate(sessions):
    sub = df[df["conversation_id"] == sid].sort_values("timestep")
    for _, row in sub.iterrows():
        mat[r, int(row["timestep"])] = intent_to_code.get(row["prompt_intent"], intent_to_code["other"])

plt.figure(figsize=(14, 6))
im = plt.imshow(mat, aspect="auto")
plt.yticks(range(len(sessions)), sessions)
plt.xticks(range(max_len), range(max_len))
plt.xlabel("Prompt timestep")
plt.ylabel("Session")
plt.title("Prompt intent flow over time (v2)")

# legend (manual)
handles = []
for name, code in intent_to_code.items():
    handles.append(plt.Line2D([0], [0], marker='s', linestyle='None', label=name))
plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


# -----------------------------------------
# B) Normalized timeline: length growth
# -----------------------------------------
# Bin normalized time into 10 bins
df["t_bin"] = pd.cut(df["t_norm"], bins=np.linspace(0, 1, 11), include_lowest=True, labels=False)

length_by_bin = df.groupby("t_bin")["prompt_length"].mean()
constraints_by_bin = df.groupby("t_bin")["num_constraints"].mean()
refine_by_bin = df.groupby("t_bin")["refinement"].mean()

plt.figure(figsize=(10, 4))
plt.plot(length_by_bin.index, length_by_bin.values)
plt.xlabel("Normalized time bin (0=start, 9=end)")
plt.ylabel("Avg prompt length (words)")
plt.title("Average prompt length over normalized session time")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(constraints_by_bin.index, constraints_by_bin.values)
plt.xlabel("Normalized time bin (0=start, 9=end)")
plt.ylabel("Avg constraints score")
plt.title("Average constraint complexity over normalized session time")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(refine_by_bin.index, refine_by_bin.values)
plt.xlabel("Normalized time bin (0=start, 9=end)")
plt.ylabel("Refinement rate")
plt.title("Refinement prompts over normalized session time")
plt.tight_layout()
plt.show()


# -----------------------------------------
# C) Session-level scatter: length vs constraints
# -----------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(summ["avg_prompt_length"], summ["avg_constraints"])
plt.xlabel("Avg prompt length")
plt.ylabel("Avg constraints score")
plt.title("Sessions: prompt length vs constraint complexity")
plt.tight_layout()
plt.show()


# -----------------------------------------
# D) Intent distribution bar chart
# -----------------------------------------
intent_counts = df["prompt_intent"].value_counts().reindex(intent_order, fill_value=0)

plt.figure(figsize=(10, 4))
plt.bar(intent_counts.index, intent_counts.values)
plt.xticks(rotation=30, ha="right")
plt.ylabel("Count")
plt.title("Overall prompt intent distribution (v2)")
plt.tight_layout()
plt.show()
