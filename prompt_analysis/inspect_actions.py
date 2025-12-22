import json
from collections import Counter

PATH = "formal1_logs.json"  # <-- your parsed actions JSON

with open(PATH, "r") as f:
    actions_per_session = json.load(f)

type_counts = Counter()
total_actions = 0

for session_id, actions in actions_per_session.items():
    for a in actions:
        t = a.get("level_1_action_type", "MISSING")
        type_counts[t] += 1
        total_actions += 1

print("Total actions:", total_actions)
print("\nTop 30 action types:")
for t, c in type_counts.most_common(30):
    print(f"{t:25s}  {c}")

chat_types = {t: c for t, c in type_counts.items() if t.startswith("chat")}
print("\nChat-related types:", chat_types)
