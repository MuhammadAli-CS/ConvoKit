import pandas as pd
import re
from convokit import Corpus
from pathlib import Path

CORPUS_DIR = "prompt_only_corpus"
OUT = Path("data/prompt_trajectories.csv")

HEDGING = r"\b(maybe|i think|perhaps|might|could)\b"
REFINEMENT = r"\b(refine|revise|improve|edit|shorter|clearer)\b"

def main():
    corpus = Corpus(filename=CORPUS_DIR)
    rows = []

    for convo in corpus.iter_conversations():
        utts = sorted(
            convo.iter_utterances(),
            key=lambda u: int(u.meta.get("action_index", 0))
        )

        for t, u in enumerate(utts):
            text = u.text or ""
            text_l = text.lower()

            rows.append({
                "conversation_id": convo.id,

                # ordering
                "timestep": t,

                # real time
                "start_time": u.meta.get("start_time"),
                "end_time": u.meta.get("end_time"),

                # text
                "text": text,

                # derived features (ALWAYS correct)
                "prompt_length": len(text.split()),
                "hedging": bool(re.search(HEDGING, text_l)),
                "refinement": bool(re.search(REFINEMENT, text_l)),
            })

    df = pd.DataFrame(rows)

    # normalized timestep (within session)
    df["t_norm"] = (
        df.groupby("conversation_id")["timestep"]
        .transform(lambda x: x / (x.max() if x.max() > 0 else 1))
    )

    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"✅ Wrote {OUT}")

if __name__ == "__main__":
    main()
