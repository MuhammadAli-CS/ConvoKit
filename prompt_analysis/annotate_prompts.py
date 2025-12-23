# annotate_prompts.py
import re
from convokit import Corpus

CORPUS_DIR = r"C:\Users\Ali\.convokit\saved-corpora\prompt_only_corpus"
OUT_DIR = "prompt_only_corpus_annotated"

INTENT_PATTERNS = {
    "idea_generation": [r"\bbrainstorm\b", r"\bmore (ideas|angles|perspectives)\b", r"\balternatives?\b", r"\bexpand\b"],
    "explain_or_clarify":   [r"\bwhat do you mean\b", r"\bexplain\b", r"\bclarify\b", r"\bwhy\b"],
    "validation":      [r"\bis this (correct|right)\b", r"\bdoes this make sense\b", r"\bam i right\b"],
    "delegation":      [r"\bwrite\b", r"\bdraft\b", r"\bgenerate\b", r"\bgive me\b"],
    "critique":        [r"\bwhat('s| is) wrong\b", r"\bweakness(es)?\b", r"\bcounterargument\b", r"\bcritic\b"],
}

HEDGING = r"\b(maybe|i think|perhaps|might|could|not sure)\b"
CERTAINTY = r"\b(clearly|obviously|definitely|certainly)\b"
AUTHORITY = r"\b(you decide|up to you|just pick)\b"
META_COG = [r"am i missing", r"what assumptions", r"what would a critic", r"overlooking"]

corpus = Corpus(filename=CORPUS_DIR)

for utt in corpus.iter_utterances():
    text = utt.text.lower()

    intent = "other"
    for k, pats in INTENT_PATTERNS.items():
        if any(re.search(p, text) for p in pats):
            intent = k
            break

    utt.add_meta("prompt_intent", intent)
    utt.add_meta("hedging", bool(re.search(HEDGING, text)))
    utt.add_meta("certainty", bool(re.search(CERTAINTY, text)))
    utt.add_meta("authority_transfer", bool(re.search(AUTHORITY, text)))
    utt.add_meta("meta_cognitive", any(re.search(p, text) for p in META_COG))

    utt.add_meta("prompt_length", len(text.split()))
    utt.add_meta("num_constraints", text.count(",") + text.count(";"))

corpus.dump(OUT_DIR)
print(f"✅ Annotated corpus saved to '{OUT_DIR}'.")
