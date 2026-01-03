# annotate_prompts.py
import re
from pathlib import Path
from convokit import Corpus

CORPUS_DIR = "prompt_only_corpus"
OUT_DIR = "prompt_only_corpus_annotated"

# ---- INTENT patterns (edit as needed) ----
INTENT_PATTERNS = {
    "idea_generation": [
        r"\bbrainstorm\b", r"\bmore (ideas|angles|perspectives)\b", r"\balternatives?\b", r"\bexpand\b"
    ],
    "explain_or_clarify": [
        r"\bwhat do you mean\b", r"\bexplain\b", r"\bclarify\b", r"\bwhy\b", r"\bhow\b"
    ],
    "validation": [
        r"\bis this (correct|right)\b", r"\bdoes this make sense\b", r"\bam i right\b", r"\bverify\b"
    ],
    "delegation_generate": [
        r"\bwrite\b", r"\bdraft\b", r"\bgenerate\b", r"\bgive me\b", r"\bcreate\b"
    ],
    "delegation_modify": [
        r"\bfix\b", r"\brewrite\b", r"\bimprove\b", r"\bedit\b", r"\bmake it\b"
    ],
    "critique": [
        r"\bwhat('s| is) wrong\b", r"\bweakness(es)?\b", r"\bcounterargument\b", r"\bcritic\b"
    ],
}

# ---- marker regex ----
HEDGING = r"\b(maybe|i think|perhaps|might|could|not sure|kinda|sorta)\b"
CERTAINTY = r"\b(clearly|obviously|definitely|certainly|for sure)\b"
AUTHORITY = r"\b(you decide|up to you|just pick|whatever you think)\b"
META_COG = [
    r"am i missing", r"what assumptions", r"what would a critic", r"overlooking", r"trade[- ]?off"
]

# ---- constraint lexicon (seed; expand later) ----
CONSTRAINT_TERMS = [
    r"\bmust\b", r"\bshould\b", r"\bdon't\b", r"\bdo not\b", r"\bavoid\b", r"\brequire\b",
    r"\bneed(s)? to\b", r"\binclude\b", r"\bexclude\b", r"\bno\b(?!\s+idea)", r"\bwithout\b",
    r"\bonly\b", r"\bat least\b", r"\bat most\b", r"\bexactly\b",
    r"\bformat\b", r"\bstructure\b", r"\bword limit\b", r"\bmax\b", r"\bminimum\b",
]

# “descriptive words” = adjectives-ish proxy (cheap)
# You can swap this later with POS tagging; for now it is stable & fast.
DESCRIPTIVE_PROXY = r"\b(\w+ly|\w+ive|\w+ful|\w+ous|\w+able|\w+al|\w+ic|\w+ish)\b"

# refinement heuristic: contains “more”, “better”, “shorter”, etc.
REFINEMENT = r"\b(more|better|shorter|longer|simpler|clearer|revise|refine|improve|fix)\b"

def count_matches(patterns, text):
    return sum(1 for p in patterns if re.search(p, text))

def main():
    corpus = Corpus(filename=CORPUS_DIR)

    for utt in corpus.iter_utterances():
        text_raw = utt.text or ""
        text = text_raw.lower().strip()

        # intent
        intent = "other"
        for label, pats in INTENT_PATTERNS.items():
            if any(re.search(p, text) for p in pats):
                intent = label
                break

        # markers
        hedging = bool(re.search(HEDGING, text))
        certainty = bool(re.search(CERTAINTY, text))
        authority = bool(re.search(AUTHORITY, text))
        meta_cog = any(re.search(p, text) for p in META_COG)
        refinement = bool(re.search(REFINEMENT, text))

        # features
        prompt_length = len(text.split())
        num_constraints_rule = count_matches(CONSTRAINT_TERMS, text)
        descriptive_words = len(re.findall(DESCRIPTIVE_PROXY, text))

        # store
        utt.add_meta("prompt_intent_v2", intent)
        utt.add_meta("hedging", hedging)
        utt.add_meta("certainty", certainty)
        utt.add_meta("authority_transfer", authority)
        utt.add_meta("meta_cognitive", meta_cog)
        utt.add_meta("refinement", refinement)

        utt.add_meta("prompt_length", prompt_length)
        utt.add_meta("num_constraints_rule", num_constraints_rule)
        utt.add_meta("descriptive_words", descriptive_words)

        # keep raw text for sklearn file
        utt.add_meta("text", text_raw)

    out_path = Path(OUT_DIR).resolve()
    corpus.dump(str(out_path))
    print(f"✅ Annotated corpus saved to '{out_path}'.")

if __name__ == "__main__":
    main()
