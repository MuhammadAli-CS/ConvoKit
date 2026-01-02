import re
import json
from pathlib import Path
from convokit import Corpus

CORPUS_DIR = r"C:\Users\Ali\.convokit\saved-corpora\prompt_only_corpus"
OUT_DIR = "prompt_only_corpus_annotated_v2"

# ----------------------------
# 1) Better intent taxonomy
# ----------------------------
# Order matters: first match wins.
INTENT_RULES = [
    ("meta_cognitive", [
        r"\bam i missing\b", r"\bwhat am i missing\b", r"\bwhat assumptions\b",
        r"\bwhat would a critic\b", r"\bwhat could go wrong\b", r"\bwhere am i wrong\b",
        r"\bwhat are the weaknesses\b", r"\bhow can i improve\b"
    ]),
    ("critique", [
        r"\bcritic(ize|ism)?\b", r"\bweakness(es)?\b", r"\bcounterargument\b",
        r"\bwhat('s| is) wrong\b", r"\bfind flaws?\b", r"\bpush back\b"
    ]),
    ("validation", [
        r"\bis this (correct|right|accurate)\b", r"\bdoes this make sense\b",
        r"\bam i right\b", r"\bcheck my\b", r"\bverify\b", r"\bcan you confirm\b"
    ]),
    ("explain_or_clarify", [
        r"\bexplain\b", r"\bclarify\b", r"\bwhat do you mean\b", r"\bwhy\b",
        r"\bhow does\b", r"\bhelp me understand\b"
    ]),
    ("idea_generation", [
        r"\bbrainstorm\b", r"\bmore (ideas|angles|perspectives)\b",
        r"\balternatives?\b", r"\bdifferent ways\b", r"\bgive me options\b"
    ]),
    # Split delegation into generate vs modify
    ("delegation_generate", [
        r"\bwrite\b", r"\bdraft\b", r"\bgenerate\b", r"\bcompose\b",
        r"\bgive me a\b", r"\bcreate\b", r"\bproduce\b"
    ]),
    ("delegation_modify", [
        r"\brewrite\b", r"\brevise\b", r"\bedit\b", r"\bpolish\b",
        r"\bmake (this|it)\b", r"\bimprove\b", r"\bfix\b", r"\brefine\b"
    ]),
]

# ----------------------------
# 2) Markers: stance + refinement
# ----------------------------
HEDGING = re.compile(r"\b(maybe|i think|perhaps|might|could|not sure|i guess)\b", re.I)
CERTAINTY = re.compile(r"\b(clearly|obviously|definitely|certainly)\b", re.I)
AUTHORITY = re.compile(r"\b(you decide|up to you|just pick|whatever you think)\b", re.I)

# Refinement markers: "shorter", "more formal", "less X", etc.
REFINEMENT_PATTERNS = [
    r"\bshorter\b", r"\blonger\b", r"\bmore concise\b", r"\bmore formal\b",
    r"\bmore professional\b", r"\bmore (clear|persuasive|specific)\b",
    r"\bless (formal|wordy|aggressive|strong)\b",
    r"\btone\b", r"\bstyle\b", r"\bmake it sound\b",
    r"\bsimplify\b", r"\badd details\b", r"\bremove\b"
]
REFINEMENT = re.compile("|".join(REFINEMENT_PATTERNS), re.I)

# Constraint cues (better than just commas):
CONSTRAINT_CUES = [
    r"\bmust\b", r"\bshould\b", r"\bdon't\b", r"\bdo not\b", r"\binclude\b", r"\bexclude\b",
    r"\bat least\b", r"\bat most\b", r"\bno more than\b", r"\bexactly\b",
    r"\buse\b", r"\bavoid\b", r"\bformat\b", r"\bbullet\b", r"\btable\b",
    r"\bkeep\b", r"\blimit\b", r"\bwithin\b", r"\bword(s)?\b"
]
CONSTRAINT_REGEX = re.compile("|".join(CONSTRAINT_CUES), re.I)

def classify_intent(text: str) -> str:
    t = text.strip().lower()
    for label, pats in INTENT_RULES:
        for p in pats:
            if re.search(p, t):
                return label
    return "other"

def count_constraints(text: str) -> int:
    # Count distinct constraint "hits" + punctuation structure signals
    hits = len(CONSTRAINT_REGEX.findall(text))
    structure = text.count("\n") + text.count(";") + text.count(":")
    return hits + structure

def count_descriptive_words(text: str) -> int:
    # A rough proxy: adjectives/adverbs-ish via suffixes + common descriptive words
    # (We keep it simple / fast / no spacy dependency)
    desc = re.findall(r"\b\w+(?:ly|ive|ous|able|ible|al|ful|less|ic)\b", text.lower())
    common = re.findall(r"\b(clear|concise|formal|professional|persuasive|specific|detailed|strong|soft|polite)\b", text.lower())
    return len(desc) + len(common)

def main():
    corpus = Corpus(filename=CORPUS_DIR)

    for utt in corpus.iter_utterances():
        text = utt.text or ""
        low = text.lower()

        intent = classify_intent(low)

        utt.add_meta("prompt_intent_v2", intent)
        utt.add_meta("hedging", bool(HEDGING.search(low)))
        utt.add_meta("certainty", bool(CERTAINTY.search(low)))
        utt.add_meta("authority_transfer", bool(AUTHORITY.search(low)))

        # meta_cognitive gets its own label already; also store boolean
        utt.add_meta("meta_cognitive", intent == "meta_cognitive")

        utt.add_meta("refinement", bool(REFINEMENT.search(low)))
        utt.add_meta("prompt_length", len(re.findall(r"\S+", text)))
        utt.add_meta("num_constraints", count_constraints(text))
        utt.add_meta("descriptive_words", count_descriptive_words(text))

    corpus.dump(OUT_DIR)
    print(f"✅ Annotated (v2) corpus saved to '{OUT_DIR}'")

if __name__ == "__main__":
    main()
