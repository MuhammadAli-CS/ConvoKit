from convokit import Corpus
from utils import PROMPT_INTENTS, EPISTEMIC_STANCES

corpus = Corpus(filename="data/prompt_corpus")

for utt in corpus.iter_utterances():
    text = utt.text.lower()

    # --- Prompt intent (VERY naive rules, replace later) ---
    if "give" in text or "more angles" in text:
        intent = "idea_generation"
    elif "is this" in text or "correct" in text:
        intent = "validation"
    elif "write" in text:
        intent = "delegation"
    elif "wrong" in text or "critique" in text:
        intent = "critique"
    else:
        intent = "clarification"

    # --- Epistemic stance ---
    if "maybe" in text or "i think" in text:
        stance = "tentative"
    elif "clearly" in text:
        stance = "assertive"
    elif "you decide" in text:
        stance = "outsourced"
    else:
        stance = "exploratory"

    # --- Delegation level ---
    delegation_level = 2 if intent == "delegation" else 0

    # --- Constraint complexity ---
    constraint_count = text.count("and") + text.count(",")

    # --- Meta-cognitive ---
    metacognitive = any(
        phrase in text
        for phrase in ["am i missing", "what assumptions", "what would a critic"]
    )

    utt.add_meta("prompt_intent", intent)
    utt.add_meta("epistemic_stance", stance)
    utt.add_meta("delegation_level", delegation_level)
    utt.add_meta("constraint_count", constraint_count)
    utt.add_meta("metacognitive", metacognitive)

corpus.dump("data/prompt_corpus_annotated")
