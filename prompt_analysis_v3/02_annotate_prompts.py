import re
from convokit import Corpus

CORPUS_DIR = "prompt_only_corpus"

HEDGING = r"\b(maybe|i think|perhaps|might|could)\b"
REFINEMENT = r"\b(refine|revise|improve|edit|shorter|clearer)\b"

def main():
    corpus = Corpus(filename=CORPUS_DIR)

    for utt in corpus.iter_utterances():
        text = (utt.text or "").lower()

        utt.add_meta("prompt_length", len(text.split()))
        utt.add_meta("hedging", bool(re.search(HEDGING, text)))
        utt.add_meta("refinement", bool(re.search(REFINEMENT, text)))
        utt.add_meta("text", utt.text)

    corpus.dump(CORPUS_DIR)
    print("Annotations added")

if __name__ == "__main__":
    main()
