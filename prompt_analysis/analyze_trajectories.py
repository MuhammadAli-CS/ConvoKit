from convokit import Corpus

corpus = Corpus(filename="data/prompt_corpus_annotated")

for convo in corpus.iter_conversations():
    print(f"\nConversation {convo.id}")

    intents = []
    stances = []
    delegation = []

    for utt in convo.iter_utterances():
        intents.append(utt.meta["prompt_intent"])
        stances.append(utt.meta["epistemic_stance"])
        delegation.append(utt.meta["delegation_level"])

    print("Intent trajectory:", intents)
    print("Stance trajectory:", stances)
    print("Delegation trajectory:", delegation)
