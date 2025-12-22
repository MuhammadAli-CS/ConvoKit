import json
from convokit import Corpus, Utterance, Speaker

with open("data/raw_sessions.json") as f:
    sessions = json.load(f)

user = Speaker(id="user")

utterances = []
conversations = []

for session in sessions:
    convo_utts = []
    for i, prompt in enumerate(session["prompts"]):
        utt = Utterance(
            id=f"{session['session_id']}_{i}",
            speaker=user,
            text=prompt,
            conversation_id=session["session_id"]
        )
        utterances.append(utt)
        convo_utts.append(utt.id)

    conversations.append((session["session_id"], convo_utts))

corpus = Corpus(
    utterances=utterances,
    speakers=[user],
    conversations=conversations
)

corpus.dump("data/prompt_corpus")