import json
import uuid
from convokit import Corpus, Utterance, Speaker

PATH = "original.json"
OUT_DIR = "prompt_only_corpus_annotated"

def extract_prompt_text(action):
    ad = action.get("action_delta")
    if isinstance(ad, list) and len(ad) >= 2:
        if isinstance(ad[1], str) and ad[1].strip():
            return ad[1].strip()

    aecw = action.get("action_end_chat_writing")
    if isinstance(aecw, str) and aecw.strip():
        return aecw.strip()

    logs = action.get("action_logs", [])
    pieces = []
    for lg in logs:
        td = lg.get("textDelta")
        if not td:
            continue
        for op in td.get("ops", []):
            ins = op.get("insert")
            if isinstance(ins, str):
                pieces.append(ins)

    text = "".join(pieces).strip()
    return text if text else None

def is_prompt(action):
    return (
        str(action.get("action_source", "")).lower() == "user"
        and "chat" in str(action.get("level_1_action_type", "")).lower()
    )

def main():
    with open(PATH, "r", encoding="utf-8") as f:
        sessions = json.load(f)

    user = Speaker(id="user")
    utterances = []

    for session_id, actions in sessions.items():
        for i, a in enumerate(actions):
            if not is_prompt(a):
                continue

            text = extract_prompt_text(a)
            if not text:
                continue

            utt = Utterance(
                id=str(uuid.uuid4()),
                speaker=user,
                text=text,
                conversation_id=session_id,
                meta={"action_index": i,
                       "start_time": a.get("action_start_time"),
                        "end_time": a.get("action_end_time")
                        }
            )
            utterances.append(utt)

    corpus = Corpus(utterances=utterances)
    corpus.dump(OUT_DIR)
    print(f"Built corpus with {len(utterances)} prompts")

if __name__ == "__main__":
    main()
