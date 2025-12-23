import json
import uuid
from convokit import Corpus, Utterance, Speaker

PATH = "original.json"
OUT = "prompt_only_corpus"

AI_TRIGGER_TYPES = {
    "present_suggestion",
    "chat_suggestion_get",
    "chat_suggestion_open",
    "accept_suggestion",
}

def extract_text_from_textdelta(action_logs):
    pieces = []
    for lg in action_logs or []:
        td = lg.get("textDelta")
        if not td:
            continue
        for op in td.get("ops", []):
            ins = op.get("insert")
            if isinstance(ins, str):
                pieces.append(ins)
    return "".join(pieces).strip()

def extract_prompt_text(action):
    aecw = action.get("action_end_chat_writing")
    if isinstance(aecw, str) and aecw.strip():
        return aecw.strip()

    ad = action.get("action_delta")
    if isinstance(ad, list) and len(ad) >= 2 and isinstance(ad[1], str) and ad[1].strip():
        return ad[1].strip()

    return extract_text_from_textdelta(action.get("action_logs"))

def is_prompt_action(actions, i):
    a = actions[i]
    if not (a.get("action_source") == "user" and a.get("level_1_action_type") == "chat_insert"):
        return False

    # keep only prompts that trigger an AI event next
    if i + 1 < len(actions):
        nxt = actions[i + 1]
        if nxt.get("action_source") == "api":
            return True
        if nxt.get("level_1_action_type") in AI_TRIGGER_TYPES:
            return True
    return False

def main():
    with open(PATH, "r", encoding="utf-8") as f:
        actions_per_session = json.load(f)

    user = Speaker(id="user")
    utterances = []
    sessions_with_prompts = set()

    for session_id, actions in actions_per_session.items():
        for i, a in enumerate(actions):
            if not is_prompt_action(actions, i):
                continue

            text = extract_prompt_text(a)
            if not text:
                continue

            utt = Utterance(
                id=str(uuid.uuid4()),
                speaker=user,
                text=text,
                conversation_id=session_id
            )
            utt.add_meta("session_id", session_id)
            utt.add_meta("action_index", i)
            utt.add_meta("start_time", a.get("action_start_time"))
            utt.add_meta("end_time", a.get("action_end_time"))

            utterances.append(utt)
            sessions_with_prompts.add(session_id)

    # IMPORTANT: initialize with utterances so internal storage exists
    corpus = Corpus(utterances=[])

    # add in one batch
    corpus.add_utterances(utterances)

    corpus.dump(OUT)
    print(f"✅ Built corpus '{OUT}' with {len(utterances)} prompts across {len(sessions_with_prompts)} sessions.")

if __name__ == "__main__":
    main()
