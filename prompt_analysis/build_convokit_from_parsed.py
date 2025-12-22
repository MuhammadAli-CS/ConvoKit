
import json
from convokit import Corpus, Utterance, Speaker
import uuid

PATH = "formal1_logs.json"  # <-- your parsed actions JSON
OUT = "prompt_only_corpus"  # folder ConvoKit will write

def extract_text_from_textdelta(action_logs):
    """Best-effort extraction of inserted text from Quill-style ops."""
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
    # 1) action_delta format like ["INSERT", "<text>", ...]
    ad = action.get("action_delta")
    if isinstance(ad, list) and len(ad) >= 2 and isinstance(ad[1], str) and ad[1].strip():
        return ad[1].strip()

    # 2) some parsers store full chat state
    aecw = action.get("action_end_chat_writing")
    if isinstance(aecw, str) and aecw.strip():
        return aecw.strip()

    # 3) fallback: reconstruct from logs
    txt = extract_text_from_textdelta(action.get("action_logs"))
    return txt

def is_prompt_action(action):
    # adjust this if your dataset uses a different name
    return action.get("action_source") == "user" and action.get("level_1_action_type") == "chat_insert"

with open(PATH, "r") as f:
    actions_per_session = json.load(f)

user = Speaker(id="user")
utterances = []
conversations = []

num_prompts = 0

for session_id, actions in actions_per_session.items():
    convo_utt_ids = []
    for i, a in enumerate(actions):
        if not is_prompt_action(a):
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
        convo_utt_ids.append(utt.id)
        num_prompts += 1

    if convo_utt_ids:
        conversations.append((session_id, convo_utt_ids))

corpus = Corpus(utterances=utterances, speakers=[user], conversations=conversations)
corpus.dump(OUT)

print(f"✅ Built corpus '{OUT}' with {num_prompts} prompts across {len(conversations)} sessions.")
