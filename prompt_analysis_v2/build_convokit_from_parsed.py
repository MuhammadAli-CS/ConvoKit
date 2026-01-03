# build_convokit_from_parsed.py
import json
import uuid
from pathlib import Path
from convokit import Corpus, Utterance, Speaker

# ---- CONFIG ----
PATH = "original.json"        # parsed actions JSON (dict: session_id -> list[actions])
OUT_DIR = "prompt_only_corpus"    # output folder

def extract_text_from_textdelta(action_logs):
    """Best-effort extraction of inserted text from Quill ops."""
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
    """
    Priority:
      1) action_delta[1] if exists and is string
      2) action_end_chat_writing
      3) reconstructed insert text from logs
    """
    ad = action.get("action_delta")
    if isinstance(ad, list) and len(ad) >= 2 and isinstance(ad[1], str) and ad[1].strip():
        return ad[1].strip()

    aecw = action.get("action_end_chat_writing")
    if isinstance(aecw, str) and aecw.strip():
        return aecw.strip()

    return extract_text_from_textdelta(action.get("action_logs"))

def is_prompt_action(action):
    return (
        action.get("action_source") == "user"
        and action.get("level_1_action_type") == "chat_insert"
    )

def main():
    with open(PATH, "r", encoding="utf-8") as f:
        actions_per_session = json.load(f)

    user = Speaker(id="user")
    utterances = []
    num_prompts = 0
    num_sessions_used = 0

    for session_id, actions in actions_per_session.items():
        convo_has_any = False

        for i, a in enumerate(actions):
            if not is_prompt_action(a):
                continue
            text = extract_prompt_text(a)
            if not text:
                continue

            convo_has_any = True
            utt = Utterance(
                id=str(uuid.uuid4()),
                speaker=user,
                text=text,
                conversation_id=session_id,
                meta={
                    "session_id": session_id,
                    "action_index": i,
                    "start_time": a.get("action_start_time"),
                    "end_time": a.get("action_end_time"),
                }
            )
            utterances.append(utt)
            num_prompts += 1

        if convo_has_any:
            num_sessions_used += 1

    corpus = Corpus(utterances=utterances)
    out_path = Path(OUT_DIR).resolve()
    corpus.dump(str(out_path))

    print(f"✅ Built corpus '{out_path}' with {num_prompts} prompts across {num_sessions_used} sessions.")

if __name__ == "__main__":
    main()
