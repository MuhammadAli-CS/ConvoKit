from convokit import Corpus, Conversation, Utterance, Speaker
import uuid


def build_prompt_corpus(level_1_actions_per_session):
    """
    Builds a ConvoKit Corpus where each utterance is a USER PROMPT
    sent to the AI (chat_insert, source=user).
    """

    corpus = Corpus()
    user = Speaker(id="user")

    for session_id, actions in level_1_actions_per_session.items():
        convo = Conversation(id=session_id)

        for idx, action in enumerate(actions):
            if (
                action["level_1_action_type"] == "chat_insert"
                and action["action_source"] == "user"
            ):
                prompt_text = action["action_delta"][1]

                if not prompt_text.strip():
                    continue

                utt = Utterance(
                    id=str(uuid.uuid4()),
                    speaker=user,
                    text=prompt_text,
                    conversation_id=session_id,
                    meta={
                        "timestamp": action["action_start_time"],
                        "session_id": session_id,
                        "timestep": idx,
                    },
                )

                corpus.add_utterance(utt)
                convo.add_utterance(utt)

        if len(convo.get_utterance_ids()) > 0:
            corpus.add_conversation(convo)

    return corpus
