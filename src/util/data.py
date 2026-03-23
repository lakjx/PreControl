from typing import List, Dict, Any


__all__ = [
    'single_turn_conv2nemo_rm_train_format', 'single_turn_conv2nemo_rm_eval_format',
    'armorm_pred2helpsteer_score',
    'helpsteer2_prompt2messages'
]


# taken from https://github.com/NVIDIA/NeMo-Aligner/blob/59f8d16d448f5fa4d4eef2e20b79a549e833837f/examples/nlp/data/steerlm/common.py
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

SYSTEM_PROMPT_TEMPLATE = "<extra_id_0>System\n{value}\n"

USER_TURN_TEMPLATE = "<extra_id_1>User\n{value}\n"

ASSISTANT_TURN_TEMPLATE = "<extra_id_1>Assistant\n{value}\n"

LABEL_PREFIX = "<extra_id_2>"


def single_turn_conv2nemo_rm_train_format(prompt: str = None, response: str = None, labels: List[int] = None) -> Dict[str, Any]:
    text = SYSTEM_PROMPT_TEMPLATE.format(value=SYSTEM_PROMPT)
    text += USER_TURN_TEMPLATE.format(value=prompt)
    text += ASSISTANT_TURN_TEMPLATE.format(value=response)
    return dict(text=text+LABEL_PREFIX, label=labels)


reward_bench_dummy_label = '__LABEL__'

def single_turn_conv2nemo_rm_eval_format(prompt: str = None, response: str = None) -> Dict[str, Any]:
    """
    A more general format c.f. the above, per NeMo-Aligner's `attribute_annotate` eval pipeline
    """
    return dict(
        conversations=[
            {'from': 'user', 'value': prompt, 'label': None},
            {'from': 'assistant', 'value': response, 'label': reward_bench_dummy_label}
        ],
        system=SYSTEM_PROMPT, mask='User', type='VALUE_TO_TEXT'
    )


def armorm_pred2helpsteer_score(pred: Dict[str, Any]) -> List[int]:
    """
    Extract the scores from the 17-objective ArmoRM that corresponds to the 5 HelpSteer2 attributes
    """
    scores = pred['obj_wise_rewards']
    scores = [s_ * 5 - 0.5 for s_ in scores[:5]]
    scores = [round(s_) for s_ in scores]
    return [max(0, min(4, s_)) for s_ in scores]


prefix_bot, prefix_user = '<extra_id_1>Assistant\n', '<extra_id_1>User\n'


def helpsteer2_prompt2messages(prompt: str) -> List[Dict[str, str]]:
    ppt = prompt
    messages = []
    if prefix_bot in ppt:  # multi-turn prompt
        while prefix_bot in ppt:
            # find the idx of the 1st occurrence
            idx = ppt.index(prefix_bot)
            # sanity check user turn & bot turn are interleaved; there must be a user prefix afterward, not before
            assert prefix_user in ppt[idx:] and prefix_user not in ppt[:idx]

            user_turn = ppt[:idx].strip()
            idx_ = ppt.index(prefix_user)
            bot_turn = ppt[idx+len(prefix_bot):idx_].strip()

            messages += [dict(role='user', content=user_turn), dict(role='assistant', content=bot_turn)]
            ppt = ppt[idx_ + len(prefix_user):]
    
    messages += [dict(role='user', content=ppt)]

    return messages

def helpsteer2_iterative_messages(prompt, response: str) -> List[Dict[str, str]]:
    ppt = prompt
    messages = []
    if prefix_bot in ppt:  # multi-turn prompt
        while prefix_bot in ppt:
            # find the idx of the 1st occurrence
            idx = ppt.index(prefix_bot)
            # sanity check user turn & bot turn are interleaved; there must be a user prefix afterward, not before
            assert prefix_user in ppt[idx:] and prefix_user not in ppt[:idx]

            user_turn = ppt[:idx].strip()
            idx_ = ppt.index(prefix_user)
            bot_turn = ppt[idx+len(prefix_bot):idx_].strip()

            messages += [dict(role='user', content=user_turn), dict(role='assistant', content=bot_turn)]
            ppt = ppt[idx_ + len(prefix_user):]
    messages += [dict(role='user', content=ppt)]

    messages += [dict(role='assistant', content=response)]
    messages += [dict(role="user", content=f"Based on the conversation above, please re-address the following question. Begin immediately with the answer content.\n{ppt}")]
    return messages

def ultrafeedback_prompt2messages(prompt: str) -> List[Dict[str, str]]:
    messages = [dict(role='user', content=prompt)]
    return messages

def ultrafeedback_iterative_messages(prompt, response:str) -> List[Dict[str, str]]:
    messages = [dict(role='user', content=prompt)]
    messages += [dict(role='assistant', content=response)]
    messages += [dict(role="user", content=f"Based on the conversation above, please re-address the following question. Begin immediately with the answer content.\n{prompt}")]
    return messages