import asyncio
import traceback
import yaml
import re
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import Any, Dict, List


def _parse_responses(responses: List[Any], min_score: float, max_score: float) -> List[float]:
    """Parsing and Filtering"""
    scores = []

    for res in responses:
        matched = re.search(r"\b\d+(\.\d+)?\b", res)
        if (matched):
            try:
                sc = float(matched.group())
            except:
                sc = 0.0
        else:
            sc = 0.0
        
        if sc >= min_score and sc <= max_score:
            scores.append(sc)

    return scores


async def get_async_g_eval_responses(
        semaphore: asyncio.Semaphore,
        user_prompt: str,
        key: Any,
        min_score: float,
        max_score: float
    ) -> Dict[str, Any]:
    """Get asynchronous OpenAI G-Eval responses (GPT-4o mini)

    [Params]
    semaphore   : asyncio.Semaphore
    user_prompt : str
    model_name  : str
    key         : Any
    min_score   : float
    max_score   : float

    [Return]
    instance : Dict[str, Any]
    e.g. dict_keys(['user_prompt', 'responses', 'input_tokens_cost', 'output_tokens_cost', 'key'])
    """
    global llm
    global input_token_price
    global output_token_price

    async with semaphore:
        try:
            user_message = HumanMessage(user_prompt)

            responses = await llm.agenerate([[user_message]])

            input_tokens = sum(res.message.usage_metadata['input_tokens'] for res in responses.generations[0]) / len(responses.generations[0])
            output_tokens = sum(res.message.usage_metadata['output_tokens'] for res in responses.generations[0])

            input_tokens_cost = input_tokens * input_token_price
            output_tokens_cost = output_tokens * output_token_price

            scores = _parse_responses([res.text for res in responses.generations[0]], min_score, max_score)

        except Exception as e:
            return {
                'user_prompt': user_prompt,
                'responses': [f"[{key}] {traceback.format_exc()}"],
                'input_tokens_cost': 0,
                'output_tokens_cost': 0,
                'key': key
            }

    return {
        'user_prompt': user_prompt,
        'responses': scores,
        'input_tokens_cost': input_tokens_cost,
        'output_tokens_cost': output_tokens_cost,
        'key': key
    }


if __name__ == 'mt_raig_eval._get_async_g_eval_response':
    API_KEY =  yaml.load(open('config/openai_api_key.yaml'), Loader=yaml.FullLoader)['api_key']

    llm = ChatOpenAI(
        model='gpt-4o-mini-2024-07-18',
        temperature=2.0,
        api_key=API_KEY,
        presence_penalty=0,
        frequency_penalty=0,
        n=20,
        top_p=1,
        max_tokens=5
    )

    input_token_price = 0.15/1e6
    output_token_price = 0.60/1e6