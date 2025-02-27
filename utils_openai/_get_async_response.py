import asyncio
import traceback
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import Any, Dict
from ._load_config import PRICING
from ._load_llm import load_llm


async def get_async_response(
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    key: Any
    ) -> Dict[str, Any]:
    """Get asynchronous OpenAI response

    [Params]
    semaphore     : asyncio.Semaphore
    system_prompt : str
    user_prompt   : str
    model_name    : str
    key           : Any

    [Return]
    instance : Dict[str, Any]
    e.g. dict_keys(['system_prompt', 'user_prompt', 'response', 'input_tokens_cost', 'output_tokens_cost', 'key'])
    """
    llm = load_llm(model_name=model_name)

    input_token_price = PRICING[model_name]['input_token_price']
    output_token_price = PRICING[model_name]['output_token_price']

    async with semaphore:
        try:
            system_message = SystemMessage(system_prompt)
            human_message = HumanMessage(user_prompt)

            response = await llm.ainvoke([system_message, human_message])

            input_tokens = response.usage_metadata['input_tokens']
            output_tokens = response.usage_metadata['output_tokens']

            input_tokens_cost = input_tokens * input_token_price
            output_tokens_cost = output_tokens * output_token_price
        
        except Exception:
            return {
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'response': f"[{key}] {traceback.format_exc()}",
                'input_tokens_cost': 0,
                'output_tokens_cost': 0,
                'key': key
            }

    return {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'response': response.content.replace('\n', ' ').strip(),
        'input_tokens_cost': input_tokens_cost,
        'output_tokens_cost': output_tokens_cost,
        'key': key
    }