import asyncio
import traceback
import yaml
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import Any, Dict


async def get_async_gpt_4o_mini_response(
        semaphore: asyncio.Semaphore,
        system_prompt: str,
        user_prompt: str,
        key: Any
    ) -> Dict[str, Any]:
    """Get asynchronous GPT-4o mini response

    [Params]
    semaphore     : asyncio.Semaphore
    system_prompt : str
    user_prompt   : str
    key           : Any

    [Return]
    instance : Dict[str, Any]
    e.g. dict_keys(['system_prompt', 'user_prompt', 'response', 'input_tokens_cost', 'output_tokens_cost', 'key'])
    """
    global llm
    global input_token_price
    global output_token_price

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


if __name__ == 'mt_raig_eval._get_async_gpt_4o_mini_response':
    API_KEY =  yaml.load(open('config/openai_api_key.yaml'), Loader=yaml.FullLoader)['api_key']

    llm = ChatOpenAI(
        model='gpt-4o-mini-2024-07-18',
        temperature=0.0,
        api_key=API_KEY
    )

    input_token_price = 0.15/1e6
    output_token_price = 0.60/1e6