import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import Any, Dict, List, Tuple, Union
from utils import extract_from_numbered_list, load_batch_size, load_prompt, serialize_table
from ._get_async_g_eval_responses import get_async_g_eval_responses
from ._get_async_gpt_4o_mini_response import get_async_gpt_4o_mini_response


async def table_aware_insight_decomposition(
    predicted_insight_set: List[str],
    retrieved_tables_set: List[List[Dict[str, Any]]]
    ) -> Tuple[List[List[str]], float]:
    """Decompose table-aware insight

    [Params]
    predicted_insight_set : List[str]
    retrieved_tables_set  : List[List[Dict[str, Any]]]

    [Returns]
    decomposed_claims_set : List[List[str]]
    cost                  : float
    """
    semaphore = asyncio.Semaphore(load_batch_size(flag='openai'))
    
    task_input_set = [
        {
            'insight': insight,
            'serialized_table_schemas': '\n'.join([serialize_table(table=table, is_cell=False) for table in tables])
        }
        for insight, tables
        in zip(predicted_insight_set, retrieved_tables_set)
    ]
    
    tasks = [
		get_async_gpt_4o_mini_response(
			semaphore=semaphore,
            system_prompt=load_prompt(role='system', task='table_aware_insight_decomposition'),
            user_prompt=load_prompt(role='user', task='table_aware_insight_decomposition').format(**task_input),
            key=idx
		)
        for idx, task_input in enumerate(task_input_set)
	]
    
    task_output_set = []
    
    for _ in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Decompose table-aware insight"):
        task_output = await _
        task_output_set.append(task_output)
    
    cost = sum([output['input_tokens_cost'] + output['output_tokens_cost'] for output in task_output_set])
    
    sorted_task_output_set = sorted(task_output_set, key=lambda x: x['key'])

    decomposed_claims_set = [
        extract_from_numbered_list(response=task_output['response'])
        for task_output in sorted_task_output_set
    ]
    
    return decomposed_claims_set, cost


async def claim_verification(
    decomposed_claims_set: List[List[str]],
    retrieved_tables_set: List[List[Dict[str, Any]]]
    ) -> Tuple[List[Dict[str, Union[float, int]]], float]:
    """Verify claims

    [Params]
    decomposed_claims_set : List[List[str]]
    retrieved_tables_set  : List[List[Dict[str, Any]]]

    [Returns]
    claim_verification_results_with_key : List[Dict[str, Union[float, int]]]
    cost                                : float
    """
    semaphore = asyncio.Semaphore(load_batch_size(flag='openai'))
    
    task_inputs_set = [
        [
            {
                'claim': claim,
                'serialized_tables': '\n'.join(serialize_table(table=table) for table in tables)
            }
            for claim in decomposed_claims
        ]
        for decomposed_claims, tables
        in zip(decomposed_claims_set, retrieved_tables_set)
    ]
    
    tasks = [
        get_async_g_eval_responses(
            semaphore=semaphore,
            user_prompt=load_prompt(role='system', task='claim_verification').format(**task_input),
            key=(idx, jdx),
            min_score=0.0,
            max_score=1.0
        )
        for idx, task_inputs in enumerate(task_inputs_set)
        for jdx, task_input in enumerate(task_inputs)
    ]
    
    task_output_set = []
    
    for _ in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Verify claims"):
        task_output = await _
        task_output_set.append(task_output)
    
    cost = sum([output['input_tokens_cost'] + output['output_tokens_cost'] for output in task_output_set])
    
    sorted_task_output_set = sorted(task_output_set, key=lambda x: x['key'])
    
    claim_verification_results_with_key = [
        {
            'result': sum(task_output['responses']) / len(task_output['responses']) if task_output['responses'] else 0.0,
            'key': task_output['key']
        }
        for task_output in sorted_task_output_set
    ]
    
    return claim_verification_results_with_key, cost