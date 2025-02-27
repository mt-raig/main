import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import Dict, List, Tuple, Union
from utils import extract_from_numbered_list, load_batch_size, load_prompt
from ._get_async_gpt_4o_mini_response import get_async_gpt_4o_mini_response


async def question_aware_insight_decomposition(
        insight_set: List[str],
        question_set: List[str]
    ) -> Tuple[List[List[str]], float]:
    """Decompose question-aware insight

    [Params]
    insight_set  : List[str],
    question_set : List[str]

    [Returns]
    decomposed_topics_set : List[List[str]]
    cost                  : float
    """
    semaphore = asyncio.Semaphore(load_batch_size(flag='openai'))
    
    task_input_set = [
        {
            'insight': insight,
            'question': question
        }
        for insight, question
        in zip(insight_set, question_set)
    ]
    
    tasks = [
        get_async_gpt_4o_mini_response(
            semaphore=semaphore,
            system_prompt=load_prompt(role='system', task='question_aware_insight_decomposition'),
            user_prompt=load_prompt(role='user', task='question_aware_insight_decomposition').format(**task_input),
            key=idx
        )
        for idx, task_input in enumerate(task_input_set)
    ]
    
    task_output_set = []
    
    for _ in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Decompose question-aware insight"):
        task_output = await _
        task_output_set.append(task_output)
    
    cost = sum([output['input_tokens_cost'] + output['output_tokens_cost'] for output in task_output_set])
    
    sorted_task_output_set = sorted(task_output_set, key=lambda x: x['key'])

    decomposed_topics_set = [
        extract_from_numbered_list(response=task_output['response'])
        for task_output in sorted_task_output_set
    ]
    
    return decomposed_topics_set, cost


async def topic_semantic_matching(
        decomposed_pred_topics_set: List[List[str]],
        decomposed_gt_topics_set: List[List[str]]
    ) -> Tuple[List[Dict[str, Union[str, int]]], float]:
    """Match topics semantically 

    [Params]
    decomposed_pred_topics_set : List[List[str]]
    decomposed_gt_topics_set   : List[List[str]]

    [Returns]
    topic_semantic_matching_results_with_key : List[Dict[str, Union[str, int]]]
    cost                                     : float
    """
    semaphore = asyncio.Semaphore(load_batch_size(flag='openai'))
    
    task_input_set = [
        {
            'pred_topics': '\n'.join(f'{index + 1}. {topic}' for index, topic in enumerate(pred_topics)),
            'gt_topics': '\n'.join(f'{index + 1}. {topic}' for index, topic in enumerate(gt_topics))
        }
        for pred_topics, gt_topics
        in zip(decomposed_pred_topics_set, decomposed_gt_topics_set)
    ]
    
    tasks = [
        get_async_gpt_4o_mini_response(
            semaphore=semaphore,
            system_prompt=load_prompt(role='system', task='topic_semantic_matching'),
            user_prompt=load_prompt(role='user', task='topic_semantic_matching').format(**task_input),
            key=idx
        )
        for idx, task_input in enumerate(task_input_set)
    ]
    
    task_output_set = []
    
    for _ in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Verify claims"):
        task_output = await _
        task_output_set.append(task_output)
    
    cost = sum([output['input_tokens_cost'] + output['output_tokens_cost'] for output in task_output_set])
    
    sorted_task_output_set = sorted(task_output_set, key=lambda x: x['key'])
    
    topic_semantic_matching_results_with_key = [
        {
            'result': task_output['response'],
            'key': task_output['key']
        }
        for task_output in sorted_task_output_set
    ]
    
    return topic_semantic_matching_results_with_key, cost