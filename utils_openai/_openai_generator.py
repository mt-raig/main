import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import Any, Dict, List, Tuple
from utils import load_prompt, serialize_table
from ._get_async_response import get_async_response


class OpenAIGenerator:
    def __init__(self, model_name: str, batch_size: int):
        """Initialization"""
        self.model_name = model_name
        self.batch_size = batch_size
    
    def preprocess_data(self, tables_set: List[List[Dict[str, Any]]], dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocessing"""
        preprocessed_input_set = []
        for tables, data in zip(tables_set, dataset):
            serialized_tables = '\n'.join([serialize_table(table=table, table_index=idx + 1) for idx, table in enumerate(tables)])
            preprocessed_input_set.append({'serialized_tables': serialized_tables, 'question': data['question']})
        return preprocessed_input_set
    
    async def _async_generate(self, task: str, input_set: List[Dict[str, Any]], key_set: List[Any]) -> Tuple[List[Dict[str, Any]], str]:
        """Asynchronous generation"""
        semaphore = asyncio.Semaphore(self.batch_size)

        tasks = [
            get_async_response(
                semaphore=semaphore,
                system_prompt=load_prompt(role='system', task=task),
                user_prompt=load_prompt(role='user', task=task).format(**task_input),
                model_name=self.model_name,
                key=key
            )
            for task_input, key in zip(input_set, key_set)
        ]

        task_output_set = []

        for _ in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc=f"{self.model_name:<30}"):
            task_output = await _
            task_output_set.append(task_output)
        
        cost = sum([output['input_tokens_cost'] + output['output_tokens_cost'] for output in task_output_set])

        return sorted(task_output_set, key=lambda x: x['key']), cost
    
    def generate(self, task: str, input_set: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        """Generation"""
        task_output_set, cost = asyncio.run(self._async_generate(
            task=task,
            input_set=input_set,
            key_set=list(range(len(input_set)))
        ))

        return task_output_set, cost