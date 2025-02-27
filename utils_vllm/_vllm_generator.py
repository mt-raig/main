from tqdm import tqdm
from typing import Any, Dict, List, Tuple
from utils import load_prompt, serialize_table
from ._get_batch_response import get_batch_response


class VLLMGenerator:
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
    
    def generate(self, task: str, input_set: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        """Generation"""
        task_output_set = []

        for idx in tqdm(
            range(0, len(input_set), self.batch_size),
            total=len(input_set) // self.batch_size + (1 if len(input_set) % self.batch_size != 0 else 0),
            desc=f"{self.model_name:<30}"
            ):
            task_outputs = get_batch_response(
                batch_size=self.batch_size,
                prompts=[load_prompt(role='user', task=task).format(**task_input) for task_input in input_set[idx : idx + self.batch_size]],
                model_name=self.model_name,
                batch_index=idx
            )

            task_output_set.extend(task_outputs)
        
        time = sum([output['time_taken'] for output in task_output_set])

        return task_output_set, time