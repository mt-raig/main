import traceback
from time import time
from typing import Any, Dict, List
from ._load_llm import load_llm


def get_batch_response(
    prompts: List[str],
    model_name: str,
    batch_index: int
    ) -> List[Dict[str, Any]]:
    """Get vLLM batch response

    [Params]
    prompts     : List[str]
    model_name  : str
    batch_index : int

    [Return]
    instance_list : List[Dict[str, Any]]
    e.g. dict_keys(['prompt', 'response', 'time_taken'])
    """
    llm = load_llm(model_name=model_name)
    
    try:
        start_time = time()
        results = llm.generate(prompts=prompts)
        end_time = time()
        
        results = [
            res.split('</think>', 1)[-1] for res in results
        ]
    
    except Exception:
        return [
            {
                'prompt': pmt,
                'response': f"[{batch_index}] {traceback.format_exc()}",
                'time_taken': -1
            }
            for pmt in prompts
        ]

    return [
        {
            'prompt': pmt,
            'response': res[0].text.replace('\n', ' ').strip(),
            'time_taken': (end_time - start_time) / len(prompts)
        }
        for pmt, res in zip(prompts, results.generations)
    ]