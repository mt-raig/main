from typing import Dict, List, Union


def compute_faithfulness_score(
    claim_verification_results_with_key: List[Dict[str, Union[float, int]]]
    ) -> List[float]:
    """ MT-RAIG Eval Faithfulness

    [Param]
    claim_verification_results_with_key : List[Dict[str, Union[float, int]]]

    [Return]
    faithfulness_score_set : List[float]
    """
    idx_len = max(_['key'][0] for _ in claim_verification_results_with_key) + 1
    
    faithfulness_scores_set = [[] for _ in range(idx_len)]
    
    for res in claim_verification_results_with_key:
        idx, _ = res['key']
        score = res['result']
        
        faithfulness_scores_set[idx].append(score)
    
    faithfulness_score_set = [sum(scores) / len(scores) if scores else 0.0 for scores in faithfulness_scores_set]
    
    return faithfulness_score_set