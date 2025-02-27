import re
from typing import Dict, List, Tuple, Union


def compute_completeness_score(
    topic_semantic_matching_results_with_key: List[Dict[str, Union[str, int]]],
    decomposed_pred_topics_set: List[List[str]],
    decomposed_gt_topics_set: List[List[str]]
    ) -> Tuple[List[float], List[float], List[float]]:
    """ MT-RAIG Eval Completeness

    [Params]
    topic_semantic_matching_results_with_key : List[Dict[str, Union[str, int]]]
    decomposed_pred_topics_set               : List[List[str]]
    decomposed_gt_topics_set                 : List[List[str]]

    [Returns]
    completeness_precision_score_set : List[float]
    completeness_recall_score_set    : List[float]
    completeness_f1_score_set  : List[float]
    """
    idx_len = max(_['key'] for _ in topic_semantic_matching_results_with_key) + 1
    
    completeness_precision_score_set = [0.0 for _ in range(idx_len)]
    completeness_recall_score_set = [0.0 for _ in range(idx_len)]
    
    for res in topic_semantic_matching_results_with_key:
        idx = res['key']
        response = res['result']
        
        match_pred = re.search(r"Matched topic subset of B: \[(.*?)\]", response)
        match_gt = re.search(r"Matched topic subset of A: \[(.*?)\]", response)
        
        if match_pred and decomposed_pred_topics_set[idx]:
            matched_pred_topics = match_pred.group(1).split(',')
            num_of_topics = len(t.strip() for t in matched_pred_topics if t.strip())
            completeness_precision_score_set[idx] = num_of_topics / len(decomposed_pred_topics_set[idx])
        
        if match_gt and decomposed_gt_topics_set[idx]:
            matched_gt_topics = match_gt.group(1).split(',')
            num_of_topics = len(t.strip() for t in matched_gt_topics if t.strip())
            completeness_recall_score_set[idx] = num_of_topics / len(decomposed_gt_topics_set[idx])
    
    completeness_f1_score_set = [
        2 * (p_score * r_score) / (p_score + r_score) if p_score + r_score > 0 else 0.0
        for p_score, r_score in zip(completeness_precision_score_set, completeness_recall_score_set)
    ]
    
    return completeness_precision_score_set, completeness_recall_score_set, completeness_f1_score_set