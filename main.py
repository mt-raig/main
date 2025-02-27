import argparse
import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Literal
from utils import load_batch_size, load_mt_raig_bench
from utils_openai import OpenAIGenerator
from utils_vllm import VLLMGenerator

from mt_raig_eval import (
    table_aware_insight_decomposition, claim_verification,
    compute_faithfulness_score,
    question_aware_insight_decomposition, topic_semantic_matching,
    compute_completeness_score
)


def generate_insight(
    retrieved_tables_set: List[List[Dict[str, Any]]],
    benchmark: List[Dict[str, Any]],
    model_name: str,
    flag: str
    ) -> List[Dict[str, Any]]:
    """Insight generation"""

    if flag == 'openai':
        generator_class = OpenAIGenerator
    elif flag == 'vllm':
        generator_class = VLLMGenerator
    else:
        generator_class = None
    
    generator = generator_class(model_name=model_name, batch_size=load_batch_size(flag=flag))
    
    input_set = generator.preprocess_data(tables_set=retrieved_tables_set, dataset=benchmark)
    generated_insight_set, _ = generator.generate(
        task=f'{flag}_generate_insight',
        input_set=input_set
    )

    return generated_insight_set


def evaluate(
    inputs: Dict[str, Any],
    demension: Literal['faithfulness', 'completeness']
    ) -> List[float]:
    """Evaluation"""
    if demension == 'faithfulness':
        decomposed_claims_set, _ = asyncio.run(table_aware_insight_decomposition(
            predicted_insight_set=inputs['predicted_insight_set'],
            retrieved_tables_set=inputs['retrieved_tables_set']
        ))
        
        claim_verification_results_with_key, _ = asyncio.run(claim_verification(
            decomposed_claims_set=decomposed_claims_set,
            retrieved_tables_set=inputs['retrieved_tables_set']
        ))
        
        score_set = compute_faithfulness_score(
            claim_verification_results_with_key=claim_verification_results_with_key
        )
    
    elif demension == 'completeness':
        decomposed_pred_topics_set, _ = asyncio.run(question_aware_insight_decomposition(
            insight_set=inputs['predicted_insight_set'],
            question_set=inputs['question_set']
        ))
        
        decomposed_gt_topics_set, _ = asyncio.run(question_aware_insight_decomposition(
            insight_set=inputs['ground_truth_insight_set'],
            question_set=inputs['question_set']
        ))
        
        topic_semantic_matching_results_with_key, _ = asyncio.run(topic_semantic_matching(
            decomposed_pred_topics_set=decomposed_pred_topics_set,
            decomposed_gt_topics_set=decomposed_gt_topics_set
        ))
        
        _, _, score_set = compute_completeness_score(
            topic_semantic_matching_results_with_key=topic_semantic_matching_results_with_key,
            decomposed_pred_topics_set=decomposed_pred_topics_set,
            decomposed_gt_topics_set=decomposed_gt_topics_set
        )
    
    else:
        score_set = None
    
    return score_set


def compute(
    type_set: List[str],
    faithfulness_score_set: List[int],
    completeness_score_set: List[int]
    ) -> str:
    """Computation"""
    scores_per_type = defaultdict(list)
    
    for type, faith_score, comp_score in zip(type_set, faithfulness_score_set, completeness_score_set):
        scores_per_type[('faithfulness', type)].append(faith_score)
        scores_per_type[('faithfulness', 'Total')].append(faith_score)
        scores_per_type[('completeness', type)].append(comp_score)
        scores_per_type[('completeness', 'Total')].append(comp_score)
    
    average_scores = {
        type_with_demension: sum(scores) / len(scores)
        for type_with_demension, scores in scores_per_type.items()
    }
    
    sorted_average_scores = dict(sorted(
        list(average_scores.items()),
        key=lambda x: (['faithfulness', 'completeness'].index(x[0][0]), x[0][1])
    ))
    
    scores_str = '\n'.join(f'{type_with_demension}: {score * 100:.2f}' for type_with_demension, score in sorted_average_scores.items())
    
    return scores_str


def main(baseline: str, model_name: str, path_dir: str):
    mt_raig_bench = load_mt_raig_bench(path_dir=path_dir)
    flag = 'openai' if baseline in ['o3-mini', 'GPT-4o'] else 'vllm' if baseline in ['DeepSeek-R1-8B'] else None

    # Insight generation
    generated_insight_set = generate_insight(
        retrieved_tables_set=mt_raig_bench.retrieved_tables_set,
        benchmark=[data for data in mt_raig_bench],
        model_name=model_name,
        flag=flag
    )

    inputs = {
        'faithfulness': {
            'predicted_insight_set': generated_insight_set,
            'retrieved_tables_set': mt_raig_bench.retrieved_tables_set
        },
        'completeness': {
            'predicted_insight_set': generated_insight_set,
            'ground_truth_insight_set': [data['insight'] for data in mt_raig_bench],
            'question_set': [data['question'] for data in mt_raig_bench]
        }
    }

    # MT-RAIG Eval Faithfulness
    faithfulness_score_set = evaluate(
        inputs=inputs['faithfulness'],
        demension='faithfulness'
    )

    # MT-RAIG Eval Completeness
    completeness_score_set = evaluate(
        inputs=inputs['completeness'],
        demension='completeness'
    )

    # Computation
    print(compute(
        type_set = [data['type'] for data in mt_raig_bench],
        faithfulness_score_set=faithfulness_score_set,
        completeness_score_set=completeness_score_set
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, choices=['o3-mini', 'GPT-4o', 'DeepSeek-R1-8B'], required=True)
    parser.add_argument('--path-dir', type=str, default='mt_raig_bench')
    args, _ = parser.parse_known_args()

    MODEL_NAME = {
        'o3-mini': 'o3-mini-2025-01-31',
        'GPT-4o': 'gpt-4o-2024-08-06',
        'DeepSeek-R1-8B': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    }

    main(
        baseline=args.baseline,
        model_name=MODEL_NAME[args.baseline],
        path_dir=args.path_dir
    )