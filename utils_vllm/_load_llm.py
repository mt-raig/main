from langchain_community.llms.vllm import VLLM


def load_llm(model_name: str) -> VLLM:
    """Load vLLM

    [Param]
    model_name : str

    [Return]
    llm : VLLM
    """
    if model_name not in VLLM_BUFFER:
        llm = VLLM(
            model=model_name,
            tensor_parallel_size=1,
            temperature=0.0
        )
        VLLM_BUFFER[model_name] = llm
    
    else:
        llm = VLLM_BUFFER[model_name]
    
    return llm


if __name__ == 'utils_vllm._load_llm':
    VLLM_BUFFER = dict()