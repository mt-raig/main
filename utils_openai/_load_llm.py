from langchain_openai import ChatOpenAI
from ._load_config import API_KEY


def load_llm(model_name: str) -> ChatOpenAI:
    """Load OpenAI LLM

    [Param]
    model_name : str

    [Return]
    llm : ChatOpenAI
    """
    if model_name not in OPENAI_LLM_BUFFER:
        if model_name.startswith('o'):
            llm = ChatOpenAI(
                model=model_name,
                reasoning_effort='high',
                api_key=API_KEY
            )
            OPENAI_LLM_BUFFER[model_name] = llm
        elif model_name.startswith('gpt'):
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                api_key=API_KEY
            )
            OPENAI_LLM_BUFFER[model_name] = llm
    
    else:
        llm = OPENAI_LLM_BUFFER[model_name]
    
    return llm


if __name__ == 'utils_openai._load_llm':
    OPENAI_LLM_BUFFER = dict()