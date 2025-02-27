def load_batch_size(
    flag: str
    ) -> int:
    """Load batch size
    
    [Return]
    batch_size : int
    """
    if flag == 'openai':
        batch_size = 300
    
    elif flag == 'vllm':
        batch_size = 3
    
    else:
        batch_size = 0
    
    return batch_size