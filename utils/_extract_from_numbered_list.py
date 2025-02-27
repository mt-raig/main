import re
from typing import List


def extract_from_numbered_list(response: str) -> List[str]:
    """Extract from numbered list
    
    [Param]
    response : str
    
    [Return]
    extracted_list : List[str]
    """
    pattern = r"(?:^|\s)\d+\.\s*(.+?)(?=\s*\d+\.\s|$)"
    
    try:
        extracted_list = [match.strip() for match in re.findall(pattern, response, re.DOTALL)]
    
    except:
        extracted_list = []
    
    return extracted_list