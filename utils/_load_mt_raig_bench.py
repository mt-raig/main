import json
from typing import Any, Dict, List, Literal, Union


class MTRAIGBENCH:
    def __init__(self, path_dir: str):
        """Initialization"""
        self.path_dir = path_dir
        self._benchmark = json.load(open(f'{path_dir}/benchmark.json', 'r'))
        self._table_corpus = json.load(open(f'{path_dir}/table_corpus.json', 'r'))
        self._match_table_with_id = {table['id']: table for table in self._table_corpus}
        self._retrieved_tables_set = self._load_retrieved_tables(path_dir=path_dir)
    
    @property
    def table_corpus(self) -> List[Dict[str, Any]]:
        """Table corpus"""
        return self._table_corpus
    
    @property
    def retrieved_tables_set(self):
        """Top-10 retrieved tables set"""
        return self._retrieved_tables_set

    def table(self, table_id: str) -> Dict[str, Any]:
        """Table matching by table ID"""
        return self._match_table_with_id[table_id]
    
    def _load_retrieved_tables(self, path_dir: str):
        """Retrieval results loading"""
        retrieved_table_ids_set = json.load(open(f'{path_dir}/dpr_top_10_retrieved_table_ids_set.json', 'r'))
        retrieved_tables_set = [[self._match_table_with_id[table_id] for table_id in table_ids] for table_ids in retrieved_table_ids_set]
        return retrieved_tables_set

    def __len__(self) -> int:
        """Benchmark size"""
        return len(self._benchmark)
    
    def __getitem__(self, key: Union[int, slice]):
        """Benchmark data"""
        return self._benchmark[key]
    
    def __str__(self) -> Literal['MT-RAIG Bench']:
        """Representation"""
        return 'MT-RAIG Bench'


def load_mt_raig_bench(path_dir: str) -> MTRAIGBENCH:
    """Load MT-RAIG Bench
    
    [Param]
    path_dir : str

    [Return]
    mt_raig_bench : MTRAIGBENCH
    """
    global mt_raig_bench
    if mt_raig_bench is None:
        mt_raig_bench = MTRAIGBENCH(path_dir=path_dir)
    
    return mt_raig_bench


if __name__ == 'utils._load_mt_raig_bench':
    mt_raig_bench = None