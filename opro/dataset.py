# opro/dataset.py
import json
import jsonlines
from typing import List, Dict, Any, Union
import pandas as pd

class Dataset:
    """Enhanced dataset interface for OPRO."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        self._validate_data()
    
    def _validate_data(self):
        """Validate dataset format."""
        if not self.data:
            raise ValueError("Dataset cannot be empty")
        
        required_keys = {'input', 'output'}
        for item in self.data:
            if not all(key in item for key in required_keys):
                raise ValueError(f"Each item must contain keys: {required_keys}")
    
    @classmethod
    def from_jsonl(cls, file_path: str) -> 'Dataset':
        """Load dataset from JSONL file."""
        data = []
        with jsonlines.open(file_path) as reader:
            for item in reader:
                data.append(item)
        return cls(data)
    
    @classmethod
    def from_json(cls, file_path: str) -> 'Dataset':
        """Load dataset from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(data)
    
    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> 'Dataset':
        """Create dataset from pandas DataFrame."""
        if not all(col in df.columns for col in ['input', 'output']):
            raise ValueError("DataFrame must contain 'input' and 'output' columns")
        return cls(df.to_dict('records'))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        return self.data[idx]