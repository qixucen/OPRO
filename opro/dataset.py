import json
import jsonlines


def context_adapt(obj: dict):
    context = []
    for i in range(len(obj["context"])):
        context.append(" ".join(obj["context"][i][1]))
    return "\n".join(context)
class Dataset:
    """Enhanced dataset interface for OPRO."""
    
    def __init__(self, dataset_name: str, dataset_path: str):
        self.dataset_name = dataset_name
        # 适配不同文件格式的读取
        try:
            if dataset_path.endswith('.jsonl'):
                with jsonlines.open(dataset_path) as reader:
                    self.data = [obj for obj in reader]
            else:
                with open(dataset_path, 'r') as f:
                    self.data = json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load dataset: {str(e)}")
        
        self.adapt()


    def adapt(self):
        """Adapt different datasets to a standard format with 'input' and 'output' keys"""
        if self.dataset_name == "hotpotqa":
            adapted_data = []
            for item in self.data:
                adapted_item = {
                    "input": f"context: {context_adapt(item)} \n question: {item['question']}",
                    "output": item["answer"]
                }
                adapted_data.append(adapted_item)
            self.data = adapted_data
            
        elif self.dataset_name == "drop":
            adapted_data = []
            for item in self.data:
                adapted_item = {
                    "input": f"context: {item['context']} \n question: {item['question']}",
                    "output": item['completion'] if 'completion' in item else item['ref_text']
                }
                adapted_data.append(adapted_item)
            self.data = adapted_data
        
        # TODO: 适配更多数据集
        elif self.dataset_name == "aime":
            pass 
            
        else:
            if not all("input" in item and "output" in item for item in self.data):
                raise ValueError(f"Unsupported dataset format: {self.dataset_name}")
            
    def get_data(self):
        return self.data
