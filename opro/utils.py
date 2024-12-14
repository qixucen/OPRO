import re


def parse_tag_content(text, prefix="<TEXT>", suffix="</TEXT>"):
    pattern = f"{prefix}(.*?){suffix}"
    results = re.findall(pattern, text, re.DOTALL)
    return results

def evaluate_f1(predicted: str, actual: str) -> float:
    """
    计算F1分数来评估文本生成质量
    
    Args:
        predicted: 模型预测的文本
        actual: 真实答案文本
    Returns:
        F1分数 (0-1之间的浮点数)
    """
    # 将文本分词
    pred_tokens = set(predicted.lower().split())
    actual_tokens = set(actual.lower().split())
    
    # 计算精确率和召回率
    common = pred_tokens & actual_tokens
    if not pred_tokens or not actual_tokens:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(actual_tokens)
    
    # 计算F1分数
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_code(predicted: str, actual: str) -> float:
    """
    评估代码生成的正确性
    
    Args:
        predicted: 模型生成的代码
        actual: 标准答案代码
    Returns:
        1.0 如果执行结果相同,否则返回 0.0
    """
    try:
        # 移除空白字符后比较
        pred_clean = "".join(predicted.split())
        actual_clean = "".join(actual.split())
        return float(pred_clean == actual_clean)
    except:
        return 0.0

def evaluate_number(predicted: str, actual: str) -> float:
    """
    评估数值答案的准确性
    
    Args:
        predicted: 模型预测的数值
        actual: 真实数值
    Returns:
        如果数值相等返回1.0,否则返回0.0
    """
    try:
        pred_num = float(predicted)
        actual_num = float(actual)
        return float(abs(pred_num - actual_num) < 1e-6)
    except:
        return 0.0
