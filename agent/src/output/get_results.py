import numpy as np
from collections import defaultdict
import difflib
import json

def load_standard_accusations(file_path):
    """
    从文件加载标准罪名列表
    
    Args:
        file_path: 标准罪名文件路径
    
    Returns:
        list: 标准罪名列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        standard_names = [line.strip() for line in f if line.strip()]
    return standard_names

def find_best_match(predicted_name, standard_names):
    """
    找到与预测罪名最匹配的标准罪名
    
    Args:
        predicted_name: 预测的罪名
        standard_names: 标准罪名列表
    
    Returns:
        str: 最匹配的标准罪名
    """
    if not predicted_name or not standard_names:
        return predicted_name
    
    # 使用difflib进行模糊匹配
    matches = difflib.get_close_matches(predicted_name, standard_names, n=1, cutoff=0.5)
    
    if matches:
        return matches[0]
    else:
        # 如果没有找到匹配，尝试更宽松的匹配
        # 检查是否包含关键词
        for std_name in standard_names:
            # 如果预测罪名包含标准罪名或标准罪名包含预测罪名
            if predicted_name in std_name or std_name in predicted_name:
                return std_name
        
        # 如果还是没有匹配，返回原始名称
        return predicted_name

def map_accusation_to_standard(predicted_name, standard_names):
    """
    将预测罪名映射到最匹配的标准罪名
    
    Args:
        predicted_name: 预测的罪名
        standard_names: 标准罪名列表
    
    Returns:
        str: 映射后的标准罪名
    """
    # 如果预测罪名已经是标准罪名，直接返回
    if predicted_name in standard_names:
        return predicted_name
    
    # 查找最佳匹配
    return find_best_match(predicted_name, standard_names)

def calculate_all_metrics_with_errors(y_true, y_pred, standard_names_file='f:\\LegalAgent\\dataset\\final_all_data\\meta\\accu.txt'):
    """
    计算所有指标：准确率和macro-F1，并记录错误信息
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        standard_names_file: 标准罪名文件路径
    
    Returns:
        dict: 包含各项指标的字典
    """
    # 加载标准罪名
    standard_names = load_standard_accusations(standard_names_file)
    
    # 映射预测的罪名到最匹配的标准罪名
    mapped_y_pred = []
    for pred in y_pred:
        mapped_pred = map_accusation_to_standard(pred, standard_names)
        mapped_y_pred.append(mapped_pred)
    
    # 计算准确率并记录错误
    correct = 0
    error_records = []  # 记录错误信息
    
    for idx, (true, pred) in enumerate(zip(y_true, mapped_y_pred)):
        if true == pred:
            correct += 1
        else:
            # 记录错误信息
            error_records.append({
                'index': idx + 1,  # 第几个样本（从1开始）
                'true_accusation': true,
                'predicted_accusation': y_pred[idx],  # 原始预测
                'mapped_prediction': pred,  # 映射后的预测
                'is_mapped': y_pred[idx] != pred  # 是否经过映射
            })
    
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0
    
    # 计算macro-F1
    all_labels = list(set(y_true + mapped_y_pred))
    f1_scores = []
    
    for label in all_labels:
        # 计算TP, FP, FN
        tp = sum(1 for t, p in zip(y_true, mapped_y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, mapped_y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, mapped_y_pred) if t == label and p != label)
        
        # 计算precision和recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # 计算F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        f1_scores.append(f1)
    
    # 计算macro-F1（所有类别F1的平均值）
    macro_f1 = np.mean(f1_scores) if f1_scores else 0
    
    
    # 统计映射情况
    mapping_stats = defaultdict(int)
    for orig, mapped in zip(y_pred, mapped_y_pred):
        if orig != mapped:
            mapping_stats[f"{orig} -> {mapped}"] += 1
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'mapped_predictions': mapped_y_pred,
        'mapping_statistics': dict(mapping_stats),
        'num_mapped': sum(mapping_stats.values()),
        'mapping_rate': sum(mapping_stats.values()) / len(y_pred) if len(y_pred) > 0 else 0,
        'error_records': error_records,  # 新增：错误记录
        'num_errors': len(error_records),
        'standard_names_count': len(standard_names),
        'total_samples': len(y_true)
    }

def evaluate_from_jsonl_with_errors(jsonl_file_path, standard_names_file='f:\\LegalAgent\\dataset\\final_all_data\\meta\\accu.txt'):
    """
    从JSONL文件读取数据并评估，记录错误信息
    
    Args:
        jsonl_file_path: JSONL文件路径
        standard_names_file: 标准罪名文件路径
    
    Returns:
        dict: 评估结果
    """
    y_true = []
    y_pred = []
    original_data = []  # 保存原始数据
    
    # 读取JSONL文件
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                data = json.loads(line.strip())
                original_data.append(data)
                if 'true_accusation' in data and 'predicted_accusation' in data:
                    y_true.append(data['true_accusation'])
                    y_pred.append(data['predicted_accusation'])
    
    # 计算指标
    results = calculate_all_metrics_with_errors(y_true, y_pred, standard_names_file)
    
    # 将错误记录与原始数据关联
    for error in results['error_records']:
        idx = error['index'] - 1  # 转换为0-based索引
        if idx < len(original_data):
            error['fact_id'] = original_data[idx].get('fact_id', '未知')
            error['original_fact'] = original_data[idx].get('original_fact', '')  
    
    # 打印摘要
    print("=" * 60)
    print("评估结果摘要")
    print("=" * 60)
    print(f"总样本数: {results['total_samples']}")
    print(f"标准罪名数量: {results['standard_names_count']}")
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"Macro-F1: {results['macro_f1']:.4f}")
    print(f"映射数量: {results['num_mapped']}")
    print(f"映射比例: {results['mapping_rate']:.2%}")
    print(f"错误数量: {results['num_errors']}")
    print(f"错误率: {results['num_errors']/results['total_samples']:.2%}")
        
    # 保存错误记录到文件
    error_file = jsonl_file_path.replace('.jsonl', '_errors.json')
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(results['error_records'], f, ensure_ascii=False, indent=2)
    print(f"错误记录已保存到: {error_file}")
    
    return results

def get_error_summary_table(results):
    """
    生成错误摘要表格
    
    Args:
        results: 评估结果
    
    Returns:
        str: 错误摘要表格
    """
    if not results['error_records']:
        return "没有错误记录"
    
    table = "错误摘要:\n"
    table += "-" * 80 + "\n"
    table += "序号 | 真实罪名                 | 预测罪名 \n"
    table += "-" * 80 + "\n"
    
    for error in results['error_records']:
        table += f"{error['index']:4d} | {error['true_accusation']:20s} | {error['predicted_accusation']:20s} \n"
    
    return table

if __name__ == "__main__":
    jsonl_path = r'F:\LegalAgent\agent\src\output\evaluation_results_qwen3_8b.jsonl'
    
    # 执行评估
    results = evaluate_from_jsonl_with_errors(jsonl_path)
    
    # 生成错误摘要表格
    error_table = get_error_summary_table(results)
    
    # 构造完整的报告文本
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("评估结果摘要")
    report_lines.append("=" * 60)
    report_lines.append(f"总样本数: {results['total_samples']}")
    report_lines.append(f"标准罪名数量: {results['standard_names_count']}")
    report_lines.append(f"准确率: {results['accuracy']:.4f}")
    report_lines.append(f"Macro-F1: {results['macro_f1']:.4f}")
    report_lines.append(f"映射数量: {results['num_mapped']}")
    report_lines.append(f"映射比例: {results['mapping_rate']:.2%}")
    report_lines.append(f"错误数量: {results['num_errors']}")
    report_lines.append(f"错误率: {results['num_errors']/results['total_samples']:.2%}")
    report_lines.append(f"错误记录已保存到: {jsonl_path.replace('.jsonl', '_errors.json')}")
    report_lines.append("")
    report_lines.append(error_table)
    
    full_report = "\n".join(report_lines)
    
    # 保存完整报告到文件
    report_file = jsonl_path.replace('.jsonl', '_evaluation_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    print("\n" + "="*60)
    print("完整评估报告已保存至:")
    print(report_file)
    