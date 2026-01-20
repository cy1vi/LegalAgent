import json
import os
import random
from collections import defaultdict
import time
from tqdm import tqdm
from agent import LegalCaseSchemaExtractor
import build_prompt as build_prompt  # 修正导入名称

# --- 配置 ---
INPUT_FILE = r"D:\deeplearning\project_learning\LegalAgent\dataset\final_all_data\first_stage\test.json"
ACCUSATION_MAPPING_FILE = r"D:\deeplearning\project_learning\LegalAgent\statistics_analyze\accusation_mapping.json"
OUTPUT_DIR = r"D:\deeplearning\project_learning\LegalAgent\statistics_analyze\discovered_factors"
TEMP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "temp_individual_outputs")
CATEGORY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "by_category")
SUMMARY_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_discovered_factors_summary.json")

SAMPLES_PER_CATEGORY = 100

def load_json(filepath):
    """安全地加载JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到配置文件 {filepath}。请检查路径是否正确。")
        raise
    except json.JSONDecodeError as e:
        print(f"错误：配置文件 {filepath} 格式不正确。{e}")
        raise

def load_cases(file_path):
    """从JSONL文件加载所有案件数据"""
    cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            cases.append(json.loads(line))
    return cases

def categorize_cases(cases, accusation_mapping):
    """根据指控将案件分类"""
    categorized = defaultdict(list)
    # 创建一个反向映射：指控 -> 类别
    accusation_to_category = {}
    for category, accusations in accusation_mapping.items():
        for acc in accusations:
            accusation_to_category[acc] = category

    for case in cases:
        meta = case.get("meta", {})
        accusations = meta.get("accusation", [])
        if accusations:
            # 通常取第一个指控作为主要指控进行分类
            main_accusation = accusations[0] 
            category = accusation_to_category.get(main_accusation, "其他类别")
            categorized[category].append(case)
        else:
            # 如果没有指控信息，则放入其他类别
            categorized["其他类别"].append(case)
            
    return categorized

def sample_cases(categorized_cases, samples_per_category):
    """从每个类别中随机采样指定数量的案件"""
    sampled = {}
    for category, case_list in categorized_cases.items():
        if len(case_list) <= samples_per_category:
            print(f"类别 '{category}' 案例数 ({len(case_list)}) 少于或等于采样数 ({samples_per_category})，将抽取全部。")
            sampled[category] = case_list
        else:
            sampled[category] = random.sample(case_list, samples_per_category)
            print(f"类别 '{category}' 已随机采样 {samples_per_category} 个案例。")
    return sampled

# --- LLM调用 ---
def llm_call(case_fact, category=None):
    """调用LLM进行因素发现，返回数据"""
    system_prompt = build_prompt.build_prompt(category)
    extractor = LegalCaseSchemaExtractor()
    # 确保不使用流式输出，因为我们需要立即获取结果
    response = extractor.extract_schema(case_fact, prompt_override=system_prompt, stream=False)
    
    # 如果返回的是生成器，将其转换为字符串
    if hasattr(response, '__iter__') and not isinstance(response, (str, list, dict)):
        response = ''.join(list(response))
    
    return response

def process_sampled_cases(sampled_cases):
    """处理采样后的案件，调用LLM并保存结果"""
    all_results = {}

    # 确保输出目录存在
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CATEGORY_OUTPUT_DIR, exist_ok=True)

    for category, cases in sampled_cases.items():
        print(f"\n--- 开始处理类别: {category} (共 {len(cases)} 个案例) ---")
        category_results = []
        
        # 为每个案例创建一个唯一的ID以便保存
        for i, case in enumerate(tqdm(cases, desc=f"处理 {category}", ncols=100)):
            case_fact = case.get("fact", "")
            case_meta = case.get("meta", {})
            case_accusations = case_meta.get("accusation", ["未知"])

            # --- 核心：调用LLM进行分析 ---
            try:
                llm_output = llm_call(case_fact, category=category)
            except Exception as e:
                print(f"警告: 调用LLM处理案例 {i} 时出错: {e}")
                # 即使出错，也记录一个空结果，保证数据完整性
                llm_output = {"error": f"LLM调用失败: {str(e)}"}

            # --- 保存单个案例的结果 ---
            result_entry = {
                "case_index_in_category": i,
                "case_id": f"{category}_sample_{i}",
                "accusation": case_accusations,
                "original_fact": case_fact,
                "llm_analysis": llm_output
            }
            
            # 保存单个案例结果到临时文件
            temp_file_path = os.path.join(TEMP_OUTPUT_DIR, f"{category}_sample_{i}.json")
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                # 处理可能的非JSON可序列化对象
                try:
                    json.dump(result_entry, f, ensure_ascii=False, indent=2)
                except TypeError as e:
                    # 如果仍然有问题，将llm_analysis转换为字符串
                    result_entry["llm_analysis"] = str(result_entry["llm_analysis"])
                    json.dump(result_entry, f, ensure_ascii=False, indent=2)
            
            category_results.append(result_entry)

        # --- 保存按类别的汇总结果 ---
        category_output_path = os.path.join(CATEGORY_OUTPUT_DIR, f"{category}_discovered_factors.json")
        with open(category_output_path, 'w', encoding='utf-8') as f:
            json.dump(category_results, f, ensure_ascii=False, indent=2)
        
        all_results[category] = category_results
        print(f"类别 '{category}' 的分析结果已保存至 {category_output_path}")

    # --- 保存所有类别的最终汇总结果 ---
    with open(SUMMARY_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n--- 所有类别的分析完成 ---")
    print(f"最终汇总结果已保存至: {SUMMARY_OUTPUT_FILE}")
    print(f"各案例详细输出位于: {TEMP_OUTPUT_DIR}")
    print(f"各类别汇总输出位于: {CATEGORY_OUTPUT_DIR}")


def main():
    """主函数"""
    print("--- 启动因素发现Agent (步骤1.3) ---")
    
    # 1. 加载数据和映射
    print("1. 加载案件数据和罪名映射...")
    all_cases = load_cases(INPUT_FILE)
    accusation_mapping = load_json(ACCUSATION_MAPPING_FILE)
    print(f"   成功加载 {len(all_cases)} 个案件。")

    # 2. 分类案件
    print("2. 根据罪名对案件进行分类...")
    categorized_cases = categorize_cases(all_cases, accusation_mapping)
    print(f"   案件已分为 {len(categorized_cases)} 个类别。")

    # 3. 采样
    print(f"3. 从每个类别中随机采样 {SAMPLES_PER_CATEGORY} 个案例...")
    sampled_cases = sample_cases(categorized_cases, SAMPLES_PER_CATEGORY)

    # 4. 处理采样案例 (调用LLM并保存结果)
    print("4. 调用LLM分析采样案例并保存结果...")
    process_sampled_cases(sampled_cases)

    print("\n--- 因素发现Agent运行结束 ---")

if __name__ == "__main__":
    main()