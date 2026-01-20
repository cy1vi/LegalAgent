import json
import os
import random
from collections import defaultdict
from tqdm import tqdm

# --- 自定义模块 ---
from agent import LegalCaseSchemaExtractor
from build_one_prompt import build_prompt_no_template as bp  

# --- 配置 ---
INPUT_FILE = r"D:\deeplearning\project_learning\LegalAgent\dataset\final_all_data\first_stage\train.json"

OUTPUT_DIR = r"D:\deeplearning\project_learning\LegalAgent\statistics_analyze\discovered_factors_by_accusation_no_template"
TEMP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "temp_individual_outputs")
CATEGORY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "by_accusation")
SUMMARY_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_discovered_factors_no_template.json")

SAMPLES_PER_ACCUSATION = 20  # 每个罪名最多采样数量


def load_cases(file_path):
    """从 JSONL 文件加载案件"""
    cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def categorize_cases_by_accusation(cases):
    """按具体罪名分类（取第一个指控为主罪）"""
    categorized = defaultdict(list)
    for case in cases:
        meta = case.get("meta", {})
        accusations = meta.get("accusation", [])
        if accusations:
            main_acc = accusations[0]
            categorized[main_acc].append(case)
        else:
            categorized["未知罪名"].append(case)
    return dict(categorized)


def sample_cases_by_accusation(categorized_cases, samples_per_accusation):
    """对每个罪名采样"""
    sampled = {}
    for accusation, case_list in categorized_cases.items():
        n = len(case_list)
        if n <= samples_per_accusation:
            sampled[accusation] = case_list
            print(f"罪名 '{accusation}' 案例数 ({n}) ≤ {samples_per_accusation}，使用全部。")
        else:
            sampled[accusation] = random.sample(case_list, samples_per_accusation)
            print(f"罪名 '{accusation}' 已采样 {samples_per_accusation} 条。")
    return sampled


def llm_call(case_fact, accusation_name):
    """安全拼接 prompt，避免格式化错误"""
    base_prompt = bp(accusation_name)
    extractor = LegalCaseSchemaExtractor()
    
    # 修改：使用 analyze_case 替代 extract_schema
    # extract_schema 会强制添加额外的格式化指令，干扰我们自定义的 prompt
    # analyze_case 则直接传递 system_prompt 和 user_content
    response = extractor.analyze_case(system_prompt=base_prompt, user_content=case_fact,stream=False)
    
    # response = extractor.extract_schema(case_fact, prompt_override=base_prompt, stream=False)  
    return response



def process_sampled_cases(sampled_cases):
    """处理每个罪名的采样案件"""
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CATEGORY_OUTPUT_DIR, exist_ok=True)

    all_results = {}

    for accusation, cases in sampled_cases.items():
        print(f"\n--- 开始处理罪名: {accusation} (共 {len(cases)} 条) ---")
        results = []

        for i, case in enumerate(tqdm(cases, desc=f"{accusation}", ncols=100)):
            fact = case.get("fact", "")
            meta = case.get("meta", {})
            original_accusations = meta.get("accusation", [accusation])

            try:
                llm_output = llm_call(fact, accusation)
                # 尝试解析 JSON
                if isinstance(llm_output, str):
                    try:
                        parsed = json.loads(llm_output)
                        llm_output = parsed
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                llm_output = {"error": f"LLM 调用异常: {str(e)}"}

            result_entry = {
                "case_index": i,
                "case_id": f"{accusation}_sample_{i}",
                "original_accusations": original_accusations,
                "fact": fact,
                "llm_analysis": llm_output
            }

            # 保存单条
            temp_path = os.path.join(TEMP_OUTPUT_DIR, f"{accusation}_sample_{i}.json")
            with open(temp_path, 'w', encoding='utf-8') as f:
                try:
                    json.dump(result_entry, f, ensure_ascii=False, indent=2)
                except TypeError:
                    result_entry["llm_analysis"] = str(result_entry["llm_analysis"])
                    json.dump(result_entry, f, ensure_ascii=False, indent=2)

            results.append(result_entry)

        # 保存该罪名汇总
        category_path = os.path.join(CATEGORY_OUTPUT_DIR, f"{accusation}_factors.json")
        with open(category_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        all_results[accusation] = results
        print(f"罪名 '{accusation}' 结果已保存至: {category_path}")

    # 保存总汇总
    with open(SUMMARY_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 所有罪名处理完成！")
    print(f"总汇总文件: {SUMMARY_OUTPUT_FILE}")


def main():
    print("🚀 启动【无模板】按罪名因素发现流程...")

    # 1. 加载案件
    print("1. 加载案件数据...")
    all_cases = load_cases(INPUT_FILE)
    print(f"   共加载 {len(all_cases)} 条案件。")

    # 2. 按罪名分组
    print("2. 按具体罪名分组...")
    categorized = categorize_cases_by_accusation(all_cases)
    print(f"   共识别出 {len(categorized)} 个罪名。")

    # 3. 采样（可选：只处理高频罪名，避免冷门罪名浪费资源）
    # 这里我们处理所有罪名，但你可以加过滤条件，例如：
    # filtered = {k: v for k, v in categorized.items() if len(v) >= 5}
    sampled = sample_cases_by_accusation(categorized, SAMPLES_PER_ACCUSATION)

    # 4. 处理
    print("3. 调用 LLM 提取因素（无模板）...")
    process_sampled_cases(sampled)

    print("\n🎉 流程结束！")


if __name__ == "__main__":
    main()