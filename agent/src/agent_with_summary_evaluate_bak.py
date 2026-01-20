import json
import logging
import os
import asyncio
from typing import Dict, Any, List, Union
from sklearn.metrics import accuracy_score, f1_score

from agent import AgenticRAG

from utils.batch_extractor import BatchCaseExtractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent_evaluation.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置文件路径
SEARCH_RESULTS_FILE = r"F:\LegalAgent\agent\src\datasets\search_results_300.jsonl"
TRAIN_DATA_FILE = r"F:\LegalAgent\dataset\final_all_data\first_stage\train.json"
OUTPUT_FILE = r"F:\LegalAgent\agent\src\output\evaluation_results.json"

class AgentEvaluator:
    def __init__(self):
        self.agent = AgenticRAG()
        self.batch_extractor = BatchCaseExtractor(max_workers=5)  # 初始化批量提取器
        self.train_data = self._load_train_data()
        self.n_references = 3  # 用于Prompt的参考案例数量



    def _load_train_data(self) -> Dict[str, Any]:
        """加载训练数据，用于查找top-5的详细信息"""
        logger.info(f"正在加载训练数据: {TRAIN_DATA_FILE}")
        data_map = {}
        try:
            with open(TRAIN_DATA_FILE, 'r', encoding='utf-8') as f:
                # 读取第一个字符判断格式
                first_char = f.read(1)
                f.seek(0)
                
                if first_char == '[':
                    # JSON List格式
                    logger.info("检测到文件格式为 JSON List")
                    data = json.load(f)
                    for idx, item in enumerate(data):
                        data_map[str(idx)] = item
                else:
                    # JSONL格式 (每行一个JSON对象)
                    logger.info("检测到文件格式为 JSONL")
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            data_map[str(idx)] = item
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析第 {idx} 行")
                            continue
                            
            logger.info(f"成功加载 {len(data_map)} 条训练数据")
            
        except Exception as e:
            logger.error(f"加载训练数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        return data_map

    def _get_top_5_details(self, top_5_ids: List[str]) -> List[Dict[str, Any]]:
        """根据ID获取top-5的详细信息"""
        details = []
        for doc_id in top_5_ids:
            doc_id_str = str(doc_id)
            if doc_id_str in self.train_data:
                details.append(self.train_data[doc_id_str])
            else:
                logger.warning(f"未在训练集中找到ID: {doc_id_str}")
        return details


    def _construct_prompt(self, target_fact: str, structured_references: List[Dict[str, Any]]) -> str:
        """
        构建Agent的输入Prompt
        :param target_fact: 目标案件的原始事实文本 (未结构化)
        :param structured_references: 已结构化的参考案例列表
        """
        
        # 构建参考案例部分 (已结构化)
        references_text = ""
        for i, ref in enumerate(structured_references):
            # 提取结构化信息
            subject = ref.get('subject', '未知')
            action = ref.get('action', '未知')
            result = ref.get('result', '未知')
            others = ref.get('others', '无')
            accusation = ref.get('original_accusation', '未知') # 在处理时注入原始罪名
            
            references_text += (
                f"参考案例 {i+1}:\n"
                f"主体: {subject}\n"
                f"行为: {action}\n"
                f"结果: {result}\n"
                f"其他: {others}\n"
                f"判决罪名: {accusation}\n\n"
            )

        # 构建目标案件部分 (原始文本)
        target_case_text = (
            f"目标案件事实:\n"
            f"{target_fact}\n"
        )

        prompt = (
            f"你是一个法律专家。请参考以下{len(structured_references)}个已进行结构化分析（主体、行为、结果）的相似案例及其判决罪名，"
            f"对目标案件的原始事实进行分析，并推断最可能的罪名。\n\n"
            f"【参考案例】\n{references_text}"
            f"【目标案件】\n{target_case_text}\n"
            f"请直接给出建议的罪名，不需要过多的分析过程。格式：建议罪名：xxx"
        )
        return prompt

    def _clean_accusation(self, accusation: Any) -> str:
        """清洗罪名，去除'罪'字等，以便比较"""
        if not accusation:
            return ""
            
        # 处理列表情况
        if isinstance(accusation, list):
            if len(accusation) > 0:
                accusation = accusation[0]
            else:
                return ""
                
        if not isinstance(accusation, str):
            accusation = str(accusation)
            
        cleaned = accusation.replace("罪", "").strip()
        return cleaned

    def run_evaluation(self):
        logger.info(f"开始评估，读取搜索结果: {SEARCH_RESULTS_FILE}")
        
        results = []
        y_true = []
        y_pred = []

        try:
            with open(SEARCH_RESULTS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        fact_id = item.get('fact_id')
                        fact = item.get('fact') # 目标案件原始事实
                        true_accusation = item.get('accusation')
                        top_5_ids = item.get('top_5_ids', [])

                        logger.info(f"正在处理 fact_id: {fact_id}")

                        # 1. 获取参考案例详情 (取前 N 个)
                        top_n_ids = top_5_ids[:self.n_references]
                        top_n_details = self._get_top_5_details(top_n_ids)
                        
                        if not top_n_details:
                            logger.warning(f"fact_id {fact_id} 没有找到有效的参考案例")

                        # 2. 对参考案例进行结构化提取 (使用 BatchCaseExtractor)
                        structured_references = []
                        batch_input = []
                        batch_accusations = []

                        for ref_item in top_n_details:
                            ref_fact = ref_item.get('fact', '')
                            ref_acc_raw = ref_item.get('meta', {}).get('accusation', [])
                            
                            # 格式化参考案例的罪名
                            if isinstance(ref_acc_raw, list):
                                ref_acc_str = ", ".join(ref_acc_raw)
                            else:
                                ref_acc_str = str(ref_acc_raw)

                            batch_input.append((ref_fact, ref_acc_str))
                            batch_accusations.append(ref_acc_str)

                        # 批量提取
                        if batch_input:
                            extracted_results = self.batch_extractor.extract_batch(batch_input)
                            
                            for i, extracted_info in enumerate(extracted_results):
                                # 将原始罪名附加到结构化信息中，以便 Prompt 使用
                                extracted_info['original_accusation'] = batch_accusations[i]
                                structured_references.append(extracted_info)

                        # 3. 构建 Prompt (目标案件使用原始 fact，参考案例使用 structured_references)
                        prompt = self._construct_prompt(fact, structured_references)

                        # 4. 调用 Agent
                        messages = [
                            {"role": "system", "content": self.agent.system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                        
                        response = self.agent.client.chat.completions.create(
                            model=self.agent.model,
                            messages=messages,
                            temperature=0.1
                        )
                        
                        agent_output = response.choices[0].message.content.strip()
                        
                        # 5. 提取预测罪名
                        predicted_accusation = ""
                        if "建议罪名：" in agent_output:
                            predicted_accusation = agent_output.split("建议罪名：")[1].split("\n")[0].strip()
                        elif "建议罪名:" in agent_output:
                            predicted_accusation = agent_output.split("建议罪名:")[1].split("\n")[0].strip()
                        else:
                            predicted_accusation = agent_output

                        # 记录结果
                        result_record = {
                            "fact_id": fact_id,
                            "original_fact": fact,
                            "true_accusation": true_accusation,
                            "structured_references": structured_references, # 记录参考案例的结构化结果
                            "agent_input_prompt": prompt,
                            "agent_output_raw": agent_output,
                            "predicted_accusation": predicted_accusation
                        }
                        results.append(result_record)
                        
                        # 准备计算指标
                        y_true.append(self._clean_accusation(true_accusation))
                        y_pred.append(self._clean_accusation(predicted_accusation))

                        # 实时写入文件 (JSONL格式)
                        with open(OUTPUT_FILE + "l", 'a', encoding='utf-8') as out_f:
                            out_f.write(json.dumps(result_record, ensure_ascii=False) + '\n')

                    except Exception as e:
                        logger.error(f"处理 fact_id {fact_id} 时出错: {e}")
                        continue

            # 计算指标
            if not y_true:
                logger.error("没有有效的结果用于计算指标")
                return

            accuracy = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average='macro')

            logger.info(f"评估完成。")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Macro-F1: {macro_f1:.4f}")

            # 保存最终结果
            final_output = {
                "metrics": {
                    "accuracy": accuracy,
                    "macro_f1": macro_f1
                },
                "details": results
            }

            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
            
            logger.info(f"结果已保存至: {OUTPUT_FILE}")

        except Exception as e:
            logger.error(f"评估过程发生严重错误: {e}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    evaluator = AgentEvaluator()
    evaluator.run_evaluation()