"""Test the agent's case analysis functionality with batch processing."""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent import AgenticRAG
from config import Config

# 设置随机种子以确保可重现性
random.seed(42)

# 获取当前脚本所在目录
SCRIPT_DIR = Path(__file__).parent
LOGS_DIR = SCRIPT_DIR / 'logs'
OUTPUT_DIR = SCRIPT_DIR / 'output' / 'agent_test_results'

# 重试配置
MAX_RETRIES = 3
RETRY_DELAY = 2  # 重试间隔（秒）

# 创建必要的目录
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 配置日志
log_file = LOGS_DIR / f'agent_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_test_cases(file_path: Path, sample_size: int = 5, min_length: int = 200) -> List[Dict[str, Any]]:
    """从测试文件中随机加载案例
    Args:
        file_path: 数据文件路径
        sample_size: 采样数量
        min_length: 最小字数要求
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 先加载所有案例并筛选长度
            cases = []
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    if len(case.get('fact', '')) >= min_length:
                        cases.append(case)
        
        logger.info(f"总案例数: {len(cases)}")
        logger.info(f"符合长度要求(>={min_length}字)的案例数: {len(cases)}")
        
        sampled_cases = random.sample(cases, min(sample_size, len(cases)))
        logger.info(f"已采样 {len(sampled_cases)} 个案例")
        return sampled_cases
    except Exception as e:
        logger.error(f"加载测试案例失败: {str(e)}")
        return []

def save_results(results: List[Dict[str, Any]], output_dir: Path):
    """保存测试结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存完整结果
    result_file = output_path / f"agent_test_results_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存简要统计
    stats_file = output_path / f"agent_test_stats_{timestamp}.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"测试时间: {timestamp}\\n")
        f.write(f"测试案例数: {len(results)}\\n\\n")
        
        # 计算平均处理时间
        avg_time = sum(r['processing_time'] for r in results) / len(results) if results else 0
        f.write(f"平均处理时间: {avg_time:.2f} 秒\\n\\n")
        
        # 输出样例分析
        f.write("=== 分析样例 ===\\n")
        if results:
            sample = results[0]
            f.write(f"原始案情: {sample['original_fact'][:200]}...\\n\\n")
            f.write(f"分析结果:\\n{sample['analysis']}\\n")
    
    logger.info(f"结果已保存至: {output_path}")

def process_single_case(agent: AgenticRAG, case: Dict[str, Any], case_id: int) -> Optional[Dict[str, Any]]:
    """
    处理单个案例的辅助函数
    Args:
        agent: AgenticRAG实例
        case: 案例数据
        case_id: 案例ID
    Returns:
        处理结果字典或None
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            fact = case.get('fact', '')
            start_time = time.time()
            
            # 获取分析结果
            analysis = agent.chat(fact)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 保存结果
            result = {
                'case_id': case_id,
                'original_fact': fact,
                'original_meta': case.get('meta', {}),
                'analysis': analysis,
                'processing_time': processing_time
            }
            
            logger.info(f"案例 {case_id} 处理完成，用时 {processing_time:.2f} 秒")
            return result
            
        except Exception as e:
            retry_count += 1
            logger.warning(f"案例 {case_id} 处理失败 (第{retry_count}次): {str(e)}")
            if retry_count < max_retries:
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"案例 {case_id} 在重试 {max_retries} 次后仍然失败")
                return None

def process_batch(agent: AgenticRAG, cases: List[Dict[str, Any]], batch_size: int = 3) -> List[Dict[str, Any]]:
    """
    并行处理案例
    Args:
        agent: AgenticRAG实例
        cases: 案例列表
        batch_size: 并发数量（同时处理的案例数）
    Returns:
        处理结果列表
    """
    results = []
    total_cases = len(cases)
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_single_case, agent, case, case_id): case_id 
            for case_id, case in enumerate(cases, 1)
        }
        
        logger.info(f"已提交 {total_cases} 个案例到线程池（并发数: {batch_size}）")
        
        # 处理完成的任务
        completed_count = 0
        batch_start = time.time()
        
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    completed_count += 1
                    logger.info(f"进度: {completed_count}/{total_cases}")
                    
            except Exception as e:
                logger.error(f"案例 {case_id} 处理异常: {str(e)}")
        
        total_time = time.time() - batch_start
        logger.info(f"\n并行处理完成！")
        logger.info(f"成功处理案例数: {len(results)}/{total_cases}")
        logger.info(f"总用时: {total_time:.2f} 秒")
        if len(results) > 0:
            logger.info(f"平均每个案例用时: {total_time/len(results):.2f} 秒")
    
    return results

def main():
    # 设置参数
    test_file = Path("F:/LegalAgent/dataset/final_all_data/final_test.jsonl")
    sample_size = 10  # 随机采样数量
    min_length = 200  # 最小字数要求
    batch_size = 3   # 批处理大小
    
    # 初始化 agent
    logger.info("初始化 AgenticRAG...")
    config = Config.from_env()
    agent = AgenticRAG(config)
    
    # 加载测试案例
    logger.info(f"从 {test_file} 加载测试案例...")
    test_cases = load_test_cases(test_file, sample_size, min_length)
    if not test_cases:
        logger.error("没有加载到测试案例")
        return
    
    # 批量处理测试案例
    start_time = time.time()
    results = process_batch(agent, test_cases, batch_size)
    total_time = time.time() - start_time
    
    logger.info(f"\n所有案例处理完成！")
    logger.info(f"总用时: {total_time:.2f} 秒")
    logger.info(f"平均每个案例用时: {total_time/len(test_cases):.2f} 秒")
    
    # 保存结果
    save_results(results, OUTPUT_DIR)
    logger.info("测试完成")

if __name__ == "__main__":
    main()

