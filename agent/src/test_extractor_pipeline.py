"""Test pipeline for case information extraction."""

import json
import logging
import random
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from utils.batch_extractor import BatchCaseExtractor
from config import Config

# 获取当前脚本所在目录
SCRIPT_DIR = Path(__file__).parent
LOGS_DIR = SCRIPT_DIR / 'logs'
OUTPUT_DIR = SCRIPT_DIR / 'output' / 'extraction_results'

# 批处理配置
BATCH_SIZE = 5
MAX_WORKERS = 3

# 创建必要的目录
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 配置日志
log_file = LOGS_DIR / f'extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_jsonl(file_path: Path, sample_size: int = 10) -> List[Dict[str, Any]]:
    """从JSONL文件中随机加载指定数量的案例."""
    data = []
    try:
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return []
            
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                logger.error("文件为空")
                return []
                
            logger.info(f"总案例数量: {len(lines)}")
            # 随机采样
            sample_size = min(sample_size, len(lines))
            sampled_lines = random.sample(lines, sample_size)
            
            for i, line in enumerate(sampled_lines, 1):
                try:
                    case_data = json.loads(line)
                    data.append(case_data)
                    logger.debug(f"成功加载第 {i} 个案例")
                except json.JSONDecodeError as je:
                    logger.error(f"JSON解析错误 (第 {i} 个案例): {str(je)}")
                    continue
                    
            logger.info(f"成功加载 {len(data)} 个案例")
            return data
            
    except Exception as e:
        logger.error(f"加载数据文件失败: {str(e)}")
        return []

def process_cases(cases: List[Dict[str, Any]], batch_extractor: BatchCaseExtractor) -> List[Dict[str, Any]]:
    """使用批处理器并行处理案例并提取结构化信息."""
    try:
        logger.info(f"开始批量处理 {len(cases)} 个案例...")
        start_time = datetime.now()
        
        # 使用批处理器并行处理案例
        results = batch_extractor.process_cases(cases)
        
        # 添加处理时间信息
        processed_cases = []
        for result in results:
            result['processing_time'] = datetime.now().isoformat()
            processed_cases.append(result)
            
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        logger.info(f"批量处理完成，总用时: {total_time:.2f} 秒")
        logger.info(f"平均每个案例用时: {total_time/len(cases):.2f} 秒")
        
        return processed_cases
        
    except Exception as e:
        logger.error(f"批量处理过程中出错: {str(e)}")
        return []

def save_results(results: List[Dict[str, Any]], output_dir: Path):
    """保存处理结果."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存完整结果
    full_output_file = output_path / f"extraction_results_{timestamp}.json"
    with open(full_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存简要统计
    stats_file = output_path / f"extraction_stats_{timestamp}.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"处理时间: {timestamp}\\n")
        f.write(f"总处理案例数: {len(results)}\\n")
        f.write("\\n=== 提取样例 ===\\n")
        if results:
            sample_case = results[0]
            f.write(f"原始案情: {sample_case['original']['fact'][:200]}...\\n")
            f.write(f"提取结果:\\n{json.dumps(sample_case['extracted'], ensure_ascii=False, indent=2)}\\n")
    
    logger.info(f"结果已保存至: {output_path}")

def main():
    # 设置参数
    input_file = Path("F:/LegalAgent/dataset/final_all_data/final_test.jsonl")
    sample_size = 10  # 随机采样数量
    
    logger.info(f"开始处理，采样数量: {sample_size}")
    
    # 初始化批处理提取器
    config = Config.from_env()
    batch_extractor = BatchCaseExtractor(batch_size=BATCH_SIZE, max_workers=MAX_WORKERS)
    
    # 加载数据
    cases = load_jsonl(input_file, sample_size)
    if not cases:
        logger.error("没有加载到有效数据")
        return
    
    # 批量处理案例
    results = process_cases(cases, batch_extractor)
    
    # 保存结果
    save_results(results, OUTPUT_DIR)
    
    logger.info("处理完成")

if __name__ == "__main__":
    main()