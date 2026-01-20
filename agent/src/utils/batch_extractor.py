"""Batch processing for case information extraction."""

import json
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from .case_extractor import CaseExtractor

logger = logging.getLogger(__name__)

class BatchCaseExtractor:
    """批量案例信息提取器"""
    
    def __init__(self, batch_size: int = 5, max_workers: int = 3):
        """
        初始化批处理提取器
        
        Args:
            batch_size: 每批处理的案例数量
            max_workers: 最大并行工作线程数
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.extractor = CaseExtractor()
        
    def extract_case_info(self, fact: str, crime_str: str) -> Dict[str, Any]:
        """单个案例的提取方法，用于兼容性"""
        return self.extractor.extract_case_info(fact, crime_str)
    
    def _extract_batch(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理一批案例"""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for case in cases:
                fact = case.get('fact', '')
                accusation = case.get('accusation', [])
                accusation_str = '、'.join(accusation) if isinstance(accusation, list) else str(accusation)
                
                future = executor.submit(
                    self.extractor.extract_case_info,
                    fact,
                    accusation_str
                )
                futures.append((case, future))
            
            for case, future in futures:
                try:
                    extracted_info = future.result()
                    results.append({
                        'original': case,
                        'extracted': extracted_info
                    })
                except Exception as e:
                    logger.error(f"提取失败: {str(e)}")
                    results.append({
                        'original': case,
                        'extracted': {
                            'subject': '',
                            'action': '',
                            'result': '',
                            'others': ''
                        }
                    })
        
        return results
    
    def process_cases(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理案例
        
        Args:
            cases: 要处理的案例列表
            
        Returns:
            处理结果列表
        """
        results = []
        total_cases = len(cases)
        
        for i in range(0, total_cases, self.batch_size):
            batch = cases[i:i + self.batch_size]
            logger.info(f"处理批次 {i//self.batch_size + 1}/{(total_cases + self.batch_size - 1)//self.batch_size}")
            
            batch_results = self._extract_batch(batch)
            results.extend(batch_results)
            
        return results