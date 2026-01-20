# universal_fact_extractor.py
import re
import json
import yaml
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any, List
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from collections import defaultdict, Counter
from config import ExtractorConfig
# ----------------------------
# 配置日志
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("extraction.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


class RuleMatcher:
    def __init__(self, rules_dict: Dict[str, Any]):
        self.rules = rules_dict
        self._precompile_regex()

    def _precompile_regex(self):
        """预编译所有正则表达式，提升匹配速度"""
        for field, rule in self.rules.items():
            rtype = rule.get("type")
            if rtype == "regex":
                patterns = rule.get("patterns", [])
                # 将字符串模式转换为编译后的正则对象
                rule["_compiled_patterns"] = [re.compile(p) for p in patterns]

    def match(self, field: str, text: str) -> Optional[Any]:
        rule = self.rules.get(field)
        if not rule:
            return None

        rtype = rule.get("type")
        
        if rtype == "keyword":
            keywords = rule.get("keywords", [])
            for kw in keywords:
                if kw in text:
                    return True
            return None

        elif rtype == "regex":
            # 使用预编译的正则对象
            patterns = rule.get("_compiled_patterns", [])
            for pat in patterns:
                if pat.search(text):
                    return True
            return None

        elif rtype == "mapping":
            mappings = rule.get("mappings", {})
            for label, config in mappings.items():
                for kw in config.get("keywords", []):
                    if kw in text:
                        return label
            return None

        return None


# ----------------------------
# 全局变量与 Worker 函数 (用于多进程)
# ----------------------------

# 全局变量，用于在子进程中缓存 Extractor 实例
_global_extractor = None

def init_worker(rules_dir: str):
    """子进程初始化函数：加载规则"""
    global _global_extractor
    # 禁止子进程打印加载日志，避免控制台刷屏
    worker_logger = logging.getLogger()
    worker_logger.setLevel(logging.WARNING)
    _global_extractor = UniversalFactExtractor(rules_dir, is_worker=True)

def process_batch(lines: List[str]) -> List[str]:
    """子进程处理函数：处理一批数据"""
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            fact = item.get("fact", "")
            # 调用全局实例进行抽取
            universal_fact = _global_extractor.extract_from_fact(fact)

            output_item = {
                "meta": item.get("meta", {}),
                "fact": fact,
                "universal_fact": universal_fact
            }
            results.append(json.dumps(output_item, ensure_ascii=False))
        except Exception:
            # 忽略错误行，避免中断
            continue
    return results


# ----------------------------
# 通用事实抽取器
# ----------------------------

class UniversalFactExtractor:
    def __init__(self, rules_dir: str = "rules", is_worker: bool = False):
        self.rules_dir = Path(rules_dir)
        
        if not is_worker:
            logger.debug(f"Loading rule files from {self.rules_dir}...")
            
        self.act_obj_matcher = RuleMatcher(self._load_yaml("act.yaml"))
        self.result_matcher = RuleMatcher(self._load_yaml("result.yaml"))
        self.participation_matcher = RuleMatcher(self._load_yaml("participation.yaml"))
        self.context_matcher = RuleMatcher(self._load_yaml("context.yaml"))
        
        if not is_worker:
            logger.debug("All rules loaded successfully.")

    def _load_yaml(self, filename: str) -> Dict:
        path = self.rules_dir / filename
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def extract_from_fact(self, fact: str) -> Dict[str, Dict]:
        clean_fact = fact.replace("\r", "").replace("\n", " ").strip()
        
        return {
            "act": {
                "has_violence": self.act_obj_matcher.match("has_violence", clean_fact),
                "violence_level": self.act_obj_matcher.match("violence_level", clean_fact),
                "has_threat": self.act_obj_matcher.match("has_threat", clean_fact),
                "is_secret": self.act_obj_matcher.match("is_secret", clean_fact),
                "is_deceptive": self.act_obj_matcher.match("is_deceptive", clean_fact),
                "has_conspiracy": self.act_obj_matcher.match("has_conspiracy", clean_fact),
                "used_tool": self.act_obj_matcher.match("used_tool", clean_fact),
            },
            "object": {
                "is_person": self.act_obj_matcher.match("is_person", clean_fact),
                "is_property": self.act_obj_matcher.match("is_property", clean_fact),
                "is_public_order": self.act_obj_matcher.match("is_public_order", clean_fact),
                "is_state_interest": self.act_obj_matcher.match("is_state_interest", clean_fact),
                "property_type": self.act_obj_matcher.match("property_type", clean_fact),
            },
            "result": {
                "injury": self.result_matcher.match("injury", clean_fact),
                "injury_level": self.result_matcher.match("injury_level", clean_fact),
                "death": self.result_matcher.match("death", clean_fact),
                "property_transferred": self.result_matcher.match("property_transferred", clean_fact),
                "amount_mentioned": self.result_matcher.match("amount_mentioned", clean_fact),
                "has_restitution": self.result_matcher.match("has_restitution", clean_fact),
                "has_forgiveness": self.result_matcher.match("has_forgiveness", clean_fact),
                "has_confession": self.result_matcher.match("has_confession", clean_fact),
            },
            "participation": {
                "has_multiple_offenders": self.participation_matcher.match("has_multiple_offenders", clean_fact),
                "has_organization": self.participation_matcher.match("has_organization", clean_fact),
                "role_description": self.participation_matcher.match("role_description", clean_fact),
            },
            "context": {
                "is_indoor": self.context_matcher.match("is_indoor", clean_fact),
                "is_public_place": self.context_matcher.match("is_public_place", clean_fact),
                "is_night": self.context_matcher.match("is_night", clean_fact),
                "is_online": self.context_matcher.match("is_online", clean_fact),
            }
        }

    def _count_lines(self, filepath: str) -> int:
        try:
            with open(filepath, "rb") as f:
                count = sum(1 for _ in f)
            return count
        except Exception:
            return 0

    def process_dataset(self, input_path: str, output_path: str, sample_limit: Optional[int] = None, batch_size: int = 2000):
        """
        使用多进程并行处理数据集
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_lines = self._count_lines(str(input_path)) if sample_limit is None else sample_limit
        
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        logger.debug(f"Starting PARALLEL extraction using {max_workers} cores.")
        logger.debug(f"Input: {input_path} -> Output: {output_path}")

        processed_count = 0
        
        def line_generator():
            with open(input_path, "r", encoding="utf-8") as fin:
                batch = []
                for i, line in enumerate(fin):
                    if sample_limit and i >= sample_limit:
                        break
                    batch.append(line)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch

        # 使用 ProcessPoolExecutor 进行并行处理
        with open(output_path, "w", encoding="utf-8") as fout:
            with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(str(self.rules_dir),)) as executor:
                
                # 提交任务
                futures = []
                # 计算总 batch 数用于进度条
                total_batches = (total_lines // batch_size) + 1 if total_lines else None
                
                pbar = tqdm(total=total_lines, desc="Processing", unit="case")
                
                for batch_results in executor.map(process_batch, line_generator()):
                    # 写入结果
                    for res_line in batch_results:
                        fout.write(res_line + "\n")
                    
                    count = len(batch_results)
                    processed_count += count
                    pbar.update(count)

                pbar.close()

        logger.info(f"✅ Completed! Processed {processed_count} cases. Output saved to {output_path}")
        
        # 生成统计报告
        report_path = str(output_path).replace(".jsonl", "_stats.txt")
        self.generate_stats_report(str(output_path), report_path)

    def generate_stats_report(self, output_path: str, report_path: str = "output/stats_report.txt"):

        stats = defaultdict(int)
        total = 0
        value_counters = defaultdict(Counter)

        logger.info(f"Generating statistics from {output_path}...")

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        uf = item.get("universal_fact", {})
                        total += 1

                        for category, fields in uf.items():
                            for field, value in fields.items():
                                full_key = f"{category}.{field}"
                                if value is not None:
                                    stats[full_key] += 1
                                    if isinstance(value, str):
                                        value_counters[full_key][value] += 1
                    except Exception as e:
                        continue

            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as rpt:
                rpt.write(f"📊 Universal Fact Extraction Statistics\n")
                rpt.write(f"{'='*50}\n")
                rpt.write(f"Total cases processed: {total}\n\n")

                categories = ["act", "object", "result", "participation", "context"]
                for cat in categories:
                    rpt.write(f"## {cat.upper()}\n")
                    cat_fields = [k for k in stats.keys() if k.startswith(cat + ".")]
                    if not cat_fields:
                        rpt.write("  (no data)\n\n")
                        continue

                    for key in sorted(cat_fields):
                        hit = stats[key]
                        rate = hit / total * 100 if total > 0 else 0
                        rpt.write(f"  {key:30} : {hit:6} hits ({rate:5.2f}%)\n")

                        if key in value_counters:
                            dist = value_counters[key]
                            dist_str = ", ".join([f"{k}({v})" for k, v in dist.most_common()])
                            rpt.write(f"    └─ Values: {dist_str}\n")
                    rpt.write("\n")

            logger.info(f"✅ Statistics report saved to {report_path}")

        except Exception as e:
            logger.error(f"Failed to generate stats report: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    extractor = UniversalFactExtractor(rules_dir=ExtractorConfig.RULES_YAML_PATH)
    extractor.process_dataset(
        input_path=ExtractorConfig.INPUT_DATASET,
        output_path= ExtractorConfig.SCHEMA_PATH,
        sample_limit=None,
        batch_size=5000 
    )