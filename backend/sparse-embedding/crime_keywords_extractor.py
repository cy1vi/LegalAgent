import json
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from collections import defaultdict, Counter
from config import ExtractorConfig
from flashtext import KeywordProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

global_extractor = None

def init_worker(keywords_path: Path):
    """
    多进程初始化函数：每个进程只加载一次关键词
    """
    global global_extractor
    global_extractor = CrimeKeywordsExtractor(keywords_path)

def worker_process_line(line: str) -> Optional[str]:
    """
    多进程处理函数：处理单行文本
    """
    global global_extractor
    try:
        line = line.strip()
        if not line:
            return None

        item = json.loads(line)
        fact = item.get("fact") or item.get("content") or item.get("description") or ""
        
        # 调用全局 extractor 进行提取
        extraction_result = global_extractor.extract(fact)

        output_item = {
            "meta": item.get("meta", {}),
            "sparse_extraction": extraction_result
        }
        
        if "id" in item:
            output_item["id"] = item["id"]

        return json.dumps(output_item, ensure_ascii=False)
    except Exception:
        return None

# ----------------------------
# 主类
# ----------------------------
class CrimeKeywordsExtractor:
    def __init__(self, keywords_path: Path):
        self.keywords_path = keywords_path
        self.crime_keywords_map = self._load_keywords()
        self.keyword_processor = KeywordProcessor()
        self.kw_to_crimes = defaultdict(list)
        
        total_kw = 0
        for crime, keywords in self.crime_keywords_map.items():
            for kw in keywords:
                self.keyword_processor.add_keyword(kw)
                self.kw_to_crimes[kw].append(crime)
                total_kw += 1
        
        if multiprocessing.current_process().name == 'MainProcess':
            logger.debug(f"Loaded {len(self.crime_keywords_map)} crime categories with {total_kw} total keywords.")

    def _load_keywords(self) -> Dict[str, List[str]]:
        if not self.keywords_path.exists():
            raise FileNotFoundError(f"Keywords file not found: {self.keywords_path}")
        
        with open(self.keywords_path, "r", encoding="utf-8") as f:
            return json.load(f)  

    def extract(self, text: str) -> Dict[str, Any]:
        if not text:
            return self._empty_result()

        found_keywords = self.keyword_processor.extract_keywords(text)
        
        if not found_keywords:
            return self._empty_result()

        keyword_counts = Counter(found_keywords)
        crime_counts = Counter()
        crime_keywords_list = defaultdict(list)
        matched_details = defaultdict(dict)

        for kw, count in keyword_counts.items():
            crimes = self.kw_to_crimes.get(kw, [])
            for crime in crimes:
                crime_counts[crime] += count
                crime_keywords_list[crime].append(kw)
                matched_details[crime][kw] = count

        return {
            "keyword_counts": dict(keyword_counts),
            "crime_counts": dict(crime_counts),
            "crime_keywords": dict(crime_keywords_list),
            "matched_crimes": list(crime_counts.keys()),
            "details": dict(matched_details)
        }

    def _empty_result(self):
        return {
            "keyword_counts": {},
            "crime_counts": {},
            "crime_keywords": {},
            "matched_crimes": [],
            "details": {}
        }

    def _count_lines(self, filepath: str) -> int:
        try:
            with open(filepath, "rb") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def process_dataset_multiprocess(self, input_path: Path, output_path: Path, sample_limit: Optional[int] = None):
        """
        多进程处理入口
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_lines = self._count_lines(str(input_path)) if sample_limit is None else sample_limit
        
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"🚀 Starting MULTIPROCESS extraction with {num_processes} cores.")
        logger.info(f"Input: {input_path} -> Output: {output_path}")

        processed = 0
        
        def line_generator():
            with open(input_path, "r", encoding="utf-8") as fin:
                for i, line in enumerate(fin):
                    if sample_limit and i >= sample_limit:
                        break
                    yield line

        # 开启进程池
        with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(self.keywords_path,)) as pool:
            with open(output_path, "w", encoding="utf-8") as fout:
                for result in tqdm(pool.imap(worker_process_line, line_generator(), chunksize=1000), total=total_lines):
                    if result:
                        fout.write(result + "\n")
                        processed += 1

        logger.info(f"✅ Completed! Processed {processed} cases.")
        
        # 生成统计报告 
        report_path = str(output_path).replace(".jsonl", "_stats.txt")
        self.generate_stats_report(str(output_path), report_path)

    def generate_stats_report(self, output_path: str, report_path: str):
        logger.info("Generating statistics...")
        global_crime_counts = Counter()
        global_keyword_counts = Counter()
        total_docs = 0
        docs_with_matches = 0

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        res = item.get("sparse_extraction", {})
                        c_counts = res.get("crime_counts", {})
                        k_counts = res.get("keyword_counts", {})
                        
                        if k_counts: docs_with_matches += 1
                        global_crime_counts.update(c_counts)
                        global_keyword_counts.update(k_counts)
                        total_docs += 1
                    except: continue

            with open(report_path, "w", encoding="utf-8") as rpt:
                rpt.write(f"📊 Crime Keywords Extraction Statistics\n{'='*50}\n")
                rpt.write(f"Total Documents: {total_docs}\n")
                rpt.write(f"Docs with Matches: {docs_with_matches}\n\n")
                rpt.write(f"## Top 50 Crimes\n")
                for k, v in global_crime_counts.most_common(50): rpt.write(f"  {k:<30}: {v}\n")
                rpt.write(f"\n## Top 50 Keywords\n")
                for k, v in global_keyword_counts.most_common(50): rpt.write(f"  {k:<30}: {v}\n")
            logger.info(f"✅ Statistics report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate stats: {e}")

if __name__ == "__main__":
    extractor = CrimeKeywordsExtractor(keywords_path=ExtractorConfig.KEYWORDS_FILE)
    
    extractor.process_dataset_multiprocess(
        input_path=ExtractorConfig.INPUT_DATASET,
        output_path=ExtractorConfig.OUTPUT_DATASET,
        sample_limit=ExtractorConfig.SAMPLE_LIMIT
    )