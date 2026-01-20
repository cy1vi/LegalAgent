# universal_fact_extractor.py
import re
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm import tqdm

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


# ----------------------------
# 规则加载器
# ----------------------------

class RuleMatcher:
    def __init__(self, rules_dict: Dict[str, Any]):
        self.rules = rules_dict

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
            patterns = rule.get("patterns", [])
            for pat in patterns:
                if re.search(pat, text):
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
# 通用事实抽取器
# ----------------------------

class UniversalFactExtractor:
    def __init__(self, rules_dir: str = "rules"):
        self.rules_dir = Path(rules_dir)

        # 加载所有规则
        logger.info("Loading rule files...")
        self.act_obj_matcher = RuleMatcher(self._load_yaml("act.yaml"))
        self.result_matcher = RuleMatcher(self._load_yaml("result.yaml"))
        self.participation_matcher = RuleMatcher(self._load_yaml("participation.yaml"))
        self.context_matcher = RuleMatcher(self._load_yaml("context.yaml"))
        logger.info("All rules loaded successfully.")

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
        """快速估算行数（用于 tqdm total）"""
        try:
            with open(filepath, "rb") as f:
                count = sum(1 for _ in f)
            return count
        except Exception:
            return 0  # 无法统计时禁用 total

    def process_dataset(self, input_path: str, output_path: str, sample_limit: Optional[int] = None):
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_lines = self._count_lines(str(input_path)) if sample_limit is None else sample_limit
        logger.info(f"Starting extraction from {input_path} → {output_path}")

        processed = 0
        error_count = 0

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            # 使用 tqdm 进度条
            pbar = tqdm(
                fin,
                total=total_lines if total_lines > 0 else None,
                desc="Processing cases",
                unit="case"
            )

            for line in pbar:
                if sample_limit and processed >= sample_limit:
                    break
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    fact = item.get("fact", "")
                    universal_fact = self.extract_from_fact(fact)

                    output_item = {
                        "meta": item["meta"],
                        "fact": fact,
                        "universal_fact": universal_fact
                    }
                    fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")

                    processed += 1
                    pbar.set_postfix({"errors": error_count})

                except json.JSONDecodeError:
                    error_count += 1
                    logger.warning(f"JSON decode error at line {processed + error_count}: {line[:100]}...")
                    continue
                except Exception as e:
                    error_count += 1
                    logger.error(f"Unexpected error at line {processed + error_count}: {e}")
                    continue

        logger.info(f"✅ Completed! Processed {processed} cases, {error_count} errors. Output saved to {output_path}")
        # 自动生成统计报告
        report_path = str(output_path).replace(".jsonl", "_stats.txt")
        self.generate_stats_report(str(output_path), report_path)

    def generate_stats_report(self, output_path: str, report_path: str = "output/stats_report.txt"):
        """
        从已生成的 universal_facts.jsonl 中统计各字段命中情况
        """
        from collections import defaultdict, Counter
        import os

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

                        # 遍历所有字段
                        for category, fields in uf.items():
                            for field, value in fields.items():
                                full_key = f"{category}.{field}"
                                if value is not None:
                                    stats[full_key] += 1
                                    # 记录具体值（用于枚举型字段）
                                    if isinstance(value, str):
                                        value_counters[full_key][value] += 1
                    except Exception as e:
                        logger.warning(f"Skip invalid line in stats: {e}")
                        continue

            # 写入报告
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as rpt:
                rpt.write(f"📊 Universal Fact Extraction Statistics\n")
                rpt.write(f"{'='*50}\n")
                rpt.write(f"Total cases processed: {total}\n\n")

                # 按类别分组输出
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

                        # 如果有值分布，也打印
                        if key in value_counters:
                            dist = value_counters[key]
                            dist_str = ", ".join([f"{k}({v})" for k, v in dist.most_common()])
                            rpt.write(f"    └─ Values: {dist_str}\n")
                    rpt.write("\n")

            logger.info(f"✅ Statistics report saved to {report_path}")

        except Exception as e:
            logger.error(f"Failed to generate stats report: {e}")
if __name__ == "__main__":
    extractor = UniversalFactExtractor(rules_dir="F:\\LegalAgent\\rules_yaml")
    extractor.process_dataset(
        input_path="F:\\LegalAgent\\dataset\\final_all_data\\first_stage\\test.json",
        output_path="output/universal_facts.jsonl",
        sample_limit=None  # 或设为 1000 测试
    )