import os
import time
import json
import threading
import uvicorn
from concurrent.futures import ThreadPoolExecutor, as_completed 
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import array

from logger import logger
from config import ExtractorConfig, GlobalConfig
from utils import SparseRetriever
from crime_keywords_extractor import CrimeKeywordsExtractor
from universal_fact_extractor import UniversalFactExtractor


class CorpusManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.line_offsets = array.array('Q')
        self.lock = threading.Lock() 
        try:
            self.file_handle = open(filepath, 'r', encoding='utf-8')
            logger.debug(f"Successfully opened corpus file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to open corpus file: {e}")
            self.file_handle = None
        

    def load_index(self):
        """扫描文件，建立行号到文件偏移量的索引"""
        offset = 0
        with open(self.filepath, 'rb') as f:
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line)

    def get_doc(self, idx: int) -> Dict[str, Any]:
        """根据行号获取原始数据"""
        if not self.file_handle or idx >= len(self.line_offsets):
            logger.error(f"Invalid index: {idx}, max: {len(self.line_offsets)}")
            return {}
            
        with self.lock:
            try:
                self.file_handle.seek(self.line_offsets[idx])
                line = self.file_handle.readline()
                doc = json.loads(line)
                return doc
            except Exception as e:
                logger.error(f"获取文档失败: {e}")
                return {}

    def close(self):
        if self.file_handle:
            self.file_handle.close()

# ---------------------------------------------------------
# 1.5. Schema 和 Keywords 管理器 (用于按需读取结构化信息和关键词)
# ---------------------------------------------------------
class SchemaKeywordsManager:
    def __init__(self, schema_filepath: str, keywords_filepath: str):
        self.schema_filepath = schema_filepath
        self.keywords_filepath = keywords_filepath
        self.schema_line_offsets = array.array('Q')
        self.keywords_line_offsets = array.array('Q')
        self.schema_file_handle = None
        self.keywords_file_handle = None
        self.lock = threading.Lock() 


    def load_index(self):
        """扫描两个文件，建立行号到文件偏移量的索引"""
        if not os.path.exists(self.schema_filepath):
            logger.error(f"Schema数据文件不存在: {self.schema_filepath}")
            return
        if not os.path.exists(self.keywords_filepath):
            logger.error(f"Keywords数据文件不存在: {self.keywords_filepath}")
            return

        logger.debug(f"正在构建Schema索引 (文件: {self.schema_filepath})...")
        start = time.time()
        try:
            offset = 0
            self.schema_line_offsets = array.array('Q')  # 修改这里
            self.schema_line_offsets.append(offset)
            
            with open(self.schema_filepath, 'rb') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    offset += len(line)
                    self.schema_line_offsets.append(offset)
            
            self.schema_line_offsets.pop()
            self.schema_file_handle = open(self.schema_filepath, 'r', encoding='utf-8')
            
            elapsed = time.time() - start
            logger.debug(f"Schema索引构建完成，共 {len(self.schema_line_offsets)} 条数据，耗时 {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"构建Schema索引失败: {e}")
            raise e

        logger.debug(f"正在构建Keywords索引 (文件: {self.keywords_filepath})...")
        start = time.time()
        try:
            offset = 0
            self.keywords_line_offsets = array.array('Q')  # 修改这里
            self.keywords_line_offsets.append(offset)
            
            with open(self.keywords_filepath, 'rb') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    offset += len(line)
                    self.keywords_line_offsets.append(offset)
            
            self.keywords_line_offsets.pop()
            self.keywords_file_handle = open(self.keywords_filepath, 'r', encoding='utf-8')
            
            elapsed = time.time() - start
            logger.debug(f"Keywords索引构建完成，共 {len(self.keywords_line_offsets)} 条数据，耗时 {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"构建Keywords索引失败: {e}")
            raise e

    def get_schema_and_keywords(self, idx: int) -> tuple[dict, dict]:
        """根据行号获取对应的schema和keywords"""
        schema_result = {}
        keywords_result = {}
        success = True

        if not self.schema_file_handle or idx >= len(self.schema_line_offsets):
            logger.warning(f"Schema文件句柄无效或索引超出范围: {idx}")
            success = False
        else:
            with self.lock:
                try:
                    offset = self.schema_line_offsets[idx]
                    self.schema_file_handle.seek(offset)
                    line = self.schema_file_handle.readline()
                    item = json.loads(line)
                    schema_result = item.get("universal_fact", {})
                except Exception as e:
                    logger.error(f"读取Schema数据失败 (Index: {idx}): {e}")
                    success = False

        if not self.keywords_file_handle or idx >= len(self.keywords_line_offsets):
            logger.warning(f"Keywords文件句柄无效或索引超出范围: {idx}")
            success = False
        else:
            with self.lock:
                try:
                    offset = self.keywords_line_offsets[idx]
                    self.keywords_file_handle.seek(offset)
                    line = self.keywords_file_handle.readline()
                    item = json.loads(line)
                    doc_keywords = {}
                    if "sparse_extraction" in item and "keyword_counts" in item["sparse_extraction"]:
                         doc_keywords = item["sparse_extraction"]["keyword_counts"]
                    elif "keyword_counts" in item: 
                        doc_keywords = item["keyword_counts"]
                    keywords_result = doc_keywords
                except Exception as e:
                    logger.error(f"读取Keywords数据失败 (Index: {idx}): {e}")
                    success = False

        if not success:
            return {}, {}
        return schema_result, keywords_result


    def close(self):
        if self.schema_file_handle:
            self.schema_file_handle.close()
        if self.keywords_file_handle:
            self.keywords_file_handle.close()

# ---------------------------------------------------------
# 2. 全局变量与生命周期
# ---------------------------------------------------------
retriever: Optional[SparseRetriever] = None
extractor: Optional[CrimeKeywordsExtractor] = None
universal_extractor: Optional[UniversalFactExtractor] = None 
corpus_manager: Optional[CorpusManager] = None
schema_keywords_manager: Optional[SchemaKeywordsManager] = None 

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动与关闭的生命周期管理"""
    global retriever, extractor, universal_extractor, corpus_manager,schema_keywords_manager

    logger.info("🚀 正在启动稀疏检索服务...")

    # 1. 初始化关键词提取器 (恢复这部分代码)
    logger.debug(f"加载关键词提取器: {ExtractorConfig.KEYWORDS_FILE}")
    try:
        extractor = CrimeKeywordsExtractor(ExtractorConfig.KEYWORDS_FILE)
    except Exception as e:
        logger.error(f"关键词提取器加载失败: {e}")
        raise e

    # 1.5 初始化通用事实提取器
    logger.debug(f"加载通用事实提取器 (规则路径: {ExtractorConfig.RULES_YAML_PATH})...")
    try:
        universal_extractor = UniversalFactExtractor(rules_dir=ExtractorConfig.RULES_YAML_PATH)

    except Exception as e:
        logger.error(f"通用事实提取器加载失败: {e}")
        raise e

    # 2. 初始化稀疏检索器
    try:
        with open(ExtractorConfig.maps_path) as f:
            one_hot_maps = json.load(f)
        with open(ExtractorConfig.fields_path) as f:
            schema_fields = json.load(f)
            
        retriever = SparseRetriever(
            schema_fields=schema_fields,
            one_hot_maps=one_hot_maps,
            crime_keywords_path=ExtractorConfig.KEYWORDS_FILE
        )
        retriever.load_precomputed_data(ExtractorConfig.DB_PATH)
        logger.debug("稀疏检索器初始化成功")
    except Exception as e:
        logger.error(f"初始化稀疏检索器失败: {e}")
        raise e

    # 3. 初始化语料管理器 (读取原始数据用于展示)
    try:
        corpus_manager = CorpusManager(str(ExtractorConfig.INPUT_DATASET))
        corpus_manager.load_index()
        # 测试读取第一条数据
        first_doc = corpus_manager.get_doc(0)
        if not first_doc:
            logger.error("无法读取语料数据，请检查数据文件!")
        else:
            logger.debug(f"成功加载语料数据，示例数据: {json.dumps(first_doc['meta'], ensure_ascii=False)}")
    except Exception as e:
        logger.error(f"语料管理器初始化失败: {e}")

    # 4. 初始化 Schema 和 Keywords 管理器 
    try:
        schema_keywords_manager = SchemaKeywordsManager(ExtractorConfig.SCHEMA_PATH, ExtractorConfig.KEYWORDS_PATH)
        schema_keywords_manager.load_index()
    except Exception as e:
        logger.warning(f"Schema和Keywords管理器初始化失败: {e}")

    logger.info("✅ 服务启动完成，准备就绪。")
    try:
        yield 
    finally:
        if corpus_manager:
            corpus_manager.close()
        if schema_keywords_manager: 
            schema_keywords_manager.close()
        logger.info("服务已关闭。")

app = FastAPI(title="Legal Sparse Retrieval Service", lifespan=lifespan)

# ---------------------------------------------------------
# 3. 数据模型 (Pydantic)
# ---------------------------------------------------------
class SearchRequest(BaseModel):
    fact: str
    top_k: int = 5

class BatchSearchRequest(BaseModel):
    facts: List[str]
    top_k: int = 5

class SearchResult(BaseModel):
    fact_id: str
    score: float
    rank: int
    fact: str = ""
    accusation: List[str] = []
    relevant_articles: List[str] = []
    imprisonment: Dict[str, Any] = {}  
    punish_of_money: float = 0.0      
    criminals: List[str] = []        
    matched_keywords: Dict[str, int] = {}
    query_schema: Optional[Dict[str, Any]] = None
    document_schema: Optional[Dict[str, Any]] = None
    document_keywords: Optional[Dict[str, int]] = None
    laws: List[str] = []
    metadata: Optional[Dict[str, Any]] = None
    orig_score: Optional[float] = None



# ---------------------------------------------------------
# 4. 接口实现
# ---------------------------------------------------------

def _process_single_search(fact: str, top_k: int) -> List[SearchResult]:
    """内部处理单条检索逻辑"""
    # 1. 提取关键词频次
    extraction = extractor.extract(fact)
    query_counts = extraction.get("keyword_counts", {})
    
    # 2. 提取结构化 Schema 
    query_schema_flat = {}
    raw_query_schema = {} 

    if universal_extractor and retriever and getattr(retriever, 'schema_fields', []):
        # 提取原始嵌套结构
        raw_query_schema = universal_extractor.extract_from_fact(fact)
        
        # 扁平化处理用于检索
        def flatten(x, name=''):
            out = {}
            if isinstance(x, dict):
                for a in x: 
                    out.update(flatten(x[a], name + a + '.'))
            else: 
                out[name[:-1]] = x
            return out
            
        query_schema_flat = flatten(raw_query_schema)

    # 3. 执行检索
    try:
        raw_results = retriever.search(query_schema_flat, query_counts, top_k=top_k)
    except ValueError as e:
        logger.error(f"检索失败: {e}")
        return []
    
    # 4. 格式化结果
    logger.debug(f"Query Schema: {raw_query_schema}")
    logger.debug(f"Query Keywords: {query_counts}")
    
    formatted_results = []
    for rank, res in enumerate(raw_results):
        try:
            doc_info = corpus_manager.get_doc(res['index']) if corpus_manager else {}
            if not doc_info:
                logger.error(f"无法获取文档信息: index={res['index']}")
                continue
                
            meta = doc_info.get("meta", {})
            if not meta:
                logger.error(f"文档缺少meta信息: index={res['index']}")
                
            # 添加调试信息
            logger.debug(f"处理文档: index={res['index']}, meta={json.dumps(meta, ensure_ascii=False)}")
            

            document_schema = {}
            document_keywords = {}
            if schema_keywords_manager:
                doc_schema, doc_keywords_dict = schema_keywords_manager.get_schema_and_keywords(res['index'])
                document_schema = doc_schema
                document_keywords = doc_keywords_dict
            else:
                logger.warning(f"SchemaKeywordsManager 未初始化，无法获取 index {res['index']} 的 schema 和 keywords。")
            # --- ---

            # 调试日志
            logger.debug(f"Doc Info: {json.dumps(doc_info, ensure_ascii=False)}")
            logger.debug(f"Meta Info: {json.dumps(meta, ensure_ascii=False)}")
            
            formatted_results.append(SearchResult(
                fact_id=str(res['id']),
                score=float(res['score']),
                rank=rank + 1,
                fact=doc_info.get("fact", ""),
                accusation=meta.get("accusation", []),
                relevant_articles=meta.get("relevant_articles", []),
                imprisonment=meta.get("term_of_imprisonment", {
                    "death_penalty": False,
                    "life_imprisonment": False,
                    "imprisonment": 0
                }),
                punish_of_money=float(meta.get("punish_of_money", 0)),
                criminals=meta.get("criminals", []),
                matched_keywords=query_counts,
                query_schema=raw_query_schema,
                document_schema=document_schema,     
                document_keywords=document_keywords,
                laws=meta.get("relevant_articles", []),
                metadata={
                    "raw_meta_sample": meta,
                    "doc_index": res.get("index"),
                    "doc_id": res.get("id")
                }
            ))
        except Exception as e:
            logger.error(f"处理文档失败 (FACT_ID: {res.get('id')}): {e}", exc_info=True)
            continue
    
    return formatted_results

@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    if not retriever or not extractor:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    try:
        start_time = time.time()
        results = _process_single_search(request.fact, request.top_k)
        elapsed = (time.time() - start_time) * 1000
        
        logger.info(f"Search processed in {elapsed:.2f}ms. Found {len(results)} results.")
        
        for r in results:
            try:
                fact_text = (r.fact or "").replace("\n", " ").strip()
                logger.info(
                    f"\n{'='*80}\n"
                    f"  Search Result [Rank {r.rank}]:\n"
                    f"  Score: {r.score:.4f}\n"
                    f"  Fact_ID: {r.fact_id}\n"
                    f"  罪名: {r.accusation}\n"
                    f"  法条: {r.relevant_articles}\n"
                    f"  刑期: {r.imprisonment}\n"
                    f"  罚金: {r.punish_of_money}元\n"
                    f"  犯罪人: {r.criminals}\n"
                    f"  Schema: {json.dumps(r.query_schema, ensure_ascii=False, indent=2)}\n"
                    f"  Keywords: {json.dumps(r.matched_keywords, ensure_ascii=False, indent=2)}\n"
                    f"  案件: {fact_text[:200]}...\n"
                    f"{'='*80}"
                )
            except Exception as e:
                logger.warning(f"打印结果详情失败: {e}")
        
        return results
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/batch_search", response_model=List[List[SearchResult]])
async def batch_search(request: BatchSearchRequest):
    if not retriever or not extractor:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    try:
        start_time = time.time()
        
        batch_results = [None] * len(request.facts)
        
        max_workers = min(16, len(request.facts))
        
        logger.info(f"Starting batch search for {len(request.facts)} items with {max_workers} threads...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_process_single_search, fact, request.top_k): i
                for i, fact in enumerate(request.facts)
            }
            
            # 获取结果
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    batch_results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
                    batch_results[idx] = [] 

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Batch search ({len(request.facts)} items) processed in {elapsed:.2f}ms. Avg: {elapsed/len(request.facts):.2f}ms/item")
        return batch_results
        
    except Exception as e:
        logger.error(f"Batch Search Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=GlobalConfig.PORT,
        log_level="info", 
        reload=False
    )
