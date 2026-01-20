import os
import time
import json
import threading
import numpy as np
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel 
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import array
from config import Config, ServiceConfig
from indexing import HNSWIndex, FAISSIndex
from logger import logger

#全局变量
vector_index = None
model = None
tokenizer = None
device = None
corpus_manager = None 
schema_keywords_manager = None

def cleanup_memory():
    """清理内存"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
            self.schema_line_offsets = array.array('Q')  
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
            self.keywords_line_offsets = array.array('Q')  
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


class CorpusManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.line_offsets = array.array('Q')  
        self.file_handle = None
        self.lock = threading.Lock() 

    def load_index(self):
        """扫描文件，建立行号到文件偏移量的索引"""
        if not os.path.exists(self.filepath):
            logger.error(f"原始数据文件不存在: {self.filepath}")
            return

        logger.debug(f"正在构建语料索引 (文件: {self.filepath})...")
        start = time.time()
        
        try:
            offset = 0
            self.line_offsets = array.array('Q')  
            self.line_offsets.append(offset)
            
            with open(self.filepath, 'rb') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    offset += len(line)
                    self.line_offsets.append(offset)
            
            self.line_offsets.pop()
            self.file_handle = open(self.filepath, 'r', encoding='utf-8')
            
            elapsed = time.time() - start
            logger.debug(f"语料索引构建完成，共 {len(self.line_offsets)} 条数据，耗时 {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"构建语料索引失败: {e}")
            raise e

    def get_doc(self, idx: int) -> Dict[str, Any]:
        """根据行号获取原始数据"""
        if not self.file_handle or idx >= len(self.line_offsets):
            return {}

        with self.lock: 
            try:
                offset = self.line_offsets[idx]
                self.file_handle.seek(offset)
                line = self.file_handle.readline()
                return json.loads(line)
            except Exception as e:
                logger.error(f"读取数据失败 (Index: {idx}): {e}")
                return {}

    def close(self):
        if self.file_handle:
            self.file_handle.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    服务生命周期：加载配置 -> 加载模型 -> 加载索引 -> 加载语料映射 -> 加载 Schema/Keywords 管理器
    """
    global vector_index, model, tokenizer, device, corpus_manager, schema_keywords_manager

    # 1. 根据配置加载嵌入模型
    logger.debug(f"正在加载嵌入模型，类型: {Config.EMBEDDING_MODEL_TYPE}")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if Config.EMBEDDING_MODEL_TYPE.lower() == "bge-m3":
            # 加载 BGE-M3 模型
            model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
            tokenizer = None  
            logger.debug(f"BGE-M3 模型加载成功 (Device: {device})")
        else:
            # 加载 Lawformer 模型
            tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
            model = AutoModel.from_pretrained(Config.MODEL_PATH).to(device)
            model.eval()
            logger.debug(f"Lawformer 模型加载成功 (Device: {device})")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise e

    # 2. 加载 Embeddings 并处理索引
    emb_path = Config.EMBEDDING_FILE
    index_path = ServiceConfig.get_index_path()

    # 确保目录存在
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    if not os.path.exists(emb_path):
         logger.error(f"致命错误: 找不到 Embeddings 文件: {emb_path}")
    else:
        try:
            # 强制为 float32 且为 C-contiguous，避免 faiss 报错
            embeddings = np.load(emb_path)
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

            dimension = embeddings.shape[1]
            logger.debug(f"Embeddings 加载成功，数量: {len(embeddings)}, 维度: {dimension}")
            
            if ServiceConfig.INDEX_TYPE == "hnsw":
                vector_index = HNSWIndex(dimension=dimension, index_file=index_path)
            else:
                vector_index = FAISSIndex(dimension=dimension, index_file=index_path)

            # 不再回退到通用索引：若模型专属索引不存在则构建新索引
            if os.path.exists(index_path):
                logger.debug(f"加载现有索引: {index_path}")
                vector_index.load(index_path)
            else:
                logger.debug(f"模型专属索引不存在，开始构建新索引: {index_path}")
                ids = [str(i) for i in range(len(embeddings))]
                vector_index.build(ids, embeddings)
                vector_index.save()
                logger.debug("索引构建完成")
                
        except Exception as e:
            logger.error(f"索引初始化失败: {str(e)}")
            raise e

    # 3. 初始化语料管理器
    try:
        corpus_manager = CorpusManager(Config.RAW_DATA_PATH)
        corpus_manager.load_index()
    except Exception as e:
        logger.warning(f"语料管理器初始化失败，搜索结果将不包含文本内容: {e}")

    # 4. 初始化 Schema 和 Keywords 管理器（用于回填 document_schema / document_keywords）
    try:
        schema_keywords_manager = SchemaKeywordsManager(Config.SCHEMA_path, Config.KEYWORDS_path)
        schema_keywords_manager.load_index()
        logger.debug("SchemaKeywordsManager 初始化成功")
    except Exception as e:
        logger.warning(f"SchemaKeywordsManager 初始化失败，document_schema/document_keywords 将为空: {e}")
        schema_keywords_manager = None
    yield
    
    # 关闭资源
    if corpus_manager:
        corpus_manager.close()
    if schema_keywords_manager:
        schema_keywords_manager.close()
    logger.info("服务正在关闭...")

app = FastAPI(title="Legal Fact Dense Retrieval Service", lifespan=lifespan)

# --- 数据模型 ---

class SearchRequest(BaseModel):
    fact: str          
    top_k: int = 10    
    threshold: Optional[float] = None

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

class BatchSearchRequest(BaseModel):
    facts: List[str]       
    top_k: int = 10



def get_query_embedding(text: str):
    if Config.EMBEDDING_MODEL_TYPE.lower() == "bge-m3":
        # 使用 BGE-M3 生成嵌入
        embeddings = model.encode([text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return embeddings['dense_vecs'][0].astype(np.float32)
    else:
        # 使用 Lawformer 生成嵌入
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=Config.MAX_SEQ_LENGTH
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        return cls_embedding

def get_batch_embeddings(texts: List[str]):
    """一次性计算多个文本的向量"""
    if Config.EMBEDDING_MODEL_TYPE.lower() == "bge-m3":
        # 使用 BGE-M3 批量生成嵌入
        embeddings = model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return embeddings['dense_vecs'].astype(np.float32)
    else:
        # 使用 Lawformer 批量生成嵌入
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=Config.MAX_SEQ_LENGTH
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embeddings.astype(np.float32)



@app.post("/batch_search", response_model=List[List[SearchResult]])
async def batch_search(request: BatchSearchRequest):
    if vector_index is None or model is None:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    try:
        start_time = time.time()
        batch_size = len(request.facts)
        
        if Config.VERBOSE:
            logger.info(f"Received Batch Request: {batch_size} items")

        # 1. 批量计算向量 
        query_vecs = get_batch_embeddings(request.facts)
        cleanup_memory()

        # 2. 逐个检索
        search_k = min(request.top_k * 3, 100)
        all_results = []
        
        for i in range(batch_size):
            vec = query_vecs[i]
            ids, scores = vector_index.search(vec.astype(np.float32), search_k)
            
            results = []
            seen_facts = set()
            
            for rank, (doc_id, score) in enumerate(zip(ids, scores)):
                if len(results) >= request.top_k:
                    break
                
                # 获取原始文档内容
                detail = {}
                if corpus_manager:
                    detail = corpus_manager.get_doc(int(doc_id))
                fact_text = detail.get('fact', 'Content not found')
                
                # 去重
                fact_hash = hash(fact_text)
                if fact_hash in seen_facts:
                    continue
                seen_facts.add(fact_hash)
                
                meta = detail.get('meta', {})
                
                # 获取 document_schema 和 document_keywords
                document_schema = {}
                document_keywords = {}
                if schema_keywords_manager is not None:
                    try:
                        doc_schema, doc_kw = schema_keywords_manager.get_schema_and_keywords(int(doc_id))
                        document_schema = doc_schema or {}
                        document_keywords = doc_kw or {}
                    except Exception as e:
                        logger.debug(f"Failed to load schema/keywords for doc_id={doc_id}: {e}")
                        document_schema = {}
                        document_keywords = {}

                results.append(SearchResult(
                    fact_id=str(doc_id),
                    score=float(score),
                    rank=len(results) + 1,
                    fact=fact_text,
                    accusation=meta.get('accusation', []),
                    relevant_articles=meta.get('relevant_articles', []),
                    imprisonment=meta.get('term_of_imprisonment', {
                        "death_penalty": False,
                        "life_imprisonment": False,
                        "imprisonment": 0
                    }),
                    punish_of_money=float(meta.get('punish_of_money', 0)),
                    criminals=meta.get('criminals', []),
                    matched_keywords={},          # 稠密无关键词匹配
                    query_schema=None,            # 稠密无查询结构化
                    document_schema=document_schema,
                    document_keywords=document_keywords,
                    laws=meta.get('relevant_articles', []),
                    metadata={
                        "raw_meta_sample": {
                            k: meta.get(k) for k in [
                                "accusation",
                                "relevant_articles",
                                "term_of_imprisonment",
                                "punish_of_money",
                                "criminals"
                            ]
                        },
                        "doc_index": int(doc_id),
                        "doc_id": str(doc_id)
                    },
                    orig_score=float(score)
                ))
            all_results.append(results)
            if (i + 1) % 100 == 0:  # 每处理100个请求清理一次
                cleanup_memory()

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Batch search completed in {elapsed:.2f}ms for {batch_size} items.")
        cleanup_memory()  
        return all_results

    except Exception as e:
        cleanup_memory()
        logger.error(f"Batch Search Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    if vector_index is None or model is None:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    try:
        start_time = time.time()
        
        if Config.VERBOSE:
            logger.info("="*50)
            logger.info(f"Received Search Request (top_k={request.top_k})")
            logger.info(f"Query: {request.fact[:200]}...") 

        # 1. 向量检索
        query_vec = get_query_embedding(request.fact)
        search_k = min(request.top_k * 3, 100) 
        ids, scores = vector_index.search(query_vec.astype(np.float32), search_k)
        
        # 2. 封装结果并回填内容
        results = []
        seen_facts = set() 
        
        for rank, (doc_id, score) in enumerate(zip(ids, scores)):
            if len(results) >= request.top_k:
                break

            detail = {}
            if corpus_manager:
                detail = corpus_manager.get_doc(int(doc_id))
            fact_text = detail.get('fact', 'Content not found')
            
            fact_hash = hash(fact_text)
            if fact_hash in seen_facts:
                continue
            seen_facts.add(fact_hash)
            
            meta = detail.get('meta', {})
            
            # 获取 document_schema 和 document_keywords
            document_schema = {}
            document_keywords = {}
            if schema_keywords_manager is not None:
                try:
                    doc_schema, doc_kw = schema_keywords_manager.get_schema_and_keywords(int(doc_id))
                    document_schema = doc_schema or {}
                    document_keywords = doc_kw or {}
                except Exception as e:
                    logger.debug(f"Failed to load schema/keywords for doc_id={doc_id}: {e}")
                    document_schema = {}
                    document_keywords = {}

            results.append(SearchResult(
                fact_id=str(doc_id),
                score=float(score),
                rank=len(results) + 1,
                fact=fact_text,
                accusation=meta.get('accusation', []),
                relevant_articles=meta.get('relevant_articles', []),
                imprisonment=meta.get('term_of_imprisonment', {
                    "death_penalty": False,
                    "life_imprisonment": False,
                    "imprisonment": 0
                }),
                punish_of_money=float(meta.get('punish_of_money', 0)),
                criminals=meta.get('criminals', []),
                matched_keywords={},          # 稠密无关键词匹配
                query_schema=None,            # 稠密无查询结构化
                document_schema=document_schema,
                document_keywords=document_keywords,
                laws=meta.get('relevant_articles', []),
                metadata={
                    "raw_meta_sample": {
                        k: meta.get(k) for k in [
                            "accusation",
                            "relevant_articles",
                            "term_of_imprisonment",
                            "punish_of_money",
                            "criminals"
                        ]
                    },
                    "doc_index": int(doc_id),
                    "doc_id": str(doc_id)
                },
                orig_score=float(score)
            ))
            
        elapsed = (time.time() - start_time) * 1000
        
        if Config.VERBOSE:
            logger.info(f"Search completed in {elapsed:.2f}ms. Found {len(results)} unique results.")
            for r in results:
                try:
                    laws = r.laws or r.relevant_articles or []
                    imprisonment = r.imprisonment or {}
                    accusation = r.accusation or []
                    criminals = r.criminals or []
                    fact_text = (r.fact or "").replace("\n", " ").strip()
                    logger.info(
                        f"Results[{r.rank}]\n"
                        f"    Score:{r.score:.4f}\n"
                        f"    罪名:{accusation}\n"
                        f"    法条:{laws}\n"
                        f"    刑期:{imprisonment}\n"
                        f"    罚金:{r.punish_of_money}元\n"
                        f"    犯罪人:{criminals}\n"
                        f"    案件:{fact_text[:500]}"
                    )
                except Exception as e:
                    logger.warning(f"打印结果详情失败: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"Search Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host=ServiceConfig.HOST, 
        port=ServiceConfig.PORT, 
        reload=False
    )