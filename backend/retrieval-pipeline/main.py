from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time
from config import PipelineConfig, SearchMode, BaseConfig
from reranker import RetrievalPipeline  
from logger import logger

app = FastAPI(title="Retrieval Pipeline API")
logger.info("正在初始化 Pipeline API 服务...")

try:
    config = PipelineConfig.from_env()
    pipeline = RetrievalPipeline(config)
    logger.info("Pipeline API 服务初始化完成")
except Exception as e:
    logger.error(f"服务初始化失败: {str(e)}", exc_info=True)
    raise


class SearchRequest(BaseModel):
    query: str
    mode: SearchMode = SearchMode.HYBRID
    top_k: Optional[int] = None
    rerank: bool = True

class BatchSearchRequest(BaseModel):
    queries: List[str]
    mode: SearchMode = SearchMode.HYBRID
    top_k: Optional[int] = None
    rerank: bool = True

@app.post("/search")
async def search(request: SearchRequest):
    """单条查询接口"""
    try:
        start_time = time.time()
        logger.info(f"收到搜索请求 - Query: {request.query[:200]}...")
        results = await pipeline.search(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            rerank=request.rerank
        )
        
        elapsed = (time.time() - start_time) * 1000
        if BaseConfig.VERBOSE:
            logger.info(f"Search completed in {elapsed:.2f}ms. Found {len(results)} unique results.")
            for r in results:
                try:
                    laws = r.get("relevant_articles", [])
                    imprisonment = r.get("imprisonment", {})
                    accusation = r.get("accusation", [])
                    criminals = r.get("criminals", [])
                    fact_text = (r.get("fact", "")).replace("\n", " ").strip()
                    score = r.get("score", 0.0)
                    rank = r.get("rank", 0)
                    
                    logger.info(
                        f"  Results[{rank}]\n"
                        f"  Score:{score:.4f}\n"
                        f"  罪名:{accusation}\n"
                        f"  法条:{laws}\n"
                        f"  刑期:{imprisonment}\n"
                        f"  罚金:{r.get('punish_of_money', 0)}元\n"
                        f"  犯罪人:{criminals}\n"
                        f"  案件:{fact_text[:500]}"
                    )
                except Exception as e:
                    logger.warning(f"打印结果详情失败: {e}")
        
        return {"results": results}
    except Exception as e:
        logger.error(f"搜索请求处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_search")
async def batch_search(request: BatchSearchRequest):
    """批量查询接口"""
    try:
        logger.info(f"收到批量搜索请求 - {len(request.queries)} 条查询")
        results = await pipeline.batch_search(
            queries=request.queries,
            mode=request.mode,
            top_k=request.top_k,
            rerank=request.rerank
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"批量搜索请求处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info" if config.debug else "error"
    )