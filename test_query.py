import requests
from typing import List, Dict, Any

def search_local(
    query: str,
    local_base_url: str = "http://localhost:8000",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    向本地检索服务发起查询，返回检索结果。

    Args:
        query (str): 用户输入的查询文本。
        local_base_url (str): 本地检索 API 的基础 URL（默认 http://localhost:8000）。
        top_k (int): 返回的最多结果数量（默认 5）。

    Returns:
        List[Dict[str, Any]]: 检索结果列表，每个元素包含：
            - doc_id (str)
            - chunk_id (str)
            - text (str)
            - score (float)
            - metadata (dict)
    """
    try:
        # 发送 POST 请求到本地检索服务
        response = requests.post(
            f"{local_base_url}/search",
            json={
                "query": query,
                "mode": "hybrid",
                "top_k": top_k,
                "rerank": True
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # 按优先级获取结果：reranked > dense > sparse
        search_results = (
            data.get("reranked_results") or
            data.get("dense_results") or
            data.get("sparse_results") or
            []
        )

        results = []
        for idx, item in enumerate(search_results):
            doc_id = item.get("doc_id", "")
            chunk_id = item.get("chunk_id", f"{doc_id}_chunk_{idx}")
            text = item.get("text", "")
            # 优先使用 rerank_score，否则用 score
            score = item.get("rerank_score", item.get("score", 0.0))
            metadata = item.get("metadata", {})

            results.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": text,
                "score": float(score),
                "metadata": metadata
            })

        return results

    except requests.exceptions.RequestException as e:
        print(f"[Error] Failed to call local search service: {e}")
        return []

# ========================
# 示例用法（可选）
# ========================
if __name__ == "__main__":
    # 示例查询
    query_text = "被告人吴某某伤人,致使易某的脸部、耳朵等部位被划伤，伤情构成轻伤一级"
    
    # 调用函数（请确保你的本地服务正在运行，例如在 http://localhost:8000）
    results = search_local(query=query_text, local_base_url="http://localhost:4242", top_k=3)

    # 打印结果
    print(f"共检索到 {len(results)} 条结果：\n")
    for i, res in enumerate(results, 1):
        print(f"【结果 {i}】(得分: {res['score']:.4f})")
        print(f"文档ID: {res['doc_id']}")
        print(f"文本: {res['text'][:200]}{'...' if len(res['text']) > 200 else ''}")
        print(f"元数据: {res['metadata']}\n")