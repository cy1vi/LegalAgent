import asyncio
import json
import aiohttp
from typing import Dict, Any, List

async def test_search(session: aiohttp.ClientSession, base_url: str, query: str, rerank: bool = True, top_k: int = 3):
    payload = {
        "query": query,
        "mode": "hybrid",
        "top_k": top_k,
        "rerank": rerank
    }
    async with session.post(f"{base_url}/search", json=payload) as response:
        response.raise_for_status()
        results = await response.json()
        return results

def safe_join(value):
    """安全地将列表或值转为字符串"""
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value) if value is not None else ""

def print_results(results: Dict[str, Any], label: str):
    print(f"\n=== {label} ===")
    
    for i, item in enumerate(results.get("results", []), 1):
        meta = item.get("metadata", {})
        
        print(f"\n--- 结果 {i} ---")
        print(f"Rerank_Score: {float(item.get('score', 0.0)):.6f}")
        print(f"Fact_ID: {item.get('id', 'N/A')}")
        print(f"罪名: {item.get('accusation', '')}")
        print(f"法条: {item.get('relevant_articles', '')}")
        print(f"刑期: {item.get('imprisonment', '')}")
        print(f"罚金: {item.get('punish_of_money', '')}")
        print(f"犯罪人: {item.get('criminals', '')}")
        print(f"Schema: {item.get('document_schema', '')}")
        print(f"Keywords: {(item.get('document_keywords', ''))}")
        print(f"案件: {item.get('fact', '')}")

async def main():
    base_url = "http://localhost:8000"
    query = "被告人王某某在KTV内对李某某实施殴打，导致轻伤二级"

    async with aiohttp.ClientSession() as session:
        try:
            print("\n🔍 正在测试 Rerank 结果...")
            results = await test_search(session, base_url, query, rerank=True, top_k=3)
            print_results(results, "Rerank 检索结果")

        except Exception as e:
            print(f"❌ 请求失败: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(main())