import requests
import time
import json

class TestConfig:
    BASE_URL = "http://localhost:4241"
    SEARCH_URL = f"{BASE_URL}/search"
    BATCH_SEARCH_URL = f"{BASE_URL}/batch_search"


def test_search(query_fact: str, top_k: int = 10):
    print("=" * 60)
    print(f"🔍 单条检索测试 (top_k={top_k})")
    print(f"Query preview: {query_fact[:60]}...")

    payload = {"fact": query_fact, "top_k": top_k}

    try:
        start_time = time.time()
        response = requests.post(TestConfig.SEARCH_URL, json=payload, timeout=30)
        api_time = (time.time() - start_time) * 1000

        if response.status_code != 200:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)
            return

        results = response.json()
        print(f"✅ 成功返回 {len(results)} 条结果 | 耗时: {api_time:.2f}ms\n")

        for i, item in enumerate(results, 1):
            fact_id = item.get("fact_id", "N/A")
            score = item.get("score", 0.0)
            rank = item.get("rank", i)
            fact = item.get("fact", "")[:120] + "..."

            # 结构化字段
            accusation = item.get("accusation", [])
            laws = item.get("laws", []) or item.get("relevant_articles", [])
            imprisonment = item.get("imprisonment", {})
            punish_money = item.get("punish_of_money", 0)

            # 新增：document_schema 和 document_keywords
            doc_schema = item.get("document_schema", {})
            doc_keywords = item.get("document_keywords", {})

            print(f"[{rank}] ID: {fact_id} | Score: {score:.4f}")
            print(f"    📌 罪名: {accusation}")
            print(f"    ⚖️ 法条: {laws}")
            print(f"    ⏳ 刑期: {imprisonment}")
            print(f"    💰 罚金: {punish_money} 元")
            print(f"    📄 案情: {fact}")
            
            if doc_schema:
                print(f"    🧩 Schema: {doc_schema}")
            if doc_keywords:
                print(f"    🔑 Keywords (top): {list(doc_keywords.keys())[:5]}")
            print("-" * 50)

    except requests.exceptions.ConnectionError:
        print(f"🔌 无法连接服务，请确认服务运行在 {TestConfig.BASE_URL}")
    except Exception as e:
        print(f"💥 测试异常: {e}")
        import traceback
        traceback.print_exc()


def test_batch_search(query_facts: list, top_k: int = 5):
    print("\n" + "=" * 60)
    print(f"📦 批量检索测试 (batch_size={len(query_facts)}, top_k={top_k})")

    payload = {"facts": query_facts, "top_k": top_k}

    try:
        start_time = time.time()
        response = requests.post(TestConfig.BATCH_SEARCH_URL, json=payload, timeout=60)
        api_time = (time.time() - start_time) * 1000

        if response.status_code != 200:
            print(f"❌ 批量请求失败: {response.status_code}")
            print(response.text)
            return

        batch_results = response.json()
        total_results = sum(len(r) for r in batch_results)
        print(f"✅ 批量完成 | 总结果数: {total_results} | 耗时: {api_time:.2f}ms\n")

        for idx, results in enumerate(batch_results):
            print(f"--- Query {idx + 1}: {query_facts[idx][:50]}...")
            if not results:
                print("    ❗ 无结果")
                continue
            for r in results[:2]:  # 只打印前2个
                print(f"    [Rank {r.get('rank', '?')}] Score: {r['score']:.4f} | 罪名: {r.get('accusation', [])}")
            print()

    except Exception as e:
        print(f"💥 批量测试异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 示例长案情
    full_query = """经审理查明,被告人吴某1、吴某2、王某1、吴某3、董某1、吴某4原系安顺市西秀区东关办事处麒麟社区居委会工作人员,
    吴某1系居委会主任,吴某2系居委会委员兼会计,王某1系居委会委员兼出纳,吴某3系居委会支部书记,董某1系居委会委员兼人口主任,吴某4系居委会委员。
    从2014年2月12日起,麒麟社区居委会与安顺市公共交通总公司签订公交车发车站管护协议,约定由麒麟社区居委会负责东站进口至公交车站出口等场地的秩序管理、公交车夜间停放的看护,
    安顺市公共交通总公司按月支付麒麟社区居委会管护费,并由麒麟社区居委会收取外来车辆临时停车费。此后,双方一直按协议各自履行权利义务,麒麟社区居委会收取的管护费由吴某2保管,
    临时停车费由王某1保管。2015年年初,麒麟社区居委会开会时,被告人吴某1提议将居委会收取的管护费及临时停车费的余款以年终补助形式发放,被告人吴某3、吴某2、王某1、董某1、吴某4均表示同意。
    此后从2015年2月至2017年9月,被告人吴某1、吴某2、王某1、吴某3、董某1、吴某4将居委会收取的管护费及临时停车费的余款196300元以发年终补助的形式进行私分,
    其中吴某1分得40500元,吴某2分得42700元,王某1分得40100元,吴某3分得24000元,董某1分得25000元,吴某4分得24000元。案发后六被告人在公安机关退清所得赃款。"""

    test_search(full_query, top_k=3)

    # 批量测试（可选）
    short_queries = [
        "被告人贪污公款十万元",
        "酒后驾驶机动车被查获",
        "非法吸收公众存款用于放贷"
    ]
    test_batch_search(short_queries, top_k=3)