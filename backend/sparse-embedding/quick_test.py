import requests
import time
import json

class TestConfig:
    SERVICE_URL = "http://localhost:4240/search"

def print_schema(schema_dict, indent=2):
    """辅助函数：美化打印 Schema"""
    if not schema_dict:
        return
    prefix = " " * indent
    for cat, fields in schema_dict.items():
        # 只打印非空字段
        valid_fields = {k: v for k, v in fields.items() if v is not None}
        if valid_fields:
            print(f"{prefix}- {cat}: {valid_fields}")

def test_search(query_fact, top_k=5):
    print("-" * 60)
    print(f"🔍 测试查询: {query_fact}...")
    
    payload = {
        "fact": query_fact,
        "top_k": top_k
    }
    
    try:
        start_time = time.time()
        response = requests.post(TestConfig.SERVICE_URL, json=payload)
        
        if response.status_code != 200:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)
            return

        results = response.json()
        api_time = (time.time() - start_time) * 1000
        print(f"✅ API 响应时间: {api_time:.2f}ms, 返回 {len(results)} 个结果")
        
        print("\n=== 检索结果详情 ===")
        
        print("reslults:", results)  
        # 1. 打印查询特征 (使用 query_schema)
        if results and len(results) > 0:
            first_res = results[0]
            print(f"🔎 [查询特征分析]")
            
            q_schema = first_res.get('query_schema')
            if q_schema:
                print("  > Query Schema:")
                print_schema(q_schema, indent=4)
            else:
                print("  > Query Schema: (未提取到有效 Schema)")
            
            kws = first_res.get('matched_keywords', {})
            if kws:
                print(f"    - Keywords: {kws}")
            else:
                # [修改] 明确打印未命中
                print(f"    - Keywords: (未命中任何关键词)") 
            print("-" * 30)

        # 2. 打印结果列表
        print("results:", results)
        for item in results:
            fid = item.get('fact_id', 'N/A')
            score = item.get('score', 0.0)
            rank = item.get('rank', 0)
            
            # 获取所有字段
            accusation = item.get('accusation', [])
            relevant_articles = item.get('relevant_articles', [])
            imprisonment = item.get('imprisonment', {})
            punish_of_money = item.get('punish_of_money', 0)
            criminals = item.get('criminals', [])
            doc_schema = item.get('document_schema')
            doc_keywords = item.get('document_keywords', {})
            
            print(f"\n[Rank {rank}] (Fact_ID: {fid}, Score: {score:.4f})")
            print(f"  > 罪名: {accusation}")
            print(f"  > 法条: {relevant_articles}")
            print(f"  > 刑期: {imprisonment}")
            print(f"  > 罚金: {punish_of_money}元")
            print(f"  > 犯罪人: {criminals}")
                
            # 展示文档 Keywords
            if doc_keywords:
                print(f"  > Document Keywords (命中词频):")
                for keyword, freq in doc_keywords.items():
                    print(f"      - {keyword}: {freq}")
            else:
                print(f"  > Document Keywords: (无)")

            # 展示文档 Schema
            if doc_schema:
                print(f"  > Document Schema (提取特征):")
                print_schema(doc_schema, indent=6)
            else:
                print(f"  > Document Schema: (无)")
                
            # 案情预览 - 增加长度并格式化
            fact = item.get('fact', 'N/A')
            preview = fact[:500].replace('\n', ' ').strip()
            print(f"  > 案情预览: \n    {preview}...")
                
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")

if __name__ == "__main__":
    print("测试罪名：非法持有、私藏枪支、弹药, 赌博")
    full_query = """巍山县人民检察院指控,一、2017年2月中旬至2017年2月22日期间,被告人刘1某吉、罗2某组织、邀约杨某某、曹某某、胡某某等20余名参赌人员在巍山县南诏镇向阳村荒田篝鱼塘鱼棚内,以?摇宝?方式聚众赌博,期间刘1某吉安排被告人罗1某放哨,赌博期间抽头渔利达9000余元。2017年2月22日,巍山县公安局南诏派出所民警在刘1某吉鱼棚内将21名参赌人员当场查获。\n二、2016年初,被告人罗3某持一支射钉枪到南诏镇向阳村荒田箐被告人刘1某吉看守的鱼塘打野鸭,后将该射钉枪存放于刘1某吉鱼棚内。2017年2月22日,巍山县公安局南诏派出所民警在查获刘1某吉等人赌博一案过程中,将该射钉枪查获。经鉴定,该射钉枪是以火药为动力的自制枪,具有致伤力,认定为枪支。\n公诉机关认为,被告人刘1某吉、罗2某以营利为目的,多次组织他人聚众赌博,并从中抽头渔利,被告人罗1某明知他人实施赌博犯罪活动,而为其提供帮助,三名被告人的行为触犯了《中华人民共和国刑法》××××之规定,应当以××追究刑事责任。被告人刘1某吉、罗3某违反枪支管理规定,无配备配置枪支资格,擅自持有、私藏枪支一支,二被告人的行为触犯了《中华人民共和国刑法》××之规定,应当以非法持有、私藏枪支罪追究刑事责任。被告人罗3某犯罪后主动投案,后如实供述自己的罪行,属自首,应根据《中华人民共和国刑法》××之规定处罚。在赌博犯罪中,被告人刘1某吉、罗2某起主要作用,属主犯,被告人罗1某起次要、辅助作用,属从犯,应根据《中华人民共和国刑法》××、××之规定处罚。被告人刘1某吉一人犯数罪,应根据《中华人民共和国刑法》××之规定数罪并罚。根据《中华人民共和国刑事诉讼法》××的规定,提起公诉,请依法判处。并提供受案登记表、立案决定书、户口证明、到案经过、辨认笔录及照片、搜查笔录及照片、证人证言、鉴定意见、被告人供述和辩解等证据供法庭质证。"""
    
    test_search(full_query, top_k=3)