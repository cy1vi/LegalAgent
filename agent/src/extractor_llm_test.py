import os
import requests
from dotenv import load_dotenv

load_dotenv()

# 从 .env 加载配置
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL")

def call_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",      
        "X-Title": "Legal Fact Extractor"       
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个法律信息提取助手。请严格按JSON格式输出，只包含以下四个字段：subject（主体）、action（行为）、result（结果）、others（其他定罪相关内容）。只输出纯 JSON。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        # 调试用：打印状态和响应
        if response.status_code != 200:
            return f"API Error {response.status_code}: {response.text}"
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error calling LLM: {e} | Response: {response.text if 'response' in locals() else 'N/A'}"

def extract_legal_fact(fact_text: str, crime_str: str) -> str:
    prompt = (
        "请从以下法律事实中提取关键信息，若文字过多可压缩，尽量提取原文内容，并以纯JSON格式返回，仅包含以下三个字段：\n"
        "- subject: 主体（如'刘某某'）\n"
        "- action: 行为（简要描述其实施的具体行为）\n"
        "- result: 结果（如'造成森林破坏'或'导致财产损失'等）"
        "- others: 其他对于定罪判刑有关的内容      \n\n"
        f"罪名如下：{crime_str}"
        "法律事实如下：\n\n"
        f"{fact_text}"
    )
    return call_llm(prompt)

if __name__ == "__main__":
    try:
        fact = """寿县人民检察院指控:一、××\n1、2016年10月8日夜,被告人李1某、孙某驾车至寿县瓦埠镇上奠村瓦房组,
        将稻田内的一台变压器拆卸后盗走铜芯。之后,李1某、孙某又驾车至寿县瓦埠镇供电所,
        将供电所门口放置的变压器拆卸后盗走铜芯。\n2、2016年11月18日夜,被告人李1某、孙某驾车至寿县瓦埠镇金源食品厂附近,
        将金源食品厂对面一个水泥台上的变压器拽下拆卸后盗走铜芯。\n3、2016年11月16日夜,被告人李1某、
        孙某驾车至寿县小2某镇老三岗村部附近,将村部旁变电房顶上的变压器拆卸后盗走铜芯。\n4、2016年12月3日夜,
        被告人李1某、孙某驾车至寿县刘2某曙光路,将曙光路旁的一台变压器拆卸后盗走铜芯。\n5、2016年12月某夜,
        被告人李1某、孙某驾车至寿县炎刘镇格义新能源公司附近,将格义新能源公司旁的一台变压器拆卸后盗走铜芯。
        \n6、2017年2月9日夜,被告人李1某、孙某驾车至寿县炎刘镇新桥大道耐力特厂附近,将该厂南侧一台变压器拆卸后盗走铜芯。
        \n7、2017年3月29日夜,被告人李1某、孙某驾车至寿县双庙集镇周岗村,将该村李4某超市对面的一台变压器拆卸后盗走铜芯。
        \n8、2017年4月6日夜,被告人李1某、孙某驾车至安徽省肥东县长临河镇洪葛村村委会附近,将该村委会钱的一台变压器拆卸后,
        将变压器铜芯及链接变压器的部分铜制电缆线盗走。\n以上李1某、孙某采用破坏性手段盗窃,盗窃犯罪金额总计151993元。
        \n二、故意毁坏财物罪\n2016年11月20日的某夜,被告人李1某、孙某驾车至寿县瓦埠镇铁佛村排灌站,
        将排灌站水泥台上的变压器拽下拆卸,意欲盗窃,因变压器芯为铝制,李1某、孙某未予窃取。该变压器因被拆卸遭到破坏,
        完全失去使用价值。经鉴定:该变压器价值11400元。\n三、掩饰、隐瞒犯罪所得罪\n2016年10月8日至2017年4月7日,
        被告人李1某孙某在每次盗窃后,均驾车至被告人李某某在合肥市瑶海区和平路的废品收购点,将盗窃的赃物全部向李某某销赃。
        李某某每次明知其收购的铜芯和电缆为李1某、孙某犯罪所得,仍然以低价全部收购,共计收购铜芯等赃物重量在1700公斤以上,
        付给李1某孙某赃款共计在24000元以上。\n针对指控的上述事实,公诉机关提供了相关证据。公诉机关认为,被告人李1某、
        孙某以非法占有为目的,多次流窜作案,采取破坏性手段盗窃,数额巨大,其行为触犯了《中华人民共和国刑法》××,犯罪事实清楚,
        证据确实、充分,应当以××追究二被告人刑事责任;被告人李1某、孙某故意毁坏公私财物,数额较大,
        其行为触犯了《中华人民共和国刑法》××,应当以故意毁坏财物罪追究二被告人刑事责任;
        被告人李某某明知是犯罪所得而予以收购,其行为触犯了《中华人民共和国刑法》××××,
        应当以掩饰、隐瞒犯罪所得罪追究其刑事责任;被告人李1某系累犯,应当从重处罚。
        \n被告人李1某对公诉机关指控的犯罪事实、罪名和证据无异议;被告人孙某对公诉机关指控的犯罪事实、罪名和证据无异议,
        其辩护人辩护意见:对孙某犯××不持异议,犯故意毁坏财物罪不能成立,应认定为盗窃未遂,孙某具有立功情节,
        当庭认罪,系初犯,建议对其从轻处罚;被告人李某某对公诉机关指控的犯罪事实、罪名和证据无异议,
        其辩护人辩护意见:对定性无异议,被告人李某某积极退赃,具有悔罪表现,有坦白情节,建议对其从轻处罚并适用××。"""
        crime = ["盗窃", "掩饰、隐瞒犯罪所得、犯罪所得收益"]
        crime_str = ", ".join(crime)
    except EOFError:
        pass
    else:
        print("\n正在调用 LLM 提取信息...\n")
        result = extract_legal_fact(fact,crime_str)
        print("提取结果：")
        print(result)