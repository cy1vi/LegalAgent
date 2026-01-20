import os
from openai import OpenAI

def create_client():
    # 从系统环境变量读取 OPENAI_API_KEY
    api_key = "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"
    if not api_key:
        raise ValueError("环境变量 OPENAI_API_KEY 未设置")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    return client


def ask_query(client, query: str) -> str:
    """向模型发送 query 并返回回答文本。"""
    resp = client.chat.completions.create(
        model="meta-llama/llama-3.2-3b-instruct",      # 根据你平台支持的模型名调整
        messages=[
            {"role": "system", 
             "content": 
"""你是一个法律专业助手。你的任务是提取法律文书案件的主体、行为以及结果,便于我进行法律文书的检索。
输出结果是json格式
{
"subject": "被告人甲",
"action": "非法占有他人财物",
"result": "骗取人民币5万元"
}
"""  
             },
            {"role": "user", "content": query}
        ]
    )
    return resp.choices[0].message.content



def main():
    client = create_client()
    print("模型已初始化，可以开始提问。")

    while True:
        try:
            query = input("\n请输入你的问题： ")
            if query.strip().lower() in ["quit", "exit"]:
                print("退出程序。")
                break

            answer = ask_query(client, query)
            print("\n模型回答：\n", answer)

        except KeyboardInterrupt:
            print("\n终止程序。")
            break


if __name__ == "__main__":
    main()
