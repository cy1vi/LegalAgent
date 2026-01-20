"""Test the agent's case analysis functionality."""

from agent import AgenticRAG
from config import Config

def main():
    # 初始化agent
    config = Config.from_env()
    agent = AgenticRAG(config)
    
    # 测试案例
    test_cases = [
        "被告人王某某在KTV内对李某某实施殴打，导致轻伤二级",
        "被告人张某某在网上发布虚假广告，诈骗多人，涉案金额50万元",
    ]
    
    for case in test_cases:
        print(f"\n=== 测试案例: {case} ===")
        response = agent.chat(case)
        print("\n=== 案件分析结果 ===")
        print(response)
        print("\n" + "="*50)

if __name__ == "__main__":
    main()