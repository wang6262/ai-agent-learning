"""
Step 3: 手写 ReAct Agent（不依赖任何框架）

学习目标:
  1. 理解 Agent 的核心循环: Think → Act → Observe → Think → ...
  2. 掌握 ReAct (Reasoning + Acting) 模式
  3. 学会处理 Agent 循环的终止条件

Agent 循环伪代码:
  while 未完成:
    调用 LLM（带上 messages 和 tools）
    if LLM 返回文本回复（无 tool_calls）:
      任务完成，输出结果 ✓
    else if LLM 返回 tool_calls:
      执行每个工具
      将工具结果追加到 messages
      继续循环
    else if 循环次数超限:
      强制停止

运行: python step03_react_agent.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 修复 Windows GBK 终端无法输出 emoji 的问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 自动加载 .env
load_dotenv(Path(__file__).parent / ".env")

# ============================================================
# 配置
# ============================================================
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

if not client.api_key:
    print("❌ 未找到 API Key！请检查 .env 文件")
    sys.exit(1)

MODEL = "qwen-plus"
MAX_TURNS = 10  # Agent 最多思考 10 轮，防止死循环

# ============================================================
# 1. 工具定义
# ============================================================

def get_current_time() -> str:
    """获取当前日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


def get_weather(city: str) -> str:
    """模拟获取城市天气"""
    weather_data = {
        "北京": {"temperature": 25, "condition": "晴", "humidity": "45%"},
        "上海": {"temperature": 28, "condition": "多云", "humidity": "65%"},
        "广州": {"temperature": 32, "condition": "雷阵雨", "humidity": "80%"},
        "深圳": {"temperature": 30, "condition": "阴", "humidity": "75%"},
        "杭州": {"temperature": 26, "condition": "小雨", "humidity": "70%"},
        "成都": {"temperature": 24, "condition": "阴转晴", "humidity": "60%"},
    }
    w = weather_data.get(city, {"temperature": 22, "condition": "未知", "humidity": "未知"})
    return json.dumps(w, ensure_ascii=False)


def calculator(expression: str) -> str:
    """安全计算数学表达式"""
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return f"错误: 表达式包含不允许的字符"
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def web_search(query: str) -> str:
    """模拟搜索互联网信息"""
    knowledge = {
        "python": "Python 由 Guido van Rossum 于 1991 年创建，是一种解释型、面向对象的高级编程语言。最新稳定版是 Python 3.12。",
        "地球": "地球是太阳系第三颗行星，直径约 12,742 公里，表面积约 5.1 亿平方公里，约 71% 被水覆盖。",
        "openai": "OpenAI 成立于 2015 年，是一家 AI 研究公司，开发了 GPT 系列模型。2022 年底发布 ChatGPT。",
        "agent": "AI Agent（智能体）是能自主感知环境、使用工具、做出决策并执行任务的 AI 系统。核心特征包括自主性、工具使用、记忆和反思。",
    }
    for key, value in knowledge.items():
        if key.lower() in query.lower():
            return value
    return f"关于 '{query}' 的搜索结果: 这是一个模拟的搜索功能，仅供参考。"


# 工具注册表
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "calculator": calculator,
    "web_search": web_search,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算，支持 + - * / ( ) 和基本函数",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网获取信息，适合查找事实、知识、概念",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        },
    },
]


# ============================================================
# 2. Agent 核心循环 — ReAct 模式
# ============================================================

SYSTEM_PROMPT = """你是一个智能助手，具备以下能力:
- 获取当前时间
- 查询城市天气
- 执行数学计算
- 搜索互联网信息

工作原则:
- 对于需要工具才能回答的问题，主动调用工具
- 工具返回结果后，用自然语言清晰地总结给用户
- 如果是复杂问题，可以分步使用多个工具"""


def run_agent(user_input: str, verbose: bool = True) -> str:
    """
    ReAct Agent 核心循环:

    这是整个 Agent 开发最关键的代码。
    理解这个循环 = 理解 Agent 如何工作。

    循环结构:
    1. LLM 思考 (Reasoning): 分析用户输入，决定下一步做什么
    2. 行动 (Acting): 如果需要，调用工具 / 如果不需要，直接回答
    3. 观察 (Observation): 获取工具执行结果
    4. 回到步骤 1，直到产生最终答案
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    for turn in range(1, MAX_TURNS + 1):
        if verbose:
            print(f"\n--- 第 {turn} 轮思考 ---")

        # 调用 LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message

        # 情况 A: LLM 直接给出文本回复 → 任务完成 ✓
        if msg.content and not msg.tool_calls:
            if verbose:
                print(f"💬 最终回答: {msg.content[:200]}...")
            return msg.content

        # 情况 B: LLM 请求调用工具 → 执行并继续循环
        if msg.tool_calls:
            # 把 LLM 的 tool_call 消息加入历史
            messages.append(msg)

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                if verbose:
                    print(f"🔧 调用工具: {name}({args})")

                # 执行工具
                func = TOOL_FUNCTIONS[name]
                result = func(**args)

                if verbose:
                    # 工具结果可能很长，只显示前 200 字符
                    display = result[:200] + "..." if len(result) > 200 else result
                    print(f"📋 工具返回: {display}")

                # 将工具结果追加到 messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # 这一轮结束了，回到循环开始，再调 LLM
            continue

        # 情况 C: 既没有文本也没有 tool_calls（不太常见，但可能发生）
        if verbose:
            print("⚠️ LLM 返回空消息，跳过")
        continue

    # 超过最大轮次，强制让 LLM 总结
    if verbose:
        print(f"\n⚠️ 已达最大轮次 ({MAX_TURNS})，强制总结")

    messages.append({
        "role": "user",
        "content": "请基于已有的信息，给我一个尽可能完整的回答。"
    })
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


# ============================================================
# 3. 交互式运行
# ============================================================

def main():
    print("=" * 60)
    print("🤖 ReAct Agent — 不依赖任何框架，从零构建")
    print("=" * 60)
    print(f"模型: {MODEL} | 最大轮次: {MAX_TURNS}")
    print("工具: 时间 · 天气 · 计算器 · 搜索")
    print()
    print("试试这些例子:")
    print("  1. 北京今天天气怎么样？适合出门吗？")
    print("  2. 上海和深圳哪个更热？温差是多少？")
    print("  3. Python 是谁发明的？")
    print("  4. 现在是几点？100 分钟后是几点？")
    print("输入 'quit' 退出\n")

    while True:
        try:
            user_input = input("🧑 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        answer = run_agent(user_input, verbose=True)
        print(f"\n🤖 最终答案:\n{answer}")
        print("\n" + "-" * 40)


if __name__ == "__main__":
    main()
