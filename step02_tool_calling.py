"""
Step 2: Tool Calling 工具调用

学习目标:
  1. 理解 Tool Call 的完整生命周期
  2. 学会定义工具的 JSON Schema
  3. 手动处理 tool_call → 执行函数 → 回传结果

核心流程:
  用户: "北京今天天气怎么样？"
    → LLM 判断: "我需要调用 get_weather 工具，参数 city='北京'"
    → 返回 tool_call (不返回文本)
    → 我们拿到 tool_call，执行 Python 函数 get_weather("北京")
    → 把结果 {"temperature": 25, "condition": "晴"} 回传给 LLM
    → LLM 用自然语言回复: "北京今天晴天，25°C"

运行: python step02_tool_calling.py
"""

import json
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

# ============================================================
# 1. 定义工具 — 关键是 JSON Schema
# ============================================================

# 每个工具是一个 dict，包含:
#   type: "function" (固定)
#   function.name: 工具名称（LLM 用它来指定调用哪个工具）
#   function.description: 工具用途（LLM 据此判断什么时候用）
#   function.parameters: JSON Schema（LLM 据此知道要传什么参数）


def get_current_time() -> str:
    """获取当前日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_weather(city: str) -> str:
    """模拟获取城市天气（实际应调用天气 API）"""
    # 模拟数据
    weather_data = {
        "北京": {"temperature": 25, "condition": "晴", "humidity": "45%"},
        "上海": {"temperature": 28, "condition": "多云", "humidity": "65%"},
        "广州": {"temperature": 32, "condition": "雷阵雨", "humidity": "80%"},
        "深圳": {"temperature": 30, "condition": "阴", "humidity": "75%"},
        "杭州": {"temperature": 26, "condition": "小雨", "humidity": "70%"},
    }
    w = weather_data.get(city, {"temperature": 22, "condition": "未知", "humidity": "未知"})
    return json.dumps(w, ensure_ascii=False)


def calculator(expression: str) -> str:
    """安全计算数学表达式"""
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return f"错误: 表达式包含不允许的字符，只支持数字和 + - * / ( ) . %"
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


# 工具函数注册表 — 名称 → 函数对象的映射
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "calculator": calculator,
}

# 工具的 JSON Schema 定义 — 这是 LLM "看懂"工具的方式
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间，不需要任何参数",
            "parameters": {
                "type": "object",
                "properties": {},  # 无参数
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息，包括温度、天气状况和湿度",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 北京、上海、广州",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算，支持加减乘除和括号。输入一个数学表达式字符串。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '(3 + 5) * 2'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


# ============================================================
# 2. 核心函数 — 调用 LLM，处理 tool_call
# ============================================================

def call_llm(messages: list) -> dict:
    """
    调用 LLM，返回消息对象。

    可能返回:
      - 普通文本回复 (message.content 有内容)
      - 工具调用请求 (message.tool_calls 有内容)
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS_SCHEMA,
        # tool_choice="auto" 是默认值，LLM 自己判断是否需要调用工具
    )
    # print(response)
    return response.choices[0].message


def execute_tool(tool_call) -> str:
    """根据 tool_call 执行对应的 Python 函数，返回结果字符串"""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    func = TOOL_FUNCTIONS[name]
    return func(**args)


# ============================================================
# 3. 演示 — 一次完整的工具调用流程
# ============================================================

print("=" * 60)
print("【演示 1】LLM 自动选择工具 — 问天气")
print("=" * 60)

messages = [
    {"role": "user", "content": "北京今天天气怎么样？"},
]

msg = call_llm(messages)
print(f"\nLLM 返回的消息类型:")
print(f"  content:    {msg.content}")        # None — 因为 LLM 决定调用工具
print(f"  tool_calls: {msg.tool_calls}")      # 有值 — 工具调用请求

if msg.tool_calls:
    tc = msg.tool_calls[0]
    print(f"\n工具调用详情:")
    print(f"  工具名:   {tc.function.name}")
    print(f"  参数:     {tc.function.arguments}")

    # 执行工具
    result = execute_tool(tc)
    print(f"  执行结果: {result}")

    # 把结果回传给 LLM
    messages.append(msg)  # LLM 的 tool_call 消息
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": result,
    })

    # 再次调用 LLM，让它用自然语言总结
    final_msg = call_llm(messages)
    print(f"\n最终回复: {final_msg.content}")

# ============================================================
# 4. 演示 — LLM 不需要工具时
# ============================================================

print("\n" + "=" * 60)
print("【演示 2】不需要工具 — LLM 直接回答问题")
print("=" * 60)

messages = [
    {"role": "user", "content": "Python 是谁发明的？"},
]

msg = call_llm(messages)
print(f"\ncontent:    {msg.content}")
print(f"tool_calls: {msg.tool_calls}")  # None — LLM 不需要工具

# ============================================================
# 5. 演示 — 多工具组合
# ============================================================

print("\n" + "=" * 60)
print("【演示 3】多工具组合 — 计算 + 天气")
print("=" * 60)

messages = [
    {"role": "user", "content": "帮我看一下北京和上海的温差是多少？"},
]

# 这个任务需要:
#   1. 先调用 get_weather("北京")  获取 25°C
#   2. 再调用 get_weather("上海")  获取 28°C
#   3. 然后用 calculator("28 - 25") 计算温差
# LLM 可以一次返回多个 tool_call

msg = call_llm(messages)
print(f"\ncontent: {msg.content}")
print(f"tool_calls 数量: {len(msg.tool_calls) if msg.tool_calls else 0}")

if msg.tool_calls:
    for i, tc in enumerate(msg.tool_calls):
        print(f"  工具 {i+1}: {tc.function.name}({tc.function.arguments})")

    # 执行所有工具调用
    messages.append(msg)
    for tc in msg.tool_calls:
        result = execute_tool(tc)
        print(f"  结果 {tc.function.name}: {result}")
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })

    # 回传结果，获取最终答案
    final_msg = call_llm(messages)
    print(f"\n最终回复: {final_msg.content}")

# ============================================================
# 关键概念总结
# ============================================================

print("\n" + "=" * 60)
print("📚 关键概念")
print("=" * 60)
print("""
1. Tool Call 生命周期:
   用户提问 → LLM 判断需要工具 → 返回 tool_calls
   → 执行 Python 函数 → 把结果用 tool 角色回传
   → LLM 理解结果 → 用自然语言回答

2. 关键消息角色:
   - tool_call 消息: role="assistant", 但 content 为空, tool_calls 有值
   - tool 结果消息:  role="tool", 包含 tool_call_id 和函数返回值

3. LLM 可以一次请求多个工具调用 (并行)
   - 工具之间无依赖 → 并行调用（如同时查两个城市天气）
   - 工具有依赖 → 分步调用（先查天气再算温差）

4. JSON Schema 是关键 — LLM 靠它理解:
   - 什么时候该用工具 (description)
   - 该传什么参数 (parameters)
   - 参数是什么类型 (type)

下一步: step03_react_agent.py — 手写 Agent 循环
""")
