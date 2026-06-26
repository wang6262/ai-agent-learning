"""
Step 6: 带反思机制的 Agent

学习目标:
  1. 理解 Self-Reflection 模式 — Agent 自己检查自己的输出质量
  2. 掌握"生成 → 检查 → 修正"的质量控制循环
  3. 学会定义反思的评估标准

反思的核心思想:
  正常的 Agent: 用户输入 → LLM 生成 → 直接输出
  带反思的 Agent: 用户输入 → LLM 生成 → LLM 自检 → 合格? → 输出
                                                    ↓ 不合格
                                                LLM 修正 → 再检查

为什么需要反思:
  - LLM 有时会遗漏关键信息
  - 计算可能出错（需要验证）
  - 回答可能不完整或偏题

运行: python step06_reflection_agent.py
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
MAX_TURNS = 10
MAX_REFLECTION_ROUNDS = 2  # 最多反思修正 2 次


# ============================================================
# 1. 工具定义
# ============================================================

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


def get_weather(city: str) -> str:
    data = {
        "北京": {"temperature": 25, "condition": "晴", "humidity": "45%"},
        "上海": {"temperature": 28, "condition": "多云", "humidity": "65%"},
        "广州": {"temperature": 32, "condition": "雷阵雨", "humidity": "80%"},
        "深圳": {"temperature": 30, "condition": "阴", "humidity": "75%"},
    }
    w = data.get(city, {"temperature": 22, "condition": "未知", "humidity": "未知"})
    return json.dumps(w, ensure_ascii=False)


def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return "表达式包含不允许的字符"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def web_search(query: str) -> str:
    knowledge = {
        "python": "Python 由 Guido van Rossum 于 1991 年创建。最新稳定版 3.12 于 2023 年 10 月发布。Python 是解释型、面向对象的高级语言。",
        "agent": "AI Agent 是能自主感知环境、使用工具、做出决策并执行任务的 AI 系统。核心: 规划、工具使用、记忆、反思。",
        "recursion": "递归是一种算法设计技巧，函数直接或间接调用自身。包含基准条件（停止递归）和递归条件（继续调用）。",
    }
    for key, value in knowledge.items():
        if key.lower() in query.lower():
            return value
    return f"关于 '{query}' 的搜索结果: 无具体信息。"


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
            "description": "获取当前日期和时间",
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
                "properties": {"city": {"type": "string", "description": "城市名称"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网获取信息",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "搜索关键词"}},
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """你是一个智能助手，可以使用工具获取信息。
回答问题要准确、完整，基于工具返回的实际数据。
如果用户的问题需要多个步骤，请分步获取信息后综合回答。"""


# ============================================================
# 2. ReAct 循环（生成初步答案）
# ============================================================

def react_loop(user_input: str, verbose: bool = True) -> tuple[str, list]:
    """执行 ReAct 循环，返回 (初步答案, 工具调用记录)"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    tool_records = []  # 记录所有工具调用和结果

    for turn in range(1, MAX_TURNS + 1):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message

        if msg.content and not msg.tool_calls:
            return msg.content, tool_records

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                if verbose:
                    print(f"  🔧 {name}({args})")
                func = TOOL_FUNCTIONS[name]
                result = func(**args)
                tool_records.append({"tool": name, "args": args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue

    messages.append({"role": "user", "content": "请直接给我最终答案"})
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content, tool_records


# ============================================================
# 3. 反思机制 — Agent 的核心进阶能力
# ============================================================

def reflect(original_question: str, answer: str, tool_results: list = None) -> dict:
    """
    让 LLM 检查自己生成的答案是否合格。

    返回:
      {"passed": True/False, "feedback": "评价内容", "score": 1-10}
    """
    context = f"""原始问题: {original_question}

生成的答案: {answer}"""

    if tool_results:
        context += f"\n\n工具调用结果: {json.dumps(tool_results, ensure_ascii=False)}"

    reflect_prompt = f"""{context}

请评估这个答案的质量，从以下维度检查:
1. 准确性: 答案是否基于工具返回的实际数据？有没有编造？
2. 完整性: 是否完整回答了用户的所有问题？
3. 清晰性: 表达是否清晰易懂？

请严格按以下 JSON 格式回复（不要加任何其他文字）:
{{
  "passed": true/false,
  "score": 1-10,
  "issues": ["问题1", "问题2"],
  "improvement_suggestions": "改进建议"
}}

如果答案准确且完整，passed 为 true；如果有明显问题，passed 为 false。"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": reflect_prompt}],
        temperature=0.3,  # 低温度，让评估更一致
    )
    content = response.choices[0].message.content

    # 尝试提取 JSON
    try:
        # 处理可能的 markdown 代码块包装
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        return json.loads(content)
    except json.JSONDecodeError:
        # 解析失败，保守处理 — 标记为通过
        return {"passed": True, "score": 7, "issues": [], "improvement_suggestions": ""}


def improve_answer(original_question: str, answer: str, feedback: dict, tool_results: list = None) -> str:
    """根据反思反馈改进答案"""
    tool_info = ""
    if tool_results:
        tool_info = f"\n\n[工具调用记录 — 以下数据是实际获取的，不是编造]\n{json.dumps(tool_results, ensure_ascii=False)}"

    improve_prompt = f"""原始问题: {original_question}

之前的答案: {answer}{tool_info}

反思反馈:
- 评分: {feedback.get('score', 'N/A')}/10
- 问题: {json.dumps(feedback.get('issues', []), ensure_ascii=False)}
- 建议: {feedback.get('improvement_suggestions', '无')}

请根据反馈重新生成一个改进后的答案。修正所有指出的问题，让答案更准确、更完整。注意: 工具调用记录中的数据是真实获取的。"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": improve_prompt}],
    )
    return response.choices[0].message.content


# ============================================================
# 4. 完整的反思 Agent 流程
# ============================================================

def run_agent_with_reflection(user_input: str, verbose: bool = True) -> str:
    """
    完整的反思 Agent 流程:

    1. ReAct 循环 → 生成初步答案
    2. 反思评估 → 检查答案质量
    3. 如果不合格 → 改进 → 再评估
    4. 合格 → 输出
    """
    if verbose:
        print("🧠 正在思考...")

    # 第一步: 生成初步答案（同时收集工具调用记录）
    answer, tool_records = react_loop(user_input, verbose=verbose)
    if verbose:
        print(f"\n📝 初稿:\n{answer[:300]}...")

    # 第二步 + 第三步: 反思 + 改进循环（把工具记录传给反思器）
    for round_num in range(1, MAX_REFLECTION_ROUNDS + 1):
        if verbose:
            print(f"\n🔍 第 {round_num} 次反思检查...")

        feedback = reflect(user_input, answer, tool_records)

        if verbose:
            print(f"   评分: {feedback.get('score', '?')}/10")
            print(f"   通过: {'✅ 是' if feedback.get('passed') else '❌ 否'}")
            issues = feedback.get("issues", [])
            if issues:
                for issue in issues:
                    print(f"   问题: {issue}")

        if feedback.get("passed"):
            if verbose:
                print(f"\n✅ 答案通过质量检查（评分: {feedback.get('score')}/10）")
            return answer

        # 不合格，改进
        if verbose:
            print(f"   🔄 改进中...")

        answer = improve_answer(user_input, answer, feedback, tool_records)
        if verbose:
            print(f"   改进后:\n{answer[:300]}...")

    return answer


# ============================================================
# 5. 交互式运行
# ============================================================

def main():
    print("=" * 60)
    print("🪞 带反思机制的 Agent")
    print("=" * 60)
    print(f"模型: {MODEL} | 最多反思 {MAX_REFLECTION_ROUNDS} 轮")
    print()
    print("反思机制让 Agent 能够:")
    print("  1. 自动检查自己回答的准确性")
    print("  2. 发现遗漏信息后自动补充")
    print("  3. 修正计算错误")
    print()
    print("试试连续问同一个问题，观察初稿和最终答案的区别:")
    print("  'Python 有哪些特点？'")
    print("  '北京和上海哪个更热？温差是多少？'")
    print("输入 'quit' 退出, 'noreflect' 切换反思开关\n")

    use_reflection = True

    while True:
        try:
            user_input = input("\n🧑 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "noreflect":
            use_reflection = not use_reflection
            print(f"反思机制: {'开 ✅' if use_reflection else '关 ❌'}")
            continue

        if use_reflection:
            answer = run_agent_with_reflection(user_input, verbose=True)
        else:
            answer, _ = react_loop(user_input, verbose=True)

        print(f"\n{'=' * 40}")
        print(f"🤖 最终答案:\n{answer}")
        print("-" * 40)


if __name__ == "__main__":
    main()
