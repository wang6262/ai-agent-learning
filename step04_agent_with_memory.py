"""
Step 4: Agent + 对话记忆

学习目标:
  1. 理解 Agent 如何"记住"之前的对话
  2. 掌握短期记忆（滑动窗口）和长期记忆（摘要）两种策略
  3. 处理 token 预算 — 上下文窗口有限，不能无限制堆积历史

记忆策略对比:
  ┌──────────────┬─────────────────────┬─────────────────────┐
  │ 策略          │ 原理                 │ 适用场景             │
  ├──────────────┼─────────────────────┼─────────────────────┤
  │ 滑动窗口      │ 只保留最近 N 轮对话   │ 短对话、快速响应     │
  │ 对话摘要      │ 定期压缩旧对话为摘要  │ 长对话、需要持久记忆 │
  │ 混合          │ 窗口 + 摘要组合      │ 实际生产环境推荐     │
  └──────────────┴─────────────────────┴─────────────────────┘

运行: python step04_agent_with_memory.py
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
MAX_HISTORY_MESSAGES = 20  # 滑动窗口大小 — 最多保留 20 条消息

# ============================================================
# 1. 工具定义（与 step03 相同，略作精简）
# ============================================================

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


def get_weather(city: str) -> str:
    data = {
        "北京": {"temperature": 25, "condition": "晴", "humidity": "45%"},
        "上海": {"temperature": 28, "condition": "多云", "humidity": "65%"},
        "广州": {"temperature": 32, "condition": "雷阵雨", "humidity": "80%"},
    }
    w = data.get(city, {"temperature": 22, "condition": "未知", "humidity": "未知"})
    return json.dumps(w, ensure_ascii=False)


def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return "表达式包含不允许的字符"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


def web_search(query: str) -> str:
    knowledge = {
        "python": "Python 由 Guido van Rossum 于 1991 年创建，是一种解释型、面向对象的高级编程语言。",
        "agent": "AI Agent 是能自主感知环境、使用工具、做出决策并执行任务的 AI 系统。",
    }
    for key, value in knowledge.items():
        if key.lower() in query.lower():
            return value
    return f"关于 '{query}' 的搜索结果: 这是一个模拟搜索。"


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

SYSTEM_PROMPT = """你是一个有帮助的智能助手。
- 使用工具获取实时信息
- 回答简洁准确
- 记住用户之前告诉你的信息（如姓名、偏好等）"""


# ============================================================
# 2. 记忆管理器
# ============================================================

class MemoryManager:
    """
    管理 Agent 的记忆。

    两种记忆:
    1. 短期记忆 (short_term): 消息列表，滑动窗口
    2. 长期记忆 (long_term): 对话摘要，压缩旧消息

    工作流程:
    - 新对话不断追加到 short_term
    - 当消息数超过窗口大小时:
      1. 取出最早的消息
      2. 调用 LLM 生成摘要
      3. 将摘要合并到 long_term
      4. 把摘要作为 system 消息注入到对话中
    """

    def __init__(self, max_messages: int = MAX_HISTORY_MESSAGES):
        self.max_messages = max_messages
        self.messages: list = []         # 短期记忆 — 完整的消息列表
        self.summary: str = ""           # 长期记忆 — 压缩后的摘要
        self.user_facts: dict = {}       # 用户事实 — 提取的关键信息

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_call_message(self, msg):
        """添加 tool_call 消息（OpenAI 消息对象）"""
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str):
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    def maybe_compress(self):
        """如果消息太多，压缩旧消息为摘要"""
        if len(self.messages) <= self.max_messages:
            return

        # 取出最早的消息（保留最近一条 system message 和 N-5 条消息）
        old_messages = self.messages[:-5]
        self.messages = self.messages[-5:]

        # 用 LLM 生成摘要
        # 消息可能是 dict 或对象，需要统一处理
        def _get_role(msg) -> str:
            return msg["role"] if isinstance(msg, dict) else getattr(msg, "role", "")

        def _get_content(msg) -> str:
            if isinstance(msg, dict):
                return msg.get("content", "") or ""
            return getattr(msg, "content", "") or ""

        summary_lines = []
        for m in old_messages:
            role = _get_role(m)
            if role in ("user", "assistant"):
                content = _get_content(m)
                if content:
                    summary_lines.append(f"{role}: {content[:200]}")
        summary_text = "\n".join(summary_lines)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": f"请用一段中文总结以下对话的关键信息和结论（50 字以内）:\n\n{summary_text}"
                }],
                max_tokens=100,
            )
            new_summary = response.choices[0].message.content
        except Exception:
            new_summary = "(摘要生成失败)"

        # 合并摘要
        if self.summary:
            self.summary = self.summary + " " + new_summary
        else:
            self.summary = new_summary

    def build_context(self) -> list:
        """
        构建发送给 LLM 的完整上下文。
        结构: [system(含摘要)] + [近期消息...]
        """
        system_content = SYSTEM_PROMPT

        # 如果有摘要，注入
        if self.summary:
            system_content += f"\n\n[对话历史摘要]\n{self.summary}"

        # 如果提取了用户事实，注入
        if self.user_facts:
            facts = "; ".join(f"{k}: {v}" for k, v in self.user_facts.items())
            system_content += f"\n\n[已知用户信息]\n{facts}"

        return [{"role": "system", "content": system_content}] + self.messages


# ============================================================
# 3. Agent 循环（带记忆）
# ============================================================

def run_agent_with_memory(memory: MemoryManager, verbose: bool = True) -> str:
    """
    ReAct 循环，每次调用前通过 memory.build_context() 获取完整上下文。
    每次调用后把消息写入 memory。
    """
    for turn in range(1, MAX_TURNS + 1):
        print(memory)
        context = memory.build_context()

        if verbose:
            print(f"\n--- 第 {turn} 轮 ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=context,
            tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message

        # LLM 直接回复 → 完成
        if msg.content and not msg.tool_calls:
            memory.add_assistant_message(msg.content)
            if verbose:
                print(f"💬 回复: {msg.content[:200]}...")
            return msg.content

        # LLM 调用工具
        if msg.tool_calls:
            memory.add_tool_call_message(msg)

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                if verbose:
                    print(f"🔧 {name}({args})")
                func = TOOL_FUNCTIONS[name]
                result = func(**args)
                if verbose:
                    print(f"📋 {result[:150]}")
                memory.add_tool_result(tc.id, result)

            continue

        continue

    # 超时
    context = memory.build_context()
    context.append({"role": "user", "content": "请基于已有信息，给我一个尽可能完整的回答。"})
    response = client.chat.completions.create(model=MODEL, messages=context)
    final = response.choices[0].message.content
    memory.add_assistant_message(final)
    return final


# ============================================================
# 4. 交互式运行
# ============================================================

def main():
    print("=" * 60)
    print("🧠 带记忆的 Agent")
    print("=" * 60)
    print(f"滑动窗口: 最多 {MAX_HISTORY_MESSAGES} 条消息")
    print("试试连续提问，看 Agent 是否记得之前的内容:")
    print("  例如: 我叫小明 → 我在学 Python → 我叫什么？")
    print("输入 'quit' 退出, 'memory' 查看记忆状态\n")

    memory = MemoryManager()

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
        if user_input.lower() == "memory":
            print(f"\n📊 记忆状态:")
            print(f"   消息数: {len(memory.messages)}")
            print(f"   摘要: {memory.summary or '(空)'}")
            print(f"   用户事实: {memory.user_facts or '(空)'}")
            continue

        memory.add_user_message(user_input)
        answer = run_agent_with_memory(memory, verbose=True)
        print(f"\n🤖 {answer}")

        # 检查是否需要压缩
        memory.maybe_compress()
        print("\n" + "-" * 40)


if __name__ == "__main__":
    main()
