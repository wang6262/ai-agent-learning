"""
Step 7: 完整 Agent 框架 — 整合所有能力

学习目标:
  1. 将前 6 步学到的所有概念整合为一个可复用的 Agent 类
  2. 理解 Agent 框架的设计模式
  3. 掌握可配置、可扩展的 Agent 架构

整合的能力:
  - ReAct 循环（step03）
  - 对话记忆管理（step04）
  - 工具安全执行 + 错误处理（step05）
  - 自我反思 + 质量检查（step06）

架构设计:
  ┌──────────────────────────────────────────────┐
  │                 Agent 类                      │
  │  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
  │  │ 记忆管理  │  │ 工具系统  │  │ 反思检查器 │ │
  │  │ Memory    │  │ ToolKit  │  │ Reflector  │ │
  │  └──────────┘  └──────────┘  └────────────┘ │
  │         │            │              │         │
  │         └────────────┼──────────────┘         │
  │                      ▼                         │
  │              ┌──────────────┐                  │
  │              │  ReAct 循环  │                  │
  │              └──────────────┘                  │
  └──────────────────────────────────────────────┘

运行: python step07_complete_agent.py
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable
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

MODEL = "qwen3.7-plus"


# ============================================================
# 1. 工具系统 — ToolKit
# ============================================================

class ToolKit:
    """
    工具注册和管理系统。

    使用方式:
      toolkit = ToolKit()
      toolkit.register("get_time", get_time_func, "获取当前时间", {"city": "string"})
    """

    def __init__(self):
        self._functions: dict[str, Callable] = {}
        self._schemas: list[dict] = []

    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: dict = None,
    ):
        """注册一个工具"""
        self._functions[name] = func

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters or {},
                    "required": list(parameters.keys()) if parameters else [],
                },
            },
        }
        self._schemas.append(schema)

    def get_schemas(self) -> list[dict]:
        return self._schemas

    def execute(self, name: str, arguments: dict, max_output_length: int = 2000) -> str:
        """安全执行工具，返回结果字符串"""
        func = self._functions.get(name)
        if not func:
            return f"错误: 未找到工具 '{name}'"

        try:
            result = func(**arguments)
            result_str = str(result)
            if len(result_str) > max_output_length:
                result_str = result_str[:max_output_length] + "\n... [已截断]"
            return result_str
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "error_type": type(e).__name__,
            }, ensure_ascii=False)


# ============================================================
# 2. 记忆管理器 — Memory
# ============================================================

class Memory:
    """对话记忆管理器"""

    def __init__(self, system_prompt: str = "", max_messages: int = 30):
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.messages: list[dict] = []
        self.summary: str = ""

    def add(self, message: dict):
        self.messages.append(message)
        self._maybe_compress()

    def _maybe_compress(self):
        """消息过多时自动压缩"""
        if len(self.messages) <= self.max_messages:
            return

        old = self.messages[:-10]  # 保留最近 10 条
        self.messages = self.messages[-10:]

        # 提取 user/assistant 内容做摘要（兼容 dict 和对象）
        lines = []
        for m in old:
            role = m["role"] if isinstance(m, dict) else getattr(m, "role", "")
            content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                lines.append(f"{role}: {content[:100]}")

        if lines:
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{
                        "role": "user",
                        "content": f"用一句话（30字内）总结这段对话:\n" + "\n".join(lines[-20:])
                    }],
                    max_tokens=60,
                )
                new_summary = response.choices[0].message.content
                self.summary = (self.summary + "; " + new_summary) if self.summary else new_summary
            except Exception:
                pass

    def build_context(self) -> list[dict]:
        """构建发送给 LLM 的消息列表"""
        system_content = self.system_prompt
        if self.summary:
            system_content += f"\n\n[历史摘要] {self.summary}"
        return [{"role": "system", "content": system_content}] + self.messages


# ============================================================
# 3. 反思检查器 — Reflector
# ============================================================

class Reflector:
    """答案质量检查器"""

    @staticmethod
    def check(question: str, answer: str, context: str = "") -> dict:
        """评估答案质量。context 包含工具调用记录，让反思器知道哪些数据来自工具而非编造。"""
        context_block = ""
        if context:
            context_block = f"\n\n[工具调用记录 — 以下数据是实际获取的，不是模型编造的]\n{context}"

        prompt = f"""原始问题: {question}
答案: {answer}{context_block}

请评估答案质量（准确性、完整性、清晰性）。
重要: 如果答案中的数据来自"工具调用记录"，说明是真实获取的，不应标记为"编造"或"虚假"。
严格按 JSON 格式回复:
{{"passed": true/false, "score": 1-10, "issues": [], "suggestion": ""}}"""

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(content)
        except Exception:
            return {"passed": True, "score": 7, "issues": [], "suggestion": ""}

    @staticmethod
    def improve(question: str, answer: str, feedback: dict, context: str = "") -> str:
        """根据反馈改进答案"""
        context_block = ""
        if context:
            context_block = f"\n\n[工具调用记录 — 可靠数据]\n{context}"

        prompt = f"""原始问题: {question}
之前的答案: {answer}
问题: {json.dumps(feedback.get('issues', []), ensure_ascii=False)}
建议: {feedback.get('suggestion', '')}{context_block}

请重新生成一个改进后的答案。注意区分: 工具返回的数据是真实的，只需改进表达方式。"""
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception:
            return answer


# ============================================================
# 4. Agent 类 — 整合一切
# ============================================================

class Agent:
    """
    完整的 Agent 类，整合了所有能力。

    使用示例:
      agent = Agent(name="助手", system_prompt="你是一个有用的助手")
      agent.add_tool("get_time", get_time, "获取时间")
      answer = agent.run("现在几点了？")
    """

    def __init__(
        self,
        name: str = "Agent",
        system_prompt: str = "你是一个智能助手。",
        model: str = MODEL,
        max_turns: int = 10,
        enable_reflection: bool = True,
        max_reflection_rounds: int = 2,
    ):
        self.name = name
        self.model = model
        self.max_turns = max_turns
        self.enable_reflection = enable_reflection
        self.max_reflection_rounds = max_reflection_rounds

        self.toolkit = ToolKit()
        self.memory = Memory(system_prompt=system_prompt)
        self.reflector = Reflector()

    def add_tool(self, name: str, func: Callable, description: str, parameters: dict = None):
        """注册一个工具"""
        self.toolkit.register(name, func, description, parameters)
        return self  # 链式调用

    def _react_loop(self, user_input: str, verbose: bool = False) -> str:
        """核心 ReAct 循环"""
        self.memory.add({"role": "user", "content": user_input})

        for turn in range(1, self.max_turns + 1):
            context = self.memory.build_context()

            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=context,
                    tools=self.toolkit.get_schemas() or None,
                )
            except Exception as e:
                if verbose:
                    print(f"  ⚠️ API 错误: {e}")
                continue

            msg = response.choices[0].message

            # 直接回复 → 完成
            if msg.content and not msg.tool_calls:
                self.memory.add({"role": "assistant", "content": msg.content})
                return msg.content

            # 工具调用
            if msg.tool_calls:
                self.memory.add({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)

                    if verbose:
                        print(f"  🔧 {name}({args})")

                    result = self.toolkit.execute(name, args)

                    if verbose:
                        print(f"  📋 {result[:120]}")

                    self.memory.add({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue

        # 超时，强制总结
        context = self.memory.build_context()
        context.append({"role": "user", "content": "请基于已有信息给出最终回答。"})
        try:
            response = client.chat.completions.create(model=self.model, messages=context)
            final = response.choices[0].message.content
            self.memory.add({"role": "assistant", "content": final})
            return final
        except Exception:
            return "抱歉，处理超时。请简化您的问题重试。"

    def _extract_tool_context(self) -> str:
        """从记忆中提取工具调用记录，供反思器参考"""
        lines = []
        for m in self.memory.messages:
            role = m["role"] if isinstance(m, dict) else getattr(m, "role", "")
            if role == "tool":
                tc_id = m.get("tool_call_id", "") if isinstance(m, dict) else getattr(m, "tool_call_id", "")
                content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
                if isinstance(content, str):
                    lines.append(f"[工具返回] {content[:300]}")
            elif role == "assistant":
                # 检查是否有 tool_calls
                tc = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
                if tc:
                    for t in tc:
                        fn = t.get("function", {}) if isinstance(t, dict) else getattr(t, "function", {})
                        name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                        args = fn.get("arguments", "") if isinstance(fn, dict) else getattr(fn, "arguments", "")
                        lines.append(f"[调用工具] {name}({args})")
        return "\n".join(lines) if lines else ""

    def run(self, user_input: str, verbose: bool = False) -> str:
        """
        执行 Agent，返回最终答案。

        流程:
        1. ReAct 循环 → 初步答案
        2. (可选) 反思检查 → 不合格则改进
        3. 返回最终答案
        """
        if verbose:
            print(f"{'='*40}")
            print(f"🤖 {self.name} 开始处理: {user_input}")

        # Step 1: ReAct 循环
        answer = self._react_loop(user_input, verbose=verbose)

        if not self.enable_reflection:
            return answer

        # Step 2: 提取工具调用上下文，防止反思器误判工具返回数据为"编造"
        tool_context = self._extract_tool_context()

        # Step 3: 反思 + 改进
        for round_num in range(1, self.max_reflection_rounds + 1):
            if verbose:
                print(f"  🔍 反思检查 (第 {round_num} 轮)...")

            feedback = self.reflector.check(user_input, answer, tool_context)

            if feedback.get("passed"):
                if verbose:
                    print(f"  ✅ 通过 (评分: {feedback.get('score')}/10)")
                return answer

            if verbose:
                print(f"  ⚠️ 未通过 (评分: {feedback.get('score')}/10)")
                for issue in feedback.get("issues", []):
                    print(f"    - {issue}")
                print(f"  🔄 改进中...")

            answer = self.reflector.improve(user_input, answer, feedback, tool_context)

        return answer

    def reset(self):
        """重置对话记忆"""
        self.memory = Memory(system_prompt=self.memory.system_prompt)


# ============================================================
# 5. 使用示例 — 构建一个完整的 Agent
# ============================================================

def create_demo_agent() -> Agent:
    """创建一个演示用的完整 Agent"""

    # -------- 工具函数 --------
    def get_time() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")

    def get_weather(city: str) -> str:
        data = {
            "北京": {"temp": 25, "condition": "晴", "humidity": "45%"},
            "上海": {"temp": 28, "condition": "多云", "humidity": "65%"},
            "广州": {"temp": 32, "condition": "雷阵雨", "humidity": "80%"},
        }
        w = data.get(city, {"temp": 22, "condition": "未知", "humidity": "N/A"})
        return json.dumps(w, ensure_ascii=False)

    def calculate(expression: str) -> str:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "错误: 包含不允许的字符"
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"错误: {e}"

    def search(query: str) -> str:
        knowledge = {
            "python": "Python 由 Guido van Rossum 于 1991 年创建，是解释型、面向对象的高级编程语言。",
            "agent": "AI Agent 是能自主感知环境、使用工具、做出决策并执行任务的 AI 系统。",
            "qwen": "Qwen（通义千问）是阿里云开发的大语言模型系列，支持多轮对话、工具调用和代码生成。",
        }
        for k, v in knowledge.items():
            if k in query.lower():
                return v
        return f"搜索 '{query}': 无结果"

    def read_file(path: str) -> str:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"文件不存在: {path}"
        try:
            content = p.read_text(encoding="utf-8")
            return content[:3000]
        except Exception as e:
            return f"读取失败: {e}"

    def list_dir(path: str = ".") -> str:
        p = Path(path).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            return f"目录不存在: {path}"
        items = []
        for entry in sorted(p.iterdir()):
            t = "[DIR]" if entry.is_dir() else "[FILE]"
            items.append(f"  {t} {entry.name}")
        return "\n".join(items) or "(空)"

    # -------- 构建 Agent --------
    agent = Agent(
        name="智能助手",
        system_prompt="""你是一个全能的智能助手。
你可以:
- 查询时间、天气
- 执行数学计算
- 搜索知识
- 浏览和读取文件

工作原则:
- 使用工具获取准确信息，不要编造
- 回答简洁、准确、完整
- 如果用户问题需要多步操作，分步完成""",
        max_turns=10,
        enable_reflection=True,
    )

    # 注册工具
    agent.add_tool("get_time", get_time, "获取当前日期和时间")
    agent.add_tool("get_weather", get_weather, "获取城市天气",
                   {"city": {"type": "string", "description": "城市名称"}})
    agent.add_tool("calculate", calculate, "执行数学计算",
                   {"expression": {"type": "string", "description": "数学表达式"}})
    agent.add_tool("search", search, "搜索知识库",
                   {"query": {"type": "string", "description": "搜索关键词"}})
    agent.add_tool("read_file", read_file, "读取文件内容",
                   {"path": {"type": "string", "description": "文件路径"}})
    agent.add_tool("list_dir", list_dir, "列出目录内容",
                   {"path": {"type": "string", "description": "目录路径"}})

    return agent


# ============================================================
# 6. 交互式运行
# ============================================================

def main():
    print("=" * 60)
    print("🏗️ 完整 Agent 框架")
    print("=" * 60)
    print("这是前 6 步所有概念的整合:")
    print("  Step1-2  → 模型调用 + 工具")
    print("  Step3    → ReAct 循环")
    print("  Step4    → 对话记忆 + 摘要")
    print("  Step5    → 安全执行 + 错误处理")
    print("  Step6    → 反思 + 质量检查")
    print()

    agent = create_demo_agent()
    print(f"Agent 已就绪: {len(agent.toolkit._functions)} 个工具已注册")
    print()
    print("试试:")
    print("  '现在几点了？'")
    print("  '北京天气怎么样？'")
    print("  然后问 '我刚才问了什么？' — 测试记忆")
    print("  '列出当前目录' — 测试文件操作")
    print("输入 'quit' 退出, 'reset' 重置记忆, 'reflect off' 关闭反思\n")

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
        if user_input.lower() == "reset":
            agent.reset()
            print("🔄 记忆已重置")
            continue
        if user_input.lower() == "reflect off":
            agent.enable_reflection = False
            print("🪞 反思已关闭")
            continue
        if user_input.lower() == "reflect on":
            agent.enable_reflection = True
            print("🪞 反思已开启")
            continue

        answer = agent.run(user_input, verbose=True)
        print(f"\n🤖 {answer}")
        print("\n" + "-" * 40)


if __name__ == "__main__":
    main()
