"""
Step 5: 健壮的多工具 Agent — 工程化实践

学习目标:
  1. 掌握工具调用失败时的处理策略（重试/降级/报错）
  2. 处理工具输出过长的问题（截断策略）
  3. 理解生产环境 Agent 的健壮性设计

健壮性设计要点:
  - 工具执行异常 → 捕获并返回错误信息给 LLM，让 LLM 决定下一步
  - 工具输出过长 → 智能截断，保留关键信息
  - 网络超时 → 设置超时 + 重试机制
  - LLM 格式错误 → 参数校验 + 友好的错误反馈

运行: python step05_robust_agent.py
"""

import json
import os
import re
import subprocess
import sys
import time
import traceback
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
MAX_TURNS = 15
MAX_TOOL_OUTPUT_LENGTH = 2000  # 工具输出最大字符数


# ============================================================
# 1. 工具执行的安全包装
# ============================================================

class ToolError(Exception):
    """工具执行错误"""
    pass


def safe_execute(func, max_length: int = MAX_TOOL_OUTPUT_LENGTH, **kwargs) -> str:
    """
    安全执行工具函数，包含:
    1. 异常捕获 — 出错不崩溃，返回错误信息给 LLM
    2. 输出截断 — 避免超长输出撑爆上下文
    3. 执行计时 — 方便调试
    """
    start = time.time()
    try:
        result = func(**kwargs)
    except Exception as e:
        # 返回详细的错误信息，LLM 可以据此调整策略
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__,
            "suggestion": "请检查参数是否正确，或尝试其他方式",
        }, ensure_ascii=False)

    elapsed = time.time() - start
    result_str = str(result)

    # 截断过长输出
    if len(result_str) > max_length:
        result_str = (
            result_str[:max_length]
            + f"\n\n... [输出被截断，总长度 {len(result_str)} 字符，"
            f"耗时 {elapsed:.2f}s]"
        )

    return result_str


# ============================================================
# 2. 工具定义 — 更多、更实用的工具
# ============================================================

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


def get_weather(city: str) -> str:
    """获取城市天气（模拟）"""
    data = {
        "北京": {"temperature": 25, "condition": "晴", "humidity": "45%", "wind": "北风 3级"},
        "上海": {"temperature": 28, "condition": "多云", "humidity": "65%", "wind": "东南风 2级"},
        "广州": {"temperature": 32, "condition": "雷阵雨", "humidity": "80%", "wind": "南风 4级"},
        "深圳": {"temperature": 30, "condition": "阴", "humidity": "75%", "wind": "东风 3级"},
    }
    w = data.get(city, {"temperature": 22, "condition": "未知", "humidity": "N/A", "wind": "N/A"})
    return json.dumps(w, ensure_ascii=False)


def calculator(expression: str) -> str:
    """数学计算器，支持更多操作"""
    allowed = set("0123456789+-*/().%^ ")
    if not all(c in allowed for c in expression):
        return f"错误: 表达式包含不允许的字符。只支持数字、运算符和括号。"
    try:
        # 安全的 eval（仅允许数学运算）
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except ZeroDivisionError:
        return "错误: 除数不能为零"
    except SyntaxError as e:
        return f"语法错误: {e}"
    except Exception as e:
        return f"计算错误: {e}"


def list_directory(path: str = ".") -> str:
    """列出目录内容"""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"错误: 路径 '{path}' 不存在"
    if not p.is_dir():
        return f"错误: '{path}' 不是目录"

    items = []
    for entry in sorted(p.iterdir()):
        prefix = "[DIR] " if entry.is_dir() else "[FILE]"
        try:
            size = entry.stat().st_size if entry.is_file() else 0
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size / 1024 / 1024:.1f}MB"
            items.append(f"  {prefix} {entry.name} ({size_str})")
        except OSError:
            items.append(f"  {prefix} {entry.name}")

    return "\n".join(items) if items else "(空目录)"


def read_file(file_path: str) -> str:
    """读取文本文件"""
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        return f"错误: 文件 '{file_path}' 不存在"
    if not p.is_file():
        return f"错误: '{file_path}' 不是文件"

    # 安全检查：拒绝二进制文件
    text_extensions = {".py", ".txt", ".md", ".json", ".yaml", ".toml",
                       ".csv", ".html", ".css", ".js", ".ts", ".xml", ".cfg", ".ini"}
    if p.suffix.lower() not in text_extensions and p.suffix:
        return f"错误: 不支持的文件类型 '{p.suffix}'。仅支持文本文件。"

    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"错误: 文件 '{file_path}' 不是 UTF-8 文本文件"
    except Exception as e:
        return f"读取错误: {e}"

    return content


def write_file(file_path: str, content: str) -> str:
    """写入文件（限制在当前目录下）"""
    p = Path(file_path).resolve()

    # 安全检查：只允许写入当前目录的子目录
    cwd = Path.cwd().resolve()
    try:
        p.relative_to(cwd)
    except ValueError:
        return f"安全错误: 只能写入当前目录 ({cwd}) 下的文件，不能写入 '{file_path}'"

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"已写入 {len(content)} 个字符到 '{p}'"
    except Exception as e:
        return f"写入失败: {e}"


def run_shell_command(command: str) -> str:
    """
    执行 shell 命令（仅允许安全的白名单命令）

    ⚠️ 安全注意: 这是一个教学示例，白名单不足以完全防止恶意利用。
    生产环境应使用 docker 沙箱或其他隔离机制。
    """
    # 白名单 — 只允许这些命令
    ALLOWED_COMMANDS = ["echo", "date", "whoami", "dir", "ls", "cat",
                        "type", "pwd", "hostname", "python --version",
                        "git status", "git log"]

    # 检查命令是否在白名单中
    cmd_lower = command.strip().lower()
    allowed = False
    for allowed_cmd in ALLOWED_COMMANDS:
        if cmd_lower.startswith(allowed_cmd.lower()):
            allowed = True
            break

    if not allowed:
        return (f"安全错误: 命令 '{command}' 不在白名单中。\n"
                f"允许的命令: {', '.join(ALLOWED_COMMANDS)}")

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=30,  # 30 秒超时
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return output.strip() or "(命令执行成功，无输出)"
    except subprocess.TimeoutExpired:
        return "错误: 命令执行超时（30秒）"
    except Exception as e:
        return f"命令执行错误: {e}"


# 工具注册表
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "calculator": calculator,
    "list_directory": list_directory,
    "read_file": read_file,
    "write_file": write_file,
    "run_shell_command": run_shell_command,
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
            "description": "执行数学表达式计算",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式，如 '3 + 5 * 2'"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出指定目录下的文件和子目录",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "目录路径，默认为当前目录"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文本文件内容（.py .txt .md .json 等）",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "文件路径"}},
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径，如 output/summary.md"},
                    "content": {"type": "string", "description": "要写入的内容"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "执行安全的 shell 命令（仅白名单命令）",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "要执行的命令"}},
                "required": ["command"],
            },
        },
    },
]

SYSTEM_PROMPT = """你是一个文件系统助手。你可以在当前目录浏览文件、读取文件内容、
写入新文件、执行安全的 shell 命令。

重要规则:
- 先列出目录了解有什么文件，再读取需要的文件
- write_file 只能写当前目录下的文件
- 工具返回的错误信息要认真对待，调整策略而不是重复同样的错误
- shell 命令只能用白名单中的命令"""


# ============================================================
# 3. Agent 循环（带重试和降级）
# ============================================================

def run_agent(user_input: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    for turn in range(1, MAX_TURNS + 1):
        if verbose:
            print(f"\n--- 第 {turn} 轮 ---")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
            )
        except Exception as e:
            # API 调用失败 — 可能是网络问题、限流等
            if verbose:
                print(f"❌ API 调用失败: {e}")
            if turn <= 2:
                # 前两次失败尝试重试
                if verbose:
                    print("🔄 重试中...")
                time.sleep(1)
                continue
            else:
                return f"抱歉，API 调用多次失败: {e}"

        msg = response.choices[0].message

        # 直接回复
        if msg.content and not msg.tool_calls:
            if verbose:
                print(f"💬 回复")
            return msg.content

        # 工具调用
        if msg.tool_calls:
            messages.append(msg)

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                if verbose:
                    args_display = {k: v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v for k, v in args.items()}
                    print(f"🔧 {name}({args_display})")

                func = TOOL_FUNCTIONS[name]
                result = safe_execute(func, **args)

                if verbose:
                    display = result[:150] + "..." if len(result) > 150 else result
                    print(f"📋 {display}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            continue

    # 超时
    messages.append({
        "role": "user",
        "content": "请基于已有信息，用一个完整的回答来总结。"
    })
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


# ============================================================
# 4. 交互式运行
# ============================================================

def main():
    print("=" * 60)
    print("🛡️ 健壮的多工具 Agent")
    print("=" * 60)
    print(f"模型: {MODEL} | 最大轮次: {MAX_TURNS}")
    print("工具: 时间 · 天气 · 计算器 · 目录列表 · 读文件 · 写文件 · Shell")
    print()
    print("试试这些例子:")
    print("  1. 列出当前目录的文件")
    print("  2. 读取 step01_hello_qwen.py 看看里面有什么")
    print("  3. 把当前时间写入一个 time.txt 文件")
    print("  4. 执行 whoami 看看我是谁")
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
        print(f"\n🤖 {answer}")
        print("\n" + "-" * 40)


if __name__ == "__main__":
    main()
