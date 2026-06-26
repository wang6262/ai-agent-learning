"""
Step 1: 第一次调用 Qwen 模型

学习目标:
  1. 理解 LLM API 的请求/响应格式
  2. 理解三种消息角色: system(设定行为) / user(用户输入) / assistant(模型回复)
  3. 掌握 OpenAI 兼容 SDK 的基本用法

运行方式:
  # 先设置 API Key
  set DASHSCOPE_API_KEY=sk-xxxxxxxx    (Windows CMD)
  $env:DASHSCOPE_API_KEY="sk-xxxx"     (Windows PowerShell)

  # 运行
  python step01_hello_qwen.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 修复 Windows GBK 终端无法输出 emoji 的问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 自动加载 .env 文件中的环境变量
load_dotenv(Path(__file__).parent / ".env")

# ============================================================
# 1. 配置 — 连接 Qwen 模型
# ============================================================

# DashScope 提供了 OpenAI 兼容接口，所以用 openai 包就能访问 Qwen
# base_url 指向阿里云 DashScope 的兼容端点
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

if not client.api_key:
    print("❌ 未找到 API Key！请检查 .env 文件中的 DASHSCOPE_API_KEY")
    print("   获取 Key: https://dashscope.aliyun.com")
    sys.exit(1)

MODEL = "qwen-plus"  # 性价比最高的模型，适合学习

# ============================================================
# 2. 第一次调用 — 最简单的对话
# ============================================================

print("=" * 60)
print("【演示 1】最简单的问答")
print("=" * 60)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        # system: 设定模型行为（非必需，但推荐）
        {"role": "system", "content": "你是一个友好的 Python 学习助手，回答简洁明了。"},
        # user: 用户的问题
        {"role": "user", "content": "用一句话解释什么是递归？"},
    ],
    temperature=0.7,  # 0=确定, 1=创意, 中间值平衡
    max_tokens=200,   # 限制回复长度，防止浪费 token
)
print(response)
# response.choices[0].message 包含模型的回复
msg = response.choices[0].message
print(msg)
print(f"\n模型回复: {msg.content}")
print(f"\n消耗 token: {response.usage.total_tokens} (输入: {response.usage.prompt_tokens}, 输出: {response.usage.completion_tokens})")

# ============================================================
# 3. 多轮对话 — 模型是无状态的，需要手动传递历史
# ============================================================

print("\n" + "=" * 60)
print("【演示 2】多轮对话（手动维护历史）")
print("=" * 60)

messages = [
    {"role": "system", "content": "你是一个 Python 助手，回答简洁。"},
]

# 第一轮
messages.append({"role": "user", "content": "Python 中 list 和 tuple 的区别是什么？"})
response = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=150)
reply = response.choices[0].message.content
messages.append({"role": "assistant", "content": reply})  # ← 把模型回复加回去
print(f"\n用户: {messages[-2]['content']}")
print(f"模型: {reply}")

# 第二轮 — 模型记得上一轮说了什么
messages.append({"role": "user", "content": "那在什么场景下该用 tuple？"})
response = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=150)
reply = response.choices[0].message.content
messages.append({"role": "assistant", "content": reply})
print(f"\n用户: {messages[-2]['content']}")
print(f"模型: {reply}")

# ============================================================
# 4. 带上下文的多轮对话 — 模型"记住"了之前的交流
# ============================================================

print("\n" + "=" * 60)
print("【演示 3】上下文理解")
print("=" * 60)

# 第一轮给一个事实
messages = [
    {"role": "user", "content": "我叫小明，我今年 25 岁。"},
]
response = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=100)
messages.append({"role": "assistant", "content": response.choices[0].message.content})

# 第二轮问一个需要回忆的问题
messages.append({"role": "user", "content": "我叫什么名字？我多大了？"})
response = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=100)
reply = response.choices[0].message.content

print(f"\n模型回复: {reply}")
print("\n👆 如果模型正确回答了你叫小明、25 岁，说明消息历史起到了记忆作用。")

# ============================================================
# 关键概念总结
# ============================================================

print("\n" + "=" * 60)
print("📚 关键概念")
print("=" * 60)
print("""
1. LLM 是无状态的 — 每次请求是独立的，不会自动记住之前的对话
2. messages 列表就是"上下文" — 把历史消息传回去，模型就能"记住"
3. 三种角色:
   - system:  设定助手的行为和边界（"你是一个 XX 助手"）
   - user:    用户说的话
   - assistant: 模型之前的回复（多轮对话时必须加回去）
4. temperature: 控制随机性，0=精准确定，1=天马行空
5. max_tokens: 限制回复长度，控制成本

下一步: step02_tool_calling.py — 让模型学会使用工具
""")
