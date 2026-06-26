# ==============================================
# 文件名：step11_web_search.py
# 基础功能：Agent 自动搜索网页 → 提取内容 → 总结回答，让 Agent 拥有"联网能力"
# 核心学习知识点：
#   1. 两种搜索方式：DuckDuckGo（免费）vs Search API（付费/稳定）
#   2. 网页内容提取：requests + BeautifulSoup + html2text 降级策略
#   3. Web Search → Read Pages → RAG Answer 三段式流水线
#   4. 搜索结果去重、安全校验（SSRF 防御）、超时控制
#   5. Agent 工具注册：将 web_search 注册到 ReAct 循环
# 适用场景：需要实时信息的问答、新闻总结、技术调查、竞品分析
# 使用方法：
#   1. pip install requests beautifulsoup4 duckduckgo_search
#   2. python step11_web_search.py
#   3. 在交互中提问（如"今天的 AI 新闻有哪些？"、"最新的 Python 3.13 特性？"）
# 进阶说明：
#   - DuckDuckGo 免费但不稳定，生产环境建议用 Tavily / SerpAPI / Brave Search
#   - 网页内容提取可升级为 Jina AI Reader / Firecrawl 等专业工具
#   - 可结合 step10 的知识库，实现"先查知识库，没有再搜网页"的降级策略
# 常用配套函数：
#   1. requests.get()：发送 HTTP GET 请求获取网页
#   2. BeautifulSoup.find_all()：解析 HTML 提取指定标签
#   3. re.sub()：用正则清理 HTML 标签、多余空白
#   4. duckduckgo_search.ddgs_text()：免费网页搜索
# ==============================================
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from dotenv import load_dotenv
from openai import OpenAI

import requests

# Windows 终端 UTF-8 兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 自动加载 .env
load_dotenv(Path(__file__).parent / ".env")

# ============================================================
# 配置区（集中管理，方便修改）
# ============================================================
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    print("❌ 未找到 API Key！请检查 .env 文件中的 DASHSCOPE_API_KEY")
    sys.exit(1)

llm_client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

LLM_MODEL = "qwen-plus"
SEARCH_ENGINE = "duckduckgo"  # 可选: "duckduckgo" | "serpapi" | "tavily"
MAX_SEARCH_RESULTS = 5         # 每次搜索返回的网页数量
PAGE_FETCH_TIMEOUT = 10        # 网页抓取超时（秒）
MAX_PAGE_LENGTH = 8000         # 单个网页最大提取字符数（防止 token 爆炸）


# ============================================================
# 1. 网页搜索 — 关键词 → URL 列表
# ============================================================

def search_duckduckgo(query: str, max_results: int = None) -> list[dict]:
    """
    【基础功能】使用 DuckDuckGo 免费搜索网页，返回标题 + URL + 摘要
    【学习知识点】
        1. DuckDuckGo 不需要 API Key，适合学习和原型开发
        2. ddgs_text() 返回生成器，每个结果包含 title / href / body
        3. 限制返回数量防止 token 爆炸
    参数：
        query: 搜索关键词（支持中英文）
        max_results: 最大返回数量
    返回值：list[dict]，每个 dict 包含 title / url / snippet
    调用示例：
        results = search_duckduckgo("Python 3.13 新特性", max_results=3)
        for r in results:
            print(f"{r['title']}\n  {r['url']}\n  {r['snippet']}\n")
    """
    if max_results is None:
        max_results = MAX_SEARCH_RESULTS

    try:
        from duckduckgo_search import DDGS

        results = []
        # DDGS().text() 返回生成器，每次 yield 一个搜索结果
        # region: 搜索区域，设为 "wt-wt" 表示全球，设为 "cn-zh" 表示中国中文
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, region="wt-wt"):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })

        return results

    except ImportError:
        print("  ⚠️ duckduckgo_search 未安装，请执行: pip install duckduckgo_search")
        return []
    except Exception as e:
        print(f"  ⚠️ DuckDuckGo 搜索异常: {e}")
        return []


def search_web(query: str, engine: str = None, max_results: int = None) -> list[dict]:
    """
    【基础功能】统一的网页搜索入口，支持切换搜索引擎
    【学习知识点】
        1. 策略模式：不同搜索引擎封装在同一接口后面，方便切换
        2. 生产环境建议 Tavily（AI 优化，直接返回可用的搜索结果）
    """
    if engine is None:
        engine = SEARCH_ENGINE
    if max_results is None:
        max_results = MAX_SEARCH_RESULTS

    engines = {
        "duckduckgo": search_duckduckgo,
        # 拓展：可以在此注册其他搜索引擎
        # "serpapi": search_serpapi,
        # "tavily": search_tavily,
    }

    search_func = engines.get(engine)
    if not search_func:
        print(f"  ⚠️ 不支持的搜索引擎: {engine}，使用 duckduckgo 兜底")
        search_func = search_duckduckgo

    return search_func(query, max_results=max_results)


# ============================================================
# 2. 网页内容提取 — URL → 纯文本
# ============================================================

def _clean_html(text: str) -> str:
    """
    清理 HTML 提取后的文本：去掉多余空行、特殊字符、HTML 实体
    """
    # 去掉 HTML 实体残留
    text = re.sub(r'&[a-z]+;', ' ', text)
    # 合并多余空白行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 合并多余空格
    text = re.sub(r'[ \t]{3,}', '  ', text)
    # 去掉 null 字符（某些网页可能包含）
    text = text.replace('\x00', '')
    return text.strip()


def fetch_page_text(url: str, timeout: int = None) -> str:
    """
    【基础功能】抓取网页并提取正文文本
    【学习知识点】
        1. HTTP GET 请求：User-Agent 模拟浏览器避免被拒
        2. BeautifulSoup 解析 HTML：提取 body 文本，跳过 script/style
        3. 降级策略：bs4 不可用时用正则粗暴提取（保证可用性）
        4. SSRF 防御：限制只能访问 http/https，禁止内网地址
        5. 超时控制：防止请求卡死影响整体流程
    参数：
        url: 网页地址（必须以 http:// 或 https:// 开头）
        timeout: 超时秒数
    返回值：提取的纯文本字符串
    调用示例：
        text = fetch_page_text("https://example.com/article")
        if text:
            print(f"提取了 {len(text)} 个字符")
    """
    if timeout is None:
        timeout = PAGE_FETCH_TIMEOUT

    # ---- 安全校验：URL 白名单（防御 SSRF 攻击）----
    # 只允许 http / https 协议
    if not re.match(r'^https?://', url):
        return ""

    # 禁止内网 IP 地址（192.168.*, 10.*, 127.*, 172.16.*）
    # 提取 URL 中的主机名部分
    host_match = re.search(r'://([^/:]+)', url)
    if host_match:
        host = host_match.group(1)
        # 检查是否是内网地址
        if re.match(r'^(127\.|192\.168\.|10\.|172\.(1[6-9]|2\d|3[01])\.)', host):
            return ""  # 放弃内网地址，防御 SSRF
        if host in ('localhost', '0.0.0.0', '[::1]'):
            return ""

    # 请求头：模拟浏览器，避免被网站拒绝
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()

        # 自动检测编码（某些网站不在 header 声明 charset）
        resp.encoding = resp.apparent_encoding if resp.apparent_encoding else "utf-8"

        html_text = resp.text

    except requests.exceptions.Timeout:
        return f"[超时] 请求 {url} 超时（{timeout}秒）"
    except requests.exceptions.SSLError:
        return f"[SSL错误] {url} 证书验证失败"
    except requests.exceptions.ConnectionError:
        return f"[连接失败] 无法连接到 {url}"
    except requests.exceptions.RequestException as e:
        return f"[请求错误] {url}: {e}"

    # ---- HTML → 纯文本提取 ----
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_text, "html.parser")

        # 移除无用的标签：脚本、样式、导航、页脚
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        # 提取 body 文本，separator="\n" 保证段落分隔
        body = soup.find("body")
        if body:
            raw_text = body.get_text(separator="\n", strip=True)
        else:
            raw_text = soup.get_text(separator="\n", strip=True)

    except ImportError:
        # ---- 降级方案：正则粗暴去标签 ----
        # 当 bs4 未安装时的兜底策略
        # 去掉 <script>...</script> 和 <style>...</style> 块
        html_text = re.sub(r'<script[^>]*>.*?</script>', '', html_text, flags=re.DOTALL | re.IGNORECASE)
        html_text = re.sub(r'<style[^>]*>.*?</style>', '', html_text, flags=re.DOTALL | re.IGNORECASE)
        # 去掉所有 HTML 标签
        raw_text = re.sub(r'<[^>]+>', '\n', html_text)
        # 解码常见 HTML 实体
        raw_text = raw_text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        raw_text = raw_text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')

    # 统一清理
    clean = _clean_html(raw_text)

    # 截断过长文本，防止 token 爆炸
    if len(clean) > MAX_PAGE_LENGTH:
        clean = clean[:MAX_PAGE_LENGTH] + "\n... [内容已截断]"

    return clean


# ============================================================
# 3. 完整的"搜索 → 阅读 → 回答"流水线
# ============================================================

class WebSearchPipeline:
    """
    【基础功能】把"搜索网页 + 阅读网页 + 回答问题"三个步骤封装成一条流水线
    【学习知识点】
        1. Pipeline 模式：将多步操作串联成统一接口
        2. 去重：搜索结果的 URL 和标题可能存在重复
        3. 异步思维：先搜索拿到 URL 列表，再逐个抓取内容
        4. 上下文拼接：将多个网页内容拼接成 RAG 式的上下文给 LLM
    """

    def __init__(self, search_engine: str = None):
        self.search_engine = search_engine or SEARCH_ENGINE
        self.last_sources = []  # 记录本次搜索的网页来源

    def run(self, query: str, max_results: int = None, verbose: bool = True) -> dict:
        """
        执行完整流水线：搜索 → 抓取网页 → LLM 总结回答

        参数：
            query: 用户问题
            max_results: 搜索网页数量
            verbose: 是否打印过程信息

        返回值：
            {"answer": str, "sources": [{"title": str, "url": str, "snippet": str}, ...]}

        调用示例：
            pipeline = WebSearchPipeline()
            result = pipeline.run("最新的 Python 版本是什么？")
            print(result["answer"])
            for src in result["sources"]:
                print(f"  - {src['title']}: {src['url']}")
        """
        if max_results is None:
            max_results = MAX_SEARCH_RESULTS

        # ---- 第 1 步：搜索网页 ----
        if verbose:
            print(f"🔍 搜索中: \"{query}\" ...")

        search_results = search_web(query, engine=self.search_engine, max_results=max_results)

        if not search_results:
            return {
                "answer": "搜索未返回任何结果，可能是网络问题或搜索服务暂时不可用。",
                "sources": [],
            }

        if verbose:
            print(f"  找到 {len(search_results)} 个网页")

        # 去重：按 URL 去重
        seen_urls = set()
        unique_results = []
        for r in search_results:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                unique_results.append(r)

        # ---- 第 2 步：逐个抓取网页内容 ----
        pages_content = []
        for i, result in enumerate(unique_results):
            if verbose:
                print(f"  📄 [{i+1}/{len(unique_results)}] {result['title'][:50]}...")

            content = fetch_page_text(result["url"])

            # 过滤掉抓取失败的页面（以 "[" 开头的错误信息）
            if content and not content.startswith("[") and len(content) > 100:
                pages_content.append({
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": result.get("snippet", ""),
                    "content": content,
                })

            # 小延迟：避免对同一网站请求过快（反爬虫礼仪）
            time.sleep(0.5)

        if not pages_content:
            return {
                "answer": "已找到相关网页，但无法抓取内容（可能是网站反爬虫或网络限制）。以下是搜索摘要：\n"
                + "\n".join([f"- {r['title']}: {r.get('snippet', '')[:100]}" for r in unique_results[:3]]),
                "sources": unique_results,
            }

        if verbose:
            total_chars = sum(len(p["content"]) for p in pages_content)
            print(f"  ✅ 成功提取 {len(pages_content)} 个网页，共 {total_chars} 字符")

        # ---- 第 3 步：拼接上下文，LLM 总结回答 ----
        # 将多个网页内容拼接成 RAG 风格的上下文
        context_parts = []
        for i, page in enumerate(pages_content):
            context_parts.append(
                f"[网页{i+1}: {page['title']}]\n"
                f"来源: {page['url']}\n"
                f"内容:\n{page['content']}\n"
            )
        context = "\n---\n".join(context_parts)

        # 构建 prompt：严格要求基于网页内容回答
        prompt = f"""根据以下网页搜索结果回答问题。

要求:
- 优先使用搜索到的信息，不要凭模型知识编造
- 如果不同网页信息矛盾，说明差异
- 如果信息不足以完整回答，说明哪些部分不确定
- 回答末尾列出参考来源（网址）

网页内容:
{context}

问题: {query}"""

        if verbose:
            print(f"  🤖 LLM 生成回答中... (上下文 {len(context)} 字符)")

        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )

        answer = response.choices[0].message.content

        # 记录来源
        self.last_sources = [
            {"title": p["title"], "url": p["url"], "snippet": p.get("snippet", "")}
            for p in pages_content
        ]

        return {"answer": answer, "sources": self.last_sources}


# ============================================================
# 4. Agent 类 — 集成 web_search 工具
# ============================================================

class Agent:
    """
    Agent = LLM + 工具集（含联网搜索） + ReAct 循环

    工具列表:
      - web_search: 搜索互联网获取实时信息
      - get_time: 获取当前时间
      - calculate: 数学计算
    """

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self._functions: dict[str, Callable] = {}
        self._schemas: list[dict] = []
        self.messages: list[dict] = []
        self.max_turns = 8

    def register_tool(self, name: str, func: Callable, description: str, parameters: dict = None):
        """注册工具到 Agent"""
        self._functions[name] = func
        self._schemas.append({
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
        })

    def _build_context(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}] + self.messages

    def _execute_tool(self, name: str, args: dict) -> str:
        func = self._functions.get(name)
        if not func:
            return f"未知工具: {name}"
        try:
            result = func(**args)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {e}"

    def run(self, user_input: str, verbose: bool = True) -> str:
        """ReAct 循环"""
        self.messages.append({"role": "user", "content": user_input})

        if verbose:
            print(f"\n{'─'*40}")
            print(f"🤖 {self.name} 正在处理...")

        for turn in range(1, self.max_turns + 1):
            context = self._build_context()
            tools = self._schemas if self._schemas else None

            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=context,
                tools=tools,
            )
            msg = response.choices[0].message

            # LLM 直接回复 → 完成
            if msg.content and not msg.tool_calls:
                self.messages.append({"role": "assistant", "content": msg.content})
                return msg.content

            # LLM 调用工具 → 执行工具 → 反馈结果
            if msg.tool_calls:
                tool_call_records = []
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)

                    if verbose:
                        print(f"  🔧 [{turn}] {name}({json.dumps(args, ensure_ascii=False)})")

                    result = self._execute_tool(name, args)

                    if verbose:
                        preview = result[:200].replace("\n", " ")
                        print(f"  📋 [{turn}] {preview}...")

                    tool_call_records.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": name, "arguments": tc.function.arguments},
                    })
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                self.messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": tool_call_records,
                })
                continue

        # 超兜底
        self.messages.append({"role": "user", "content": "请基于已有信息给出最终回答。"})
        response = llm_client.chat.completions.create(model=LLM_MODEL, messages=self._build_context())
        final = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": final})
        return final

    def reset(self):
        self.messages = []


# ============================================================
# 5. 构建带联网搜索的 Agent
# ============================================================

def build_agent() -> Agent:
    """构建 Agent，web_search 作为核心工具"""

    pipeline = WebSearchPipeline()

    # ---- 工具函数定义 ----

    def web_search(query: str) -> str:
        """
        【Agent 工具】搜索互联网获取实时信息，自动搜索 + 阅读网页 + 总结回答。

        用法：当用户询问实时新闻、最新动态、你不知道的信息时调用。
        参数 query: 搜索关键词，建议提取用户问题中的核心词，如"Python 3.13 发布日期"
        """
        result = pipeline.run(query, verbose=True)
        # 返回格式：回答 + 来源链接
        sources_text = "\n\n参考来源:\n" + "\n".join([
            f"- [{s['title']}]({s['url']})" for s in result["sources"]
        ])
        return result["answer"] + sources_text

    def get_time() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")

    def calculate(expression: str) -> str:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "错误: 包含不允许的字符"
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"计算错误: {e}"

    agent = Agent(
        name="联网智能助手",
        system_prompt=f"""你是一个可以联网搜索的智能助手。

你的能力:
- web_search 工具: 搜索互联网获取最新信息（实时新闻、技术文档、百科知识等）
- get_time 工具: 获取当前日期时间
- calculate 工具: 执行数学计算

工作原则:
- 当前日期是 {datetime.now().strftime('%Y-%m-%d')}，你内置的知识截止到训练日期，之后的事情你不知道
- 当用户询问新闻、最新动态、实时信息、你不知道的内容时，必须调用 web_search
- 当用户问时间相关问题时调用 get_time
- 一般常识性问题可以直接回答，不需要搜索
- 搜索结果可能不完整或矛盾，如实说明情况
- 回答时引用来源链接""",
    )

    # 注册工具 —— web_search 放在第一位
    agent.register_tool(
        "web_search",
        web_search,
        "搜索互联网获取实时信息。当你不确定答案、需要最新信息、或被问到新闻/时事时使用。"
        "参数 query: 搜索查询关键词",
        {"query": {"type": "string", "description": "搜索查询关键词"}},
    )
    agent.register_tool("get_time", get_time, "获取当前日期和时间")
    agent.register_tool(
        "calculate", calculate, "执行数学计算",
        {"expression": {"type": "string", "description": "数学表达式"}},
    )

    return agent


# ============================================================
# 6. 交互式主程序
# ============================================================

def main():
    print("=" * 60)
    print("🌐 联网搜索 Agent — 可以实时搜索网页的智能助手")
    print("=" * 60)
    print(f"LLM: {LLM_MODEL}")
    print(f"搜索引擎: {SEARCH_ENGINE}（免费，无需 API Key）")
    print(f"工具: web_search, get_time, calculate")
    print()

    agent = build_agent()
    print("✅ Agent 就绪！")
    print()
    print("试试这些问题（Agent 会自动判断是否需要联网搜索）:")
    print("  'Python 最新版本是多少？有什么新特性？'")
    print("  '今天的 AI 领域有什么重要新闻？'")
    print("  '杭州今天天气怎么样？' ← 需要联网")
    print("  '3.14 * 256 等于多少？' ← 不需要联网，直接用计算器")
    print("  '现在几点了？' ← 不需要联网")
    print()
    print("输入 'quit' 退出, 'reset' 重置对话\n")

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
            print("🔄 对话已重置")
            continue

        answer = agent.run(user_input, verbose=True)
        print(f"\n🤖 {answer}")
        print("\n" + "-" * 40)


# ============================================================
# 7. 概念总结
# ============================================================

def print_summary():
    print("""
╔══════════════════════════════════════════════════════════════╗
║              联网搜索 Agent 架构核心要点                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. 搜索 → 阅读 → 回答 三段式流水线                           ║
║     搜索: 关键词 → 搜索引擎 → URL 列表                        ║
║     阅读: URL → requests.get() → HTML → 纯文本               ║
║     回答: 网页内容拼接为上下文 → LLM 总结                     ║
║                                                              ║
║  2. 搜索引擎选择                                              ║
║     学习阶段: DuckDuckGo（免费，无需 API Key）                ║
║     生产环境: Tavily（AI 优化） / SerpAPI（Google）           ║
║     国内环境: 可替换为 Bing Search API 或自建爬虫             ║
║                                                              ║
║  3. 安全防护（工程必备）                                      ║
║     SSRF 防御: 禁止访问内网 IP（192.168.* / 10.* / 127.*）   ║
║     超时控制: requests.get(timeout=10) 防卡死                ║
║     反爬虫礼仪: 请求间延迟 0.5s                              ║
║                                                              ║
║  4. Agent 联网 vs RAG 知识库                                  ║
║     RAG 知识库: 内部私有信息（公司文档、产品手册）            ║
║     联网搜索:   外部公开信息（新闻、文档、百科）              ║
║     组合使用:   先查知识库 → 没有再搜网页（降级策略）         ║
║                                                              ║
║  5. 技术选型建议                                              ║
║     - 快速验证: DuckDuckGo + BeautifulSoup（本项目）          ║
║     - 稳定生产: Tavily Search API（AI 优化搜索结果）          ║
║     - 深度分析: Jina AI Reader（网页转 Markdown）             ║
║     - 大规模: Firecrawl / Spider（专业爬虫平台）              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
    print_summary()
