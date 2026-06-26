# ==============================================
# 文件名：step12_multi_agent.py
# 基础功能：从零构建多智能体系统，用 3 种协作模式让多个 Agent 分工完成复杂任务
# 核心学习知识点：
#   1. 三种多智能体通信拓扑：链式传递 / 星型层级 / 对等辩论
#   2. Agent-as-Tool 模式：把 Agent 封装为另一个 Agent 的工具（最简洁的多智能体实现）
#   3. 多智能体 vs 单智能体的权衡：专业化收益 vs 延迟/成本开销
#   4. 编排器(Orchestrator)设计：纯函数编排 vs Manager-Agent 编排
#   5. 闭包(Closure)实战：用闭包实现 Agent 工厂 → 工具函数的延迟创建
# 适用场景：复杂任务分解、多视角决策、内容生产流水线、代码审查
# 使用方法：
#   1. 确保 .env 中配置 DASHSCOPE_API_KEY
#   2. python step12_multi_agent.py
#   3. 选择演示模式 1/2/3，或回车使用默认场景
# 进阶说明：
#   - 本文件从零构建，不依赖 autogen/crewai 等框架
#   - Agent 基类可替换为 step07 完整版（带 Memory/Reflector），多智能体代码无需修改
#   - 三种模式可组合：经理模式中某个工人可以是一个流水线
# 常用配套函数：
#   1. agent_as_tool(): 将 Agent 工厂封装为工具函数（多智能体核心技巧）
#   2. run_pipeline(): 通用流水线执行器
#   3. run_debate(): 辩论编排器
#   4. batch_run(): 并行执行多个 Agent（进阶用法）
# ==============================================
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable
from dotenv import load_dotenv
from openai import OpenAI

# ---- Windows 终端 UTF-8 兼容 ----
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(Path(__file__).parent / ".env")

# ============================================================
# 配置区（集中管理，方便修改）
# ============================================================
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    print("❌ 未找到 API Key！请检查 .env 文件中的 DASHSCOPE_API_KEY")
    sys.exit(1)

# LLM 客户端：与 step01~step11 完全一致的初始化方式
llm_client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_MODEL = "qwen-plus"


# ============================================================
# 第 1 部分：Agent 基类（复用 step09/step11 简化版）
# ============================================================

class Agent:
    """
    【基础功能】可注册工具的智能体，通过 ReAct 循环自主决策调用工具
    【学习知识点】
        1. 统一接口：所有多智能体模式共用同一个 Agent 类 —— 这就是"面向接口编程"
        2. 本类是 step07 完整 Agent 的简化版，省略了 Memory/ToolKit/Reflector
        3. 聚焦多智能体协作逻辑，而非 Agent 内部实现细节

    【进阶说明】
        可以直接替换为 step07 的完整 Agent 类（带对话记忆、反思机制），
        本文件的所有多智能体协作代码无需任何修改。
        原理：多智能体只依赖 Agent.run() 和 Agent.register_tool() 两个接口，
              不关心 Agent 内部如何实现。

    调用示例：
        # 基础调用（零基础）
        agent = Agent(name="助手", system_prompt="你是一个有用助手")
        reply = agent.run("你好")
        print(reply)

        # 带工具的调用（进阶）
        agent = Agent(name="助手", system_prompt="你是数学助手")
        agent.register_tool("calc", lambda x: eval(x), "计算", {"x": {"type": "string"}})
        reply = agent.run("3*4+5 等于多少？")

    同场景常用替代函数：
        1. step07_complete_agent.Agent：完整版，带 Memory/Reflector
        2. autogen.ConversableAgent：微软 AutoGen 框架的 Agent
        3. crewai.Agent：CrewAI 框架的 Agent
    """

    def __init__(self, name: str, system_prompt: str):
        """
        【基础功能】初始化 Agent，给它一个名字和角色设定
        【学习知识点】
            system_prompt 是 Agent 的核心：它定义了 Agent 的"人设"和能力边界
            在多智能体系统中，每个 Agent 的 system_prompt 决定它的专业分工
        """
        self.name = name
        self.system_prompt = system_prompt
        # _functions: 工具名 → 工具函数 的映射表
        self._functions: dict[str, Callable] = {}
        # _schemas: OpenAI 函数调用格式的工具描述列表
        self._schemas: list[dict] = []
        # messages: 对话历史（列表中的每个元素是 {"role": ..., "content": ...}）
        self.messages: list[dict] = []
        # max_turns: 最大推理轮数（防止死循环）
        self.max_turns = 8

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: dict = None,
    ):
        """
        【基础功能】给 Agent 注册一个工具，让它能调用外部函数
        【学习知识点】
            1. OpenAI Function Calling 标准格式：每个工具需要 name / description / parameters
            2. parameters 是 JSON Schema 格式，描述工具的输入参数
            3. 链式调用：return self 支持 agent.register_tool(...).register_tool(...)

        调用示例：
            agent.register_tool("get_time", lambda: "2025-01-01", "获取当前时间")
        """
        self._functions[name] = func
        # 构建 OpenAI 标准的工具 Schema
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
        return self  # 链式调用支持

    def _build_context(self) -> list[dict]:
        """构建完整的对话上下文：system prompt + 历史消息"""
        return [{"role": "system", "content": self.system_prompt}] + self.messages

    def _execute_tool(self, name: str, args: dict) -> str:
        """
        【基础功能】执行指定工具并返回结果字符串
        【学习知识点】
            所有工具返回值统一转成字符串，方便拼接到 LLM 上下文中
        """
        func = self._functions.get(name)
        if not func:
            return f"错误：未找到工具 '{name}'。可用工具：{list(self._functions.keys())}"
        try:
            result = func(**args)
            return str(result)
        except Exception as e:
            return f"工具执行失败：{e}"

    def run(self, user_input: str, verbose: bool = True) -> str:
        """
        【基础功能】ReAct 循环：思考 → 行动 → 观察 → 思考...直到得出最终答案
        【学习知识点】
            1. ReAct = Reasoning + Acting，是 Agent 的核心工作流程
            2. 每轮 LLM 可以选择：直接回复 OR 调用工具
            3. max_turns 防止无限循环（工程必备的防御措施）
        """
        # 步骤1：把用户输入加入对话历史
        self.messages.append({"role": "user", "content": user_input})

        if verbose:
            print(f"\n{'─'*35}")
            print(f"🤖 {self.name} 思考中...")

        # 步骤2：ReAct 循环
        for turn in range(1, self.max_turns + 1):
            context = self._build_context()
            tools = self._schemas if self._schemas else None

            # 调用 LLM：传入工具列表，LLM 自主决定是否调用工具
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=context,
                tools=tools,
            )
            msg = response.choices[0].message

            # 情况A：LLM 直接回复（不调用工具）→ 任务完成
            if msg.content and not msg.tool_calls:
                self.messages.append({"role": "assistant", "content": msg.content})
                return msg.content

            # 情况B：LLM 决定调用工具 → 执行工具 → 把结果反馈给 LLM
            if msg.tool_calls:
                tool_call_records = []
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)

                    if verbose:
                        print(f"  🔧 [{turn}] 调用: {name}({json.dumps(args, ensure_ascii=False)})")

                    result = self._execute_tool(name, args)

                    if verbose:
                        preview = result[:150].replace("\n", " ")
                        print(f"  📋 [{turn}] 结果: {preview}...")

                    # 记录工具调用（OpenAI 要求的格式）
                    tool_call_records.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": name, "arguments": tc.function.arguments},
                    })
                    # 把工具结果加入对话历史
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                # 把 assistant 的工具调用决策也记录到历史
                self.messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": tool_call_records,
                })
                # 继续下一轮循环：LLM 可以继续调工具，也可以给出最终答案
                continue

        # 超过 max_turns 兜底：强制要求 LLM 给出最终答案
        self.messages.append({"role": "user", "content": "请基于已有信息给出最终答案"})
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=self._build_context(),
        )
        final = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": final})
        return final

    def reset(self):
        """重置对话历史（开始新对话时使用）"""
        self.messages = []


# ============================================================
# 第 2 部分：Agent-as-Tool 包装器（多智能体通信的核心技巧）
# ============================================================

def agent_as_tool(
    agent_factory,
    tool_name: str,
    tool_description: str,
    query_param_desc: str = "向该专家提出的问题",
):
    """
    【基础功能】将子 Agent 包装为主 Agent 的一个工具函数

    【学习知识点】
        1. 闭包(Closure)：内部函数 tool_func 捕获了外部的 agent_factory，
           每次调用 tool_func 时，才创建 Agent 实例 —— 这叫"延迟创建"
        2. 这是多智能体系统中最简洁、最实用的通信方式：
           主 Agent 通过 tool_call 调用子 Agent，子 Agent 的返回结果自动反馈给主 Agent
        3. 无状态设计：每次调用创建新 Agent → 不会残留上次调用的对话记忆

    【进阶说明】
        为什么用 agent_factory（工厂函数）而不是直接传 Agent 实例？
        - 如果直接传实例，每次 tool 调用会累积对话历史，导致记忆污染
        - 工厂函数每次创建新实例，"干净"地开始每次调用
        - 这就是"工厂模式"在多智能体中的应用

    参数：
        agent_factory: 无参数函数，返回 Agent 实例（如 lambda: Agent(...)）
        tool_name: 在主 Agent 中注册的工具名称
        tool_description: 工具的用途描述（LLM 据此决定何时调用）
        query_param_desc: query 参数的描述

    返回值：(tool_func, schema_dict) 二元组

    调用示例：
        # 把航班专家包装为工具
        func, schema = agent_as_tool(
            lambda: Agent(name="航班专家", system_prompt="你是航班专家..."),
            "ask_flight_expert",
            "向航班专家咨询航班方案",
        )
        manager.register_tool("ask_flight_expert", func, schema["description"],
                              schema["parameters"]["properties"])
    """

    def tool_func(query: str) -> str:
        """
        工具函数本体：创建子 Agent → 传入问题 → 返回结果
        这就是多智能体协作的最小单元
        """
        worker = agent_factory()  # 每次调用创建新实例（无状态）
        # 子 Agent 独立运行，不受主 Agent 对话历史影响
        result = worker.run(query, verbose=True)
        # 返回前加上子 Agent 的身份标识，方便主 Agent 引用
        return f"[{worker.name} 的回答]\n{result}"

    # 构建 OpenAI Function Calling 标准的参数 Schema
    schema = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": query_param_desc,
                    },
                },
                "required": ["query"],
            },
        },
    }

    return tool_func, schema


# ============================================================
# 第 3 部分：模式 1 — 顺序流水线（Sequential Pipeline）
# ============================================================

def demo_sequential_pipeline(topic: str = None):
    """
    【基础功能】演示顺序流水线：研究员 Agent → 作家 Agent
    【学习知识点】
        1. 流水线模式：上一个 Agent 的输出 = 下一个 Agent 的输入
        2. 每个 Agent 有独立的 system_prompt，专注做好一件事
        3. 适用场景：
           - 研究 → 写作（本文演示）
           - 翻译 → 润色
           - 代码编写 → 代码审查
           - 数据分析 → 报告生成

    【进阶说明】
        流水线的扩展方向：
        - 扇出(Fan-out)：研究员输出多个子主题 → 并行启动多个作家
        - 条件分支：根据中间结果的质量决定是否重试或换下一个 Agent
        - 质量控制：在每个阶段之间插入"审查 Agent"

    通信方式：消息传递（上一个 Agent 的返回值直接作为下一个 Agent 的输入）
    拓扑结构：Agent_A → Agent_B → Agent_C（链式）

    调用示例：
        article = demo_sequential_pipeline("人工智能发展历史")
        article = demo_sequential_pipeline()  # 交互式输入主题
    """
    if topic is None:
        topic = input("请输入研究主题（回车使用默认主题）> ").strip()
        if not topic:
            topic = "量子计算的发展与应用"

    # ---- 创建研究员 Agent ----
    # 注意：研究员不需要工具，完全依赖 LLM 的内置知识
    researcher = Agent(
        name="研究员",
        system_prompt="""你是资深信息研究员，目标是为后续写作提供扎实的资料。

请按以下结构输出研究笔记（Markdown 格式）：
1. **核心概念**：这个主题是什么？关键的术语和定义
2. **发展历程**：重要的时间节点和里程碑
3. **关键技术/应用**：有哪些重要的技术和实际应用场景
4. **未来趋势与挑战**：发展方向和当前面临的问题

要求：信息丰富但不冗长，每条 1-3 句话，总计控制在 800 字以内。""",
    )

    # ---- 创建作家 Agent ----
    # 作家的 system_prompt 明确要求"基于研究资料撰写"，防止编造
    writer = Agent(
        name="科普作家",
        system_prompt="""你是资深科普作家，面向零基础读者撰写通俗文章。

文章结构要求：
- **标题**：吸引人且有信息量
- **引言**（1段）：用一个有趣的问题或现象引入
- **正文**（3-4小节）：每节有小标题，层层递进
- **结语**（1段）：总结 + 对未来的一句话展望

写作原则：
- 用大白话解释专业概念，多用比喻
- 只基于提供的研究资料，不编造信息
- 如果研究资料不够详细，标注"据目前资料"
- 文章总长度 600-1000 字""",
    )

    # ---- 执行流水线 ----
    print(f"\n{'='*55}")
    print(f"🔄 顺序流水线启动: 研究员 → 科普作家")
    print(f"{'='*55}")

    # 阶段 1：研究员收集信息
    print(f"\n⏳ 阶段 1/2：{researcher.name} 收集信息中...")
    print(f"{'─'*40}")
    research_notes = researcher.run(
        f"请研究以下主题，整理关键信息：{topic}",
        verbose=True,
    )
    print(f"\n📋 研究笔记 ({len(research_notes)} 字符)：")
    print(research_notes[:400] + ("\n... [已截断]" if len(research_notes) > 400 else ""))

    # 阶段 2：作家根据研究资料撰写文章
    print(f"\n⏳ 阶段 2/2：{writer.name} 撰写文章中...")
    print(f"{'─'*40}")
    article = writer.run(
        f"""请根据以下研究资料写一篇面向大众的科普文章。

研究资料：
{research_notes}

写作主题：{topic}""",
        verbose=True,
    )

    print(f"\n{'='*55}")
    print(f"📰 最终文章：")
    print(f"{'='*55}")
    print(article)

    return article


# ============================================================
# 第 4 部分：模式 2 — 经理-工人（Manager-Worker）
# ============================================================

def demo_manager_worker(user_request: str = None):
    """
    【基础功能】演示经理-工人模式：1 个经理 + 3 个专家协作完成旅行规划
    【学习知识点】
        1. 层级结构(星型拓扑)：经理是中心节点，专家是叶子节点
        2. 经理负责"分解任务 + 协调专家 + 综合结果"
        3. 专家只需关注自己的领域，输出专业建议
        4. Agent-as-Tool 是这种模式最自然的实现方式

    【进阶说明】
        经理-工人 vs 单 Agent 多工具：
        ┌─────────────────┬──────────────────┬──────────────────┐
        │                  │ 单Agent 多工具    │ 经理-工人         │
        ├─────────────────┼──────────────────┼──────────────────┤
        │ 工具行为         │ 工具是纯函数      │ 工具内部有 LLM   │
        │ 专业性           │ 依赖工具逻辑      │ 每个工人有独立   │
        │                  │                   │ system_prompt    │
        │ 适用场景         │ 简单数据查询      │ 需要深度推理的   │
        │                  │ (天气/计算/时间)  │ 子任务            │
        │ LLM 调用次数     │ 少                │ 多（每个工人     │
        │                  │                   │ 也调 LLM）       │
        │ 成本             │ 低                │ 高               │
        └─────────────────┴──────────────────┴──────────────────┘

    通信方式：工具调用（经理通过 ReAct 循环调用工人工具）
    拓扑结构：经理（中心）→ 航班专家 / 酒店专家 / 交通专家（星型）

    调用示例：
        plan = demo_manager_worker("帮我规划北京到三亚3天旅行，预算5000")
        plan = demo_manager_worker()  # 交互式输入需求
    """
    if user_request is None:
        user_request = input("请输入旅行需求（回车使用默认）> ").strip()
        if not user_request:
            user_request = "帮我规划从北京到三亚的旅行，12月15日出发，3天2晚，双人出行，预算5000元"

    # ---- 创建 3 个专家 Agent 的工厂函数 ----
    # 每个工厂函数返回一个配置好 system_prompt 的专家 Agent

    def create_flight_expert():
        """创建航班专家的工厂函数"""
        return Agent(
            name="航班专家",
            system_prompt="""你是航班预订专家。根据用户需求推荐最佳航班方案。

考虑因素：
- 出发地到目的地的最优航线
- 直飞 vs 转机（转机至少留2小时间隔）
- 合理的时间安排（不要太早太晚）
- 性价比

输出格式（每个方案包含）：
- 航司 + 参考航班号
- 出发/到达时间
- 舱位 + 预估单人往返价格
- 推荐理由（一句话）

提供 2-3 个选择。""",
        )

    def create_hotel_expert():
        """创建酒店专家的工厂函数"""
        return Agent(
            name="酒店专家",
            system_prompt="""你是酒店预订专家。根据目的地和需求推荐酒店。

考虑因素：
- 位置便利性（离海滩/景点/市区距离）
- 星级和真实口碑
- 房间类型是否满足需求（大床/双床/家庭）
- 预算匹配

输出格式（推荐 2-3 家）：
- 酒店名称 + 星级
- 位置描述
- 推荐房型 + 每晚价格
- 特色亮点（如"步行3分钟到海滩"）
- 一句话推荐理由""",
        )

    def create_transport_expert():
        """创建交通专家的工厂函数"""
        return Agent(
            name="交通专家",
            system_prompt="""你是当地交通规划专家。推荐目的地当地的出行方案。

考虑因素：
- 机场到酒店的交通（接机/打车/大巴）+ 费用
- 景点间交通（租车 vs 打车 vs 公交）
- 3天行程的交通总预算估算
- 便利性和灵活性平衡

输出格式：
- 机场接驳方案（含费用）
- 日常出行推荐（含每日估算）
- 3天交通总预算
- 注意事项（如景区限行、打车软件推荐）""",
        )

    # ---- 创建经理 Agent ----
    manager = Agent(
        name="旅行规划经理",
        system_prompt="""你是专业旅行规划经理。收到用户需求后，按以下步骤工作：

1. **分析需求**：提取出发地、目的地、日期、人数、预算、特殊偏好
2. **咨询专家**：必须依次调用以下工具获取专业建议（不要自己编造）：
   - ask_flight_expert：获取航班方案
   - ask_hotel_expert：获取酒店推荐
   - ask_transport_expert：获取当地交通方案
3. **综合规划**：将所有专家建议整合成完整的旅行计划

最终输出格式（Markdown）：
## 🗺️ 旅行计划总览
（出发地→目的地、日期、人数、总预算）

## ✈️ 航班方案
（引用航班专家的推荐）

## 🏨 住宿推荐
（引用酒店专家的推荐）

## 🚗 当地交通
（引用交通专家的推荐）

## 💰 预算总览
（列表汇总各项费用：航班 + 酒店 + 交通 + 餐饮估算 + 其他）

## 💡 温馨提示
（3-5条实用建议）

注意：每个部分必须标注信息来源，如"（航班专家建议）"。""",
    )

    # ---- 将 3 个专家包装为经理的工具 ----
    # 这是多智能体系统的关键步骤：Agent → Tool
    flight_func, flight_schema = agent_as_tool(
        create_flight_expert,
        "ask_flight_expert",
        "向航班专家咨询航班方案。必须提供具体的出发地、目的地、日期、预算信息。",
        "查询内容，如：北京到三亚12月15日出发12月17日返回，双人往返，预算2000元",
    )
    hotel_func, hotel_schema = agent_as_tool(
        create_hotel_expert,
        "ask_hotel_expert",
        "向酒店专家咨询住宿方案。必须提供目的地、入住天数、预算、偏好信息。",
        "查询内容，如：三亚3天2晚双人住宿，预算1500元，偏好近海滩",
    )
    transport_func, transport_schema = agent_as_tool(
        create_transport_expert,
        "ask_transport_expert",
        "向交通专家咨询当地出行方案。提供目的地、行程天数、出行偏好。",
        "查询内容，如：三亚3天当地交通方案，机场到亚龙湾，预算500元",
    )

    # 注册到经理 Agent
    manager.register_tool(
        "ask_flight_expert", flight_func,
        flight_schema["function"]["description"],
        flight_schema["function"]["parameters"]["properties"],
    )
    manager.register_tool(
        "ask_hotel_expert", hotel_func,
        hotel_schema["function"]["description"],
        hotel_schema["function"]["parameters"]["properties"],
    )
    manager.register_tool(
        "ask_transport_expert", transport_func,
        transport_schema["function"]["description"],
        transport_schema["function"]["parameters"]["properties"],
    )

    # ---- 执行 ----
    print(f"\n{'='*55}")
    print(f"👔 经理-工人模式启动")
    print(f"{'='*55}")
    print(f"📝 用户需求：{user_request}")
    print(f"👥 可用专家：航班专家 | 酒店专家 | 交通专家")
    print(f"👔 经理正在分析需求并协调专家...")
    print(f"{'─'*40}")

    plan = manager.run(user_request, verbose=True)

    print(f"\n{'='*55}")
    print(f"📋 最终旅行计划：")
    print(f"{'='*55}")
    print(plan)

    return plan


# ============================================================
# 第 5 部分：模式 3 — 辩论（Debate）
# ============================================================

def demo_debate(topic: str = None):
    """
    【基础功能】演示辩论模式：正方 Agent vs 反方 Agent → 裁判 Agent 综合判断
    【学习知识点】
        1. 对抗协作：两个 Agent 持对立立场互相挑战，深入挖掘问题的多个维度
        2. 多轮辩论：每轮双方看到对方的论点后进行针对性反驳
        3. 裁判综合：第三方 Agent 站在中立视角评估双方论点的优劣
        4. 辩论 vs 反思：辩论是用另一个 Agent 来"反思"，比自我反思更客观

    【进阶说明】
        辩论模式的核心价值：
        - 处理没有标准答案的问题（伦理、策略、技术选型）
        - 两个 Agent 互相"校准"彼此的偏见和盲区
        - 裁判的总结通常比任何单一 Agent 的回答更全面、更平衡

        成本分析：
        - 每次辩论至少 (2 * rounds + 1) 次 LLM 调用
        - 适合重要决策场景，日常问答不划算

    通信方式：辩论记录（累计抄本作为每轮的共享上下文）
    拓扑结构：正方 ↔ 反方（对等）→ 裁判（汇总）

    调用示例：
        result = demo_debate("Python 适合做大型企业项目吗？")
        result = demo_debate()  # 交互式输入辩题
    """
    if topic is None:
        topic = input("请输入辩题（回车使用默认）> ").strip()
        if not topic:
            topic = "人工智能是否应该被广泛用于军事领域？"

    print(f"\n{'='*55}")
    print(f"⚖️  辩论模式启动")
    print(f"{'='*55}")
    print(f"辩题: {topic}")
    print(f"选手: 正方辩手 | 反方辩手 | 裁判")

    # ---- 创建三个辩论角色 ----
    # 正方：坚决支持
    pro_agent = Agent(
        name="正方辩手",
        system_prompt=f"""你是一位优秀的辩手。当前辩题是：{topic}

你是**【正方】**，观点是支持/赞成该命题。

辩论规则：
- 开场陈述(第1轮)：提出 3 个有力的论点，每个论点附带理由和例证
- 自由辩论(第2轮)：针对反方观点进行反驳，指出逻辑漏洞或反面例证
- 保持逻辑严密，不进行人身攻击
- 输出格式清晰，使用"论点1/2/3"编号""",
    )

    # 反方：坚决反对
    con_agent = Agent(
        name="反方辩手",
        system_prompt=f"""你是一位优秀的辩手。当前辩题是：{topic}

你是**【反方】**，观点是反对该命题。

辩论规则：
- 开场陈述(第1轮)：提出 3 个有力的反对论点，每个论点附带理由和例证
- 自由辩论(第2轮)：针对正方观点进行反驳，指出逻辑漏洞或反面例证
- 保持逻辑严密，不进行人身攻击
- 输出格式清晰，使用"论点1/2/3"编号""",
    )

    # 裁判：中立评估
    judge = Agent(
        name="裁判",
        system_prompt=f"""你是公正的辩论裁判。辩题：{topic}

你需要基于完整的辩论记录给出评估：

评估标准：
1. **论点质量**：逻辑性、证据支撑、说服力
2. **反驳能力**：是否准确抓住对方漏洞、回应是否有力
3. **全面性**：是否覆盖了辩题的关键维度

输出格式：
## 📊 评分
| 评分项 | 正方 | 反方 |
|--------|------|------|
| 论点质量 | X/10 | X/10 |
| 反驳能力 | X/10 | X/10 |
| 整体说服力 | X/10 | X/10 |

## 🔍 分析
- 正方优点：
- 正方弱点：
- 反方优点：
- 反方弱点：

## 🏆 综合判断
（2-3段的总结，包括：这个辩题的关键洞察是什么？双方各在什么场景下更有道理？对决策者的建议）""",
    )

    # ---- 执行辩论 ----
    debate_transcript = []  # 累计辩论记录（共享上下文）

    # 第 1 轮：开场陈述（双方阐述各自观点）
    print(f"\n🎤 第 1 轮：开场陈述")
    print(f"{'─'*50}")

    pro_opening = pro_agent.run(
        f"请就「{topic}」发表你的开场陈述。你是正方（支持方），提出你的核心论点。",
        verbose=False,
    )
    debate_transcript.append(f"═══ 正方开场陈述 ═══\n{pro_opening}")
    print(f"🔵 [正方]\n{pro_opening}\n")

    con_opening = con_agent.run(
        f"请就「{topic}」发表你的开场陈述。你是反方（反对方），提出你的核心论点。",
        verbose=False,
    )
    debate_transcript.append(f"═══ 反方开场陈述 ═══\n{con_opening}")
    print(f"🔴 [反方]\n{con_opening}\n")

    # 第 2 轮：自由辩论（双方针对对方论点反驳）
    print(f"🎤 第 2 轮：自由辩论（反驳对方）")
    print(f"{'─'*50}")

    pro_rebuttal = pro_agent.run(
        f"""反方的开场陈述如下：
---
{con_opening}
---
作为正方，请针对反方提出的论点进行反驳。指出其论证中的逻辑漏洞、论据不足或忽视的视角。""",
        verbose=False,
    )
    debate_transcript.append(f"═══ 正方反驳 ═══\n{pro_rebuttal}")
    print(f"🔵 [正方反驳]\n{pro_rebuttal}\n")

    con_rebuttal = con_agent.run(
        f"""正方的开场陈述如下：
---
{pro_opening}
---
作为反方，请针对正方提出的论点进行反驳。指出其论证中的逻辑漏洞、论据不足或忽视的视角。""",
        verbose=False,
    )
    debate_transcript.append(f"═══ 反方反驳 ═══\n{con_rebuttal}")
    print(f"🔴 [反方反驳]\n{con_rebuttal}\n")

    # 裁判判决
    print(f"⚖️ 裁判判决")
    print(f"{'─'*50}")

    full_transcript = "\n\n".join(debate_transcript)
    verdict = judge.run(
        f"""以下是关于「{topic}」的完整辩论记录：

{full_transcript}

请基于以上辩论记录给出你的综合评估和判断。""",
        verbose=False,
    )

    print(f"👨‍⚖️ [裁判]\n{verdict}")

    return {
        "topic": topic,
        "transcript": debate_transcript,
        "verdict": verdict,
    }


# ============================================================
# 第 6 部分：交互式主程序
# ============================================================

def main():
    print("=" * 60)
    print("🤖 多智能体系统 — Multi-Agent System")
    print("=" * 60)
    print(f"LLM: {LLM_MODEL}")
    print()

    while True:
        # ---- 菜单 ----
        print("\n" + "─" * 60)
        print("请选择演示模式：")
        print()
        print("  1. 🔄 顺序流水线 (Sequential Pipeline)")
        print("     研究员 → 作家：先研究主题，再撰写科普文章")
        print()
        print("  2. 👔 经理-工人 (Manager-Worker)")
        print("     旅行规划经理协调航班/酒店/交通专家")
        print()
        print("  3. ⚖️ 辩论 (Debate)")
        print("     正反方辩论 + 裁判综合判断")
        print()
        print("  q. 退出")
        print("─" * 60)

        try:
            choice = input("选择 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not choice:
            continue

        if choice.lower() == "q":
            print("再见！")
            break

        if choice == "1":
            demo_sequential_pipeline()
        elif choice == "2":
            demo_manager_worker()
        elif choice == "3":
            demo_debate()
        else:
            print("⚠️ 请输入 1 / 2 / 3 / q")

    # 结束后打印学习总结
    print_summary()


# ============================================================
# 第 7 部分：概念总结
# ============================================================

def print_summary():
    print("""
╔══════════════════════════════════════════════════════════════╗
║              多智能体系统 核心要点总结                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. 多智能体 vs 单智能体                                      ║
║     单Agent + 多工具：简单查询型任务，成本低                  ║
║     多Agent 协作：需要深度推理的子任务，每个Agent独立思考     ║
║     选择原则：简单任务不加Agent，复杂任务分而治之             ║
║                                                              ║
║  2. 三种通信拓扑                                              ║
║     链式(Pipeline)：Agent A → Agent B → ...                   ║
║         适合：有明确先后顺序的任务（研究→写作）               ║
║     星型(Manager-Worker)：Manager ←→ Workers                  ║
║         适合：需要多领域专家协作的复杂任务                    ║
║     对等+裁判(Debate)：Agent A ↔ Agent B → Judge              ║
║         适合：没有标准答案的决策/分析问题                     ║
║                                                              ║
║  3. Agent-as-Tool 核心技巧                                    ║
║     用 agent_as_tool() 将子Agent包成工具函数                  ║
║     关键设计：工厂函数 + 闭包 = 延迟创建 + 无状态             ║
║     这是最简洁的多智能体实现方式                              ║
║                                                              ║
║  4. 成本与延迟权衡                                            ║
║     每增加一个 Agent = 增加 N 次 LLM 调用                     ║
║     流水线2Agent ≈ 2次LLM调用（顺序执行，延迟叠加）           ║
║     经理+3工人 ≈ 4次LLM调用（可并行调用工人）                 ║
║     辩论2轮 ≈ 5次LLM调用（顺序执行）                          ║
║     省钱技巧：能用单Agent解决的不加Agent                      ║
║                                                              ║
║  5. 进阶方向                                                  ║
║     - 并行化：多个工人同时执行，减少总延迟                    ║
║     - 质量门禁：每个阶段之间加入审查Agent                     ║
║     - 动态路由：Manager根据任务自动选择最合适的工人           ║
║     - 记忆共享：多个Agent共享一个知识库（你已经会了！）       ║
║     - 生产框架：AutoGen / CrewAI / LangGraph                  ║
║                                                              ║
║  学习路径回顾:                                                ║
║     step01-07 → 单 Agent 开发全栈                             ║
║     step08-09 → RAG + 知识库                                  ║
║     step10    → 多格式文档摄入                                ║
║     step11    → 联网搜索 Agent                                ║
║     step12    → 多智能体协作 ← 你在这里                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
