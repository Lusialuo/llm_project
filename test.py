# pip install -U langchain langchain-openai langchain-community
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_deepseek_llm(cfg: Dict[str, Any]) -> ChatOpenAI:
    """
    用 OpenAI 兼容接口创建 DeepSeek/火山(ARK) 上的 Chat 模型客户端
    cfg = {
      "base_url": "...",
      "model": "...",
      "api_key": "...",
      "verify_ssl": False
    }
    """
    # langchain_openai / openai>=1.x 里通常用 http_client 来控制 verify
    # 这里用 httpx.Client(verify=...) 实现 verify_ssl=False
    import httpx
    http_client = httpx.Client(verify=cfg.get("verify_ssl", True))
    return ChatOpenAI(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        http_client=http_client,
    )

def create_deepseek_agent(cfg: Dict[str, Any]) -> AgentExecutor:
    llm = create_deepseek_llm(cfg)
    # 没有工具也能跑（纯对话 agent）；如果你有工具，把 tools=[...] 传进来
    tools = []
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个严谨的AI助手。"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def call_agent(agent: AgentExecutor, user_input: str) -> str:
    res = agent.invoke({"input": user_input})
    return res["output"]

if __name__ == "__main__":
    cfg = {
        "base_url": "https://ark-cn-beijing.bytedance.net/api/v3",
        "model": "ep-20251229164935-7mw76",
        "api_key": "598d7c22-d2af-4a92-bce2-5c9be3170b6a",
        "verify_ssl": False,
    }
    agent = create_deepseek_agent(cfg)
    out = call_agent(agent, "用三句话解释RAG是什么。")
    print(out)
