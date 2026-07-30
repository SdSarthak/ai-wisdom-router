from langchain_community.llms import Ollama
from backend.config import LLM_MODEL, OLLAMA_BASE_URL

_llm = None


def get_llm() -> Ollama:
    global _llm
    if _llm is None:
        _llm = Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.7,
        )
    return _llm


def generate(system_prompt: str, user_message: str, history: list = None) -> str:
    llm = get_llm()
    messages = [("system", system_prompt)]
    if history:
        for turn in history[-6:]:
            messages.append((turn["role"], turn["content"]))
    messages.append(("human", user_message))

    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    lc_messages = []
    for role, content in messages:
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "human":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    return llm.invoke(lc_messages)
