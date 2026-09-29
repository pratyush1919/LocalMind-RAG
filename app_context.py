from P1.llm_factory import (
    get_embedding_model,
    get_llama_llm,
    get_langchain_llm
)
EMBED_MODEL = get_embedding_model()
LLM = get_llama_llm(
    temperature=0.1
)
from functools import lru_cache
@lru_cache(maxsize=16)
def get_agent_llm(temperature: float = 0.1,streaming: bool = False,):
    return get_langchain_llm(temperature=temperature, streaming=streaming,)

