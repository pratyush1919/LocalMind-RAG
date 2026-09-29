"""
Centralised LLM + Embedding factory.
One env var (LLM_BACKEND) switches between local Ollama and cloud Groq.
Every agent in the system imports from here — no hard-coded models anywhere.
"""
import os
from dotenv import load_dotenv
load_dotenv()
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
print(LLM_BACKEND)
print(os.getenv("OLLAMA_MODEL"))
def get_llama_llm(temperature: float = 0.1, streaming: bool = False):
    """Return a LlamaIndex-compatible LLM."""
    if LLM_BACKEND == "groq":
        from llama_index.llms.groq import Groq
        return Groq(
            model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
        )
    else:  # ollama (default)
        from llama_index.llms.ollama import Ollama
        # return Ollama(
        #     model=os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_K_M"),
        #     base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        #     temperature=temperature,
        #     request_timeout=120.0,
        #     streaming=streaming,
        # )
        return Ollama(
            model=os.getenv("OLLAMA_MODEL"),
            base_url=os.getenv("OLLAMA_BASE_URL"),
            temperature=temperature,
            request_timeout=120.0,
            streaming=streaming,
            context_window=8192,
        )
    
def get_langchain_llm(
    temperature: float = 0.1,
    streaming: bool = False,
    request_timeout: float = None,
):
    """Return a LangChain-compatible LLM."""
    if LLM_BACKEND == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
            streaming=streaming,
        )
    else:  # ollama (default)
        from langchain_ollama import ChatOllama
        timeout = request_timeout if request_timeout is not None else float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "mistral-rag"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
            streaming=streaming,
            client_kwargs={"timeout": timeout},
        )

def get_embedding_model():
    """
    Return a LlamaIndex HuggingFaceEmbedding wrapping BAAI/bge-small-en-v1.5.
 
    I am using bge-small because:
    - 33 MB on disk — tiny
    - Bi-encoder architecture: query and document embedded independently (fast, scalable)
    - CPU-friendly — no GPU needed for embeddings
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        embed_batch_size=32,
    )
