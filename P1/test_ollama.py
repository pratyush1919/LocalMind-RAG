# test_ollama.py

from P1.llm_factory import get_llama_llm

llm = get_llama_llm()

print("Model loaded")

response = llm.complete("Say hello in one sentence.")

print(response)