from vector_store import load_index, get_retriever

index = load_index("chunks")

retriever = get_retriever(
    index,
    similarity_top_k=3
)

results = retriever.retrieve(
    "What are Python string methods?"
)

for r in results:
    print(r.score)
    print(r.node.text[:500])