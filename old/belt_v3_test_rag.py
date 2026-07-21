from belt_v3_rag import rag_search


while True:
    query = input("\nAsk a question, or type 'quit': ").strip()

    if query.lower() == "quit":
        break

    if not query:
        continue

    results = rag_search(
        query=query,
        top_k=3
    )

    if not results:
        print("No relevant information found.")
        continue

    for rank, result in enumerate(results, start=1):
        print(
            f"\nResult {rank} | "
            f"Hybrid: {result['score']:.4f} | "
            f"Embedding: {result['embedding_score']:.4f} | "
            f"TF-IDF: {result['tfidf_score']:.4f}"
        )
        print(result["chunk"])