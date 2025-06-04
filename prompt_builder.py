def build_prompt(query: str, docs: list[str]) -> str:
    # If no documents are provided, return a simple prompt
    if not docs:
        return f"Answer the question: {query}\n\nNo relevant context found."
    # Join the documents with a separator
    context = "\n---\n".join(docs)
    return f"Given the following context:\n{context}\n\nAnswer the question: {query}"