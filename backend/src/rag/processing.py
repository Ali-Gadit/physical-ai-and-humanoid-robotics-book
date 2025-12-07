def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Splits text into smaller chunks with a specified overlap.
    Aims to preserve sentence boundaries where possible.
    """
    if not text:
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    i = 0
    while i < len(words):
        chunk_end = min(i + chunk_size, len(words))
        chunk = " ".join(words[i:chunk_end])
        chunks.append(chunk)

        if chunk_end == len(words):
            break

        i += (chunk_size - chunk_overlap)
        if i >= len(words):
            break # Ensure no empty chunks or negative overlap

    return chunks
