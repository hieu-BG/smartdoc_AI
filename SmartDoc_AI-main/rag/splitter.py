from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_chunks_from_pages(pages_text, chunk_size=500, chunk_overlap=50):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    for page in pages_text:
        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "source": page["source"],
                "text": chunk
            })

    return chunks