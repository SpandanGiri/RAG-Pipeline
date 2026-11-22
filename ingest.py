from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import time

PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 80
DELAY_SECONDS = 65  

def build_chunks(root):
   
    docs = DirectoryLoader(
        root, glob="**/*.md",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    ).load()


    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )

    sections = []
    for d in docs:
        for s in header_splitter.split_text(d.page_content):
            s.metadata["source"] = d.metadata["source"]  # splitter drops it, so re-attach
            sections.append(s)

    char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return char_splitter.split_documents(sections)



if __name__ == "__main__":

    chunks = build_chunks("corpus/docs")
    print(f"Total chunks to embed: {len(chunks)}")


    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",  # or "gemini-embedding-2"
        output_dimensionality=768,     # optional: keeps Chroma smaller & faster (default is 3072)
    )

    # Replacing it because it tries to load all the chunks at the same time to Gemini API

    # Chroma.from_documents(
    #         documents=chunks,
    #         embedding=embeddings,
    #         persist_directory=PERSIST_DIR,
    # )


    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    

for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vector_store.add_documents(batch)
        print(f"Indexed {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

        if i + BATCH_SIZE < len(chunks):
            time.sleep(DELAY_SECONDS)

print(f"Saved {len(chunks)} chunks to {PERSIST_DIR}")
