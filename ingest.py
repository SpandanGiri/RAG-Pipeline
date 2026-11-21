from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",  # or "gemini-embedding-2"
        output_dimensionality=768,     # optional: keeps Chroma smaller & faster (default is 3072)
    )

    Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
    )
    
    print(f"Saved {len(chunks)} chunks to {PERSIST_DIR}")
