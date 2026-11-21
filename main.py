from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know."

Context:
{context}

Question: {question}
""")


def load_rag(docs_dir: str):

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    return retriever, llm

def ask(question: str, retriever, llm):
    retrieved_docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    messages = PROMPT.format_messages(context=context, question=question)
    response = llm.invoke(messages)

    print("Answer:\n", response.text)
    print("\nSources used:")
    for doc in retrieved_docs:
        print("-", doc.metadata.get("source"), "|", doc.metadata.get("h2", ""))


if __name__ == "__main__":
    retriever,llm = load_rag("corpus/docs")
    ask("How do I handle 429 rate limit errors?", retriever, llm)