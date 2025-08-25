import os
import streamlit as st
import time
from dotenv import load_dotenv

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI

# ----------------- API KEY SAFETY -----------------
load_dotenv()  # loads from .env if available
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
if 'GEMINI_API_KEY' not in os.environ:
    raise RuntimeError("GEMINI_API_KEY not set - add it to .env or environment variables.")
# --------------------------------------------------

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
index_dir = "faiss_gemini_index"

main_placeholder = st.empty()

# Gemini LLM (instead of OpenAI)
llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # create Gemini embeddings + FAISS index
    embeddings_gemini = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",   # ✅ correct model name
    google_api_key=os.getenv("GEMINI_API_KEY")  # ✅ explicitly pass API key
)
    vectorstore_gemini = FAISS.from_documents(docs, embeddings_gemini)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save FAISS index locally
    vectorstore_gemini.save_local(index_dir)

# --------- helper to clean messy sources ----------
def normalize_sources(sources_str: str):
    """Split messy sources into clean list of unique links"""
    if not sources_str:
        return []
    parts = [s.strip() for s in sources_str.replace("\n", " ").replace(",", " ").split(" ") if s.strip()]
    unique_links = []
    for p in parts:
        if p not in unique_links:
            unique_links.append(p)
    return unique_links
# --------------------------------------------------

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(index_dir) and os.listdir(index_dir):
        embeddings_gemini = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",   # ✅ fixed here too
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        vectorstore = FAISS.load_local(
            index_dir,
            embeddings_gemini,
            allow_dangerous_deserialization=True
        )
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        # Display sources, cleaned
        clean_sources = normalize_sources(result.get("sources", ""))
        if clean_sources:
            st.subheader("Sources:")
            for source in clean_sources:
                st.write(f"- {source}")
