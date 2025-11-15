import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_FOLDER = "data"
persist_directory = "chroma_db"

print("📌 Loading PDFs from Data folder...")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
all_docs = []

for file in os.listdir(DATA_FOLDER):
    if file.endswith(".pdf"):
        print(f"📄 Processing: {file}")
        loader = PyPDFLoader(os.path.join(DATA_FOLDER, file))
        pages = loader.load()
        all_docs.extend(pages)

print(f"📑 Total pages loaded: {len(all_docs)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = text_splitter.split_documents(all_docs)

print(f"📝 Total chunks created: {len(chunks)}")

vectordb = Chroma.from_documents(
    chunks,
    embedding=embeddings,      # ⚠️ NOTICE: parameter name changed
    persist_directory=persist_directory
)

vectordb.persist()
print("🎯 PDF data indexed to Chroma DB successfully!")
