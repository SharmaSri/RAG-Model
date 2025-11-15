from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)





# 1️⃣ Load data from 'data/' folder
data_path = "data"
docs = []

for filename in os.listdir(data_path):
    file_path = os.path.join(data_path, filename)
    if filename.endswith(".txt"):
        loader = TextLoader(file_path)
    elif filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        continue
    docs.extend(loader.load())

print(f"✅ Loaded {len(docs)} documents")

# 2️⃣ Split text into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"✅ Created {len(chunks)} text chunks")

# 3️⃣ Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4️⃣ Store in Vector DB (Chroma)
db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="chroma_db")
db.persist()

print("✅ Vector DB successfully created in 'chroma_db/'")
