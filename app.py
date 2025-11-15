import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ---- LangChain + HF imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

from transformers import pipeline
from langchain_core.prompts import PromptTemplate

# === Config ===
DATA_FOLDER = "data"
PERSIST_DIR = "chroma_db"
os.makedirs(DATA_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder="templates")
CORS(app)

# === Initialize embeddings + Chroma DB ===
print("🔄 Loading embeddings and connecting to Chroma...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})   # <= reduce retrieved chunk count
print("✅ Vector DB ready")

# === Load local HF model ===
print("⚙️ Loading local HuggingFace model (FLAN-T5-BASE)...")
local_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_new_tokens=200,         # smaller = prevents overflow
    repetition_penalty=1.15
)
llm = HuggingFacePipeline(pipeline=local_model)

# === Prompt ===
prompt_template = """
You are a helpful assistant. Using the context below, answer the question in 10-12 lines.
Be detailed, structured and DO NOT mention the word "context" in output.

<context>
{context}
</context>

Question: {input}
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

# === Combine docs + LLM ===
def combine_docs(docs, question):
    context = "\n\n".join(d.page_content for d in docs)

    # ------- SAFETY: Reduce context to avoid >512 length error --------
    context = context[:1800]        # truncate very long retrieved text

    final_prompt = prompt.format(context=context, input=question)
    return llm.invoke(final_prompt)   # invoke works safely with HF pipeline

# === Ingestion helper ===
def ingest_pdf_path(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    chunks = splitter.split_documents(pages)

    vectordb = Chroma.from_documents(
        chunks, embedding_function=embeddings, persist_directory=PERSIST_DIR
    )
    vectordb.persist()

    global db, retriever
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    return len(chunks)

# === Routes ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    docs = retriever.invoke(query)

    try:
        answer = combine_docs(docs, query)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"answer": answer})

@app.route("/ingest", methods=["POST"])
def ingest():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    tmp_path = os.path.join(DATA_FOLDER, f.filename)
    f.save(tmp_path)

    try:
        chunks = ingest_pdf_path(tmp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": f"Ingested {f.filename} ({chunks} chunks)"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
