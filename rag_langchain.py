import warnings
from transformers import pipeline
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

warnings.filterwarnings("ignore", category=UserWarning)

# === 1) Load Vector DB ===
print("🔄 Loading Chroma DB and embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})
print("✅ Vector DB ready")

# === 2) Load local HuggingFace LLM ===
print("⚙️ Loading FLAN-T5-BASE model...")
local_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_length=1536,
    temperature=0.3,
    repetition_penalty=1.2
)
llm = HuggingFacePipeline(pipeline=local_model)

# === 3) Prompt ===
prompt_template = """
You are a knowledgeable assistant. Based on the context provided, write a detailed summary.
Your response must be 10–15 lines, well-structured, and include all key ideas.
Do NOT repeat the question. Do NOT say "according to the context".

<context>
{context}
</context>

Question: {question}
Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# === 4) Helper to format docs ===
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# === 5) Build RAG pipeline ===
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# === 6) Ask user ===
query = input("\n💬 Enter your question: ")
answer = rag_chain.invoke(query)

print("\n🔍 Query:", query)
print("🧠 Answer:\n", answer)
