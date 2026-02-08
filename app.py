# --- app.py ---
import gradio as gr
import os, sys
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import StreamingStdOutCallbackHandler

# 1. Setup project root
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.append(project_root)

from src.complaintRagChain import ComplaintRAGChain

# ================================
# 1. SETUP THE CPU MODEL (SMART DOWNLOAD)
# ================================
print("=== Checking Model Status ===")

models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# This function is "smart": 
# - If file exists in models_dir: It skips and returns path instantly.
# - If file is missing: It downloads it with a progress bar.
model_path = hf_hub_download(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="Phi-3-mini-4k-instruct-q4.gguf",
    local_dir=models_dir
)

print(f"✅ Model path verified: {model_path}")

# Initialize the CPU-optimized LLM
print("🧠 Loading LLM into RAM...")
# Inside your app.py
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=1024,           # Lowered context = much faster startup
    n_threads=6,          # Set to 6 or 8 for better CPU utilization
    n_batch=512,          # Processes the prompt in bigger chunks
    max_tokens=200,       # Prevents the AI from talking too much
    stop=["<|eot_id|>", "<|start_header_id|>", "user", "User:"],
    temperature=0.0,      # Most efficient for factual answers
    verbose=False,
    streaming=True
)

# ================================
# 2. INITIALIZE RAG SYSTEM
# ================================
print("📥 Loading Vector Store...")
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_path = os.path.join(project_root, "vector_store", "full_faiss_index")

vector_db = FAISS.load_local(
    index_path, 
    embeddings_model, 
    allow_dangerous_deserialization=True
)

analyst = ComplaintRAGChain(llm=llm, vector_db=vector_db)
print("🚀 System Online.")

# ================================
# 3. DEFINE GRADIO INTERFACE
# ================================
# ================================
# 3. DEFINE GRADIO INTERFACE
# ================================
def predict(message, history):
    # This calls your analyst logic
    response = analyst.query(message)
    return response["result"]

# Removed 'type="messages"' to fix the TypeError
demo = gr.ChatInterface(
    fn=predict, 
    title="🏦 CrediTrust Complaint Analyst",
    description="I check local complaint data to answer your financial queries.",
    # If your Gradio is older, it uses a simple (user, bot) tuple list for history automatically
)

if __name__ == "__main__":
    # share=False ensures it stays on your local machine
    # inbrowser=True opens the tab automatically
    demo.launch(inbrowser=True)