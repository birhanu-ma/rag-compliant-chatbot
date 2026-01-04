import pandas as pd
import os
import sys
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
# Note the underscore and the 's' at the end
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document

# 1. SETUP HUGGINGFACE CACHE (Must happen before HF imports)
PROJECT_ROOT = Path(os.getcwd()).parent
MODELS_DIR = PROJECT_ROOT / "models"

def setup_hf_cache():
    """Configures HF to store models in the project directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(MODELS_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODELS_DIR)
    print(f"✓ HuggingFace cache set to: {MODELS_DIR}")

setup_hf_cache()

class VectorStoreManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Initializing Embedding Model: {model_name}...")
        # This will now download to your /models/ folder
        self.embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{model_name}"
        )
        
        # Chunking Strategy: Balanced for complaints
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,      # Small enough for specific meaning
            chunk_overlap=60,    # Ensures context isn't cut between chunks
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def prepare_stratified_sample(self, df, target_size=12000):
        """Ensures proportional representation of products."""
        print(f"Creating stratified sample of {target_size} records...")
        return df.groupby('Product', group_keys=False).apply(
            lambda x: x.sample(n=int(len(x)/len(df) * target_size))
        )

    def build_and_save_index(self, df, index_path="../vector_store/complaints_index"):
        """Chunks, embeds, and saves the vector store."""
        documents = []
        
        print(f"Chunking {len(df)} narratives...")
        for _, row in df.iterrows():
            # Use the cleaned text from Task 1
            text = str(row['normalized_text'])
            chunks = self.text_splitter.split_text(text)
            
            for chunk in chunks:
                # Metadata is key for tracing back to the original complaint
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "complaint_id": row['Complaint ID'],
                        "product": row['Product'],
                        "issue": row['Issue'],
                        "state": row['State']
                    }
                )
                documents.append(doc)

        print(f"Generating embeddings for {len(documents)} chunks (this may take a few minutes)...")
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Ensure the output directory exists
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(index_path)
        
        print(f"✅ Vector store saved to {index_path}")
        return vector_store

# --- EXAMPLE USAGE IN NOTEBOOK ---
# processor = VectorStoreManager()
# sampled_df = processor.prepare_stratified_sample(df_cleaned)
# vector_db = processor.build_and_save_index(sampled_df)