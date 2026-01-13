import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class ComplaintVectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initializes paths and the embedding model."""
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.parquet_path = os.path.join(self.current_dir, "../data/complaint_embeddings.parquet")
        self.store_path = os.path.join(self.current_dir, "../vector_store/full_faiss_index")
        
        # Initialize the embedding model
        self.embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_db = None

    def build_index(self):
        """Loads a subset of parquet data safely without loading the whole file."""
        import pyarrow.dataset as ds
        
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(f"Parquet file not found at {self.parquet_path}")

        print("📂 Memory-Safe Loading: Streaming a small subset of the data...")
        
        # This creates a 'scanner' that doesn't load the file yet
        dataset = ds.dataset(self.parquet_path, format="parquet")
        
        # We take just the first 10,000 rows as a record batch
        # This is the most memory-efficient way in pyarrow
        table = dataset.head(10000)
        df = table.to_pandas()

        print(f"🏗️ Building FAISS index with {len(df)} documents...")
        self.vector_db = FAISS.from_embeddings(
            text_embeddings=zip(df['document'].tolist(), df['embedding'].tolist()),
            embedding=self.embeddings_model,
            metadatas=df['metadata'].tolist()
        )
        
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        self.vector_db.save_local(self.store_path)
        print(f"✅ Index saved successfully to {self.store_path}")
        return self.vector_db
    def load_index(self):
        """Loads an existing FAISS index from the local disk."""
        if not os.path.exists(self.store_path):
            print("⚠️ Local index not found. Building it now...")
            return self.build_index()
        
        print("📥 Loading existing FAISS index...")
        self.vector_db = FAISS.load_local(
            self.store_path, 
            self.embeddings_model, 
            allow_dangerous_deserialization=True # Required for loading local FAISS files
        )
        return self.vector_db

# Instance for easy import
store_manager = ComplaintVectorStore()
vector_db = store_manager.load_index()