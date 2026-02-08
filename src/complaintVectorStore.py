import os
import pandas as pd
import pyarrow.parquet as pq
import gc
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class ComplaintVectorStore:
    def __init__(self, full_parquet_path, custom_parquet_path, store_path, model_name="all-MiniLM-L6-v2"):
        self.full_parquet = full_parquet_path
        self.custom_parquet = custom_parquet_path
        self.store_path = store_path
        
        self.embeddings_model = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{model_name}")
        self.vector_db = None

    def _load_parquet_safe(self, path, limit=None):
        """Streaming loader to prevent ArrowMemoryError."""
        if not os.path.exists(path):
            print(f"⚠️ Skipping: {path} (Not found)")
            return None
        
        print(f"📂 Reading: {os.path.basename(path)}...")
        pfile = pq.ParquetFile(path)
        
        # Load in chunks to be gentle on RAM
        chunks = []
        rows = 0
        for i in range(pfile.num_row_groups):
            chunk = pfile.read_row_group(i).to_pandas()
            chunks.append(chunk)
            rows += len(chunk)
            if limit and rows >= limit: break
            
        return pd.concat(chunks, ignore_index=True).head(limit) if limit else pd.concat(chunks, ignore_index=True)

    def build_index(self, full_limit=5000):
        """The main engine: Load data and create FAISS index."""
        dfs = []

        # 1. Load data safely
        df_full = self._load_parquet_safe(self.full_parquet, limit=full_limit)
        if df_full is not None: dfs.append(df_full)

        df_custom = self._load_parquet_safe(self.custom_parquet)
        if df_custom is not None: dfs.append(df_custom)

        if not dfs:
            raise ValueError("❌ No data found! Check your file paths.")

        final_df = pd.concat(dfs, ignore_index=True)
        
        # Clear temporary data from memory immediately
        del dfs
        gc.collect() 

        print(f"🏗️ Building FAISS index with {len(final_df)} documents...")
        
        # Build the vector store
        self.vector_db = FAISS.from_embeddings(
            text_embeddings=zip(final_df['document'].tolist(), final_df['embedding'].tolist()),
            embedding=self.embeddings_model,
            metadatas=final_df['metadata'].tolist()
        )
        
        # Save to disk
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        self.vector_db.save_local(self.store_path)
        
        print(f"✅ Merged index saved to: {self.store_path}")
        
        # Final cleanup
        del final_df
        gc.collect()
        
        return self.vector_db