import pandas as pd
import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm # To see progress

class ComplaintEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_id = f"sentence-transformers/{model_name}"
        print(f"Initializing Embedding Model: {self.model_id}...")
        # encode_kwargs tells the model to process data in chunks (batches)
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=self.model_id,
            encode_kwargs={'batch_size': 64} 
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def process_and_save(self, df, output_filename="custom_complaint_embeddings.parquet"):
        """Chunks text, generates vectors in BATCHES, and saves to Parquet."""
        
        raw_chunks_data = []
        
        print(f"📦 Step 1: Chunking {len(df)} complaints...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text = str(row['normalized_text'])
            chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                # We store the raw info first, without embedding yet
                raw_chunks_data.append({
                    "complaint_id": row.get('Complaint ID', 'N/A'),
                    "product_category": row.get('Product', 'N/A'),
                    "document": chunk,
                    "metadata": {
                        "issue": row.get('Issue', 'N/A'),
                        "state": row.get('State', 'N/A'),
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                })

        # --- THE SPEED FIX: BATCH EMBEDDING ---
        print(f"🧠 Step 2: Generating vectors for {len(raw_chunks_data)} chunks in batches...")
        
        # Extract all text strings into a list
        texts_to_embed = [item['document'] for item in raw_chunks_data]
        
        # embed_documents is MUCH faster than embed_query in a loop
        # It processes many strings at once
        all_vectors = self.embeddings_model.embed_documents(texts_to_embed)

        # Re-attach vectors to our data
        for i, vector in enumerate(all_vectors):
            raw_chunks_data[i]['embedding'] = vector

        # --- STEP 3: SAVE ---
        output_df = pd.DataFrame(raw_chunks_data)
        save_dir = Path("../data/processed/")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / output_filename
        
        output_df.to_parquet(save_path, engine='pyarrow', index=False)
        print(f"✅ Success! Vectorized data saved to: {save_path}")
        return save_path