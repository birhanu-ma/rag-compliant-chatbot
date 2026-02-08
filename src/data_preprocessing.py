import pandas as pd
import re
import nltk
import os
import zipfile
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- 1. ENSURE RESOURCES ---
def ensure_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            # Check if the resource exists
            nltk.data.find(f'tokenizers/{res}' if res == 'punkt' else f'corpora/{res}')
        except (LookupError, zipfile.BadZipFile): 
            print(f"📥 NLTK resource '{res}' is missing or corrupt. Downloading...")
            nltk.download(res, quiet=False)

# Run the check
ensure_nltk_resources()

# --- 2. CLEANING CLASS ---
class ComplaintFinalProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.target_products = [
            'Credit card', 
            'Personal loan', 
            'Savings account', 
            'Money transfers',
            'Credit card or prepaid card'
        ]
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text_noise(self, text):
        """Noise Removal: URLs, Emails, Phones, HTML, and XXXX Redactions"""
        text = str(text).lower()
        # Remove redacted XXXX patterns
        text = re.sub(r'\b[x]{2,}\d*\b', '', text) 
        text = re.sub(r'\b[x]{2,}/[x]{2,}/\d*\b', '', text)
        # Remove URLs, Emails, Phone numbers
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
        text = re.sub(r'\b\d{3}[-.\s]?\d{4}\b', '', text) 
        text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)
        # Remove HTML and Punctuation
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def normalize_text(self, text):
        """Normalization: Tokenization, Stopwords, and Lemmatization"""
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        # Double-pass Lemmatization
        lemmas = [self.lemmatizer.lemmatize(t, pos='v') for t in tokens]
        lemmas = [self.lemmatizer.lemmatize(t, pos='n') for t in lemmas]
        return " ".join(lemmas)

    def run_cleaning_pipeline(self):
        print("🚀 Starting Final Preprocessing Pipeline...")
        # Step A & B: Filter and Drop NAs
        self.df = self.df[self.df['Product'].isin(self.target_products)]
        self.df = self.df.dropna(subset=['Consumer complaint narrative'])
        
        # Step C & D: Clean and Normalize
        print("-> Stripping Noise and Applying Normalization...")
        self.df['processed_text'] = self.df['Consumer complaint narrative'].apply(self.clean_text_noise)
        self.df['normalized_text'] = self.df['processed_text'].apply(self.normalize_text)
        
        # Step E: Pruning
        self.df = self.df[self.df['normalized_text'].apply(lambda x: len(str(x).split()) > 2)]
        print(f"✓ Pipeline Complete. {len(self.df)} records ready.")
        return self.df

    def save_processed_data(self, output_path):
        """Safely creates the directory if it doesn't exist and saves the CSV"""
        # Extract the directory path (e.g., '../data/processed')
        directory = os.path.dirname(output_path)
        
        # Create directory if it does not exist
        if directory and not os.path.exists(directory):
            print(f"📂 Creating missing directory: {directory}")
            os.makedirs(directory, exist_ok=True)
            
        self.df.to_csv(output_path, index=False)
        print(f"💾 File successfully saved to: {output_path}")

