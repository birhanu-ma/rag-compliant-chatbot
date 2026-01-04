import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- FIX FOR THE HANGING IMPORT ---
# We check if they exist before downloading to prevent the 40-minute wait
import zipfile  # Add this import at the top

def ensure_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    for res in resources:
        try:
            # Check if the resource exists and is valid
            nltk.data.find(f'tokenizers/{res}' if res == 'punkt' else f'corpora/{res}')
        except (LookupError, zipfile.BadZipFile): 
            # If missing OR corrupted (BadZipFile), download it again
            print(f"NLTK resource '{res}' is missing or corrupt. Downloading...")
            nltk.download(res, quiet=False) # quiet=False shows progress so you know it's working
# ----------------------------------

class ComplaintFinalProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Specific products for the deliverable
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
        """Processes Noise Removal: URLs, Emails, Phones, HTML, and XXXX Redactions"""
        text = str(text).lower()
        
        # 1. REMOVE REDACTED XXXX PATTERNS
        # Matches 'xxxx', 'xx/xx/xxxx', and 'xxxx2021'
        text = re.sub(r'\b[x]{2,}\d*\b', '', text) 
        # Matches 'xx/xx/' 
        text = re.sub(r'\b[x]{2,}/[x]{2,}/\d*\b', '', text)
        
        # 2. Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # 3. REMOVE EMAILS
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
        
        # 4. Remove Phone Numbers
        text = re.sub(r'\b\d{3}[-.\s]?\d{4}\b', '', text) 
        text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)
        
        # 5. Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # 6. Remove Punctuation and Special Characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # 7. Remove Boilerplate
        text = text.replace("i am writing to file a complaint", "")
        
        return text.strip()
    
    def normalize_text(self, text):
        """Processes Normalization: Tokenization, Stopwords, and Lemmatization"""
        tokens = word_tokenize(text)
        
        # Remove Stopwords
        tokens = [t for t in tokens if t not in self.stop_words]
        
        # Lemmatize (Double-pass for Verbs and Nouns)
        # This turns 'paying' -> 'pay' and 'banks' -> 'bank'
        lemmas = [self.lemmatizer.lemmatize(t, pos='v') for t in tokens]
        lemmas = [self.lemmatizer.lemmatize(t, pos='n') for t in lemmas]
        
        return " ".join(lemmas)

    def run_cleaning_pipeline(self):
        print("Starting Final Preprocessing Pipeline...")

        # STEP A: Product Filteration
        self.df = self.df[self.df['Product'].isin(self.target_products)]
        
        # STEP B: Drop Missing Narratives (Required for RAG)
        self.df = self.df.dropna(subset=['Consumer complaint narrative'])
        
        # STEP C: Noise Cleaning (Privacy + Regex)
        print("-> Stripping Noise (URLs, Emails, Phones, HTML)...")
        self.df['processed_text'] = self.df['Consumer complaint narrative'].apply(self.clean_text_noise)
        
        # STEP D: Normalization (Semantic Processing)
        print("-> Applying Tokenization, Stopword removal, and Lemmatization...")
        self.df['normalized_text'] = self.df['processed_text'].apply(self.normalize_text)
        
        # STEP E: Vocabulary Pruning
        # Removes rows that are too short to be useful for AI training
        self.df = self.df[self.df['normalized_text'].apply(lambda x: len(str(x).split()) > 2)]

        print(f"✓ Pipeline Complete. {len(self.df)} records fully cleaned and ready.")
        return self.df