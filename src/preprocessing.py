"""
Preprocessing Module for Information Retrieval System
Handles text normalization, tokenization, stemming, and stopword removal.
"""

import re
import string
from typing import List, Set, Optional
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

download_nltk_resources()


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for IR systems.
    
    Features:
    - Case normalization
    - Punctuation removal
    - Tokenization
    - Stopword removal
    - Stemming/Lemmatization
    - Number handling
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        use_stemming: bool = True,
        use_lemmatization: bool = False,
        min_word_length: int = 2,
        custom_stopwords: Optional[Set[str]] = None
    ):
        """
        Initialize the preprocessor with configurable options.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_numbers: Remove numeric characters
            remove_stopwords: Remove common stopwords
            use_stemming: Apply Porter stemming
            use_lemmatization: Apply WordNet lemmatization
            min_word_length: Minimum word length to keep
            custom_stopwords: Additional stopwords to remove
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.min_word_length = min_word_length
        
        # Initialize stemmer and lemmatizer
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Build stopwords set
        self.stopwords = set(stopwords.words('english'))
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.special_chars_pattern = re.compile(r'[^\w\s]')
        self.number_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def clean_text(self, text: str) -> str:
        """
        Clean raw text by removing URLs, emails, HTML tags, and extra whitespace.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove emails
        text = self.email_pattern.sub(' ', text)
        
        # Remove HTML tags
        text = self.html_pattern.sub(' ', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        return text.strip()
    
    def normalize(self, text: str) -> str:
        """
        Normalize text through case conversion and character handling.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text string
        """
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = self.special_chars_pattern.sub(' ', text)
        
        if self.remove_numbers:
            text = self.number_pattern.sub(' ', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback to simple split if NLTK fails
            tokens = text.split()
        
        return tokens
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens based on stopwords and length criteria.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered list of tokens
        """
        filtered = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_word_length:
                continue
            
            # Skip stopwords
            if self.remove_stopwords and token.lower() in self.stopwords:
                continue
            
            # Skip pure numbers if configured
            if self.remove_numbers and token.isdigit():
                continue
            
            filtered.append(token)
        
        return filtered
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming or lemmatization to tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of stemmed/lemmatized tokens
        """
        if self.use_stemming:
            return [self.stemmer.stem(token) for token in tokens]
        elif self.use_lemmatization:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    
    def preprocess(self, text: str) -> List[str]:
        """
        Full preprocessing pipeline: clean, normalize, tokenize, filter, and stem.
        
        Args:
            text: Raw input text
            
        Returns:
            List of preprocessed tokens
        """
        # Step 1: Clean the text
        text = self.clean_text(text)
        
        # Step 2: Normalize
        text = self.normalize(text)
        
        # Step 3: Tokenize
        tokens = self.tokenize(text)
        
        # Step 4: Filter
        tokens = self.filter_tokens(tokens)
        
        # Step 5: Stem/Lemmatize
        tokens = self.stem_tokens(tokens)
        
        return tokens
    
    def preprocess_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess multiple texts.
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of preprocessed token lists
        """
        return [self.preprocess(text) for text in texts]
    
    def get_vocabulary(self, texts: List[str]) -> Set[str]:
        """
        Extract vocabulary from a collection of texts.
        
        Args:
            texts: List of raw texts
            
        Returns:
            Set of unique tokens
        """
        vocabulary = set()
        for text in texts:
            tokens = self.preprocess(text)
            vocabulary.update(tokens)
        return vocabulary
    
    def get_term_frequencies(self, text: str) -> Counter:
        """
        Get term frequencies for a single document.
        
        Args:
            text: Input text
            
        Returns:
            Counter with term frequencies
        """
        tokens = self.preprocess(text)
        return Counter(tokens)


class QueryPreprocessor(TextPreprocessor):
    """
    Specialized preprocessor for queries with query expansion capabilities.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query_cache = {}
    
    def preprocess_query(self, query: str, use_cache: bool = True) -> List[str]:
        """
        Preprocess a query with optional caching.
        
        Args:
            query: Raw query string
            use_cache: Whether to use cached results
            
        Returns:
            List of preprocessed query terms
        """
        if use_cache and query in self.query_cache:
            return self.query_cache[query]
        
        tokens = self.preprocess(query)
        
        if use_cache:
            self.query_cache[query] = tokens
        
        return tokens
    
    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear()


if __name__ == "__main__":
    # Demo of preprocessing
    preprocessor = TextPreprocessor()
    
    sample_text = """
    Information Retrieval (IR) is the science of searching for information in documents,
    searching for documents themselves, and also searching for metadata that describes data,
    and for databases of texts, images or sounds. Check https://example.com for more info!
    """
    
    print("Original Text:")
    print(sample_text)
    print("\nPreprocessed Tokens:")
    tokens = preprocessor.preprocess(sample_text)
    print(tokens)
    print(f"\nNumber of tokens: {len(tokens)}")
