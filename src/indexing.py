"""
Indexing Module for Information Retrieval System
Implements inverted index with TF-IDF and BM25 scoring.
"""

import math
import json
import pickle
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import os

from .preprocessing import TextPreprocessor


@dataclass
class Document:
    """Represents a document in the collection."""
    doc_id: str
    title: str
    content: str
    tokens: List[str] = field(default_factory=list)
    term_frequencies: Counter = field(default_factory=Counter)
    doc_length: int = 0
    
    def __post_init__(self):
        if not self.tokens:
            preprocessor = TextPreprocessor()
            self.tokens = preprocessor.preprocess(self.content)
        self.term_frequencies = Counter(self.tokens)
        self.doc_length = len(self.tokens)


@dataclass
class PostingEntry:
    """An entry in a posting list."""
    doc_id: str
    term_frequency: int
    positions: List[int] = field(default_factory=list)


class InvertedIndex:
    """
    Inverted Index implementation with support for TF-IDF and BM25 scoring.
    
    Features:
    - Efficient term-to-document mapping
    - Document frequency tracking
    - Positional indexing for phrase queries
    - Persistence (save/load functionality)
    """
    
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize the inverted index.
        
        Args:
            preprocessor: TextPreprocessor instance for text processing
        """
        self.preprocessor = preprocessor or TextPreprocessor()
        
        # Core index structures
        self.index: Dict[str, List[PostingEntry]] = defaultdict(list)
        self.documents: Dict[str, Document] = {}
        self.document_lengths: Dict[str, int] = {}
        
        # Statistics
        self.total_documents = 0
        self.total_terms = 0
        self.avg_doc_length = 0.0
        self.vocabulary: Set[str] = set()
        
        # Document frequency for each term
        self.doc_frequencies: Dict[str, int] = defaultdict(int)
        
        # Collection frequency for each term
        self.collection_frequencies: Dict[str, int] = defaultdict(int)
    
    def add_document(self, doc_id: str, title: str, content: str) -> None:
        """
        Add a document to the index.
        
        Args:
            doc_id: Unique document identifier
            title: Document title
            content: Document content
        """
        # Preprocess the content
        tokens = self.preprocessor.preprocess(content)
        
        # Create document object
        doc = Document(
            doc_id=doc_id,
            title=title,
            content=content,
            tokens=tokens
        )
        
        # Store document
        self.documents[doc_id] = doc
        self.document_lengths[doc_id] = doc.doc_length
        self.total_documents += 1
        self.total_terms += doc.doc_length
        
        # Track term positions
        term_positions: Dict[str, List[int]] = defaultdict(list)
        for position, token in enumerate(tokens):
            term_positions[token].append(position)
        
        # Update index
        for term, positions in term_positions.items():
            # Create posting entry
            entry = PostingEntry(
                doc_id=doc_id,
                term_frequency=len(positions),
                positions=positions
            )
            self.index[term].append(entry)
            
            # Update vocabulary and frequencies
            self.vocabulary.add(term)
            self.doc_frequencies[term] += 1
            self.collection_frequencies[term] += len(positions)
        
        # Update average document length
        self._update_avg_doc_length()
    
    def _update_avg_doc_length(self) -> None:
        """Update the average document length."""
        if self.total_documents > 0:
            self.avg_doc_length = self.total_terms / self.total_documents
    
    def get_posting_list(self, term: str) -> List[PostingEntry]:
        """
        Get the posting list for a term.
        
        Args:
            term: The term to look up
            
        Returns:
            List of PostingEntry objects
        """
        # Preprocess the term
        processed_terms = self.preprocessor.preprocess(term)
        if not processed_terms:
            return []
        
        processed_term = processed_terms[0]
        return self.index.get(processed_term, [])
    
    def get_document_frequency(self, term: str) -> int:
        """
        Get the document frequency for a term.
        
        Args:
            term: The term to look up
            
        Returns:
            Number of documents containing the term
        """
        processed_terms = self.preprocessor.preprocess(term)
        if not processed_terms:
            return 0
        
        processed_term = processed_terms[0]
        return self.doc_frequencies.get(processed_term, 0)
    
    def get_term_frequency(self, term: str, doc_id: str) -> int:
        """
        Get the term frequency in a specific document.
        
        Args:
            term: The term to look up
            doc_id: The document identifier
            
        Returns:
            Frequency of the term in the document
        """
        processed_terms = self.preprocessor.preprocess(term)
        if not processed_terms:
            return 0
        
        processed_term = processed_terms[0]
        posting_list = self.index.get(processed_term, [])
        
        for entry in posting_list:
            if entry.doc_id == doc_id:
                return entry.term_frequency
        return 0
    
    def compute_idf(self, term: str) -> float:
        """
        Compute the Inverse Document Frequency for a term.
        
        IDF = log(N / df) where N is total docs and df is document frequency
        
        Args:
            term: The term to compute IDF for
            
        Returns:
            IDF value
        """
        processed_terms = self.preprocessor.preprocess(term)
        if not processed_terms:
            return 0.0
        
        processed_term = processed_terms[0]
        df = self.doc_frequencies.get(processed_term, 0)
        
        if df == 0:
            return 0.0
        
        return math.log(self.total_documents / df)
    
    def compute_tf_idf(self, term: str, doc_id: str) -> float:
        """
        Compute TF-IDF score for a term in a document.
        
        TF-IDF = (1 + log(tf)) * log(N/df)
        
        Args:
            term: The term
            doc_id: The document identifier
            
        Returns:
            TF-IDF score
        """
        tf = self.get_term_frequency(term, doc_id)
        if tf == 0:
            return 0.0
        
        idf = self.compute_idf(term)
        log_tf = 1 + math.log(tf)
        
        return log_tf * idf
    
    def compute_bm25(
        self,
        term: str,
        doc_id: str,
        k1: float = 1.5,
        b: float = 0.75
    ) -> float:
        """
        Compute BM25 score for a term in a document.
        
        BM25 = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl/avgdl)))
        
        Args:
            term: The term
            doc_id: The document identifier
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            
        Returns:
            BM25 score
        """
        tf = self.get_term_frequency(term, doc_id)
        if tf == 0:
            return 0.0
        
        # Get document length
        dl = self.document_lengths.get(doc_id, 0)
        if dl == 0:
            return 0.0
        
        # Compute IDF using BM25 formula: log((N - df + 0.5) / (df + 0.5))
        processed_terms = self.preprocessor.preprocess(term)
        if not processed_terms:
            return 0.0
        
        processed_term = processed_terms[0]
        df = self.doc_frequencies.get(processed_term, 0)
        
        if df == 0:
            return 0.0
        
        idf = math.log((self.total_documents - df + 0.5) / (df + 0.5) + 1)
        
        # Compute BM25
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (dl / self.avg_doc_length))
        
        return idf * (numerator / denominator)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_documents": self.total_documents,
            "total_terms": self.total_terms,
            "vocabulary_size": len(self.vocabulary),
            "avg_document_length": self.avg_doc_length,
            "index_size_terms": len(self.index)
        }
    
    def save(self, filepath: str) -> None:
        """
        Save the index to disk.
        
        Args:
            filepath: Path to save the index
        """
        data = {
            "index": {k: [(e.doc_id, e.term_frequency, e.positions) for e in v] 
                      for k, v in self.index.items()},
            "documents": {k: {"doc_id": v.doc_id, "title": v.title, "content": v.content,
                             "tokens": v.tokens, "doc_length": v.doc_length}
                         for k, v in self.documents.items()},
            "document_lengths": self.document_lengths,
            "total_documents": self.total_documents,
            "total_terms": self.total_terms,
            "avg_doc_length": self.avg_doc_length,
            "vocabulary": list(self.vocabulary),
            "doc_frequencies": dict(self.doc_frequencies),
            "collection_frequencies": dict(self.collection_frequencies)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str) -> None:
        """
        Load the index from disk.
        
        Args:
            filepath: Path to load the index from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Restore index
        self.index = defaultdict(list)
        for term, entries in data["index"].items():
            for doc_id, tf, positions in entries:
                self.index[term].append(PostingEntry(doc_id, tf, positions))
        
        # Restore documents
        self.documents = {}
        for doc_id, doc_data in data["documents"].items():
            doc = Document(
                doc_id=doc_data["doc_id"],
                title=doc_data["title"],
                content=doc_data["content"],
                tokens=doc_data["tokens"]
            )
            doc.doc_length = doc_data["doc_length"]
            self.documents[doc_id] = doc
        
        # Restore other fields
        self.document_lengths = data["document_lengths"]
        self.total_documents = data["total_documents"]
        self.total_terms = data["total_terms"]
        self.avg_doc_length = data["avg_doc_length"]
        self.vocabulary = set(data["vocabulary"])
        self.doc_frequencies = defaultdict(int, data["doc_frequencies"])
        self.collection_frequencies = defaultdict(int, data["collection_frequencies"])


class BooleanIndex:
    """
    Boolean retrieval index supporting AND, OR, NOT operations.
    """
    
    def __init__(self, inverted_index: InvertedIndex):
        """
        Initialize Boolean index.
        
        Args:
            inverted_index: The underlying inverted index
        """
        self.inverted_index = inverted_index
    
    def get_documents_containing(self, term: str) -> Set[str]:
        """Get set of document IDs containing a term."""
        posting_list = self.inverted_index.get_posting_list(term)
        return {entry.doc_id for entry in posting_list}
    
    def boolean_and(self, terms: List[str]) -> Set[str]:
        """
        Perform Boolean AND on multiple terms.
        
        Args:
            terms: List of terms
            
        Returns:
            Set of document IDs containing all terms
        """
        if not terms:
            return set()
        
        result = self.get_documents_containing(terms[0])
        for term in terms[1:]:
            result = result.intersection(self.get_documents_containing(term))
        
        return result
    
    def boolean_or(self, terms: List[str]) -> Set[str]:
        """
        Perform Boolean OR on multiple terms.
        
        Args:
            terms: List of terms
            
        Returns:
            Set of document IDs containing at least one term
        """
        result = set()
        for term in terms:
            result = result.union(self.get_documents_containing(term))
        return result
    
    def boolean_not(self, term: str) -> Set[str]:
        """
        Perform Boolean NOT on a term.
        
        Args:
            term: The term to negate
            
        Returns:
            Set of document IDs NOT containing the term
        """
        all_docs = set(self.inverted_index.documents.keys())
        docs_with_term = self.get_documents_containing(term)
        return all_docs - docs_with_term


if __name__ == "__main__":
    # Demo of indexing
    index = InvertedIndex()
    
    # Sample documents
    docs = [
        ("doc1", "Introduction to IR", "Information retrieval is the science of searching for information."),
        ("doc2", "Text Mining", "Text mining involves extracting useful information from text data."),
        ("doc3", "Search Engines", "Search engines use information retrieval techniques to find relevant documents.")
    ]
    
    for doc_id, title, content in docs:
        index.add_document(doc_id, title, content)
    
    print("Index Statistics:")
    print(json.dumps(index.get_statistics(), indent=2))
    
    print("\nPosting list for 'information':")
    for entry in index.get_posting_list("information"):
        print(f"  Doc: {entry.doc_id}, TF: {entry.term_frequency}")
    
    print("\nTF-IDF for 'information' in doc1:", index.compute_tf_idf("information", "doc1"))
    print("BM25 for 'information' in doc1:", index.compute_bm25("information", "doc1"))
