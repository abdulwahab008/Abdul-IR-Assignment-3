"""
Retrieval Engine for Information Retrieval System
Implements query processing and document ranking.
"""

import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import heapq

from .preprocessing import TextPreprocessor, QueryPreprocessor
from .indexing import InvertedIndex, BooleanIndex


class ScoringMethod(Enum):
    """Supported scoring methods."""
    TF_IDF = "tf_idf"
    BM25 = "bm25"
    BOOLEAN = "boolean"


@dataclass
class SearchResult:
    """Represents a single search result."""
    doc_id: str
    title: str
    score: float
    snippet: str = ""
    rank: int = 0


@dataclass
class QueryResult:
    """Represents the complete result of a query."""
    query: str
    results: List[SearchResult]
    total_hits: int
    query_time: float  # in milliseconds
    scoring_method: str


class RetrievalEngine:
    """
    Main retrieval engine that processes queries and returns ranked results.
    
    Features:
    - Multiple scoring methods (TF-IDF, BM25, Boolean)
    - Query preprocessing
    - Result ranking
    - Snippet generation
    """
    
    def __init__(
        self,
        index: InvertedIndex,
        scoring_method: ScoringMethod = ScoringMethod.BM25,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75
    ):
        """
        Initialize the retrieval engine.
        
        Args:
            index: The inverted index
            scoring_method: Scoring method to use
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
        """
        self.index = index
        self.scoring_method = scoring_method
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        
        self.query_preprocessor = QueryPreprocessor()
        self.boolean_index = BooleanIndex(index)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        scoring_method: Optional[ScoringMethod] = None
    ) -> QueryResult:
        """
        Search for documents matching the query.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            scoring_method: Override default scoring method
            
        Returns:
            QueryResult object containing ranked results
        """
        start_time = time.time()
        
        method = scoring_method or self.scoring_method
        
        # Process the query
        query_terms = self.query_preprocessor.preprocess_query(query)
        
        if not query_terms:
            return QueryResult(
                query=query,
                results=[],
                total_hits=0,
                query_time=0.0,
                scoring_method=method.value
            )
        
        # Get scored documents based on method
        if method == ScoringMethod.BOOLEAN:
            scored_docs = self._boolean_search(query_terms)
        elif method == ScoringMethod.TF_IDF:
            scored_docs = self._tfidf_search(query_terms)
        else:  # BM25
            scored_docs = self._bm25_search(query_terms)
        
        # Sort by score and get top-k
        top_results = heapq.nlargest(top_k, scored_docs.items(), key=lambda x: x[1])
        
        # Build search results
        results = []
        for rank, (doc_id, score) in enumerate(top_results, 1):
            doc = self.index.documents[doc_id]
            snippet = self._generate_snippet(doc.content, query_terms)
            
            results.append(SearchResult(
                doc_id=doc_id,
                title=doc.title,
                score=score,
                snippet=snippet,
                rank=rank
            ))
        
        end_time = time.time()
        query_time = (end_time - start_time) * 1000  # Convert to ms
        
        return QueryResult(
            query=query,
            results=results,
            total_hits=len(scored_docs),
            query_time=query_time,
            scoring_method=method.value
        )
    
    def _boolean_search(self, query_terms: List[str]) -> Dict[str, float]:
        """
        Perform Boolean AND search.
        
        Args:
            query_terms: Preprocessed query terms
            
        Returns:
            Dictionary of doc_id -> score (1.0 for all matches)
        """
        matching_docs = self.boolean_index.boolean_and(query_terms)
        return {doc_id: 1.0 for doc_id in matching_docs}
    
    def _tfidf_search(self, query_terms: List[str]) -> Dict[str, float]:
        """
        Perform TF-IDF scored search.
        
        Args:
            query_terms: Preprocessed query terms
            
        Returns:
            Dictionary of doc_id -> TF-IDF score
        """
        doc_scores: Dict[str, float] = defaultdict(float)
        
        for term in query_terms:
            posting_list = self.index.get_posting_list(term)
            for entry in posting_list:
                score = self.index.compute_tf_idf(term, entry.doc_id)
                doc_scores[entry.doc_id] += score
        
        return dict(doc_scores)
    
    def _bm25_search(self, query_terms: List[str]) -> Dict[str, float]:
        """
        Perform BM25 scored search.
        
        Args:
            query_terms: Preprocessed query terms
            
        Returns:
            Dictionary of doc_id -> BM25 score
        """
        doc_scores: Dict[str, float] = defaultdict(float)
        
        for term in query_terms:
            posting_list = self.index.get_posting_list(term)
            for entry in posting_list:
                score = self.index.compute_bm25(
                    term, entry.doc_id, 
                    k1=self.bm25_k1, 
                    b=self.bm25_b
                )
                doc_scores[entry.doc_id] += score
        
        return dict(doc_scores)
    
    def _generate_snippet(
        self,
        content: str,
        query_terms: List[str],
        snippet_length: int = 200
    ) -> str:
        """
        Generate a snippet from document content highlighting query terms.
        
        Args:
            content: Document content
            query_terms: Query terms to highlight
            snippet_length: Maximum snippet length
            
        Returns:
            Snippet string
        """
        content_lower = content.lower()
        
        # Find the first occurrence of any query term
        best_pos = -1
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos
        
        if best_pos == -1:
            # No query term found, return start of document
            snippet = content[:snippet_length]
        else:
            # Center snippet around the query term
            start = max(0, best_pos - snippet_length // 2)
            end = min(len(content), start + snippet_length)
            snippet = content[start:end]
            
            # Add ellipsis if truncated
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
        
        return snippet.strip()
    
    def parse_boolean_query(self, query: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Parse a Boolean query into AND, OR, and NOT terms.
        
        Supports syntax: term1 AND term2 OR term3 NOT term4
        
        Args:
            query: Boolean query string
            
        Returns:
            Tuple of (AND terms, OR terms, NOT terms)
        """
        query_upper = query.upper()
        
        and_terms = []
        or_terms = []
        not_terms = []
        
        # Split by operators
        parts = query_upper.replace(" AND ", " & ").replace(" OR ", " | ").replace(" NOT ", " ! ").split()
        
        current_op = "AND"
        for part in parts:
            if part == "&":
                current_op = "AND"
            elif part == "|":
                current_op = "OR"
            elif part == "!":
                current_op = "NOT"
            else:
                term = part.lower()
                if current_op == "AND":
                    and_terms.append(term)
                elif current_op == "OR":
                    or_terms.append(term)
                else:
                    not_terms.append(term)
        
        return and_terms, or_terms, not_terms
    
    def advanced_boolean_search(self, query: str) -> QueryResult:
        """
        Perform advanced Boolean search with AND, OR, NOT operators.
        
        Args:
            query: Boolean query string
            
        Returns:
            QueryResult object
        """
        start_time = time.time()
        
        and_terms, or_terms, not_terms = self.parse_boolean_query(query)
        
        # Start with all documents if no AND terms, otherwise AND result
        if and_terms:
            result_docs = self.boolean_index.boolean_and(and_terms)
        else:
            result_docs = set(self.index.documents.keys())
        
        # Add OR results
        if or_terms:
            or_docs = self.boolean_index.boolean_or(or_terms)
            result_docs = result_docs.union(or_docs)
        
        # Remove NOT results
        for term in not_terms:
            not_docs = self.boolean_index.get_documents_containing(term)
            result_docs = result_docs - not_docs
        
        # Build results
        results = []
        for rank, doc_id in enumerate(result_docs, 1):
            doc = self.index.documents[doc_id]
            all_terms = and_terms + or_terms
            snippet = self._generate_snippet(doc.content, all_terms) if all_terms else ""
            
            results.append(SearchResult(
                doc_id=doc_id,
                title=doc.title,
                score=1.0,
                snippet=snippet,
                rank=rank
            ))
        
        end_time = time.time()
        
        return QueryResult(
            query=query,
            results=results,
            total_hits=len(results),
            query_time=(end_time - start_time) * 1000,
            scoring_method="boolean"
        )


class HybridRetriever:
    """
    Hybrid retrieval combining multiple scoring methods.
    """
    
    def __init__(
        self,
        index: InvertedIndex,
        tfidf_weight: float = 0.3,
        bm25_weight: float = 0.7
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            index: The inverted index
            tfidf_weight: Weight for TF-IDF scores
            bm25_weight: Weight for BM25 scores
        """
        self.index = index
        self.tfidf_weight = tfidf_weight
        self.bm25_weight = bm25_weight
        
        self.tfidf_engine = RetrievalEngine(index, ScoringMethod.TF_IDF)
        self.bm25_engine = RetrievalEngine(index, ScoringMethod.BM25)
    
    def search(self, query: str, top_k: int = 10) -> QueryResult:
        """
        Perform hybrid search combining TF-IDF and BM25.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            QueryResult object
        """
        start_time = time.time()
        
        # Get results from both methods
        tfidf_result = self.tfidf_engine.search(query, top_k=100)
        bm25_result = self.bm25_engine.search(query, top_k=100)
        
        # Normalize and combine scores
        tfidf_scores = self._normalize_scores(
            {r.doc_id: r.score for r in tfidf_result.results}
        )
        bm25_scores = self._normalize_scores(
            {r.doc_id: r.score for r in bm25_result.results}
        )
        
        # Combine scores
        combined_scores: Dict[str, float] = {}
        all_docs = set(tfidf_scores.keys()) | set(bm25_scores.keys())
        
        for doc_id in all_docs:
            tfidf_score = tfidf_scores.get(doc_id, 0.0)
            bm25_score = bm25_scores.get(doc_id, 0.0)
            combined_scores[doc_id] = (
                self.tfidf_weight * tfidf_score + 
                self.bm25_weight * bm25_score
            )
        
        # Get top-k
        top_results = heapq.nlargest(top_k, combined_scores.items(), key=lambda x: x[1])
        
        # Build results
        query_preprocessor = QueryPreprocessor()
        query_terms = query_preprocessor.preprocess_query(query)
        
        results = []
        for rank, (doc_id, score) in enumerate(top_results, 1):
            doc = self.index.documents[doc_id]
            snippet = self.tfidf_engine._generate_snippet(doc.content, query_terms)
            
            results.append(SearchResult(
                doc_id=doc_id,
                title=doc.title,
                score=score,
                snippet=snippet,
                rank=rank
            ))
        
        end_time = time.time()
        
        return QueryResult(
            query=query,
            results=results,
            total_hits=len(combined_scores),
            query_time=(end_time - start_time) * 1000,
            scoring_method=f"hybrid(tfidf={self.tfidf_weight}, bm25={self.bm25_weight})"
        )
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1] range using min-max normalization."""
        if not scores:
            return {}
        
        min_score = min(scores.values())
        max_score = max(scores.values())
        
        if max_score == min_score:
            return {doc_id: 1.0 for doc_id in scores}
        
        return {
            doc_id: (score - min_score) / (max_score - min_score)
            for doc_id, score in scores.items()
        }


if __name__ == "__main__":
    from .indexing import InvertedIndex
    
    # Create and populate index
    index = InvertedIndex()
    docs = [
        ("doc1", "Introduction to IR", "Information retrieval is the science of searching for information in documents."),
        ("doc2", "Text Mining Basics", "Text mining involves extracting useful information from text data using various techniques."),
        ("doc3", "Search Engines", "Search engines use information retrieval techniques to find relevant documents quickly."),
        ("doc4", "Machine Learning", "Machine learning algorithms can improve information retrieval effectiveness.")
    ]
    
    for doc_id, title, content in docs:
        index.add_document(doc_id, title, content)
    
    # Test retrieval engine
    engine = RetrievalEngine(index, ScoringMethod.BM25)
    
    print("=== BM25 Search ===")
    result = engine.search("information retrieval", top_k=3)
    print(f"Query: {result.query}")
    print(f"Total hits: {result.total_hits}")
    print(f"Query time: {result.query_time:.2f}ms")
    for r in result.results:
        print(f"  {r.rank}. [{r.doc_id}] {r.title} (score: {r.score:.4f})")
    
    print("\n=== Hybrid Search ===")
    hybrid = HybridRetriever(index)
    result = hybrid.search("information retrieval", top_k=3)
    for r in result.results:
        print(f"  {r.rank}. [{r.doc_id}] {r.title} (score: {r.score:.4f})")
