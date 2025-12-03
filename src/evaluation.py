"""
Evaluation Module for Information Retrieval System
Implements standard IR evaluation metrics.
"""

import time
import math
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import json

from .retrieval import QueryResult, SearchResult


@dataclass
class EvaluationResult:
    """Stores evaluation results for a single query."""
    query_id: str
    query: str
    precision: float
    recall: float
    f1_score: float
    average_precision: float
    reciprocal_rank: float
    ndcg: float
    num_retrieved: int
    num_relevant: int
    num_relevant_retrieved: int


@dataclass
class SystemEvaluation:
    """Stores aggregate evaluation results for the system."""
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_average_precision: float  # MAP
    mean_reciprocal_rank: float    # MRR
    mean_ndcg: float
    total_queries: int
    avg_query_time_ms: float
    per_query_results: List[EvaluationResult]


class Evaluator:
    """
    Evaluator for Information Retrieval systems.
    
    Implements standard IR metrics:
    - Precision, Recall, F1
    - Mean Average Precision (MAP)
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (NDCG)
    """
    
    def __init__(self, relevance_judgments: Dict[str, Set[str]]):
        """
        Initialize evaluator with relevance judgments.
        
        Args:
            relevance_judgments: Dictionary mapping query_id -> set of relevant doc_ids
        """
        self.relevance_judgments = relevance_judgments
    
    def precision_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Compute Precision@K.
        
        Args:
            retrieved: Ordered list of retrieved doc_ids
            relevant: Set of relevant doc_ids
            k: Cutoff rank
            
        Returns:
            Precision@K value
        """
        if k == 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & relevant)
        
        return relevant_retrieved / k
    
    def recall_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Compute Recall@K.
        
        Args:
            retrieved: Ordered list of retrieved doc_ids
            relevant: Set of relevant doc_ids
            k: Cutoff rank
            
        Returns:
            Recall@K value
        """
        if len(relevant) == 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & relevant)
        
        return relevant_retrieved / len(relevant)
    
    def f1_score(self, precision: float, recall: float) -> float:
        """
        Compute F1 score.
        
        Args:
            precision: Precision value
            recall: Recall value
            
        Returns:
            F1 score
        """
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision(
        self,
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Compute Average Precision (AP).
        
        AP = (1/|R|) * sum(P@k * rel(k)) for k in 1..n
        
        Args:
            retrieved: Ordered list of retrieved doc_ids
            relevant: Set of relevant doc_ids
            
        Returns:
            Average Precision value
        """
        if len(relevant) == 0:
            return 0.0
        
        ap_sum = 0.0
        relevant_count = 0
        
        for k, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                relevant_count += 1
                precision_at_k = relevant_count / k
                ap_sum += precision_at_k
        
        return ap_sum / len(relevant)
    
    def reciprocal_rank(
        self,
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Compute Reciprocal Rank (RR).
        
        RR = 1 / rank of first relevant document
        
        Args:
            retrieved: Ordered list of retrieved doc_ids
            relevant: Set of relevant doc_ids
            
        Returns:
            Reciprocal Rank value
        """
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / rank
        
        return 0.0
    
    def dcg_at_k(
        self,
        retrieved: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """
        Compute Discounted Cumulative Gain at K.
        
        DCG@K = sum(rel_i / log2(i + 1)) for i in 1..k
        
        Args:
            retrieved: Ordered list of retrieved doc_ids
            relevance_scores: Dictionary mapping doc_id -> relevance score
            k: Cutoff rank
            
        Returns:
            DCG@K value
        """
        dcg = 0.0
        
        for i, doc_id in enumerate(retrieved[:k], 1):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / math.log2(i + 1)
        
        return dcg
    
    def ndcg_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Compute Normalized DCG at K.
        
        NDCG@K = DCG@K / IDCG@K
        
        Args:
            retrieved: Ordered list of retrieved doc_ids
            relevant: Set of relevant doc_ids
            k: Cutoff rank
            
        Returns:
            NDCG@K value
        """
        # Binary relevance: 1 for relevant, 0 for not relevant
        relevance_scores = {doc_id: 1.0 for doc_id in relevant}
        
        dcg = self.dcg_at_k(retrieved, relevance_scores, k)
        
        # Ideal ranking: all relevant documents first
        ideal_retrieved = list(relevant)[:k]
        idcg = self.dcg_at_k(ideal_retrieved, relevance_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_query(
        self,
        query_id: str,
        query_result: QueryResult,
        k: Optional[int] = None
    ) -> EvaluationResult:
        """
        Evaluate a single query result.
        
        Args:
            query_id: Query identifier
            query_result: QueryResult object
            k: Cutoff rank (defaults to all retrieved)
            
        Returns:
            EvaluationResult object
        """
        relevant = self.relevance_judgments.get(query_id, set())
        retrieved = [r.doc_id for r in query_result.results]
        
        if k is None:
            k = len(retrieved)
        
        # Compute metrics
        precision = self.precision_at_k(retrieved, relevant, k)
        recall = self.recall_at_k(retrieved, relevant, k)
        f1 = self.f1_score(precision, recall)
        ap = self.average_precision(retrieved, relevant)
        rr = self.reciprocal_rank(retrieved, relevant)
        ndcg = self.ndcg_at_k(retrieved, relevant, k)
        
        # Count relevant retrieved
        retrieved_set = set(retrieved[:k])
        relevant_retrieved = len(retrieved_set & relevant)
        
        return EvaluationResult(
            query_id=query_id,
            query=query_result.query,
            precision=precision,
            recall=recall,
            f1_score=f1,
            average_precision=ap,
            reciprocal_rank=rr,
            ndcg=ndcg,
            num_retrieved=len(retrieved[:k]),
            num_relevant=len(relevant),
            num_relevant_retrieved=relevant_retrieved
        )
    
    def evaluate_system(
        self,
        query_results: Dict[str, QueryResult],
        k: Optional[int] = None
    ) -> SystemEvaluation:
        """
        Evaluate the entire system across all queries.
        
        Args:
            query_results: Dictionary mapping query_id -> QueryResult
            k: Cutoff rank for evaluation
            
        Returns:
            SystemEvaluation object
        """
        per_query_results = []
        total_query_time = 0.0
        
        for query_id, query_result in query_results.items():
            eval_result = self.evaluate_query(query_id, query_result, k)
            per_query_results.append(eval_result)
            total_query_time += query_result.query_time
        
        n = len(per_query_results)
        if n == 0:
            return SystemEvaluation(
                mean_precision=0.0,
                mean_recall=0.0,
                mean_f1=0.0,
                mean_average_precision=0.0,
                mean_reciprocal_rank=0.0,
                mean_ndcg=0.0,
                total_queries=0,
                avg_query_time_ms=0.0,
                per_query_results=[]
            )
        
        return SystemEvaluation(
            mean_precision=sum(r.precision for r in per_query_results) / n,
            mean_recall=sum(r.recall for r in per_query_results) / n,
            mean_f1=sum(r.f1_score for r in per_query_results) / n,
            mean_average_precision=sum(r.average_precision for r in per_query_results) / n,
            mean_reciprocal_rank=sum(r.reciprocal_rank for r in per_query_results) / n,
            mean_ndcg=sum(r.ndcg for r in per_query_results) / n,
            total_queries=n,
            avg_query_time_ms=total_query_time / n,
            per_query_results=per_query_results
        )


class EfficiencyEvaluator:
    """
    Evaluates system efficiency metrics.
    """
    
    def __init__(self, index):
        """
        Initialize efficiency evaluator.
        
        Args:
            index: The inverted index
        """
        self.index = index
    
    def measure_index_size(self) -> Dict[str, Any]:
        """
        Measure the memory footprint of the index.
        
        Returns:
            Dictionary with size metrics
        """
        import sys
        
        # Estimate sizes
        index_size = sys.getsizeof(self.index.index)
        vocab_size = sys.getsizeof(self.index.vocabulary)
        docs_size = sum(
            sys.getsizeof(doc.content) + sys.getsizeof(doc.tokens)
            for doc in self.index.documents.values()
        )
        
        total_size = index_size + vocab_size + docs_size
        
        return {
            "index_size_bytes": index_size,
            "vocabulary_size_bytes": vocab_size,
            "documents_size_bytes": docs_size,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "num_documents": self.index.total_documents,
            "vocabulary_terms": len(self.index.vocabulary)
        }
    
    def measure_query_latency(
        self,
        retrieval_engine,
        queries: List[str],
        num_runs: int = 3
    ) -> Dict[str, float]:
        """
        Measure query latency statistics.
        
        Args:
            retrieval_engine: The retrieval engine to test
            queries: List of test queries
            num_runs: Number of runs per query for averaging
            
        Returns:
            Dictionary with latency statistics
        """
        latencies = []
        
        for query in queries:
            query_latencies = []
            for _ in range(num_runs):
                start = time.time()
                retrieval_engine.search(query)
                end = time.time()
                query_latencies.append((end - start) * 1000)  # ms
            
            latencies.append(sum(query_latencies) / num_runs)
        
        return {
            "mean_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "median_latency_ms": sorted(latencies)[len(latencies) // 2],
            "queries_per_second": 1000 / (sum(latencies) / len(latencies))
        }
    
    def measure_indexing_speed(
        self,
        documents: List[Tuple[str, str, str]]
    ) -> Dict[str, float]:
        """
        Measure indexing speed.
        
        Args:
            documents: List of (doc_id, title, content) tuples
            
        Returns:
            Dictionary with indexing speed metrics
        """
        from .indexing import InvertedIndex
        
        test_index = InvertedIndex()
        
        start = time.time()
        for doc_id, title, content in documents:
            test_index.add_document(doc_id, title, content)
        end = time.time()
        
        total_time = end - start
        
        return {
            "total_indexing_time_s": total_time,
            "documents_indexed": len(documents),
            "documents_per_second": len(documents) / total_time if total_time > 0 else 0,
            "avg_time_per_doc_ms": (total_time * 1000) / len(documents) if documents else 0
        }


def print_evaluation_report(system_eval: SystemEvaluation) -> str:
    """
    Generate a formatted evaluation report.
    
    Args:
        system_eval: SystemEvaluation object
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("INFORMATION RETRIEVAL SYSTEM EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    report.append("AGGREGATE METRICS")
    report.append("-" * 40)
    report.append(f"Total Queries Evaluated: {system_eval.total_queries}")
    report.append(f"Mean Precision:          {system_eval.mean_precision:.4f}")
    report.append(f"Mean Recall:             {system_eval.mean_recall:.4f}")
    report.append(f"Mean F1 Score:           {system_eval.mean_f1:.4f}")
    report.append(f"Mean Average Precision:  {system_eval.mean_average_precision:.4f}")
    report.append(f"Mean Reciprocal Rank:    {system_eval.mean_reciprocal_rank:.4f}")
    report.append(f"Mean NDCG:               {system_eval.mean_ndcg:.4f}")
    report.append(f"Avg Query Time:          {system_eval.avg_query_time_ms:.2f} ms")
    report.append("")
    report.append("PER-QUERY RESULTS")
    report.append("-" * 40)
    
    for result in system_eval.per_query_results:
        report.append(f"\nQuery: {result.query}")
        report.append(f"  Precision: {result.precision:.4f}")
        report.append(f"  Recall:    {result.recall:.4f}")
        report.append(f"  F1:        {result.f1_score:.4f}")
        report.append(f"  AP:        {result.average_precision:.4f}")
        report.append(f"  RR:        {result.reciprocal_rank:.4f}")
        report.append(f"  NDCG:      {result.ndcg:.4f}")
        report.append(f"  Retrieved: {result.num_retrieved}, Relevant: {result.num_relevant}, "
                     f"Rel-Ret: {result.num_relevant_retrieved}")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Demo evaluation
    from .indexing import InvertedIndex
    from .retrieval import RetrievalEngine, ScoringMethod
    
    # Create index
    index = InvertedIndex()
    docs = [
        ("doc1", "Information Retrieval", "Information retrieval is the science of searching."),
        ("doc2", "Text Mining", "Text mining extracts information from text."),
        ("doc3", "Search Engines", "Search engines retrieve documents using IR techniques."),
        ("doc4", "Machine Learning", "Machine learning can improve retrieval systems."),
        ("doc5", "Data Mining", "Data mining discovers patterns in large datasets.")
    ]
    
    for doc_id, title, content in docs:
        index.add_document(doc_id, title, content)
    
    # Create retrieval engine
    engine = RetrievalEngine(index, ScoringMethod.BM25)
    
    # Define relevance judgments
    relevance_judgments = {
        "q1": {"doc1", "doc3"},
        "q2": {"doc2", "doc5"}
    }
    
    # Run queries
    queries = {
        "q1": engine.search("information retrieval search"),
        "q2": engine.search("text data mining")
    }
    
    # Evaluate
    evaluator = Evaluator(relevance_judgments)
    system_eval = evaluator.evaluate_system(queries)
    
    print(print_evaluation_report(system_eval))
