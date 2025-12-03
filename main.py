#!/usr/bin/env python3
"""
Command-Line Interface for Information Retrieval System
CS 516: Information Retrieval and Text Mining - Fall 2025
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import TextPreprocessor
from src.indexing import InvertedIndex
from src.retrieval import RetrievalEngine, HybridRetriever, ScoringMethod
from src.evaluation import Evaluator, EfficiencyEvaluator, print_evaluation_report
from src.data_loader import DocumentLoader, RelevanceJudgmentLoader, create_sample_dataset


class IRSystem:
    """
    Main Information Retrieval System class.
    Provides a unified interface for indexing, searching, and evaluation.
    """
    
    def __init__(self, index_path: str = None):
        """
        Initialize the IR system.
        
        Args:
            index_path: Path to load existing index from
        """
        self.preprocessor = TextPreprocessor(
            lowercase=True,
            remove_punctuation=True,
            remove_numbers=False,
            remove_stopwords=True,
            use_stemming=True,
            min_word_length=2
        )
        
        self.index = InvertedIndex(self.preprocessor)
        
        if index_path and os.path.exists(index_path):
            print(f"Loading index from {index_path}...")
            self.index.load(index_path)
            print(f"Loaded {self.index.total_documents} documents")
        
        self.retrieval_engine = None
        self.hybrid_retriever = None
    
    def build_index(self, data_path: str) -> None:
        """
        Build index from documents.
        
        Args:
            data_path: Path to documents (file or directory)
        """
        print(f"Loading documents from {data_path}...")
        loader = DocumentLoader(data_path)
        documents = loader.load()
        
        print(f"Indexing {len(documents)} documents...")
        start_time = time.time()
        
        for i, (doc_id, title, content) in enumerate(documents):
            self.index.add_document(doc_id, title, content)
            if (i + 1) % 100 == 0:
                print(f"  Indexed {i + 1}/{len(documents)} documents")
        
        elapsed = time.time() - start_time
        print(f"Indexing completed in {elapsed:.2f} seconds")
        
        # Initialize retrieval engines
        self._init_retrievers()
        
        # Print statistics
        stats = self.index.get_statistics()
        print("\nIndex Statistics:")
        print(f"  Documents: {stats['total_documents']}")
        print(f"  Vocabulary size: {stats['vocabulary_size']}")
        print(f"  Avg document length: {stats['avg_document_length']:.2f} tokens")
    
    def _init_retrievers(self):
        """Initialize retrieval engines."""
        self.retrieval_engine = RetrievalEngine(
            self.index,
            scoring_method=ScoringMethod.BM25,
            bm25_k1=1.5,
            bm25_b=0.75
        )
        self.hybrid_retriever = HybridRetriever(
            self.index,
            tfidf_weight=0.3,
            bm25_weight=0.7
        )
    
    def save_index(self, path: str) -> None:
        """Save index to disk."""
        print(f"Saving index to {path}...")
        self.index.save(path)
        print("Index saved successfully")
    
    def search(
        self,
        query: str,
        method: str = "bm25",
        top_k: int = 10
    ) -> dict:
        """
        Search for documents.
        
        Args:
            query: Search query
            method: Scoring method (bm25, tfidf, boolean, hybrid)
            top_k: Number of results to return
            
        Returns:
            Search results dictionary
        """
        if self.retrieval_engine is None:
            self._init_retrievers()
        
        if method == "hybrid":
            result = self.hybrid_retriever.search(query, top_k)
        elif method == "boolean":
            result = self.retrieval_engine.advanced_boolean_search(query)
        else:
            scoring = ScoringMethod.BM25 if method == "bm25" else ScoringMethod.TF_IDF
            result = self.retrieval_engine.search(query, top_k, scoring)
        
        return {
            "query": result.query,
            "method": result.scoring_method,
            "total_hits": result.total_hits,
            "query_time_ms": result.query_time,
            "results": [
                {
                    "rank": r.rank,
                    "doc_id": r.doc_id,
                    "title": r.title,
                    "score": r.score,
                    "snippet": r.snippet
                }
                for r in result.results
            ]
        }
    
    def evaluate(
        self,
        queries_path: str,
        qrels_path: str,
        method: str = "bm25",
        top_k: int = 10
    ) -> dict:
        """
        Evaluate the system.
        
        Args:
            queries_path: Path to queries file
            qrels_path: Path to relevance judgments file
            method: Scoring method
            top_k: Number of results per query
            
        Returns:
            Evaluation results dictionary
        """
        if self.retrieval_engine is None:
            self._init_retrievers()
        
        # Load queries
        with open(queries_path, 'r') as f:
            queries = json.load(f)
        
        # Load relevance judgments
        loader = RelevanceJudgmentLoader(qrels_path)
        relevance_judgments = loader.load()
        
        # Run queries
        query_results = {}
        for query_id, query_text in queries.items():
            if method == "hybrid":
                result = self.hybrid_retriever.search(query_text, top_k)
            elif method == "boolean":
                result = self.retrieval_engine.advanced_boolean_search(query_text)
            else:
                scoring = ScoringMethod.BM25 if method == "bm25" else ScoringMethod.TF_IDF
                result = self.retrieval_engine.search(query_text, top_k, scoring)
            query_results[query_id] = result
        
        # Evaluate
        evaluator = Evaluator(relevance_judgments)
        system_eval = evaluator.evaluate_system(query_results, top_k)
        
        # Print report
        print(print_evaluation_report(system_eval))
        
        return {
            "mean_precision": system_eval.mean_precision,
            "mean_recall": system_eval.mean_recall,
            "mean_f1": system_eval.mean_f1,
            "map": system_eval.mean_average_precision,
            "mrr": system_eval.mean_reciprocal_rank,
            "ndcg": system_eval.mean_ndcg,
            "avg_query_time_ms": system_eval.avg_query_time_ms
        }
    
    def interactive_search(self, method: str = "bm25"):
        """Run interactive search mode."""
        print("\n" + "=" * 60)
        print("INFORMATION RETRIEVAL SYSTEM - Interactive Mode")
        print("=" * 60)
        print(f"Scoring method: {method}")
        print("Type 'quit' or 'exit' to stop")
        print("Type 'help' for commands")
        print("=" * 60 + "\n")
        
        while True:
            try:
                query = input("Query > ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n")
                break
            
            if not query:
                continue
            
            if query.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            if query.lower() == 'help':
                print("\nCommands:")
                print("  quit/exit/q - Exit the system")
                print("  help        - Show this help")
                print("  stats       - Show index statistics")
                print("  method:X    - Change method (bm25, tfidf, boolean, hybrid)")
                print("\nFor Boolean queries, use AND, OR, NOT operators")
                print("Example: information AND retrieval NOT machine\n")
                continue
            
            if query.lower() == 'stats':
                stats = self.index.get_statistics()
                print(f"\nIndex Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
                continue
            
            if query.lower().startswith('method:'):
                method = query.split(':')[1].strip().lower()
                print(f"Switched to {method} scoring\n")
                continue
            
            # Perform search
            results = self.search(query, method=method, top_k=10)
            
            print(f"\nFound {results['total_hits']} results ({results['query_time_ms']:.2f}ms)")
            print("-" * 60)
            
            if not results['results']:
                print("No matching documents found.\n")
                continue
            
            for r in results['results']:
                print(f"{r['rank']}. [{r['doc_id']}] {r['title']}")
                print(f"   Score: {r['score']:.4f}")
                if r['snippet']:
                    print(f"   {r['snippet'][:100]}...")
                print()


def main():
    parser = argparse.ArgumentParser(
        description="Information Retrieval System - CS 516 Fall 2025",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample dataset
  python main.py --create-sample --data-dir ./data/sample

  # Build index from documents
  python main.py --build-index --data-path ./data/sample/documents.json

  # Interactive search
  python main.py --interactive --index-path ./index.pkl

  # Single query
  python main.py --query "information retrieval" --method bm25

  # Run evaluation
  python main.py --evaluate --queries ./data/sample/queries.json --qrels ./data/sample/qrels.json
        """
    )
    
    # Actions
    parser.add_argument('--create-sample', action='store_true',
                        help='Create sample dataset')
    parser.add_argument('--build-index', action='store_true',
                        help='Build index from documents')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive search mode')
    parser.add_argument('--query', type=str,
                        help='Run a single query')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the system')
    
    # Paths
    parser.add_argument('--data-dir', type=str, default='./data/sample',
                        help='Directory for sample data')
    parser.add_argument('--data-path', type=str,
                        help='Path to documents (file or directory)')
    parser.add_argument('--index-path', type=str, default='./index.pkl',
                        help='Path to save/load index')
    parser.add_argument('--queries', type=str,
                        help='Path to queries file for evaluation')
    parser.add_argument('--qrels', type=str,
                        help='Path to relevance judgments file')
    
    # Options
    parser.add_argument('--method', type=str, default='bm25',
                        choices=['bm25', 'tfidf', 'boolean', 'hybrid'],
                        help='Scoring method')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of results to return')
    parser.add_argument('--save-index', action='store_true',
                        help='Save index after building')
    
    args = parser.parse_args()
    
    # Create sample dataset
    if args.create_sample:
        create_sample_dataset(args.data_dir)
        return
    
    # Initialize system
    index_path = args.index_path if os.path.exists(args.index_path) else None
    ir_system = IRSystem(index_path)
    
    # Build index
    if args.build_index:
        if not args.data_path:
            parser.error("--build-index requires --data-path")
        ir_system.build_index(args.data_path)
        if args.save_index:
            ir_system.save_index(args.index_path)
    
    # Interactive mode
    if args.interactive:
        if ir_system.index.total_documents == 0:
            print("Error: No documents indexed. Build index first with --build-index")
            return
        ir_system.interactive_search(args.method)
    
    # Single query
    elif args.query:
        if ir_system.index.total_documents == 0:
            print("Error: No documents indexed. Build index first with --build-index")
            return
        
        results = ir_system.search(args.query, args.method, args.top_k)
        print(json.dumps(results, indent=2))
    
    # Evaluation
    elif args.evaluate:
        if not args.queries or not args.qrels:
            parser.error("--evaluate requires --queries and --qrels")
        
        if ir_system.index.total_documents == 0:
            print("Error: No documents indexed. Build index first with --build-index")
            return
        
        results = ir_system.evaluate(args.queries, args.qrels, args.method, args.top_k)
        print("\nEvaluation Summary:")
        print(json.dumps(results, indent=2))
    
    # No action specified
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
