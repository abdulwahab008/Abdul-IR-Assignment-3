#!/usr/bin/env python3
"""
Helper script to create relevance judgments for evaluation.
Run this after building the index to interactively judge document relevance.

Usage:
    python create_qrels.py --index-path ./index.pkl --queries ./data/queries.json --output ./data/qrels.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.indexing import InvertedIndex
from src.retrieval import RetrievalEngine, ScoringMethod
from src.preprocessing import TextPreprocessor


def create_relevance_judgments(index_path: str, queries_path: str, output_path: str, top_k: int = 20):
    """
    Interactively create relevance judgments.
    
    Args:
        index_path: Path to the saved index
        queries_path: Path to queries JSON file
        output_path: Path to save relevance judgments
        top_k: Number of top documents to judge per query
    """
    # Load index
    print(f"Loading index from {index_path}...")
    preprocessor = TextPreprocessor()
    index = InvertedIndex(preprocessor)
    index.load(index_path)
    print(f"Loaded {index.total_documents} documents")
    
    # Create retrieval engine
    engine = RetrievalEngine(index, ScoringMethod.BM25)
    
    # Load queries
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    
    # Load existing judgments if any
    relevance_judgments = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            relevance_judgments = json.load(f)
    
    print("\n" + "=" * 70)
    print("RELEVANCE JUDGMENT CREATION")
    print("=" * 70)
    print("For each query, you'll see the top results.")
    print("Mark documents as relevant (y), not relevant (n), or skip (s).")
    print("Type 'q' to quit and save progress.")
    print("=" * 70 + "\n")
    
    for query_id, query_text in queries.items():
        if query_id in relevance_judgments and relevance_judgments[query_id]:
            print(f"\nSkipping {query_id} (already has judgments)")
            continue
        
        print(f"\n{'=' * 70}")
        print(f"Query {query_id}: {query_text}")
        print("=" * 70)
        
        # Run query
        result = engine.search(query_text, top_k=top_k)
        
        relevant_docs = []
        
        for r in result.results:
            print(f"\n[{r.rank}] Document: {r.doc_id}")
            print(f"    Title: {r.title}")
            print(f"    Score: {r.score:.4f}")
            print(f"    Snippet: {r.snippet[:200]}...")
            
            while True:
                response = input("\n    Relevant? (y/n/s=skip/q=quit): ").strip().lower()
                
                if response == 'q':
                    # Save and quit
                    relevance_judgments[query_id] = relevant_docs
                    with open(output_path, 'w') as f:
                        json.dump(relevance_judgments, f, indent=2)
                    print(f"\nSaved to {output_path}")
                    return
                elif response == 'y':
                    relevant_docs.append(r.doc_id)
                    print("    -> Marked as RELEVANT")
                    break
                elif response == 'n':
                    print("    -> Marked as NOT relevant")
                    break
                elif response == 's':
                    print("    -> Skipped")
                    break
                else:
                    print("    Invalid input. Use y/n/s/q")
        
        relevance_judgments[query_id] = relevant_docs
        print(f"\nQuery {query_id}: {len(relevant_docs)} relevant documents identified")
    
    # Save final judgments
    with open(output_path, 'w') as f:
        json.dump(relevance_judgments, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"COMPLETED! Relevance judgments saved to {output_path}")
    print("=" * 70)
    
    # Print summary
    print("\nSummary:")
    for qid, docs in relevance_judgments.items():
        print(f"  {qid}: {len(docs)} relevant documents")


def auto_generate_qrels(index_path: str, queries_path: str, output_path: str, threshold: float = 0.5):
    """
    Auto-generate relevance judgments based on score threshold.
    Documents with normalized scores above threshold are marked as relevant.
    
    This is a quick way to create pseudo-relevance judgments for testing.
    """
    print("Auto-generating pseudo-relevance judgments...")
    
    # Load index
    preprocessor = TextPreprocessor()
    index = InvertedIndex(preprocessor)
    index.load(index_path)
    
    engine = RetrievalEngine(index, ScoringMethod.BM25)
    
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    
    relevance_judgments = {}
    
    for query_id, query_text in queries.items():
        result = engine.search(query_text, top_k=50)
        
        if not result.results:
            relevance_judgments[query_id] = []
            continue
        
        # Normalize scores
        max_score = max(r.score for r in result.results)
        if max_score > 0:
            # Take top documents with score > threshold * max_score
            relevant = [r.doc_id for r in result.results 
                       if r.score >= threshold * max_score]
            # Limit to top 10
            relevance_judgments[query_id] = relevant[:10]
        else:
            relevance_judgments[query_id] = []
    
    with open(output_path, 'w') as f:
        json.dump(relevance_judgments, f, indent=2)
    
    print(f"Saved pseudo-relevance judgments to {output_path}")
    for qid, docs in relevance_judgments.items():
        print(f"  {qid}: {len(docs)} relevant documents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create relevance judgments for IR evaluation")
    parser.add_argument('--index-path', type=str, default='./index.pkl',
                        help='Path to the saved index')
    parser.add_argument('--queries', type=str, default='./data/queries.json',
                        help='Path to queries JSON file')
    parser.add_argument('--output', type=str, default='./data/qrels.json',
                        help='Path to save relevance judgments')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of documents to judge per query')
    parser.add_argument('--auto', action='store_true',
                        help='Auto-generate pseudo-relevance judgments')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Score threshold for auto-generation (0-1)')
    
    args = parser.parse_args()
    
    if args.auto:
        auto_generate_qrels(args.index_path, args.queries, args.output, args.threshold)
    else:
        create_relevance_judgments(args.index_path, args.queries, args.output, args.top_k)
