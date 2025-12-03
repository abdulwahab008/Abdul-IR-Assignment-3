"""
Data Loading Module for Information Retrieval System
Handles loading documents from various file formats.
"""

import os
import json
import csv
from typing import List, Tuple, Dict, Optional, Generator
from pathlib import Path
import glob


class DocumentLoader:
    """
    Loads documents from various file formats.
    
    Supported formats:
    - Plain text files (.txt)
    - JSON files (.json)
    - CSV files (.csv)
    - Directory of text files
    - Kaggle News Articles dataset (Articles.csv)
    """
    
    def __init__(self, data_path: str, dataset_type: str = "auto"):
        """
        Initialize document loader.
        
        Args:
            data_path: Path to data file or directory
            dataset_type: Type of dataset - "auto", "kaggle_news", "generic"
        """
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type
    
    def load(self) -> List[Tuple[str, str, str]]:
        """
        Load documents based on path type and file format.
        
        Returns:
            List of (doc_id, title, content) tuples
        """
        if self.data_path.is_dir():
            return self._load_directory()
        elif self.data_path.suffix == '.json':
            return self._load_json()
        elif self.data_path.suffix == '.csv':
            # Check if it's the Kaggle News Articles dataset
            if self.dataset_type == "kaggle_news" or self._is_kaggle_news_dataset():
                return self._load_kaggle_news()
            return self._load_csv()
        elif self.data_path.suffix == '.txt':
            return self._load_single_text()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
    
    def _is_kaggle_news_dataset(self) -> bool:
        """Check if the CSV is the Kaggle News Articles dataset."""
        try:
            with open(self.data_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                # Check for Kaggle News Articles dataset columns
                return 'Article' in headers and 'Heading' in headers and 'NewsType' in headers
        except Exception:
            return False
    
    def _load_kaggle_news(self) -> List[Tuple[str, str, str]]:
        """
        Load the Kaggle News Articles dataset.
        
        Dataset columns: Article, Heading, Date, NewsType
        Source: https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles
        """
        documents = []
        
        with open(self.data_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            
            for idx, row in enumerate(reader):
                # Create unique document ID
                doc_id = f"news_{idx:05d}"
                
                # Get heading (title)
                heading = row.get('Heading', '').strip()
                if not heading:
                    heading = f"News Article {idx}"
                
                # Get article content
                article = row.get('Article', '').strip()
                
                # Get metadata
                date = row.get('Date', '').strip()
                news_type = row.get('NewsType', '').strip()
                
                # Combine content with metadata for richer context
                content_parts = []
                if article:
                    content_parts.append(article)
                if news_type:
                    content_parts.append(f"Category: {news_type}")
                if date:
                    content_parts.append(f"Date: {date}")
                
                content = " ".join(content_parts)
                
                if content.strip():
                    documents.append((doc_id, heading, content))
        
        print(f"Loaded {len(documents)} news articles from Kaggle dataset")
        return documents
    
    def _load_directory(self) -> List[Tuple[str, str, str]]:
        """Load all text files from a directory."""
        documents = []
        
        # Find all text files
        patterns = ['*.txt', '*.text']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(str(self.data_path / '**' / pattern), recursive=True))
        
        for file_path in sorted(files):
            file_path = Path(file_path)
            doc_id = file_path.stem
            title = file_path.stem.replace('_', ' ').replace('-', ' ').title()
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            
            if content:
                documents.append((doc_id, title, content))
        
        return documents
    
    def _load_json(self) -> List[Tuple[str, str, str]]:
        """
        Load documents from JSON file.
        
        Expected format:
        [
            {"id": "doc1", "title": "Title", "content": "..."},
            ...
        ]
        or
        {"doc1": {"title": "Title", "content": "..."}, ...}
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        if isinstance(data, list):
            for item in data:
                doc_id = str(item.get('id', item.get('doc_id', len(documents))))
                title = item.get('title', doc_id)
                content = item.get('content', item.get('text', ''))
                if content:
                    documents.append((doc_id, title, content))
        elif isinstance(data, dict):
            for doc_id, item in data.items():
                if isinstance(item, dict):
                    title = item.get('title', doc_id)
                    content = item.get('content', item.get('text', ''))
                else:
                    title = doc_id
                    content = str(item)
                if content:
                    documents.append((doc_id, title, content))
        
        return documents
    
    def _load_csv(self) -> List[Tuple[str, str, str]]:
        """
        Load documents from CSV file.
        
        Expected columns: id, title, content (or text)
        """
        documents = []
        
        with open(self.data_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Try different column name conventions
                doc_id = row.get('id', row.get('doc_id', row.get('ID', str(len(documents)))))
                title = row.get('title', row.get('Title', doc_id))
                content = row.get('content', row.get('text', row.get('Content', row.get('Text', ''))))
                
                if content:
                    documents.append((doc_id, title, content))
        
        return documents
    
    def _load_single_text(self) -> List[Tuple[str, str, str]]:
        """Load a single text file as one document."""
        doc_id = self.data_path.stem
        title = self.data_path.stem.replace('_', ' ').replace('-', ' ').title()
        
        with open(self.data_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
        
        return [(doc_id, title, content)] if content else []
    
    def load_streaming(self, batch_size: int = 100) -> Generator[List[Tuple[str, str, str]], None, None]:
        """
        Load documents in batches for memory efficiency.
        
        Args:
            batch_size: Number of documents per batch
            
        Yields:
            Batches of (doc_id, title, content) tuples
        """
        documents = self.load()
        
        for i in range(0, len(documents), batch_size):
            yield documents[i:i + batch_size]


class RelevanceJudgmentLoader:
    """
    Loads relevance judgments for evaluation.
    
    Supported formats:
    - TREC qrels format
    - JSON format
    - CSV format
    """
    
    def __init__(self, filepath: str):
        """
        Initialize loader.
        
        Args:
            filepath: Path to relevance judgments file
        """
        self.filepath = Path(filepath)
    
    def load(self) -> Dict[str, set]:
        """
        Load relevance judgments.
        
        Returns:
            Dictionary mapping query_id -> set of relevant doc_ids
        """
        if self.filepath.suffix == '.json':
            return self._load_json()
        elif self.filepath.suffix == '.csv':
            return self._load_csv()
        else:
            return self._load_trec()
    
    def _load_trec(self) -> Dict[str, set]:
        """
        Load TREC-format qrels.
        
        Format: query_id iter doc_id relevance
        """
        judgments: Dict[str, set] = {}
        
        with open(self.filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    query_id = parts[0]
                    doc_id = parts[2]
                    relevance = int(parts[3])
                    
                    if query_id not in judgments:
                        judgments[query_id] = set()
                    
                    if relevance > 0:
                        judgments[query_id].add(doc_id)
        
        return judgments
    
    def _load_json(self) -> Dict[str, set]:
        """Load JSON-format relevance judgments."""
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        
        return {k: set(v) for k, v in data.items()}
    
    def _load_csv(self) -> Dict[str, set]:
        """Load CSV-format relevance judgments."""
        judgments: Dict[str, set] = {}
        
        with open(self.filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                query_id = row.get('query_id', row.get('qid', ''))
                doc_id = row.get('doc_id', row.get('docid', ''))
                relevance = int(row.get('relevance', row.get('rel', '1')))
                
                if query_id and doc_id and relevance > 0:
                    if query_id not in judgments:
                        judgments[query_id] = set()
                    judgments[query_id].add(doc_id)
        
        return judgments


def create_sample_dataset(output_dir: str, num_docs: int = 20) -> None:
    """
    Create a sample dataset for testing.
    
    Args:
        output_dir: Directory to save sample data
        num_docs: Number of sample documents to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample documents about various topics
    sample_docs = [
        {
            "id": "doc001",
            "title": "Introduction to Information Retrieval",
            "content": "Information retrieval (IR) is the science of searching for information in documents, searching for documents themselves, and also searching for metadata that describes data. Information retrieval systems are used to reduce information overload and help users find the information they need. Modern IR systems use advanced algorithms like TF-IDF and BM25 for ranking documents."
        },
        {
            "id": "doc002",
            "title": "Text Preprocessing Techniques",
            "content": "Text preprocessing is a crucial step in natural language processing and information retrieval. Common techniques include tokenization, which splits text into words or tokens. Stemming reduces words to their root form, while lemmatization converts words to their dictionary form. Stop word removal eliminates common words that carry little meaning."
        },
        {
            "id": "doc003",
            "title": "TF-IDF Weighting Scheme",
            "content": "Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic used to reflect the importance of a word in a document within a collection. The TF-IDF value increases proportionally with the number of times a word appears in a document but is offset by the frequency of the word in the corpus. This helps to adjust for the fact that some words appear more frequently in general."
        },
        {
            "id": "doc004",
            "title": "BM25 Ranking Algorithm",
            "content": "BM25 (Best Matching 25) is a probabilistic retrieval model used by search engines to rank matching documents according to their relevance to a given query. BM25 is a bag-of-words retrieval function that ranks documents based on query terms appearing in each document. It incorporates term frequency saturation and document length normalization."
        },
        {
            "id": "doc005",
            "title": "Boolean Retrieval Model",
            "content": "The Boolean retrieval model is an information retrieval model in which queries are specified using Boolean operators AND, OR, and NOT. Documents are retrieved if they satisfy the Boolean expression. This model is precise but does not support ranking. Boolean retrieval was one of the first information retrieval models developed."
        },
        {
            "id": "doc006",
            "title": "Inverted Index Data Structure",
            "content": "An inverted index is a database index storing a mapping from content to its locations in a table or document. The inverted index is the most popular data structure used in information retrieval systems. It contains a dictionary of terms and for each term, a postings list that records which documents the term occurs in."
        },
        {
            "id": "doc007",
            "title": "Query Processing in Search Engines",
            "content": "Query processing involves analyzing and transforming a user query into a form that can be matched against the index. Steps include parsing the query, applying the same preprocessing as documents, expanding queries with synonyms, and formulating the retrieval model query. Efficient query processing is crucial for search engine performance."
        },
        {
            "id": "doc008",
            "title": "Evaluation Metrics for IR Systems",
            "content": "Information retrieval systems are evaluated using various metrics. Precision measures the fraction of retrieved documents that are relevant. Recall measures the fraction of relevant documents that are retrieved. F1 score is the harmonic mean of precision and recall. Mean Average Precision (MAP) and Normalized Discounted Cumulative Gain (NDCG) are popular ranking metrics."
        },
        {
            "id": "doc009",
            "title": "Vector Space Model",
            "content": "The vector space model represents documents and queries as vectors in a high-dimensional space. Each dimension corresponds to a term in the vocabulary. Document similarity is computed using cosine similarity. This model allows for ranking documents by their degree of similarity to the query rather than just Boolean matching."
        },
        {
            "id": "doc010",
            "title": "Search Engine Architecture",
            "content": "A search engine architecture consists of several components: the crawler that fetches web pages, the indexer that builds the inverted index, the query processor that handles user queries, and the ranker that orders results by relevance. Additional components include a cache for frequently accessed data and a front-end for user interaction."
        },
        {
            "id": "doc011",
            "title": "Machine Learning in Information Retrieval",
            "content": "Machine learning has transformed information retrieval. Learning to rank algorithms use machine learning to improve search ranking. Neural networks and deep learning models like BERT can understand semantic meaning. These models can capture complex patterns that traditional IR models cannot, leading to more relevant search results."
        },
        {
            "id": "doc012",
            "title": "Natural Language Processing Basics",
            "content": "Natural language processing (NLP) is a field of artificial intelligence that deals with the interaction between computers and humans using natural language. NLP techniques are used in information retrieval for query understanding, document analysis, and semantic matching. Common NLP tasks include named entity recognition, part-of-speech tagging, and sentiment analysis."
        },
        {
            "id": "doc013",
            "title": "Document Clustering and Classification",
            "content": "Document clustering groups similar documents together without predefined categories. Document classification assigns documents to predefined categories. Both techniques are useful for organizing large document collections. Common algorithms include k-means clustering and naive Bayes classification. These techniques help users navigate large document collections."
        },
        {
            "id": "doc014",
            "title": "Relevance Feedback Mechanisms",
            "content": "Relevance feedback is a technique used to improve search results based on user feedback. Users mark documents as relevant or non-relevant, and the system uses this information to refine the query. Pseudo-relevance feedback assumes top-ranked documents are relevant and uses them automatically. Relevance feedback can significantly improve retrieval effectiveness."
        },
        {
            "id": "doc015",
            "title": "Web Crawling and Indexing",
            "content": "Web crawlers are programs that systematically browse the World Wide Web to collect web pages. Crawlers start with a set of seed URLs and follow links to discover new pages. The fetched pages are then processed and added to the search engine index. Crawlers must be polite and respect robots.txt files that specify crawling rules."
        },
        {
            "id": "doc016",
            "title": "PageRank Algorithm",
            "content": "PageRank is an algorithm used by Google Search to rank web pages in search engine results. It works by counting the number and quality of links to a page to determine its importance. The underlying assumption is that more important websites receive more links. PageRank revolutionized web search by considering link structure."
        },
        {
            "id": "doc017",
            "title": "Text Similarity Measures",
            "content": "Text similarity measures quantify how similar two pieces of text are. Cosine similarity measures the angle between document vectors. Jaccard similarity measures the overlap of term sets. Edit distance measures the minimum number of edits to transform one string to another. These measures are fundamental to information retrieval and text mining."
        },
        {
            "id": "doc018",
            "title": "Semantic Search Technology",
            "content": "Semantic search improves search accuracy by understanding the meaning behind queries and documents. Unlike keyword-based search, semantic search considers context, synonyms, and user intent. Technologies like word embeddings and knowledge graphs enable semantic understanding. Semantic search delivers more relevant results for complex queries."
        },
        {
            "id": "doc019",
            "title": "Cross-Language Information Retrieval",
            "content": "Cross-language information retrieval (CLIR) enables users to search documents in languages different from the query language. CLIR systems use translation techniques to bridge the language gap. Approaches include query translation, document translation, and using multilingual representations. CLIR is essential for accessing global information resources."
        },
        {
            "id": "doc020",
            "title": "Question Answering Systems",
            "content": "Question answering systems return direct answers to natural language questions instead of document lists. These systems combine information retrieval with natural language understanding. Modern QA systems use deep learning models to extract answers from text passages. QA systems are increasingly used in virtual assistants and search engines."
        }
    ]
    
    # Save as JSON
    json_path = os.path.join(output_dir, 'documents.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sample_docs, f, indent=2)
    
    # Also save as individual text files
    text_dir = os.path.join(output_dir, 'text_files')
    os.makedirs(text_dir, exist_ok=True)
    
    for doc in sample_docs:
        filepath = os.path.join(text_dir, f"{doc['id']}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{doc['title']}\n\n{doc['content']}")
    
    # Create sample queries and relevance judgments
    sample_queries = {
        "q1": "information retrieval systems",
        "q2": "text preprocessing and tokenization",
        "q3": "TF-IDF and BM25 ranking",
        "q4": "machine learning search engines",
        "q5": "web crawling indexing"
    }
    
    queries_path = os.path.join(output_dir, 'queries.json')
    with open(queries_path, 'w', encoding='utf-8') as f:
        json.dump(sample_queries, f, indent=2)
    
    # Sample relevance judgments
    relevance_judgments = {
        "q1": ["doc001", "doc006", "doc008", "doc010"],
        "q2": ["doc002", "doc012"],
        "q3": ["doc003", "doc004", "doc009"],
        "q4": ["doc011", "doc013"],
        "q5": ["doc015", "doc016"]
    }
    
    qrels_path = os.path.join(output_dir, 'qrels.json')
    with open(qrels_path, 'w', encoding='utf-8') as f:
        json.dump(relevance_judgments, f, indent=2)
    
    print(f"Sample dataset created in {output_dir}")
    print(f"  - {len(sample_docs)} documents saved")
    print(f"  - {len(sample_queries)} sample queries")
    print(f"  - Relevance judgments for evaluation")


if __name__ == "__main__":
    # Create sample dataset
    create_sample_dataset("../data/sample")
