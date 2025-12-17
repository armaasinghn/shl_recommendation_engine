"""
Embedding-Based Retriever
==========================
Uses sentence transformers for semantic search with FAISS vector store.
"""

import json
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss

from .config import ASSESSMENTS_FILE
from .models import Assessment, RecommendationResult


class EmbeddingRetriever:
    """
    Semantic search using sentence transformers and FAISS.
    """
    
    def __init__(
        self, 
        assessments_file: Optional[str] = None,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize embedding retriever.
        
        Args:
            assessments_file: Path to assessments JSON
            model_name: Sentence transformer model
        """
        self.assessments_file = assessments_file or ASSESSMENTS_FILE
        self.model_name = model_name
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded (embedding dimension: {self.model.get_sentence_embedding_dimension()})")
        
        self.assessments: List[Assessment] = []
        self.index: Optional[faiss.Index] = None
        
        self._load_assessments()
        self._build_index()
    
    def _load_assessments(self):
        """Load assessments from JSON."""
        with open(self.assessments_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assessments = [Assessment.from_dict(item) for item in data]
        print(f"✅ Loaded {len(self.assessments)} assessments")
    
    def _build_index(self):
        """Build FAISS index from assessment embeddings."""
        print("\nBuilding vector index...")
        
        # Generate embeddings for all assessments
        texts = [self._get_assessment_text(a) for a in self.assessments]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity
        self.index.add(embeddings.astype('float32'))
        
        print(f"✅ Vector index built with {self.index.ntotal} vectors")
    
    def _get_assessment_text(self, assessment: Assessment) -> str:
        """Get searchable text for an assessment."""
        parts = [
            assessment.name,
            assessment.description,
            " ".join(assessment.keywords),
            " ".join(assessment.test_type_full)
        ]
        return " ".join(parts)
    
    def search(
        self, 
        query: str, 
        num_results: int = 10
    ) -> List[RecommendationResult]:
        """
        Search for relevant assessments using semantic similarity.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of RecommendationResult objects
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search FAISS index
        scores, indices = self.index.search(
            query_embedding.astype('float32'), 
            num_results
        )
        
        # Build results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.assessments):
                assessment = self.assessments[idx]
                result = RecommendationResult(
                    assessment=assessment,
                    score=float(score),
                    match_reasons=[f"Semantic similarity: {score:.3f}"]
                )
                results.append(result)
        
        return results


class HybridEmbeddingRetriever:
    """
    Combines embedding-based search with TF-IDF for robust retrieval.
    
    Strategy:
    - Embedding-based search: 70% weight (semantic understanding)
    - TF-IDF search: 30% weight (keyword matching)
    """
    
    def __init__(self, assessments_file: Optional[str] = None):
        from .search_engine import AssessmentSearchEngine
        
        print("\n" + "=" * 70)
        print("INITIALIZING HYBRID RETRIEVER")
        print("=" * 70)
        
        # Embedding retriever (semantic)
        print("\n[1/2] Initializing Embedding Retriever...")
        self.embedding_retriever = EmbeddingRetriever(assessments_file)
        
        # TF-IDF retriever (keyword)
        print("\n[2/2] Initializing TF-IDF Retriever...")
        self.tfidf_retriever = AssessmentSearchEngine()
        self.tfidf_retriever.load_assessments(assessments_file)
        self.tfidf_retriever.fit()
        
        print("\n✅ Hybrid retriever initialized")
        print("=" * 70)
    
    def search(
        self, 
        query: str, 
        num_results: int = 10,
        embedding_weight: float = 0.7
    ) -> List[RecommendationResult]:
        """
        Hybrid search combining embeddings and TF-IDF.
        
        Args:
            query: Search query
            num_results: Number of results
            embedding_weight: Weight for embedding scores (0-1)
            
        Returns:
            Blended and ranked results
        """
        tfidf_weight = 1.0 - embedding_weight
        
        # Get results from both retrievers
        embedding_results = self.embedding_retriever.search(query, num_results * 2)
        
        from .query_analyzer import analyze_query
        query_info = analyze_query(query)
        tfidf_results = self.tfidf_retriever.search(query_info, num_results * 2)
        
        # Blend scores
        scores = {}
        url_to_assessment = {}
        
        # Add embedding scores
        for result in embedding_results:
            url = result.assessment.url
            scores[url] = scores.get(url, 0) + (result.score * embedding_weight)
            url_to_assessment[url] = result.assessment
        
        # Add TF-IDF scores
        for result in tfidf_results:
            url = result.assessment.url
            scores[url] = scores.get(url, 0) + (result.score * tfidf_weight)
            url_to_assessment[url] = result.assessment
        
        # Sort by blended score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        final_results = []
        for url, score in sorted_items[:num_results]:
            result = RecommendationResult(
                assessment=url_to_assessment[url],
                score=score,
                match_reasons=[
                    f"Hybrid score: {score:.3f}",
                    f"(Embedding: {embedding_weight*100:.0f}%, TF-IDF: {tfidf_weight*100:.0f}%)"
                ]
            )
            final_results.append(result)
        
        return final_results
