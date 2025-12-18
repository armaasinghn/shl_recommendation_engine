"""
Embedding-Based Retriever - Optimized
======================================
Enhanced blending strategy for better recall
"""

import json
import numpy as np
from typing import List, Optional
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss

from .config import ASSESSMENTS_FILE
from .models import Assessment, RecommendationResult


class EmbeddingRetriever:
    """Semantic search using sentence transformers and FAISS."""
    
    def __init__(
        self, 
        assessments_file: Optional[str] = None,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
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
        with open(self.assessments_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assessments = [Assessment.from_dict(item) for item in data]
        print(f"✅ Loaded {len(self.assessments)} assessments")
    
    def _build_index(self):
        print("\nBuilding vector index...")
        texts = [self._get_assessment_text(a) for a in self.assessments]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        print(f"✅ Vector index built with {self.index.ntotal} vectors")
    
    def _get_assessment_text(self, assessment: Assessment) -> str:
        parts = [
            assessment.name,
            assessment.description,
            " ".join(assessment.keywords),
            " ".join(assessment.test_type_full)
        ]
        return " ".join(parts)
    
    def search(self, query: str, num_results: int = 10) -> List[RecommendationResult]:
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), num_results)
        
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
    Optimized hybrid retriever with adaptive weighting.
    
    Strategy:
    - 🔥 NEW: Adaptive weighting based on query type
    - Embedding: 60-75% (was 70% fixed)
    - TF-IDF: 25-40% (was 30% fixed)
    """
    
    def __init__(self, assessments_file: Optional[str] = None):
        from .search_engine import AssessmentSearchEngine
        
        print("\n" + "=" * 70)
        print("INITIALIZING HYBRID RETRIEVER")
        print("=" * 70)
        
        print("\n[1/2] Initializing Embedding Retriever...")
        self.embedding_retriever = EmbeddingRetriever(assessments_file)
        
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
        embedding_weight: Optional[float] = None
    ) -> List[RecommendationResult]:
        """
        Hybrid search with adaptive weighting.
        
        🔥 NEW: Auto-adjust weights based on query characteristics
        """
        
        # 🔥 Adaptive weighting logic
        if embedding_weight is None:
            query_lower = query.lower()
            
            # More specific queries (with technical terms) → higher TF-IDF weight
            technical_terms = len([w for w in ['java', 'python', 'sql', 'excel', 'jira'] 
                                  if w in query_lower])
            
            if technical_terms >= 2:
                embedding_weight = 0.60  # More TF-IDF for technical queries
            elif len(query.split()) < 10:
                embedding_weight = 0.65  # Balanced for short queries
            else:
                embedding_weight = 0.75  # More embeddings for long descriptive queries
        
        tfidf_weight = 1.0 - embedding_weight
        
        # Get results from both
        embedding_results = self.embedding_retriever.search(query, num_results * 2)
        
        from .query_analyzer import analyze_query
        query_info = analyze_query(query)
        tfidf_results = self.tfidf_retriever.search(query_info, num_results * 2)
        
        # Blend scores
        scores = {}
        url_to_assessment = {}
        
        for result in embedding_results:
            url = result.assessment.url
            scores[url] = scores.get(url, 0) + (result.score * embedding_weight)
            url_to_assessment[url] = result.assessment
        
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
                    f"(Emb: {embedding_weight*100:.0f}%, TF-IDF: {tfidf_weight*100:.0f}%)"
                ]
            )
            final_results.append(result)
        
        return final_results
