"""
Recommender Module - Hybrid Approach
"""

import logging
from typing import List, Dict, Any, Optional

from .config import MIN_RECOMMENDATIONS, MAX_RECOMMENDATIONS, DEFAULT_RECOMMENDATIONS
from .models import RecommendationResult
from .query_analyzer import analyze_query, normalize_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHLRecommender:
    
    def __init__(
        self, 
        assessments_file: Optional[str] = None,
        use_embeddings: bool = True
    ):
        """
        Initialize recommender.
        
        Args:
            assessments_file: Path to assessments JSON
            use_embeddings: Whether to use hybrid embedding+TF-IDF retrieval
        """
        self.use_embeddings = use_embeddings
        
        if use_embeddings:
            try:
                from .embedding_retriever import HybridEmbeddingRetriever
                self.retriever = HybridEmbeddingRetriever(assessments_file)
                logger.info("✅ Using Hybrid Embedding + TF-IDF Retriever")
            except Exception as e:
                logger.warning(f"⚠️  Embedding retriever failed: {e}")
                logger.info("Falling back to TF-IDF only")
                self._init_tfidf_only(assessments_file)
        else:
            self._init_tfidf_only(assessments_file)
    
    def _init_tfidf_only(self, assessments_file):
        """Initialize TF-IDF only retriever."""
        from .search_engine import AssessmentSearchEngine
        self.retriever = AssessmentSearchEngine()
        self.retriever.load_assessments(assessments_file)
        self.retriever.fit()
        self.use_embeddings = False
        logger.info("✅ Using TF-IDF Retriever")
    
    def get_recommendations(
        self, 
        query: str, 
        num_results: int = DEFAULT_RECOMMENDATIONS
    ) -> List[Dict[str, Any]]:
        """Get assessment recommendations."""
        
        num_results = max(MIN_RECOMMENDATIONS, min(MAX_RECOMMENDATIONS, num_results))
        
        if not query or not query.strip():
            logger.warning("Empty query")
            return []
        
        # Get results
        if self.use_embeddings:
            results = self.retriever.search(query, num_results)
        else:
            query_info = analyze_query(query)
            results = self.retriever.search(query_info, num_results)
        
        # Convert to API format
        recommendations = [r.to_api_response() for r in results]
        logger.info(f"Returning {len(recommendations)} recommendations")
        return recommendations
    
    def get_csv_recommendations(
        self, 
        query: str, 
        num_results: int = 5
    ) -> List[Dict[str, str]]:
        """CSV-safe format."""
        clean_query = normalize_query(query)
        api_results = self.get_recommendations(query, num_results)
        return [{"query": clean_query, "assessment_url": r["url"]} for r in api_results]
    
    def get_recommendations_with_scores(
        self, 
        query: str, 
        num_results: int = DEFAULT_RECOMMENDATIONS
    ) -> List[RecommendationResult]:
        """With scores for evaluation."""
        
        num_results = max(MIN_RECOMMENDATIONS, min(MAX_RECOMMENDATIONS, num_results))
        
        if not query or not query.strip():
            return []
        
        if self.use_embeddings:
            return self.retriever.search(query, num_results)
        else:
            query_info = analyze_query(query)
            return self.retriever.search(query_info, num_results)
    
    @property
    def num_assessments(self) -> int:
        """Number of assessments."""
        if hasattr(self.retriever, 'assessments'):
            return len(self.retriever.assessments)
        elif hasattr(self.retriever, 'embedding_retriever'):
            return len(self.retriever.embedding_retriever.assessments)
        return 0
    
    def get_all_assessments(self) -> List[Dict[str, Any]]:
        """All assessments."""
        if hasattr(self.retriever, 'assessments'):
            return [a.to_dict() for a in self.retriever.assessments]
        elif hasattr(self.retriever, 'embedding_retriever'):
            return [a.to_dict() for a in self.retriever.embedding_retriever.assessments]
        return []


_recommender: Optional[SHLRecommender] = None


def get_recommender(use_embeddings: bool = True) -> SHLRecommender:
    global _recommender
    if _recommender is None:
        _recommender = SHLRecommender(use_embeddings=use_embeddings)
    return _recommender


def recommend(query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    return get_recommender().get_recommendations(query, num_results)


def recommend_csv(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    return get_recommender().get_csv_recommendations(query, num_results)
