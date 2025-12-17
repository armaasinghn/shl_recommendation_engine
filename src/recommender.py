"""
Recommender Module with LLM Support
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from .config import MIN_RECOMMENDATIONS, MAX_RECOMMENDATIONS, DEFAULT_RECOMMENDATIONS
from .models import RecommendationResult
from .query_analyzer import analyze_query, normalize_query
from .search_engine import AssessmentSearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHLRecommender:
    
    def __init__(self, assessments_file: Optional[str] = None, use_llm: bool = False, blend_mode: bool = False):
        self.use_llm = use_llm
        self.blend_mode = blend_mode
        self.tfidf_engine = AssessmentSearchEngine()
        
        try:
            count = self.tfidf_engine.load_assessments(assessments_file)
            self.tfidf_engine.fit()
            logger.info(f"Loaded and indexed {count} assessments")
        except FileNotFoundError:
            logger.warning("Assessments file not found")
            self.tfidf_engine.assessments = []
            self.tfidf_engine.is_fitted = False
        
        self.llm_retriever = None
        if use_llm:
            try:
                from .llm_retriever import HybridRetriever
                self.llm_retriever = HybridRetriever(assessments_file)
                logger.info("✅ LLM retriever initialized")
            except Exception as e:
                logger.warning(f"⚠️  LLM unavailable: {e}")
                self.use_llm = False
    
    def get_recommendations(self, query: str, num_results: int = DEFAULT_RECOMMENDATIONS) -> List[Dict[str, Any]]:
        num_results = max(MIN_RECOMMENDATIONS, min(MAX_RECOMMENDATIONS, num_results))
        
        if not query or not query.strip():
            logger.warning("Empty query")
            return []
        
        if self.use_llm and self.llm_retriever:
            try:
                results = asyncio.run(self.llm_retriever.retrieve(query, num_results, use_llm=True, blend=self.blend_mode))
                recommendations = [r.to_api_response() for r in results]
                logger.info(f"Returning {len(recommendations)} recommendations (LLM)")
                return recommendations
            except Exception as e:
                logger.error(f"LLM failed: {e}")
        
        query_info = analyze_query(query)
        try:
            results = self.tfidf_engine.search(query_info, num_results)
        except ValueError as e:
            logger.error(f"Search error: {e}")
            return []
        
        recommendations = [r.to_api_response() for r in results]
        logger.info(f"Returning {len(recommendations)} recommendations (TF-IDF)")
        return recommendations
    
    def get_csv_recommendations(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        clean_query = normalize_query(query)
        api_results = self.get_recommendations(query, num_results)
        return [{"query": clean_query, "assessment_url": r["url"]} for r in api_results]
    
    def get_recommendations_with_scores(self, query: str, num_results: int = DEFAULT_RECOMMENDATIONS) -> List[RecommendationResult]:
        num_results = max(MIN_RECOMMENDATIONS, min(MAX_RECOMMENDATIONS, num_results))
        if not query or not query.strip():
            return []
        
        if self.use_llm and self.llm_retriever:
            try:
                return asyncio.run(self.llm_retriever.retrieve(query, num_results, use_llm=True, blend=self.blend_mode))
            except Exception as e:
                logger.error(f"LLM failed: {e}")
        
        query_info = analyze_query(query)
        return self.tfidf_engine.search(query_info, num_results)
    
    def analyze_query_details(self, query: str) -> Dict[str, Any]:
        query_info = analyze_query(query)
        return {
            "original_query": query_info.original_query,
            "processed_tokens": query_info.processed_tokens,
            "expanded_tokens": query_info.expanded_tokens,
            "duration_constraint": {"min": query_info.duration_min, "max": query_info.duration_max},
            "required_test_types": query_info.required_test_types,
            "job_level": query_info.job_level,
            "needs_balance": query_info.needs_balance,
            "detected_skills": query_info.detected_skills,
            "detected_domains": query_info.detected_domains
        }
    
    @property
    def num_assessments(self) -> int:
        return len(self.tfidf_engine.assessments)
    
    def get_all_assessments(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.tfidf_engine.assessments]


_recommender: Optional[SHLRecommender] = None

def get_recommender(use_llm: bool = False) -> SHLRecommender:
    global _recommender
    if _recommender is None:
        _recommender = SHLRecommender(use_llm=use_llm)
    return _recommender

def recommend(query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    return get_recommender().get_recommendations(query, num_results)

def recommend_csv(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    return get_recommender().get_csv_recommendations(query, num_results)
