"""
Search Engine Module
====================
Implements semantic search using TF-IDF and keyword matching.

The search engine:
1. Indexes all assessments
2. Computes relevance scores using multiple signals
3. Returns ranked results

This is a lightweight implementation suitable for 400+ assessments.
"""

import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .config import ASSESSMENTS_FILE, TEST_TYPE_CODES
from .models import Assessment, QueryInfo, RecommendationResult
from .text_processor import (
    TFIDFVectorizer, 
    preprocess, 
    compute_keyword_match_score,
    expand_with_synonyms
)


class AssessmentSearchEngine:
    """
    Search engine for SHL assessments using TF-IDF and keyword matching.
    
    Features:
    - TF-IDF based semantic similarity
    - Keyword matching with boost for exact matches
    - Duration filtering
    - Test type filtering and balancing
    - Job level matching
    """
    
    def __init__(self):
        self.assessments: List[Assessment] = []
        self.vectorizer = TFIDFVectorizer()
        self.assessment_vectors: List[Dict[str, float]] = []
        self.is_fitted = False
    
    def load_assessments(self, filepath: Optional[str] = None) -> int:
        """
        Load assessments from JSON file.
        
        Args:
            filepath: Path to JSON file (uses default if None)
            
        Returns:
            Number of assessments loaded
        """
        filepath = filepath or ASSESSMENTS_FILE
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assessments = [Assessment.from_dict(item) for item in data]
        return len(self.assessments)
    
    def fit(self) -> 'AssessmentSearchEngine':
        """
        Build the search index from loaded assessments.
        
        Returns:
            Self for chaining
        """
        if not self.assessments:
            raise ValueError("No assessments loaded. Call load_assessments() first.")
        
        # Create searchable documents
        documents = [a.get_searchable_text() for a in self.assessments]
        
        # Fit TF-IDF vectorizer
        self.vectorizer.fit(documents)
        
        # Pre-compute vectors for all assessments
        self.assessment_vectors = [
            self.vectorizer.transform(doc) for doc in documents
        ]
        
        self.is_fitted = True
        return self
    
    def _compute_relevance_score(self, 
                                  query_info: QueryInfo, 
                                  assessment: Assessment,
                                  assessment_idx: int) -> float:
        """
        Compute relevance score for an assessment given a query.
        
        Combines multiple signals:
        1. TF-IDF cosine similarity (40%)
        2. Keyword match score (40%)
        3. Test type match bonus (10%)
        4. Job level match bonus (10%)
        
        Args:
            query_info: Analyzed query information
            assessment: Assessment to score
            assessment_idx: Index in assessments list
            
        Returns:
            Relevance score (0-1)
        """
        # Signal 1: TF-IDF similarity
        query_text = ' '.join(query_info.expanded_tokens)
        query_vector = self.vectorizer.transform(query_text)
        tfidf_score = self.vectorizer.compute_similarity(
            query_vector, 
            self.assessment_vectors[assessment_idx]
        )
        
        # Signal 2: Keyword match score
        keyword_score = compute_keyword_match_score(
            query_info.expanded_tokens,
            assessment.keywords,
            assessment.get_searchable_text()
        )
        
        # Signal 3: Test type match bonus
        test_type_score = 0.0
        if query_info.required_test_types:
            matches = len(set(query_info.required_test_types) & set(assessment.test_type))
            test_type_score = matches / len(query_info.required_test_types)
        
        # Signal 4: Job level match bonus
        job_level_score = 0.0
        if query_info.job_level and assessment.job_levels:
            level_mapping = {
                'entry': ['Entry-Level', 'Graduate'],
                'mid': ['Mid-Professional', 'Professional Individual Contributor'],
                'senior': ['Mid-Professional', 'Professional Individual Contributor'],
                'manager': ['Manager', 'Front Line Manager', 'Supervisor'],
                'executive': ['Director', 'Executive', 'C-Suite']
            }
            expected_levels = level_mapping.get(query_info.job_level, [])
            if any(level in assessment.job_levels for level in expected_levels):
                job_level_score = 1.0
            elif 'All Levels' in assessment.job_levels:
                job_level_score = 0.5
        
        # Combine scores with weights
        final_score = (
            0.40 * tfidf_score +
            0.40 * keyword_score +
            0.10 * test_type_score +
            0.10 * job_level_score
        )
        
        return final_score
    
    def _filter_by_duration(self, 
                            assessments: List[Tuple[Assessment, float]],
                            query_info: QueryInfo) -> List[Tuple[Assessment, float]]:
        """
        Filter assessments by duration constraints.
        
        Now less restrictive - if no results match, return all results.
        
        Args:
            assessments: List of (assessment, score) tuples
            query_info: Query with duration constraints
            
        Returns:
            Filtered list
        """
        if not query_info.has_duration_constraint():
            return assessments
        
        # Be more lenient with duration - add 50% buffer
        min_dur = query_info.duration_min
        max_dur = query_info.duration_max
        
        if max_dur and max_dur < 999:
            max_dur = int(max_dur * 1.5)  # 50% buffer
        
        filtered = []
        for assessment, score in assessments:
            duration = assessment.duration
            passes = True
            if min_dur is not None and duration < min_dur * 0.5:  # Allow 50% below min
                passes = False
            if max_dur is not None and max_dur < 999 and duration > max_dur:
                passes = False
            if passes:
                filtered.append((assessment, score))
        
        # If too restrictive, return original list
        if len(filtered) < 3 and len(assessments) >= 3:
            return assessments
        
        return filtered if filtered else assessments
    
    def _balance_test_types(self,
                            results: List[Tuple[Assessment, float]],
                            query_info: QueryInfo,
                            num_results: int) -> List[Tuple[Assessment, float]]:
        """
        Balance results between technical and behavioral assessments.
        
        When a query needs both technical and soft skills assessment,
        ensure the results contain a good mix.
        
        Args:
            results: Ranked results
            query_info: Query information
            num_results: Number of results to return
            
        Returns:
            Balanced result list
        """
        if not query_info.needs_balance or len(results) <= num_results:
            return results[:num_results]
        
        # Separate by type
        technical = []  # K, S types
        behavioral = []  # P, B, C types
        cognitive = []  # A type
        other = []
        
        for assessment, score in results:
            types = set(assessment.test_type)
            if types & {'K', 'S'}:
                technical.append((assessment, score))
            elif types & {'P', 'B', 'C'}:
                behavioral.append((assessment, score))
            elif 'A' in types:
                cognitive.append((assessment, score))
            else:
                other.append((assessment, score))
        
        # Build balanced result
        balanced = []
        
        # Allocate slots: ~40% technical, ~40% behavioral, ~20% cognitive/other
        tech_slots = max(2, num_results * 4 // 10)
        behav_slots = max(2, num_results * 4 // 10)
        other_slots = num_results - tech_slots - behav_slots
        
        # Fill slots
        balanced.extend(technical[:tech_slots])
        balanced.extend(behavioral[:behav_slots])
        balanced.extend(cognitive[:other_slots])
        
        # If we don't have enough, fill from remaining
        remaining_slots = num_results - len(balanced)
        if remaining_slots > 0:
            all_remaining = (
                technical[tech_slots:] + 
                behavioral[behav_slots:] + 
                cognitive[other_slots:] +
                other
            )
            all_remaining.sort(key=lambda x: x[1], reverse=True)
            balanced.extend(all_remaining[:remaining_slots])
        
        # Sort final results by score
        balanced.sort(key=lambda x: x[1], reverse=True)
        
        return balanced[:num_results]
    
    def search(self, 
               query_info: QueryInfo, 
               num_results: int = 10) -> List[RecommendationResult]:
        """
        Search for relevant assessments.
        
        Args:
            query_info: Analyzed query information
            num_results: Maximum number of results (1-10)
            
        Returns:
            List of RecommendationResult objects
        """
        if not self.is_fitted:
            raise ValueError("Search engine not fitted. Call fit() first.")
        
        # Score all assessments
        scored = []
        for idx, assessment in enumerate(self.assessments):
            score = self._compute_relevance_score(query_info, assessment, idx)
            if score > 0.01:  # Minimum threshold
                scored.append((assessment, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by duration
        scored = self._filter_by_duration(scored, query_info)
        
        # Balance if needed
        scored = self._balance_test_types(scored, query_info, num_results)
        
        # Convert to RecommendationResult
        results = []
        for assessment, score in scored[:num_results]:
            result = RecommendationResult(
                assessment=assessment,
                score=score,
                match_reasons=self._get_match_reasons(query_info, assessment)
            )
            results.append(result)
        
        return results
    
    def _get_match_reasons(self, 
                           query_info: QueryInfo, 
                           assessment: Assessment) -> List[str]:
        """Generate human-readable match reasons."""
        reasons = []
        
        # Check keyword matches
        query_tokens = set(query_info.expanded_tokens)
        assessment_keywords = set(k.lower() for k in assessment.keywords)
        matches = query_tokens & assessment_keywords
        if matches:
            reasons.append(f"Matches keywords: {', '.join(list(matches)[:3])}")
        
        # Check test type
        if query_info.required_test_types:
            type_matches = set(query_info.required_test_types) & set(assessment.test_type)
            if type_matches:
                type_names = [TEST_TYPE_CODES.get(t, t) for t in type_matches]
                reasons.append(f"Test type: {', '.join(type_names)}")
        
        # Check duration
        if query_info.has_duration_constraint():
            reasons.append(f"Duration: {assessment.duration} minutes")
        
        return reasons


# Singleton instance for use across the application
_search_engine: Optional[AssessmentSearchEngine] = None


def get_search_engine() -> AssessmentSearchEngine:
    """
    Get or create the singleton search engine instance.
    
    Returns:
        Initialized and fitted search engine
    """
    global _search_engine
    
    if _search_engine is None:
        _search_engine = AssessmentSearchEngine()
        _search_engine.load_assessments()
        _search_engine.fit()
    
    return _search_engine


def search_assessments(query: str, num_results: int = 10) -> List[RecommendationResult]:
    """
    Convenience function to search assessments.
    
    Args:
        query: User query string
        num_results: Number of results (1-10)
        
    Returns:
        List of recommendation results
    """
    from .query_analyzer import analyze_query
    
    engine = get_search_engine()
    query_info = analyze_query(query)
    
    return engine.search(query_info, num_results)
