"""
Search Engine Module
====================
Implements semantic search using TF-IDF and keyword matching.

Enhancements included:
- Wider candidate pool for better recall
- Skill overlap re-ranking
- Stronger duration penalty
- Balanced technical vs behavioral results
"""

import json
from typing import List, Tuple, Optional
from pathlib import Path

from .config import ASSESSMENTS_FILE, TEST_TYPE_CODES
from .models import Assessment, QueryInfo, RecommendationResult
from .text_processor import (
    TFIDFVectorizer,
    compute_keyword_match_score
)


class AssessmentSearchEngine:
    """
    Search engine for SHL assessments using TF-IDF + heuristic scoring.
    """

    def __init__(self):
        self.assessments: List[Assessment] = []
        self.vectorizer = TFIDFVectorizer()
        self.assessment_vectors = []
        self.is_fitted = False

    # ============================================================
    # LOADING & INDEXING
    # ============================================================

    def load_assessments(self, filepath: Optional[str] = None) -> int:
        filepath = filepath or ASSESSMENTS_FILE
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assessments = [Assessment.from_dict(item) for item in data]
        return len(self.assessments)

    def fit(self):
        if not self.assessments:
            raise ValueError("No assessments loaded.")

        documents = [a.get_searchable_text() for a in self.assessments]
        self.vectorizer.fit(documents)
        self.assessment_vectors = [
            self.vectorizer.transform(doc) for doc in documents
        ]
        self.is_fitted = True

    # ============================================================
    # SCORING LOGIC
    # ============================================================

    def _compute_relevance_score(
        self,
        query_info: QueryInfo,
        assessment: Assessment,
        idx: int
    ) -> float:

        # TF-IDF similarity (semantic)
        query_text = " ".join(query_info.expanded_tokens)
        query_vec = self.vectorizer.transform(query_text)
        tfidf_score = self.vectorizer.compute_similarity(
            query_vec,
            self.assessment_vectors[idx]
        )

        # Keyword match score
        keyword_score = compute_keyword_match_score(
            query_info.expanded_tokens,
            assessment.keywords,
            assessment.get_searchable_text()
        )

        # Skill overlap boost (🔥 HIGH IMPACT)
        skill_overlap = len(
            set(query_info.detected_skills) &
            set(assessment.keywords)
        )

        # Test type bonus
        test_type_score = 0.0
        if query_info.required_test_types:
            matches = set(query_info.required_test_types) & set(assessment.test_type)
            test_type_score = len(matches) / len(query_info.required_test_types)

        # Job level bonus
        job_level_score = 0.0
        if query_info.job_level and assessment.job_levels:
            if "All Levels" in assessment.job_levels:
                job_level_score = 0.5

        # Combine scores
        final_score = (
            0.40 * tfidf_score +
            0.40 * keyword_score +
            0.10 * test_type_score +
            0.10 * job_level_score +
            0.10 * skill_overlap
        )

        # 🔥 Strong duration penalty
        if query_info.duration_max is not None and assessment.duration:
            if assessment.duration > query_info.duration_max:
                final_score *= 0.4

        return final_score

    # ============================================================
    # SEARCH PIPELINE
    # ============================================================

    def search(self, query_info: QueryInfo, num_results: int = 10) -> List[RecommendationResult]:
        if not self.is_fitted:
            raise ValueError("Search engine not fitted.")

        scored: List[Tuple[Assessment, float]] = []

        for idx, assessment in enumerate(self.assessments):
            score = self._compute_relevance_score(query_info, assessment, idx)
            if score > 0.01:
                scored.append((assessment, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # 🔥 Wider candidate pool (VERY IMPORTANT)
        candidate_k = max(num_results * 3, 30)
        scored = scored[:candidate_k]

        # Balance test types if needed
        scored = self._balance_test_types(scored, query_info, num_results)

        # Build final results
        results = []
        for assessment, score in scored[:num_results]:
            results.append(
                RecommendationResult(
                    assessment=assessment,
                    score=score,
                    match_reasons=self._get_match_reasons(query_info, assessment)
                )
            )

        return results

    # ============================================================
    # BALANCING LOGIC
    # ============================================================

    def _balance_test_types(
        self,
        results: List[Tuple[Assessment, float]],
        query_info: QueryInfo,
        num_results: int
    ) -> List[Tuple[Assessment, float]]:

        if not query_info.needs_balance:
            return results[:num_results]

        technical, behavioral, other = [], [], []

        for a, s in results:
            types = set(a.test_type)
            if types & {"K", "S"}:
                technical.append((a, s))
            elif types & {"P", "B", "C"}:
                behavioral.append((a, s))
            else:
                other.append((a, s))

        balanced = []
        balanced.extend(technical[: num_results // 2])
        balanced.extend(behavioral[: num_results // 2])

        remaining = num_results - len(balanced)
        if remaining > 0:
            leftovers = technical[num_results // 2:] + behavioral[num_results // 2:] + other
            leftovers.sort(key=lambda x: x[1], reverse=True)
            balanced.extend(leftovers[:remaining])

        return balanced[:num_results]

    # ============================================================
    # MATCH EXPLANATIONS
    # ============================================================

    def _get_match_reasons(self, query_info: QueryInfo, assessment: Assessment) -> List[str]:
        reasons = []

        token_matches = set(query_info.expanded_tokens) & set(assessment.keywords)
        if token_matches:
            reasons.append(f"Keyword match: {', '.join(list(token_matches)[:3])}")

        if query_info.required_test_types:
            matches = set(query_info.required_test_types) & set(assessment.test_type)
            if matches:
                reasons.append(
                    "Test type: " + ", ".join(TEST_TYPE_CODES.get(t, t) for t in matches)
                )

        if query_info.duration_max:
            reasons.append(f"Duration: {assessment.duration} minutes")

        return reasons


# ============================================================
# SINGLETON ACCESS
# ============================================================

_search_engine: Optional[AssessmentSearchEngine] = None


def get_search_engine() -> AssessmentSearchEngine:
    global _search_engine
    if _search_engine is None:
        _search_engine = AssessmentSearchEngine()
        _search_engine.load_assessments()
        _search_engine.fit()
    return _search_engine


def search_assessments(query: str, num_results: int = 10) -> List[RecommendationResult]:
    from .query_analyzer import analyze_query
    engine = get_search_engine()
    return engine.search(analyze_query(query), num_results)