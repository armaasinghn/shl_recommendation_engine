"""
Data Models Module
==================
Defines the core data structures used throughout the application.

Classes:
    - Assessment: Represents an SHL assessment product
    - QueryInfo: Contains parsed information from user query
    - RecommendationResult: Contains recommendation with score
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class Assessment:
    """
    Represents an SHL assessment product.
    
    Attributes:
        name: Assessment name
        url: URL to the assessment page
        description: Detailed description
        duration: Duration in minutes
        remote_support: Whether remote testing is supported
        adaptive_support: Whether adaptive testing is supported
        test_type: List of test type codes (A, B, C, D, E, K, P, S)
        test_type_full: List of full test type names
        job_levels: Applicable job levels
        keywords: Keywords for search matching
    """
    name: str
    url: str
    description: str
    duration: int
    remote_support: str
    adaptive_support: str
    test_type: List[str]
    test_type_full: List[str]
    job_levels: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format as specified in requirements."""
        return {
            "url": self.url,
            "name": self.name,
            "adaptive_support": self.adaptive_support,
            "description": self.description,
            "duration": self.duration,
            "remote_support": self.remote_support,
            "test_type": self.test_type_full
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "duration": self.duration,
            "remote_support": self.remote_support,
            "adaptive_support": self.adaptive_support,
            "test_type": self.test_type,
            "test_type_full": self.test_type_full,
            "job_levels": self.job_levels,
            "keywords": self.keywords
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Assessment':
        """Create Assessment instance from dictionary."""
        return cls(
            name=data.get('name', ''),
            url=data.get('url', ''),
            description=data.get('description', ''),
            duration=data.get('duration', 0),
            remote_support=data.get('remote_support', 'No'),
            adaptive_support=data.get('adaptive_support', 'No'),
            test_type=data.get('test_type', []),
            test_type_full=data.get('test_type_full', []),
            job_levels=data.get('job_levels', []),
            keywords=data.get('keywords', [])
        )
    
    def get_searchable_text(self) -> str:
        """Generate text for search indexing."""
        parts = [
            self.name,
            self.description,
            ' '.join(self.keywords),
            ' '.join(self.test_type_full),
            ' '.join(self.job_levels)
        ]
        return ' '.join(parts).lower()


@dataclass
class QueryInfo:
    """
    Contains parsed information extracted from user query.
    
    Attributes:
        original_query: Original query string
        processed_tokens: Preprocessed tokens
        expanded_tokens: Tokens after synonym expansion
        duration_min: Minimum duration constraint (minutes)
        duration_max: Maximum duration constraint (minutes)
        required_test_types: Required test type codes
        job_level: Detected job level
        needs_balance: Whether to balance technical/behavioral
        detected_skills: Technical skills mentioned
        detected_domains: Business domains mentioned
    """
    original_query: str
    processed_tokens: List[str] = field(default_factory=list)
    expanded_tokens: List[str] = field(default_factory=list)
    duration_min: Optional[int] = None
    duration_max: Optional[int] = None
    required_test_types: List[str] = field(default_factory=list)
    job_level: Optional[str] = None
    needs_balance: bool = False
    detected_skills: List[str] = field(default_factory=list)
    detected_domains: List[str] = field(default_factory=list)
    
    def has_duration_constraint(self) -> bool:
        """Check if query has duration constraints."""
        return self.duration_min is not None or self.duration_max is not None
    
    def check_duration(self, duration: int) -> bool:
        """Check if given duration satisfies constraints."""
        if self.duration_min is not None and duration < self.duration_min:
            return False
        if self.duration_max is not None and duration > self.duration_max:
            return False
        return True


@dataclass
class RecommendationResult:
    """
    Contains a recommendation with its relevance score.
    
    Attributes:
        assessment: The recommended assessment
        score: Relevance score (0-1)
        match_reasons: Why this assessment was recommended
    """
    assessment: Assessment
    score: float
    match_reasons: List[str] = field(default_factory=list)
    
    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        response = self.assessment.to_api_response()
        response['relevance_score'] = round(self.score, 3)
        return response


@dataclass
class EvaluationResult:
    """
    Contains evaluation metrics for a set of predictions.
    
    Attributes:
        query: The query being evaluated
        predicted_urls: Predicted assessment URLs
        ground_truth_urls: Actual correct URLs
        recall_at_k: Recall@K score
        k: The K value used
    """
    query: str
    predicted_urls: List[str]
    ground_truth_urls: List[str]
    recall_at_k: float
    k: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "num_predicted": len(self.predicted_urls),
            "num_ground_truth": len(self.ground_truth_urls),
            "recall_at_k": round(self.recall_at_k, 4),
            "k": self.k
        }
