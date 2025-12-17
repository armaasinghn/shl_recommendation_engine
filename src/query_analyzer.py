"""
Query Analyzer Module
=====================
Analyzes user queries to extract:
- Duration constraints
- Required test types
- Job level requirements
- Technical skills mentioned
- Business domains
"""

import re
from typing import List, Optional, Tuple

from .config import (
    TEST_TYPE_KEYWORDS,
    JOB_LEVEL_KEYWORDS,
    DOMAIN_KEYWORDS,
    SYNONYMS
)
from .models import QueryInfo
from .text_processor import preprocess, expand_with_synonyms


# ============================================================
# QUERY NORMALIZATION (FOR CSV + EVALUATION SAFETY)
# ============================================================

def normalize_query(query: str) -> str:
    """
    Normalize query for CSV output:
    - lowercase
    - remove newlines
    - truncate length
    - preserve semantic intent
    """
    q = query.lower()
    q = re.sub(r'\s+', ' ', q).strip()
    return q[:120]


# ============================================================
# DURATION EXTRACTION
# ============================================================

DURATION_PATTERNS = [
    (r'(\d+)\s*(?:to|-)\s*(\d+)\s*(?:min|mins|minutes)',
     lambda m: (int(m.group(1)), int(m.group(2)))),

    (r'(?:within|under|less than)\s*(\d+)\s*(?:min|mins|minutes)',
     lambda m: (0, int(m.group(1)))),

    (r'(?:about|around|approximately)\s*(\d+)\s*(?:min|mins|minutes)',
     lambda m: (max(0, int(m.group(1)) - 10), int(m.group(1)) + 10)),

    (r'(\d+)\s*(?:min|mins|minutes)',
     lambda m: (max(0, int(m.group(1)) - 5), int(m.group(1)) + 5)),

    (r'(\d+)\s*(?:to|-)\s*(\d+)\s*hour',
     lambda m: (int(m.group(1)) * 60, int(m.group(2)) * 60)),

    (r'(\d+)\s*hour',
     lambda m: (int(m.group(1)) * 60 - 10, int(m.group(1)) * 60 + 10)),
]


def extract_duration_constraint(query: str) -> Tuple[Optional[int], Optional[int]]:
    query_lower = query.lower()

    for pattern, extractor in DURATION_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            try:
                return extractor(match)
            except Exception:
                continue

    return (None, None)


# ============================================================
# TEST TYPE EXTRACTION
# ============================================================

def extract_required_test_types(query: str) -> List[str]:
    query_lower = query.lower()
    required = []

    for test_type, keywords in TEST_TYPE_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            required.append(test_type)

    return required


# ============================================================
# JOB LEVEL EXTRACTION
# ============================================================

def extract_job_level(query: str) -> Optional[str]:
    query_lower = query.lower()
    for level in ['executive', 'manager', 'senior', 'mid', 'entry']:
        for keyword in JOB_LEVEL_KEYWORDS.get(level, []):
            if keyword in query_lower:
                return level
    return None


# ============================================================
# TECHNICAL SKILLS EXTRACTION
# ============================================================

TECHNICAL_SKILLS = [
    'python', 'java', 'javascript', 'sql', 'html', 'css',
    'react', 'angular', 'node', 'aws', 'azure', 'gcp',
    'docker', 'kubernetes', 'git', 'linux',
    'machine learning', 'data science',
    'jira', 'confluence', 'agile',
    'seo', 'digital marketing', 'content writing'
]


def extract_technical_skills(query: str) -> List[str]:
    query_lower = query.lower()
    found = []

    for skill in TECHNICAL_SKILLS:
        if len(skill.split()) == 1:
            if re.search(rf'\b{re.escape(skill)}\b', query_lower):
                found.append(skill)
        else:
            if skill in query_lower:
                found.append(skill)

    return found


# ============================================================
# DOMAIN EXTRACTION
# ============================================================

def extract_domains(query: str) -> List[str]:
    query_lower = query.lower()
    domains = []

    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            domains.append(domain)

    return domains


# ============================================================
# BALANCE DETECTION
# ============================================================

def needs_balanced_results(query: str, required_types: List[str]) -> bool:
    query_lower = query.lower()

    technical = 'K' in required_types
    behavioral = 'P' in required_types or any(
        k in query_lower for k in [
            'communication', 'team', 'collaboration',
            'leadership', 'customer', 'interpersonal'
        ]
    )

    return technical and behavioral


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def analyze_query(query: str) -> QueryInfo:
    processed_tokens = preprocess(query)
    expanded_tokens = expand_with_synonyms(processed_tokens)

    duration_min, duration_max = extract_duration_constraint(query)
    required_types = extract_required_test_types(query)
    job_level = extract_job_level(query)
    skills = extract_technical_skills(query)
    domains = extract_domains(query)
    balance_needed = needs_balanced_results(query, required_types)

    return QueryInfo(
        original_query=query,
        processed_tokens=processed_tokens,
        expanded_tokens=expanded_tokens,
        duration_min=duration_min,
        duration_max=duration_max,
        required_test_types=required_types,
        job_level=job_level,
        needs_balance=balance_needed,
        detected_skills=skills,
        detected_domains=domains
    )