"""
Query Analyzer Module - Enhanced for Maximum Recall
====================================================
Extracts constraints, skills, and intent from queries.
"""

import re
from typing import List, Optional, Tuple

from .config import TEST_TYPE_KEYWORDS, JOB_LEVEL_KEYWORDS, DOMAIN_KEYWORDS
from .models import QueryInfo
from .text_processor import preprocess, expand_with_synonyms


# Enhanced Role to Skill Mapping
ROLE_SKILL_MAP = {
    # Sales roles
    "sales": ["communication", "negotiation", "customer", "persuasion", "interpersonal", "presentation"],
    "sales representative": ["communication", "negotiation", "customer", "persuasion", "cold calling"],
    "account executive": ["sales", "negotiation", "customer", "business development"],
    "business development": ["sales", "negotiation", "communication", "prospecting"],
    
    # Customer service roles
    "customer support": ["communication", "english", "customer", "service", "phone", "empathy"],
    "customer service": ["communication", "english", "customer", "service", "phone", "patience"],
    "support": ["communication", "english", "customer", "service"],
    "call center": ["communication", "english", "phone", "customer", "typing"],
    "contact center": ["communication", "english", "phone", "customer", "multitasking"],
    
    # Technical roles
    "software developer": ["programming", "coding", "technical", "problem solving"],
    "developer": ["programming", "coding", "technical", "software"],
    "engineer": ["programming", "coding", "technical", "problem solving", "analytical"],
    "research engineer": ["programming", "ai", "ml", "technical", "problem solving", "analytical", "innovation"],
    "data scientist": ["python", "sql", "machine learning", "statistics", "analytical"],
    "data analyst": ["sql", "excel", "analytics", "numerical", "data", "visualization"],
    "business analyst": ["sql", "excel", "requirements", "analytical", "communication"],
    "qa engineer": ["testing", "automation", "selenium", "quality"],
    "devops": ["aws", "docker", "linux", "automation", "cloud"],
    
    # Management roles
    "product manager": ["jira", "confluence", "agile", "stakeholder", "sdlc", "requirements", "communication"],
    "project manager": ["jira", "planning", "stakeholder", "communication", "leadership"],
    "manager": ["leadership", "management", "team", "communication", "decision making"],
    "team lead": ["leadership", "technical", "mentoring", "communication"],
    
    # Executive roles
    "executive": ["leadership", "strategy", "decision making", "business acumen"],
    "director": ["leadership", "strategy", "management", "vision"],
    "coo": ["leadership", "operations", "strategy", "global", "cross-cultural"],
    "ceo": ["leadership", "vision", "strategy", "business acumen"],
    
    # Marketing roles
    "marketing": ["digital", "content", "creative", "communication", "analytics"],
    "content writer": ["writing", "english", "creative", "seo", "content", "grammar"],
    "copywriter": ["writing", "creative", "marketing", "content"],
    "digital marketing": ["seo", "analytics", "content", "social media", "advertising"],
    
    # Administrative roles
    "admin": ["administrative", "clerical", "office", "data entry", "organization"],
    "administrative assistant": ["administrative", "clerical", "office", "typing", "organization"],
    "assistant": ["administrative", "clerical", "office", "support", "communication"],
    "secretary": ["administrative", "typing", "communication", "organization"],
    
    # Analyst roles
    "analyst": ["analytical", "numerical", "problem solving", "data", "excel"],
    "consultant": ["communication", "analytical", "problem solving", "client", "presentation"],
    "presales": ["sales", "technical", "communication", "presentation", "demo"],
    
    # Finance roles
    "accountant": ["numerical", "finance", "excel", "attention to detail"],
    "financial analyst": ["numerical", "excel", "finance", "analytical"],
    "banker": ["numerical", "customer", "finance", "communication"],
}


# Duration Patterns - More comprehensive
DURATION_PATTERNS = [
    # Range patterns
    (r'(\d+)\s*(?:to|-)\s*(\d+)\s*(?:min|mins|minutes)', lambda m: (int(m.group(1)), int(m.group(2)))),
    (r'(\d+)\s*-\s*(\d+)\s*(?:min|mins|minutes)', lambda m: (int(m.group(1)), int(m.group(2)))),
    
    # Max/within patterns
    (r'(?:within|under|less than|max|maximum|upto|up to)\s*(\d+)\s*(?:min|mins|minutes)', lambda m: (0, int(m.group(1)))),
    (r'(?:not more than|no more than)\s*(\d+)\s*(?:min|mins|minutes)', lambda m: (0, int(m.group(1)))),
    
    # Approximate patterns
    (r'(?:about|around|approximately|roughly)\s*(\d+)\s*(?:min|mins|minutes)', lambda m: (max(0, int(m.group(1)) - 15), int(m.group(1)) + 15)),
    
    # Single value patterns
    (r'(\d+)\s*(?:min|mins|minutes)', lambda m: (max(0, int(m.group(1)) - 10), int(m.group(1)) + 10)),
    
    # Hour patterns
    (r'(\d+)\s*(?:to|-)\s*(\d+)\s*(?:hour|hours|hr|hrs)', lambda m: (int(m.group(1)) * 60, int(m.group(2)) * 60)),
    (r'(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs)', lambda m: (int(float(m.group(1)) * 60 - 15), int(float(m.group(1)) * 60 + 15))),
]

# Technical Skills List - Expanded
TECHNICAL_SKILLS = [
    # Programming languages
    'python', 'java', 'javascript', 'js', 'sql', 'html', 'css', 'c#', 'c++',
    'typescript', 'ruby', 'php', 'swift', 'kotlin', 'scala', 'go', 'rust',
    
    # Frameworks & Libraries
    'react', 'angular', 'vue', 'node', 'nodejs', 'express', 'django', 'flask',
    'spring', '.net', 'dotnet', 'jquery', 'bootstrap',
    
    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes', 'jenkins', 'terraform',
    'ci/cd', 'devops',
    
    # Data & ML
    'machine learning', 'ml', 'ai', 'data science', 'tensorflow', 'pytorch',
    'pandas', 'numpy', 'scikit-learn', 'nlp', 'deep learning',
    
    # Tools
    'git', 'github', 'linux', 'unix', 'tableau', 'power bi', 'excel',
    'jira', 'confluence', 'agile', 'scrum',
    
    # Testing
    'selenium', 'testing', 'qa', 'automation', 'junit', 'pytest',
    
    # Marketing/Content
    'seo', 'digital marketing', 'content writing', 'copywriting', 'google analytics'
]


def extract_duration_constraint(query: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract duration constraints from query."""
    query_lower = query.lower()
    for pattern, extractor in DURATION_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            try:
                return extractor(match)
            except:
                continue
    return (None, None)
def normalize_query(query: str) -> str:
    """
    Normalize query for safe comparison and CSV output.
    """
    if not query:
        return ""
    query = query.lower()
    query = " ".join(query.split())
    return query[:200]

def extract_required_test_types(query: str) -> List[str]:
    """Extract required test types from query."""
    query_lower = query.lower()
    required = []
    for test_type, keywords in TEST_TYPE_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            required.append(test_type)
    return required


def extract_job_level(query: str) -> Optional[str]:
    """Extract job level from query."""
    query_lower = query.lower()
    for level in ['executive', 'manager', 'senior', 'mid', 'entry']:
        for keyword in JOB_LEVEL_KEYWORDS.get(level, []):
            if keyword in query_lower:
                return level
    return None


def extract_technical_skills(query: str) -> List[str]:
    """Extract technical skills from query."""
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


def extract_domains(query: str) -> List[str]:
    """Extract business domains from query."""
    query_lower = query.lower()
    domains = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            domains.append(domain)
    return domains


def needs_balanced_results(query: str, required_types: List[str]) -> bool:
    """Check if results should balance technical and behavioral."""
    query_lower = query.lower()
    
    # Check for technical indicators
    technical = 'K' in required_types or any(
        skill in query_lower for skill in TECHNICAL_SKILLS[:20]
    )
    
    # Check for behavioral indicators
    behavioral_keywords = [
        'communication', 'team', 'collaboration', 'leadership',
        'customer', 'interpersonal', 'soft skill', 'personality',
        'behavioral', 'cultural', 'teamwork'
    ]
    behavioral = 'P' in required_types or any(k in query_lower for k in behavioral_keywords)
    
    return technical and behavioral


def analyze_query(query: str) -> QueryInfo:
    """Complete query analysis with role expansion."""
    processed_tokens = preprocess(query)
    expanded_tokens = expand_with_synonyms(processed_tokens)

    # Role-based expansion
    query_lower = query.lower()
    for role, skills in ROLE_SKILL_MAP.items():
        if role in query_lower:
            expanded_tokens.extend(skills)

    # Deduplicate
    expanded_tokens = list(set(expanded_tokens))

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
