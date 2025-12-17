"""
Configuration Module
====================
Contains all configuration constants, mappings, and settings.

This centralizes all configurable parameters for easy tuning and maintenance.
"""

import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ASSESSMENTS_FILE = os.path.join(DATA_DIR, 'shl_assessments.json')

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 5000))
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# ============================================================================
# RECOMMENDATION SETTINGS
# ============================================================================

MIN_RECOMMENDATIONS = 1
MAX_RECOMMENDATIONS = 10
DEFAULT_RECOMMENDATIONS = 10

# ============================================================================
# TEST TYPE MAPPINGS
# ============================================================================

TEST_TYPE_CODES = {
    'A': 'Ability & Aptitude',
    'B': 'Biodata & Situational Judgement',
    'C': 'Competencies',
    'D': 'Development & 360',
    'E': 'Assessment Exercises',
    'K': 'Knowledge & Skills',
    'P': 'Personality & Behavior',
    'S': 'Simulations'
}

# Reverse mapping: Full name to code
TEST_TYPE_NAMES_TO_CODES = {v: k for k, v in TEST_TYPE_CODES.items()}

# ============================================================================
# KEYWORDS FOR TEST TYPE DETECTION
# ============================================================================

TEST_TYPE_KEYWORDS = {
    'K': [
        'technical', 'knowledge', 'skill', 'programming', 'coding', 'java',
        'python', 'sql', 'javascript', 'excel', 'software', 'development',
        '.net', 'react', 'angular', 'aws', 'data', 'machine learning', 'ai',
        'html', 'css', 'node', 'api', 'database', 'tableau', 'power bi',
        'jira', 'confluence', 'agile', 'scrum', 'docker', 'git', 'linux',
        'mongodb', 'selenium', 'testing', 'qa', 'devops', 'cloud', 'seo',
        'marketing', 'content', 'writing', 'excel', 'word', 'powerpoint'
    ],
    'P': [
        'personality', 'behavioral', 'behaviour', 'opq', 'interpersonal',
        'collaboration', 'teamwork', 'leadership style', 'work style',
        'cultural fit', 'soft skills', 'communication', 'motivation',
        'emotional intelligence', 'eq'
    ],
    'A': [
        'cognitive', 'aptitude', 'reasoning', 'numerical', 'verbal', 
        'logical', 'analytical', 'problem solving', 'intelligence',
        'ability test', 'deductive', 'inductive', 'g+'
    ],
    'B': [
        'situational', 'judgement', 'judgment', 'scenarios', 'biodata',
        'situational judgment', 'sjt'
    ],
    'C': [
        'competency', 'competencies', 'behavior', 'leadership competency'
    ],
    'S': [
        'simulation', 'practical', 'hands-on', 'call center', 'typing',
        'contact center', 'phone simulation'
    ],
    'D': [
        'development', '360', 'feedback', 'coaching'
    ],
    'E': [
        'exercise', 'assessment center', 'role play', 'presentation'
    ]
}

# ============================================================================
# JOB LEVEL KEYWORDS
# ============================================================================

JOB_LEVEL_KEYWORDS = {
    'entry': [
        'entry', 'entry-level', 'entry level', 'fresher', 'graduate', 
        'new graduate', 'campus', '0-2 years', '0-1 years', 'junior', 
        'beginner', 'trainee', 'intern', 'freshers'
    ],
    'mid': [
        'mid-level', 'mid level', 'experienced', '3-5 years', '2-5 years',
        '3-4 years', '4-5 years', '5 years', 'professional', 'associate'
    ],
    'senior': [
        'senior', 'lead', 'advanced', '5+ years', '7+ years', '10+ years',
        'principal', 'staff', 'expert'
    ],
    'manager': [
        'manager', 'supervisor', 'team lead', 'management', 'team leader',
        'people manager'
    ],
    'executive': [
        'executive', 'director', 'coo', 'ceo', 'cfo', 'cto', 'cxo', 
        'c-suite', 'vp', 'vice president', 'chief'
    ]
}

# ============================================================================
# STOPWORDS FOR TEXT PROCESSING
# ============================================================================

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now',
    'want', 'need', 'looking', 'find', 'recommend', 'suggest', 'give',
    'help', 'please', 'would', 'like', 'could', 'also', 'well', 'good',
    'great', 'best', 'able', 'hire', 'hiring', 'seeking', 'require',
    'required', 'requirements', 'search', 'searching', 'job', 'position',
    'role', 'candidate', 'candidates', 'applicant', 'applicants'
}

# ============================================================================
# SYNONYM MAPPINGS FOR QUERY EXPANSION
# ============================================================================

SYNONYMS = {
    'developer': ['programmer', 'engineer', 'coder', 'software developer'],
    'programmer': ['developer', 'engineer', 'coder'],
    'coding': ['programming', 'development'],
    'programming': ['coding', 'development'],
    'sql': ['database', 'queries', 'structured query language'],
    'python': ['py', 'python3'],
    'java': ['core java', 'j2ee'],
    'javascript': ['js', 'ecmascript'],
    'js': ['javascript'],
    'analyst': ['analysis', 'analytics'],
    'manager': ['management', 'leader', 'supervisor'],
    'sales': ['selling', 'revenue', 'business development'],
    'customer': ['client', 'consumer'],
    'communication': ['interpersonal', 'verbal', 'written'],
    'leadership': ['management', 'executive', 'leader'],
    'cognitive': ['aptitude', 'reasoning', 'ability', 'mental'],
    'personality': ['behavioral', 'behaviour', 'opq', 'psychometric'],
    'admin': ['administrative', 'office', 'clerical'],
    'finance': ['financial', 'accounting', 'banking'],
    'marketing': ['digital marketing', 'advertising', 'brand'],
    'content': ['writing', 'copywriting'],
    'graduate': ['fresher', 'entry level', 'campus', 'new graduate'],
    'senior': ['experienced', 'advanced', 'expert'],
    'support': ['service', 'help desk', 'customer care'],
    'call center': ['contact center', 'bpo', 'customer support'],
    'data analyst': ['data analysis', 'analytics', 'bi analyst'],
    'product manager': ['pm', 'product owner', 'product management']
}

# ============================================================================
# DOMAIN/INDUSTRY KEYWORDS
# ============================================================================

DOMAIN_KEYWORDS = {
    'technology': [
        'software', 'developer', 'programmer', 'engineer', 'coding',
        'programming', 'tech', 'it', 'computer', 'application'
    ],
    'sales': [
        'sales', 'selling', 'revenue', 'quota', 'pipeline', 'leads',
        'conversion', 'account', 'territory'
    ],
    'finance': [
        'finance', 'financial', 'banking', 'accounting', 'audit',
        'tax', 'treasury', 'investment'
    ],
    'marketing': [
        'marketing', 'digital', 'brand', 'campaign', 'advertising',
        'seo', 'content', 'social media'
    ],
    'hr': [
        'hr', 'human resources', 'recruitment', 'talent', 'hiring',
        'onboarding', 'employee'
    ],
    'customer_service': [
        'customer', 'support', 'service', 'call center', 'contact center',
        'helpdesk', 'client'
    ],
    'operations': [
        'operations', 'logistics', 'supply chain', 'warehouse',
        'inventory', 'procurement'
    ],
    'management': [
        'manager', 'management', 'leadership', 'supervisor', 'team lead',
        'director', 'executive'
    ]
}
