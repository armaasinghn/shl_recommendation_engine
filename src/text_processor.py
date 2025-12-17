"""
Text Processor Module
=====================
Handles all text preprocessing, tokenization, and normalization.

Functions:
    - tokenize: Split text into tokens
    - preprocess: Clean and normalize text
    - expand_with_synonyms: Add synonyms for better recall
    - compute_tfidf: Calculate TF-IDF vectors
"""

import re
import math
from typing import List, Dict, Set, Tuple
from collections import Counter

from .config import STOPWORDS, SYNONYMS


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase words.
    
    Args:
        text: Input text string
        
    Returns:
        List of lowercase tokens
        
    Example:
        >>> tokenize("Hello World! Python-3")
        ['hello', 'world', 'python-3']
    """
    if not text:
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace special characters but keep hyphens, dots, and alphanumerics
    text = re.sub(r'[^a-z0-9\s\-\.\+\#]', ' ', text)
    
    # Split on whitespace
    tokens = text.split()
    
    # Filter out very short tokens
    tokens = [t.strip('.-') for t in tokens if len(t.strip('.-')) > 1]
    
    return tokens


def preprocess(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Preprocess text: tokenize and optionally remove stopwords.
    
    Args:
        text: Input text string
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        List of preprocessed tokens
        
    Example:
        >>> preprocess("I am looking for a Java developer")
        ['java', 'developer']
    """
    tokens = tokenize(text)
    
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    
    return tokens


def expand_with_synonyms(tokens: List[str]) -> List[str]:
    """
    Expand token list with synonyms for better recall.
    
    Args:
        tokens: List of tokens
        
    Returns:
        Expanded list including synonyms
        
    Example:
        >>> expand_with_synonyms(['developer', 'java'])
        ['developer', 'java', 'programmer', 'engineer', 'coder', ...]
    """
    expanded = set(tokens)
    
    for token in tokens:
        if token in SYNONYMS:
            for synonym in SYNONYMS[token]:
                # Add individual words from multi-word synonyms
                expanded.update(synonym.lower().split())
    
    return list(expanded)


def extract_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """
    Extract n-grams from token list.
    
    Args:
        tokens: List of tokens
        n: N-gram size
        
    Returns:
        List of n-grams as strings
        
    Example:
        >>> extract_ngrams(['java', 'developer', 'needed'], 2)
        ['java developer', 'developer needed']
    """
    if len(tokens) < n:
        return []
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.append(ngram)
    
    return ngrams


class TFIDFVectorizer:
    """
    Simple TF-IDF vectorizer for document similarity.
    
    This is a lightweight implementation that doesn't require sklearn,
    making deployment easier.
    """
    
    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.documents: List[List[str]] = []
    
    def fit(self, documents: List[str]) -> 'TFIDFVectorizer':
        """
        Fit the vectorizer on a corpus of documents.
        
        Args:
            documents: List of document strings
            
        Returns:
            Self for chaining
        """
        # Tokenize all documents
        self.documents = [preprocess(doc) for doc in documents]
        
        # Build vocabulary
        all_tokens = set()
        for doc_tokens in self.documents:
            all_tokens.update(doc_tokens)
        
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        
        # Compute IDF
        num_docs = len(self.documents)
        doc_freq = Counter()
        
        for doc_tokens in self.documents:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        # IDF with smoothing
        for token in self.vocabulary:
            df = doc_freq.get(token, 0)
            self.idf[token] = math.log((num_docs + 1) / (df + 1)) + 1
        
        return self
    
    def transform(self, text: str) -> Dict[str, float]:
        """
        Transform a text into TF-IDF vector.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping tokens to TF-IDF scores
        """
        tokens = preprocess(text)
        tf = Counter(tokens)
        
        # Compute TF-IDF
        tfidf = {}
        for token, count in tf.items():
            if token in self.vocabulary:
                # Normalized TF * IDF
                tf_score = count / len(tokens) if tokens else 0
                tfidf[token] = tf_score * self.idf.get(token, 1)
        
        return tfidf
    
    def compute_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Compute cosine similarity between two TF-IDF vectors.
        
        Args:
            vec1: First TF-IDF vector
            vec2: Second TF-IDF vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if not vec1 or not vec2:
            return 0.0
        
        # Get common tokens
        common_tokens = set(vec1.keys()) & set(vec2.keys())
        
        if not common_tokens:
            return 0.0
        
        # Compute dot product
        dot_product = sum(vec1[t] * vec2[t] for t in common_tokens)
        
        # Compute magnitudes
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)


def compute_keyword_match_score(query_tokens: List[str], 
                                 assessment_keywords: List[str],
                                 assessment_text: str) -> float:
    """
    Compute keyword matching score between query and assessment.
    
    Args:
        query_tokens: Preprocessed query tokens
        assessment_keywords: Assessment's keywords
        assessment_text: Full searchable text of assessment
        
    Returns:
        Match score (0-1)
    """
    if not query_tokens:
        return 0.0
    
    query_set = set(query_tokens)
    keyword_set = set(k.lower() for k in assessment_keywords)
    assessment_tokens = set(preprocess(assessment_text))
    
    # Direct keyword matches (weighted higher)
    keyword_matches = len(query_set & keyword_set)
    
    # Text matches
    text_matches = len(query_set & assessment_tokens)
    
    # Compute weighted score
    max_possible = len(query_set)
    if max_possible == 0:
        return 0.0
    
    # Keywords are worth more than general text matches
    score = (keyword_matches * 2 + text_matches) / (max_possible * 3)
    
    return min(1.0, score)


def highlight_matches(text: str, query_tokens: List[str]) -> str:
    """
    Highlight matching terms in text (for UI display).
    
    Args:
        text: Original text
        query_tokens: Query tokens to highlight
        
    Returns:
        Text with HTML highlighting
    """
    if not query_tokens:
        return text
    
    result = text
    for token in query_tokens:
        pattern = re.compile(re.escape(token), re.IGNORECASE)
        result = pattern.sub(f'<mark>{token}</mark>', result)
    
    return result
