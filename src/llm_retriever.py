"""
LLM-Based Retriever Module
"""

import json
from typing import List, Optional

from .config import ASSESSMENTS_FILE
from .models import Assessment, RecommendationResult


class LLMRetriever:
    
    def __init__(self, assessments_file: Optional[str] = None):
        self.assessments_file = assessments_file or ASSESSMENTS_FILE
        self.assessments: List[Assessment] = []
        self.structured_table = ""
        self._load_and_structure()
    
    def _load_and_structure(self):
        with open(self.assessments_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assessments = [Assessment.from_dict(item) for item in data]
        self.structured_table = self._create_assessment_table()
    
    def _create_assessment_table(self) -> str:
        lines = ["# SHL Assessment Catalog", "", "| ID | Name | Duration | Types | Description | Keywords |", "|---|---|---|---|---|---|"]
        
        for idx, assessment in enumerate(self.assessments):
            test_types = ", ".join(assessment.test_type_full[:3])
            keywords = ", ".join(assessment.keywords[:8])
            desc = assessment.description[:100].replace("|", "-")
            row = f"| {idx} | {assessment.name} | {assessment.duration}min | {test_types} | {desc} | {keywords} |"
            lines.append(row)
        
        return "\n".join(lines)
    
    async def retrieve(self, query: str, num_results: int = 10) -> List[RecommendationResult]:
        prompt = self._build_prompt(query, num_results)
        
        try:
            response = await self._call_claude_api(prompt)
            results = self._parse_response(response, num_results)
            return results
        except Exception as e:
            print(f"⚠️  LLM retrieval failed: {e}")
            return self._fallback_retrieve(num_results)
    
    def _build_prompt(self, query: str, num_results: int) -> str:
        return f"""You are an expert HR assessment consultant. Analyze this job requirement and recommend the most relevant SHL assessments.

JOB REQUIREMENT:
{query}

AVAILABLE ASSESSMENTS:
{self.structured_table}

TASK:
1. Analyze the job requirements (technical skills, soft skills, experience level, role type)
2. Select the {num_results} MOST relevant assessments by ID
3. Rank them by relevance (most relevant first)

Consider:
- Technical skills (programming languages, tools, frameworks)
- Soft skills (communication, teamwork, leadership, customer service)
- Job level (entry/junior, mid-level, senior, manager, executive)
- Role type (developer, analyst, sales, support, etc.)
- Assessment duration appropriateness

Respond with ONLY a JSON array of assessment IDs in ranked order. No explanation needed.
Example: [5, 12, 3, 18, 7, 22, 1, 15, 9, 11]

Your response (exactly {num_results} IDs):"""
    
    async def _call_claude_api(self, prompt: str) -> str:
        import aiohttp
        
        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers={"Content-Type": "application/json"}) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"API error {resp.status}: {text[:200]}")
                
                data = await resp.json()
                content = data.get("content", [])
                text = ""
                for item in content:
                    if item.get("type") == "text":
                        text += item.get("text", "")
                
                return text
    
    def _parse_response(self, response: str, num_results: int) -> List[RecommendationResult]:
        try:
            response = response.strip()
            if "```" in response:
                lines = response.split("\n")
                response = "\n".join([l for l in lines if not l.strip().startswith("```")])
            response = response.strip()
            
            indices = json.loads(response)
            results = []
            for rank, idx in enumerate(indices[:num_results]):
                if 0 <= idx < len(self.assessments):
                    assessment = self.assessments[idx]
                    score = 1.0 - (rank * 0.08)
                    result = RecommendationResult(assessment=assessment, score=max(0.2, score), match_reasons=[f"LLM rank #{rank+1}"])
                    results.append(result)
            
            return results
        except Exception as e:
            print(f"⚠️  Parse error: {e}")
            print(f"Response was: {response[:200]}")
            return self._fallback_retrieve(num_results)
    
    def _fallback_retrieve(self, num_results: int) -> List[RecommendationResult]:
        results = []
        for assessment in self.assessments[:num_results]:
            result = RecommendationResult(assessment=assessment, score=0.3, match_reasons=["Fallback"])
            results.append(result)
        return results


class HybridRetriever:
    
    def __init__(self, assessments_file: Optional[str] = None):
        from .search_engine import AssessmentSearchEngine
        
        self.llm_retriever = LLMRetriever(assessments_file)
        self.tfidf_retriever = AssessmentSearchEngine()
        self.tfidf_retriever.load_assessments(assessments_file)
        self.tfidf_retriever.fit()
    
    async def retrieve(self, query: str, num_results: int = 10, use_llm: bool = True, blend: bool = False) -> List[RecommendationResult]:
        if not use_llm:
            from .query_analyzer import analyze_query
            query_info = analyze_query(query)
            return self.tfidf_retriever.search(query_info, num_results)
        
        if blend:
            llm_results = await self.llm_retriever.retrieve(query, num_results * 2)
            from .query_analyzer import analyze_query
            query_info = analyze_query(query)
            tfidf_results = self.tfidf_retriever.search(query_info, num_results * 2)
            return self._blend_results(llm_results, tfidf_results, num_results)
        
        return await self.llm_retriever.retrieve(query, num_results)
    
    def _blend_results(self, llm_results: List[RecommendationResult], tfidf_results: List[RecommendationResult], num_results: int) -> List[RecommendationResult]:
        scores = {}
        url_to_assessment = {}
        
        for result in llm_results:
            url = result.assessment.url
            scores[url] = scores.get(url, 0) + (result.score * 0.7)
            url_to_assessment[url] = result.assessment
        
        for result in tfidf_results:
            url = result.assessment.url
            scores[url] = scores.get(url, 0) + (result.score * 0.3)
            url_to_assessment[url] = result.assessment
        
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        final = []
        for url, score in sorted_items[:num_results]:
            result = RecommendationResult(assessment=url_to_assessment[url], score=score, match_reasons=["Hybrid: LLM + TF-IDF"])
            final.append(result)
        
        return final
