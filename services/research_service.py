"""
Research Service for DAMN BOT AI System
Advanced research capabilities with multi-source analysis
"""
from typing import Set, Dict, List, Any
import asyncio
import json
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
import scholarly
import arxiv
from core.config import get_settings
from services.web_search import WebSearchService
from services.llm_orchestrator import LLMOrchestrator
from cachetools import TTLCache

logger = logging.getLogger(__name__)

class ResearchService:
    """Advanced research service with multi-source capabilities"""
    
    def __init__(self, settings, web_search_service, llm_orchestrator):
        self.settings = settings
        self.web_search = web_search_service
        self.llm_orchestrator = llm_orchestrator
        self.session = None
        self.content_cache = TTLCache(maxsize=1000, ttl=3600)
    async def _extract_content(self, url: str) -> str:
        """Cached content extraction"""
        if url in self.content_cache:
            return self.content_cache[url]
            
        content = await self._fetch_content(url)
        self.content_cache[url] = content
        return content
    def _filter_results(self, results: List[Dict], min_relevance: float = 0.3) -> List[Dict]:
        """Filter out low-quality results"""
        return [
            r for r in results
            if r.get('relevance_score', 0) > min_relevance
            and not self._is_low_quality(r)
        ]

    def _is_low_quality(self, result: Dict) -> bool:
        """Detect low-quality results"""
        url = result.get('url', '')
        title = result.get('title', '')
        
        # Filter out social media and forums for time-sensitive queries
        if any(d in url for d in ["reddit.com", "twitter.com", "facebook.com"]):
            return True
            
        # Filter out very short titles
        if len(title.split()) < 3:
            return True
        
        return False  
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def conduct_research(
        self,
        topic: str,
        depth: str = "medium",
        sources: Optional[List[str]] = None,
        max_sources: int = 10
    ) -> Dict[str, Any]:
        """
        Conduct comprehensive research on a topic
        
        Args:
            topic: Research topic
            depth: Research depth (shallow, medium, deep)
            sources: Specific sources to include
            max_sources: Maximum number of sources
            
        Returns:
            Research results with analysis
        """
        try:
            # if "time" in topic.lower() and ("now" in topic.lower() or "current" in topic.lower()):
            #     return await self._handle_time_query(topic)
            logger.info(f"Starting research on topic: {topic}")
            
            # Initialize research data
            research_data = {
                "topic": topic,
                "depth": depth,
                "timestamp": datetime.now().isoformat(),
                "sources": [],
                "analysis": {},
                "summary": "",
                "key_findings": [],
                "recommendations": [],
                "confidence_score": 0.0
            }
            base_results = await self.web_search.search(topic, max_sources * 2)
            base_urls = {r['url'] for r in base_results.get("results", [])}
            
            # Gather sources based on depth
            if depth == "shallow":
                sources_data = await self._shallow_research(topic, max_sources, base_results)
            elif depth == "medium":
                sources_data = await self._medium_research(topic, max_sources, base_results, base_urls)
            else:  # deep
                sources_data = await self._deep_research(topic, max_sources, base_results, base_urls)
            
            research_data["sources"] = sources_data
            
            # Analyze gathered information
            analysis = await self._analyze_research_data(topic, sources_data)
            research_data["analysis"] = analysis
            
            # Generate comprehensive summary
            summary = await self._generate_research_summary(topic, sources_data, analysis)
            research_data["summary"] = summary["content"]
            research_data["key_findings"] = summary["key_findings"]
            research_data["recommendations"] = summary["recommendations"]
            research_data["confidence_score"] = summary["confidence_score"]
            
            logger.info(f"Research completed for topic: {topic}")
            return research_data
            
        except Exception as e:
            logger.error(f"Research failed: {str(e)}")
            return {
                "topic": topic,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    # async def _handle_time_query(self, topic: str) -> Dict[str, Any]:
    #     """Special handling for time-related queries"""
    #     # Extract location using LLM
    #     location = await self.llm_orchestrator.extract_entity(topic, "GPE")
        
    #     # Get current time using world time API
    #     time_data = await self._get_current_time(location or "Tokyo")
        
    #     return {
    #         "topic": topic,
    #         "summary": f"Current time in {location}: {time_data['datetime']}",
    #         "key_findings": [time_data],
    #         "sources": [],
    #         "confidence_score": 1.0
    #     }
    
    async def _shallow_research(self, topic: str, max_sources: int, base_results: Dict) -> List[Dict[str, Any]]:
        sources = []
        for result in base_results.get("results", [])[:max_sources]:
            source = {
                "type": "web",
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "content": await self._extract_content(result.get("url", "")),
                "relevance_score": result.get("relevance_score", 0.5),
                "timestamp": datetime.now().isoformat()
            }
            sources.append(source)
        return sources

    async def _medium_research(self, topic: str, max_sources: int, base_results: Dict, base_urls: Set) -> List[Dict[str, Any]]:
        sources = []
        content_cache = {}
        
        # Process web results
        for result in base_results.get("results", [])[:max_sources//2]:
            url = result.get("url", "")
            content = await self._extract_content(url)
            content_cache[url] = content
            
            sources.append({
                "type": "web",
                "title": result.get("title", ""),
                "url": url,
                "snippet": result.get("snippet", ""),
                "content": content,
                "relevance_score": result.get("relevance_score", 0.5),
                "timestamp": datetime.now().isoformat()
            })
        
        # Academic papers (filtered)
        academic_count = max_sources // 3
        academic_results = await self._search_academic_papers(topic, academic_count, base_urls)
        sources.extend(academic_results)
        
        # News articles (from base results)
        news_count = max_sources - len(sources)
        for result in base_results.get("results", []):
            if news_count <= 0:
                break
                
            url = result.get("url", "")
            if self._is_news_source(url) and url not in content_cache:
                content = await self._extract_content(url)
                content_cache[url] = content
                
                sources.append({
                    "type": "news",
                    "title": result.get("title", ""),
                    "url": url,
                    "snippet": result.get("snippet", ""),
                    "content": content,
                    "relevance_score": result.get("relevance_score", 0.6),
                    "timestamp": datetime.now().isoformat()
                })
                news_count -= 1
                
        return sources

    async def _deep_research(self, topic: str, max_sources: int, base_results: Dict, base_urls: Set) -> List[Dict[str, Any]]:
        sources = []
        content_cache = {}
        categories = {
            "web": int(max_sources * 0.4),
            "academic": int(max_sources * 0.3),
            "news": int(max_sources * 0.15),
            "book": int(max_sources * 0.1),
            "expert": int(max_sources * 0.05)
        }
        
        # Process base results
        for result in base_results.get("results", []):
            url = result.get("url", "")
            content = await self._extract_content(url)
            content_cache[url] = content
            
            # Classify and assign to categories
            source_type = self._classify_source(url, result.get("title", ""))
            if categories.get(source_type, 0) > 0:
                sources.append({
                    "type": source_type,
                    "title": result.get("title", ""),
                    "url": url,
                    "snippet": result.get("snippet", ""),
                    "content": content,
                    "relevance_score": result.get("relevance_score", 0.5),
                    "timestamp": datetime.now().isoformat()
                })
                categories[source_type] -= 1
        
        # Fill remaining academic slots
        if categories["academic"] > 0:
            academic_results = await self._search_academic_papers(topic, categories["academic"], base_urls)
            sources.extend(academic_results)
        
        # Fill other categories if needed
        for cat, count in categories.items():
            if count > 0 and cat != "academic":
                # Implement specific searches only if necessary
                pass
                
        return sources[:max_sources]
    def _classify_source(self, url: str, title: str) -> str:
        """Classify source based on URL and title patterns"""
        url = url.lower()
        title = title.lower()
        
        if any(kw in url or kw in title for kw in ["arxiv", "research", "paper", "academia"]):
            return "academic"
        if any(kw in url or kw in title for kw in ["news", "reuters", "apnews", "bbc", "cnn"]):
            return "news"
        if any(kw in url or kw in title for kw in ["book", "publication", "library", "isbn"]):
            return "book"
        if any(kw in url or kw in title for kw in ["expert", "interview", "opinion", "analysis"]):
            return "expert"
        return "web"

    def _is_news_source(self, url: str) -> bool:
        return any(news_domain in url for news_domain in [
            "reuters.com", "apnews.com", "bbc.com", "cnn.com", 
            "nytimes.com", "washingtonpost.com", "theguardian.com"
        ])

    async def _search_academic_papers(self, topic: str, max_results: int, existing_urls: Set) -> List[Dict[str, Any]]:
        """Optimized academic paper search with filtering"""
        try:
            # Skip academic search for time-sensitive queries
            if any(kw in topic.lower() for kw in ["time", "current", "now", "today"]):
                return []
                
            sources = []
            search = arxiv.Search(
                query=topic,
                max_results=max_results * 2,  # Get extra to filter
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in search.results():
                if len(sources) >= max_results:
                    break
                    
                # Skip if paper URL already in base results
                if paper.entry_id in existing_urls:
                    continue
                    
                sources.append({
                    "type": "academic",
                    "title": paper.title,
                    "url": paper.entry_id,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary,
                    "published": paper.published.isoformat(),
                    "categories": paper.categories,
                    "relevance_score": 0.8,
                    "timestamp": datetime.now().isoformat()
                })
            
            return sources
        except Exception as e:
            logger.error(f"Academic search failed: {str(e)}")
            return []
    
    async def _search_news(self, topic: str, max_results: int) -> List[Dict[str, Any]]:
        """Search for recent news articles"""
        try:
            # Use web search with news-specific query
            news_query = f"{topic} news recent"
            web_results = await self.web_search.search(news_query, max_results)
            
            sources = []
            for result in web_results.get("results", []):
                source = {
                    "type": "news",
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", ""),
                    "content": await self._extract_content(result.get("url", "")),
                    "relevance_score": result.get("relevance_score", 0.6),
                    "timestamp": datetime.now().isoformat()
                }
                sources.append(source)
            
            return sources
            
        except Exception as e:
            logger.error(f"News search failed: {str(e)}")
            return []
    
    async def _search_books(self, topic: str, max_results: int) -> List[Dict[str, Any]]:
        """Search for books and publications"""
        try:
            # Use Google Scholar for book search
            books_query = f"{topic} book publication"
            web_results = await self.web_search.search(books_query, max_results)
            
            sources = []
            for result in web_results.get("results", []):
                if any(keyword in result.get("url", "").lower() for keyword in ["book", "publication", "journal"]):
                    source = {
                        "type": "book",
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("snippet", ""),
                        "content": await self._extract_content(result.get("url", "")),
                        "relevance_score": result.get("relevance_score", 0.7),
                        "timestamp": datetime.now().isoformat()
                    }
                    sources.append(source)
            
            return sources
            
        except Exception as e:
            logger.error(f"Books search failed: {str(e)}")
            return []
    
    async def _search_expert_opinions(self, topic: str, max_results: int) -> List[Dict[str, Any]]:
        """Search for expert opinions and interviews"""
        try:
            # Search for expert opinions
            expert_query = f"{topic} expert opinion interview analysis"
            web_results = await self.web_search.search(expert_query, max_results)
            
            sources = []
            for result in web_results.get("results", []):
                if any(keyword in result.get("title", "").lower() for keyword in ["expert", "interview", "opinion", "analysis"]):
                    source = {
                        "type": "expert",
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("snippet", ""),
                        "content": await self._extract_content(result.get("url", "")),
                        "relevance_score": result.get("relevance_score", 0.75),
                        "timestamp": datetime.now().isoformat()
                    }
                    sources.append(source)
            
            return sources
            
        except Exception as e:
            logger.error(f"Expert opinions search failed: {str(e)}")
            return []
    
    async def _extract_content(self, url: str) -> str:
        """Extract content from a URL"""
        try:
            if not self.session:
                return ""
                
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text()
                    
                    # Clean up text
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    # Limit content length
                    return text[:5000] if len(text) > 5000 else text
                    
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {str(e)}")
            
        return ""
    
    async def _analyze_research_data(self, topic: str, sources: List[Dict[str, Any]], max_sources: int = 10) -> Dict[str, Any]:
        try:
            # Step 1: Build context from top sources
            sources_text = ""
            for i, source in enumerate(sources[:max_sources]):
                source_type = source.get('type', 'unknown')
                title = source.get('title', 'Untitled')
                content = source.get('content') or source.get('snippet') or ''
                content_excerpt = content.strip()[:1000] if content else '[No content available]'
                
                sources_text += (
                    f"\n\n### Source {i + 1} ({source_type})\n"
                    f"**Title**: {title}\n"
                    f"**Content Excerpt**:\n{content_excerpt}"
                )

            # Step 2: Compose the structured prompt
            prompt = (
                f"You're an expert research analyst. Analyze the following research data about **\"{topic}\"**:\n"
                f"{sources_text}\n\n"
                "Please provide a structured JSON analysis with the following keys:\n"
                "1. `main_themes`\n"
                "2. `conflicting_viewpoints`\n"
                "3. `data_quality_assessment`\n"
                "4. `source_credibility_evaluation`\n"
                "5. `knowledge_gaps`\n"
                "6. `emerging_trends`\n\n"
                "Respond ONLY in JSON format."
            )

            # Step 3: Query LLM through orchestrator
            analysis_result = await self.llm_orchestrator.generate_response(
                prompt=prompt,
                task_type="research",  # Matches orchestrator's task-specific preferences
                temperature=0.3
            )

            # Step 4: Parse JSON output
            try:
                content = analysis_result.get("text") or analysis_result.get("content", "")
                analysis = json.loads(content)
            except json.JSONDecodeError as json_err:
                logger.warning(f"Failed to parse JSON response: {json_err}")
                analysis = {"raw_analysis": content, "warning": "Could not parse structured JSON."}

            return analysis

        except Exception as e:
            logger.error(f"Research analysis failed: {str(e)}")
            return {"error": str(e)}
    async def _generate_research_summary(
        self,
        topic: str,
        sources: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive research summary"""
        try:
            # Prepare summary prompt
            sources_summary = ""
            for source in sources[:15]:  # Top 15 sources
                sources_summary += f"- {source['title']} ({source['type']})\n"
            
            prompt = f"""
            Create a comprehensive research summary for "{topic}" based on:
            
            Sources analyzed:
            {sources_summary}
            
            Analysis results:
            {json.dumps(analysis, indent=2)}
            
            Provide:
            1. Executive summary (2-3 paragraphs)
            2. Key findings (5-7 bullet points)
            3. Recommendations (3-5 actionable items)
            4. Confidence score (0.0-1.0) based on source quality and consensus
            5. Areas for further research
            
            Format as JSON with clear structure.
            """
            
            # Generate summary
            summary_result = await self.llm_orchestrator.generate_response(
                prompt=prompt,
                provider="openai",
                model="gpt-4-turbo-preview",
                temperature=0.2
            )
            def extract_json_from_text(text: str) -> str:
                    """Extracts the first JSON object found in a text string."""
                    json_blocks = re.findall(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL)
                    if json_blocks:
                        return json_blocks[0]
                    raise ValueError("No valid JSON found in LLM response.")
            raw_text = summary_result.get("content") or summary_result.get("text") or ""
            
            try:
                json_str = extract_json_from_text(raw_text)
                summary = json.loads(json_str)
            except Exception as e:
                logger.warning(f"Failed to parse LLM summary response as JSON: {e}")
                summary = {
                    "content": raw_text,
                    "key_findings": [],
                    "recommendations": [],
                    "confidence_score": 0.5,
                    "warning": "Could not parse structured JSON."
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return {
                "content": f"Summary generation failed: {str(e)}",
                "key_findings": [],
                "recommendations": [],
                "confidence_score": 0.0
            }

    
    async def fact_check(self, claim: str) -> Dict[str, Any]:
        """Fact-check a specific claim"""
        try:
            # Search for evidence
            search_results = await self.web_search.search(f"fact check {claim}", 10)
            
            # Analyze evidence
            evidence_text = ""
            for result in search_results.get("results", []):
                evidence_text += f"{result.get('title', '')}: {result.get('snippet', '')}\n"
            
            prompt = f"""
            Fact-check the following claim: "{claim}"
            
            Evidence found:
            {evidence_text}
            
            Provide:
            1. Verdict (True, False, Partially True, Unverified)
            2. Confidence level (0.0-1.0)
            3. Supporting evidence
            4. Contradicting evidence
            5. Sources used
            
            Format as JSON.
            """
            
            result = await self.llm_orchestrator.generate_response(
                prompt=prompt,
                provider="openai",
                model="gpt-4-turbo-preview",
                temperature=0.1
            )
            
            try:
                return json.loads(result["content"])
            except:
                return {"raw_result": result["content"]}
                
        except Exception as e:
            logger.error(f"Fact checking failed: {str(e)}")
            return {"error": str(e)}
