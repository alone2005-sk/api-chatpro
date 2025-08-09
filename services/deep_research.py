"""
Deep Research Service - Advanced research capabilities with multiple sources
"""

import asyncio
import aiohttp
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import quote_plus, urljoin
import hashlib

from core.config import get_settings
from core.database import DatabaseManager
from core.logger import get_logger
from services.llm_orchestrator import LLMOrchestrator

logger = get_logger(__name__)

class DeepResearchService:
    """Advanced research service with multiple sources and deep analysis"""
    
    def __init__(self, llm_orchestrator: LLMOrchestrator, db_manager: DatabaseManager):
        self.settings = get_settings()
        self.llm_orchestrator = llm_orchestrator
        self.db_manager = db_manager
        
        # Research sources configuration
        self.search_providers = {
            "serper": {
                "enabled": bool(self.settings.SERPER_API_KEY),
                "api_key": self.settings.SERPER_API_KEY,
                "endpoint": "https://google.serper.dev/search",
                "priority": 1,
                "free_tier": True
            },
            "tavily": {
                "enabled": bool(self.settings.TAVILY_API_KEY),
                "api_key": self.settings.TAVILY_API_KEY,
                "endpoint": "https://api.tavily.com/search",
                "priority": 2,
                "free_tier": True
            },
            "duckduckgo": {
                "enabled": True,
                "endpoint": "https://api.duckduckgo.com/",
                "priority": 3,
                "free_tier": True
            }
        }
        
        # Research depth configurations
        self.research_configs = {
            "quick": {
                "max_sources": 5,
                "max_depth": 1,
                "analysis_level": "basic",
                "fact_checking": False,
                "cross_reference": False,
                "time_limit": 30  # seconds
            },
            "comprehensive": {
                "max_sources": 15,
                "max_depth": 2,
                "analysis_level": "detailed",
                "fact_checking": True,
                "cross_reference": True,
                "time_limit": 120  # seconds
            },
            "deep": {
                "max_sources": 30,
                "max_depth": 3,
                "analysis_level": "comprehensive",
                "fact_checking": True,
                "cross_reference": True,
                "expert_analysis": True,
                "time_limit": 300  # seconds
            }
        }
        
        # Content extractors
        self.content_extractors = {
            "web": self._extract_web_content,
            "academic": self._extract_academic_content,
            "news": self._extract_news_content,
            "social": self._extract_social_content
        }
        
        # Research cache
        self.research_cache = {}
        
        # HTTP session
        self.session = None
    
    async def initialize(self):
        """Initialize the research service"""
        try:
            logger.info("Initializing Deep Research Service...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
            
            # Test search providers
            await self._test_search_providers()
            
            logger.info("✅ Deep Research Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Deep Research Service: {str(e)}")
            raise
    
    async def conduct_research(
        self,
        query: str,
        chat_id: str,
        research_type: str = "comprehensive",
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Conduct comprehensive research on a query"""
        try:
            start_time = datetime.now()
            logger.info(f"Starting {research_type} research: {query[:100]}...")
            
            # Check cache first
            cache_key = self._generate_cache_key(query, research_type, context)
            cached_result = self._get_cached_research(cache_key)
            if cached_result:
                return cached_result
            
            # Create research session in database
            research_session = await self.db_manager.create_research_session(
                chat_id=chat_id,
                query=query,
                research_type=research_type
            )
            
            config = self.research_configs.get(research_type, self.research_configs["comprehensive"])
            
            # Phase 1: Query analysis and expansion
            expanded_queries = await self._analyze_and_expand_query(query, context)
            
            # Phase 2: Multi-source information gathering
            raw_sources = await self._gather_information_sources(
                expanded_queries, config, research_session.id
            )
            
            # Phase 3: Content extraction and processing
            processed_sources = await self._process_source_content(
                raw_sources, config
            )
            
            # Phase 4: Information synthesis and analysis
            synthesized_data = await self._synthesize_information(
                processed_sources, query, config
            )
            
            # Phase 5: Fact checking (if enabled)
            if config.get("fact_checking"):
                verified_data = await self._verify_information(synthesized_data)
            else:
                verified_data = synthesized_data
            
            # Phase 6: Generate comprehensive report
            research_report = await self._generate_research_report(
                query, verified_data, processed_sources, config
            )
            
            # Update research session
            processing_time = (datetime.now() - start_time).total_seconds()
            await self.db_manager.update_research_session(
                research_session.id,
                status="completed",
                sources_found=len(processed_sources),
                confidence_score=research_report.get("confidence_score", 0.0),
                completed_at=datetime.now(),
                results=research_report
            )
            
            result = {
                "session_id": research_session.id,
                "query": query,
                "research_type": research_type,
                "report": research_report,
                "sources": processed_sources,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            self._cache_research(cache_key, result)
            
            logger.info(f"Research completed in {processing_time:.2f}s with {len(processed_sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Research failed for query '{query}': {str(e)}")
            return {
                "error": str(e),
                "query": query,
                "research_type": research_type
            }
    
    async def _analyze_and_expand_query(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> List[str]:
        """Analyze and expand the research query"""
        try:
            expansion_prompt = f"""
            Analyze this research query and generate expanded search terms for comprehensive research:
            
            Original Query: "{query}"
            Context: {json.dumps(context or {}, indent=2)}
            
            Generate:
            1. 3-5 alternative phrasings of the main query
            2. 3-5 related subtopics to explore
            3. 3-5 specific technical terms or keywords
            4. 2-3 broader context queries
            5. 2-3 recent developments or trends related to the topic
            
            Return as JSON array of search queries, prioritized by relevance.
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=expansion_prompt,
                task_type="analytical",
                context={"query_expansion": True}
            )
            
            try:
                expanded_queries = json.loads(response.get("text", "[]"))
                if isinstance(expanded_queries, list):
                    return [query] + expanded_queries[:15]  # Original + up to 15 expanded
            except json.JSONDecodeError:
                pass
            
            # Fallback: simple keyword expansion
            return [query] + self._simple_query_expansion(query)
            
        except Exception as e:
            logger.error(f"Query expansion failed: {str(e)}")
            return [query]
    
    def _simple_query_expansion(self, query: str) -> List[str]:
        """Simple query expansion fallback"""
        expanded = []
        
        # Add "how to" variant
        if not query.lower().startswith(("how", "what", "why", "when", "where")):
            expanded.append(f"how to {query}")
        
        # Add "what is" variant
        if not query.lower().startswith("what"):
            expanded.append(f"what is {query}")
        
        # Add "best practices" variant
        expanded.append(f"{query} best practices")
        
        # Add "tutorial" variant
        expanded.append(f"{query} tutorial")
        
        # Add "examples" variant
        expanded.append(f"{query} examples")
        
        return expanded[:5]
    
    async def _gather_information_sources(
        self,
        queries: List[str],
        config: Dict[str, Any],
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Gather information from multiple sources"""
        try:
            all_sources = []
            max_sources = config["max_sources"]
            sources_per_query = max(1, max_sources // len(queries))
            
            # Search with each query
            search_tasks = []
            for query in queries[:5]:  # Limit to 5 queries to avoid rate limits
                for provider_name, provider_config in self.search_providers.items():
                    if provider_config["enabled"]:
                        task = self._search_with_provider(
                            provider_name, provider_config, query, sources_per_query
                        )
                        search_tasks.append(task)
            
            # Execute searches concurrently
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Process results
            for result in search_results:
                if isinstance(result, Exception):
                    logger.warning(f"Search task failed: {str(result)}")
                    continue
                
                if isinstance(result, list):
                    all_sources.extend(result)
            
            # Remove duplicates and rank sources
            unique_sources = self._deduplicate_sources(all_sources)
            ranked_sources = self._rank_sources(unique_sources, queries[0])
            
            # Save sources to database
            for source in ranked_sources[:max_sources]:
                await self.db_manager.save_research_source(
                    session_id=session_id,
                    title=source.get("title", ""),
                    url=source.get("url", ""),
                    content=source.get("snippet", ""),
                    source_type=source.get("type", "web"),
                    relevance_score=source.get("relevance_score", 0.0),
                    credibility_score=source.get("credibility_score", 0.0)
                )
            
            return ranked_sources[:max_sources]
            
        except Exception as e:
            logger.error(f"Information gathering failed: {str(e)}")
            return []
    
    async def _search_with_provider(
        self,
        provider_name: str,
        provider_config: Dict[str, Any],
        query: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Search with specific provider"""
        try:
            if provider_name == "serper":
                return await self._search_serper(provider_config, query, max_results)
            elif provider_name == "tavily":
                return await self._search_tavily(provider_config, query, max_results)
            elif provider_name == "duckduckgo":
                return await self._search_duckduckgo(query, max_results)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Search with {provider_name} failed: {str(e)}")
            return []
    
    async def _search_serper(
        self,
        config: Dict[str, Any],
        query: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Search using Serper API"""
        try:
            headers = {
                "X-API-KEY": config["api_key"],
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": query,
                "num": min(max_results, 10)
            }
            
            async with self.session.post(
                config["endpoint"],
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    sources = []
                    
                    for result in data.get("organic", []):
                        sources.append({
                            "title": result.get("title", ""),
                            "url": result.get("link", ""),
                            "snippet": result.get("snippet", ""),
                            "type": "web",
                            "provider": "serper",
                            "relevance_score": 0.8,  # Default high relevance
                            "credibility_score": self._calculate_credibility_score(result.get("link", ""))
                        })
                    
                    return sources
                else:
                    logger.warning(f"Serper API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Serper search failed: {str(e)}")
            return []
    
    async def _search_tavily(
        self,
        config: Dict[str, Any],
        query: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Search using Tavily API"""
        try:
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": query,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": False,
                "max_results": min(max_results, 10)
            }
            
            async with self.session.post(
                config["endpoint"],
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    sources = []
                    
                    for result in data.get("results", []):
                        sources.append({
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "snippet": result.get("content", ""),
                            "type": "web",
                            "provider": "tavily",
                            "relevance_score": result.get("score", 0.5),
                            "credibility_score": self._calculate_credibility_score(result.get("url", ""))
                        })
                    
                    return sources
                else:
                    logger.warning(f"Tavily API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Tavily search failed: {str(e)}")
            return []
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (free, no API key required)"""
        try:
            # DuckDuckGo Instant Answer API
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            async with self.session.get(
                "https://api.duckduckgo.com/",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    sources = []
                    
                    # Extract from related topics
                    for topic in data.get("RelatedTopics", [])[:max_results]:
                        if isinstance(topic, dict) and "Text" in topic:
                            sources.append({
                                "title": topic.get("Text", "")[:100] + "...",
                                "url": topic.get("FirstURL", ""),
                                "snippet": topic.get("Text", ""),
                                "type": "web",
                                "provider": "duckduckgo",
                                "relevance_score": 0.6,
                                "credibility_score": self._calculate_credibility_score(topic.get("FirstURL", ""))
                            })
                    
                    return sources
                else:
                    logger.warning(f"DuckDuckGo API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            return []
    
    def _calculate_credibility_score(self, url: str) -> float:
        """Calculate credibility score based on URL"""
        if not url:
            return 0.3
        
        url_lower = url.lower()
        
        # High credibility domains
        high_credibility = [
            ".edu", ".gov", ".org",
            "wikipedia.org", "stackoverflow.com", "github.com",
            "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "scholar.google.com",
            "nature.com", "science.org", "ieee.org"
        ]
        
        # Medium credibility domains
        medium_credibility = [
            "medium.com", "dev.to", "hackernoon.com",
            "techcrunch.com", "wired.com", "arstechnica.com"
        ]
        
        # Check for high credibility
        for domain in high_credibility:
            if domain in url_lower:
                return 0.9
        
        # Check for medium credibility
        for domain in medium_credibility:
            if domain in url_lower:
                return 0.7
        
        # Default credibility
        return 0.5
    
    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate sources"""
        seen_urls = set()
        unique_sources = []
        
        for source in sources:
            url = source.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
        
        return unique_sources
    
    def _rank_sources(self, sources: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
        """Rank sources by relevance and credibility"""
        query_words = set(original_query.lower().split())
        
        for source in sources:
            # Calculate relevance based on title and snippet
            title_words = set(source.get("title", "").lower().split())
            snippet_words = set(source.get("snippet", "").lower().split())
            
            title_overlap = len(query_words.intersection(title_words))
            snippet_overlap = len(query_words.intersection(snippet_words))
            
            # Combine relevance factors
            relevance = (title_overlap * 2 + snippet_overlap) / len(query_words)
            source["relevance_score"] = min(relevance, 1.0)
            
            # Calculate final score
            credibility = source.get("credibility_score", 0.5)
            source["final_score"] = (relevance * 0.6 + credibility * 0.4)
        
        # Sort by final score
        return sorted(sources, key=lambda x: x.get("final_score", 0), reverse=True)
    
    async def _process_source_content(
        self,
        sources: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process and extract content from sources"""
        try:
            processed_sources = []
            
            for source in sources:
                try:
                    # Extract full content if needed
                    if config.get("analysis_level") in ["detailed", "comprehensive"]:
                        full_content = await self._extract_full_content(source)
                        source["full_content"] = full_content
                    
                    # Analyze content quality
                    quality_score = self._analyze_content_quality(source)
                    source["quality_score"] = quality_score
                    
                    # Extract key information
                    key_info = await self._extract_key_information(source)
                    source["key_information"] = key_info
                    
                    processed_sources.append(source)
                    
                except Exception as e:
                    logger.warning(f"Failed to process source {source.get('url', '')}: {str(e)}")
                    continue
            
            return processed_sources
            
        except Exception as e:
            logger.error(f"Source content processing failed: {str(e)}")
            return sources
    
    async def _extract_full_content(self, source: Dict[str, Any]) -> str:
        """Extract full content from source URL"""
        try:
            url = source.get("url", "")
            if not url:
                return source.get("snippet", "")
            
            # Simple content extraction (in production, use proper web scraping)
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html_content = await response.text()
                    # Basic HTML tag removal
                    import re
                    text_content = re.sub(r'<[^>]+>', '', html_content)
                    # Clean up whitespace
                    text_content = re.sub(r'\s+', ' ', text_content).strip()
                    return text_content[:2000]  # Limit content length
                else:
                    return source.get("snippet", "")
                    
        except Exception as e:
            logger.warning(f"Content extraction failed for {source.get('url', '')}: {str(e)}")
            return source.get("snippet", "")
    
    def _analyze_content_quality(self, source: Dict[str, Any]) -> float:
        """Analyze content quality"""
        quality_score = 0.5  # Base score
        
        # Check content length
        content = source.get("snippet", "") + source.get("full_content", "")
        if len(content) > 200:
            quality_score += 0.1
        if len(content) > 500:
            quality_score += 0.1
        
        # Check for structured content
        if any(marker in content.lower() for marker in ["1.", "2.", "•", "-", "step"]):
            quality_score += 0.1
        
        # Check for technical terms
        technical_terms = ["api", "algorithm", "framework", "implementation", "solution"]
        if any(term in content.lower() for term in technical_terms):
            quality_score += 0.1
        
        # Check credibility score
        quality_score += source.get("credibility_score", 0.5) * 0.2
        
        return min(quality_score, 1.0)
    
    async def _extract_key_information(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information from source"""
        try:
            content = source.get("snippet", "") + " " + source.get("full_content", "")
            
            extraction_prompt = f"""
            Extract key information from this source:
            
            Title: {source.get('title', '')}
            URL: {source.get('url', '')}
            Content: {content[:1000]}
            
            Extract:
            1. Main topic/subject
            2. Key facts or statistics
            3. Important concepts or terms
            4. Actionable insights
            5. Credibility indicators (author, date, citations)
            
            Return as JSON with structured information.
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=extraction_prompt,
                task_type="analytical",
                context={"information_extraction": True}
            )
            
            try:
                key_info = json.loads(response.get("text", "{}"))
                return key_info
            except json.JSONDecodeError:
                return {"raw_analysis": response.get("text", "")}
                
        except Exception as e:
            logger.error(f"Key information extraction failed: {str(e)}")
            return {}
    
    async def _synthesize_information(
        self,
        sources: List[Dict[str, Any]],
        original_query: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize information from all sources"""
        try:
            # Prepare source summaries
            source_summaries = []
            for i, source in enumerate(sources[:10]):  # Limit to top 10 sources
                summary = {
                    "index": i + 1,
                    "title": source.get("title", ""),
                    "url": source.get("url", ""),
                    "key_points": source.get("snippet", "")[:200],
                    "credibility": source.get("credibility_score", 0.5),
                    "relevance": source.get("relevance_score", 0.5)
                }
                source_summaries.append(summary)
            
            synthesis_prompt = f"""
            Synthesize comprehensive research findings for this query:
            
            Original Query: "{original_query}"
            Research Level: {config.get('analysis_level', 'detailed')}
            
            Sources ({len(source_summaries)} total):
            {json.dumps(source_summaries, indent=2)}
            
            Create a comprehensive synthesis that includes:
            1. Executive Summary (2-3 sentences)
            2. Key Findings (5-8 main points)
            3. Supporting Evidence (with source references)
            4. Conflicting Information (if any)
            5. Knowledge Gaps (areas needing more research)
            6. Confidence Assessment (high/medium/low with reasoning)
            7. Actionable Insights
            8. Recommendations for further research
            
            Return as structured JSON with clear sections.
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=synthesis_prompt,
                task_type="analytical",
                context={"research_synthesis": True},
                max_tokens=2000
            )
            
            try:
                synthesis = json.loads(response.get("text", "{}"))
                
                # Add metadata
                synthesis["synthesis_metadata"] = {
                    "sources_analyzed": len(sources),
                    "synthesis_model": response.get("model", "unknown"),
                    "synthesis_time": datetime.now().isoformat(),
                    "confidence_factors": self._calculate_confidence_factors(sources)
                }
                
                return synthesis
                
            except json.JSONDecodeError:
                return {
                    "raw_synthesis": response.get("text", ""),
                    "synthesis_metadata": {
                        "sources_analyzed": len(sources),
                        "parsing_error": True
                    }
                }
                
        except Exception as e:
            logger.error(f"Information synthesis failed: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_confidence_factors(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence factors for the research"""
        if not sources:
            return {"overall_confidence": 0.1, "factors": []}
        
        # Calculate various confidence factors
        avg_credibility = sum(s.get("credibility_score", 0.5) for s in sources) / len(sources)
        avg_relevance = sum(s.get("relevance_score", 0.5) for s in sources) / len(sources)
        avg_quality = sum(s.get("quality_score", 0.5) for s in sources) / len(sources)
        
        # Source diversity
        providers = set(s.get("provider", "unknown") for s in sources)
        diversity_score = min(len(providers) / 3, 1.0)  # Max score when 3+ providers
        
        # Overall confidence
        overall_confidence = (
            avg_credibility * 0.3 +
            avg_relevance * 0.3 +
            avg_quality * 0.2 +
            diversity_score * 0.2
        )
        
        return {
            "overall_confidence": round(overall_confidence, 2),
            "factors": {
                "average_credibility": round(avg_credibility, 2),
                "average_relevance": round(avg_relevance, 2),
                "average_quality": round(avg_quality, 2),
                "source_diversity": round(diversity_score, 2),
                "total_sources": len(sources),
                "unique_providers": len(providers)
            }
        }
    
    async def _verify_information(self, synthesized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify and fact-check synthesized information"""
        try:
            key_findings = synthesized_data.get("key_findings", [])
            if not key_findings:
                return synthesized_data
            
            verification_prompt = f"""
            Fact-check and verify these research findings:
            
            Key Findings:
            {json.dumps(key_findings, indent=2)}
            
            For each finding, assess:
            1. Factual accuracy (verifiable/likely/uncertain)
            2. Potential biases or limitations
            3. Supporting evidence strength
            4. Contradictory information
            5. Confidence level (high/medium/low)
            
            Return verification results as JSON with detailed assessments.
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=verification_prompt,
                task_type="analytical",
                context={"fact_checking": True}
            )
            
            try:
                verification_results = json.loads(response.get("text", "{}"))
                synthesized_data["verification"] = verification_results
                synthesized_data["fact_checked"] = True
            except json.JSONDecodeError:
                synthesized_data["verification"] = {"raw_verification": response.get("text", "")}
                synthesized_data["fact_checked"] = False
            
            return synthesized_data
            
        except Exception as e:
            logger.error(f"Information verification failed: {str(e)}")
            synthesized_data["verification_error"] = str(e)
            return synthesized_data
    
    async def _generate_research_report(
        self,
        query: str,
        verified_data: Dict[str, Any],
        sources: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final comprehensive research report"""
        try:
            report_prompt = f"""
            Generate a comprehensive research report based on this analysis:
            
            Research Query: "{query}"
            Research Level: {config.get('analysis_level', 'detailed')}
            
            Synthesized Data:
            {json.dumps(verified_data, indent=2)[:3000]}
            
            Create a professional research report with:
            1. Executive Summary
            2. Research Methodology
            3. Key Findings (with source citations)
            4. Analysis and Insights
            5. Limitations and Caveats
            6. Conclusions
            7. Recommendations
            8. Further Research Suggestions
            
            Format as a structured report suitable for professional use.
            """
            
            response = await self.llm_orchestrator.generate_response(
                prompt=report_prompt,
                task_type="analytical",
                context={"report_generation": True},
                max_tokens=2500
            )
            
            # Calculate overall confidence score
            confidence_factors = verified_data.get("synthesis_metadata", {}).get("confidence_factors", {})
            overall_confidence = confidence_factors.get("overall_confidence", 0.5)
            
            report = {
                "report_text": response.get("text", ""),
                "confidence_score": overall_confidence,
                "methodology": {
                    "research_type": config.get("analysis_level", "detailed"),
                    "sources_analyzed": len(sources),
                    "fact_checked": verified_data.get("fact_checked", False),
                    "cross_referenced": config.get("cross_reference", False)
                },
                "quality_metrics": {
                    "source_diversity": confidence_factors.get("factors", {}).get("source_diversity", 0),
                    "average_credibility": confidence_factors.get("factors", {}).get("average_credibility", 0),
                    "coverage_completeness": min(len(sources) / config.get("max_sources", 15), 1.0)
                },
                "generated_at": datetime.now().isoformat(),
                "model_used": response.get("model", "unknown")
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Research report generation failed: {str(e)}")
            return {
                "error": str(e),
                "confidence_score": 0.1,
                "report_text": "Failed to generate research report."
            }
    
    async def _test_search_providers(self):
        """Test availability of search providers"""
        logger.info("Testing search provider availability...")
        
        for name, config in self.search_providers.items():
            try:
                if config["enabled"]:
                    # Test with a simple query
                    test_results = await self._search_with_provider(
                        name, config, "test query", 1
                    )
                    
                    if test_results:
                        logger.info(f"  {name}: ✅ Available")
                    else:
                        logger.warning(f"  {name}: ⚠️ No results returned")
                        config["enabled"] = False
                else:
                    logger.info(f"  {name}: ❌ Disabled (no API key)")
                    
            except Exception as e:
                logger.warning(f"  {name}: ❌ Test failed - {str(e)}")
                config["enabled"] = False
    
    def _generate_cache_key(
        self,
        query: str,
        research_type: str,
        context: Dict[str, Any] = None
    ) -> str:
        """Generate cache key for research results"""
        key_data = f"{query}:{research_type}:{json.dumps(context or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_research(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached research results"""
        if cache_key in self.research_cache:
            cached_data, timestamp = self.research_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.settings.RESEARCH_CACHE_TTL:
                cached_data["cached"] = True
                return cached_data
            else:
                del self.research_cache[cache_key]
        return None
    
    def _cache_research(self, cache_key: str, result: Dict[str, Any]):
        """Cache research results"""
        self.research_cache[cache_key] = (result.copy(), datetime.now())
        
        # Limit cache size
        if len(self.research_cache) > 100:
            oldest_key = min(
                self.research_cache.keys(),
                key=lambda k: self.research_cache[k][1]
            )
            del self.research_cache[oldest_key]
    
    async def get_research_stats(self) -> Dict[str, Any]:
        """Get research service statistics"""
        enabled_providers = [
            name for name, config in self.search_providers.items()
            if config["enabled"]
        ]
        
        return {
            "enabled_providers": enabled_providers,
            "total_providers": len(self.search_providers),
            "cache_size": len(self.research_cache),
            "research_configs": list(self.research_configs.keys())
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for research service"""
        enabled_providers = sum(
            1 for config in self.search_providers.values()
            if config["enabled"]
        )
        
        return {
            "status": "healthy" if enabled_providers > 0 else "degraded",
            "enabled_providers": enabled_providers,
            "total_providers": len(self.search_providers),
            "cache_size": len(self.research_cache),
            "session_active": self.session is not None and not self.session.closed
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
