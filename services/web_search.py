import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re

from urllib.parse import quote, urljoin, urlparse
import hashlib
from bs4 import BeautifulSoup
import feedparser
from core.logger import get_logger
from core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

class WebSearchService:
    """Advanced web search and scraping service"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.search_engines = {
            'google': {
                'url': 'https://www.googleapis.com/customsearch/v1',
                'requires_api_key': True
            },
            'bing': {
                'url': 'https://api.bing.microsoft.com/v7.0/search',
                'requires_api_key': True
            },
            'duckduckgo': {
                'url': 'https://api.duckduckgo.com/',
                'requires_api_key': False
            },
            'brave' : {
    'url': 'https://api.search.brave.com/res/v1/web/search',
    'requires_api_key': True
           }
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json'  # Explicitly request JSON
        }
        
        self.cache = {}
        self.cache_duration = timedelta(hours=1)

    async def initialize(self):
        """Optional startup logic."""
        pass
    async def _search_brave(self, query: str, max_results: int) -> Dict[str, Any]:
        try:
            if not hasattr(settings, 'BRAVE_API_KEY'):
                return {'results': []}
            
            headers = {
                'Accept': 'application/json',
                'X-Subscription-Token': settings.BRAVE_API_KEY
            }
            params = {'q': query, 'count': min(max_results, 20)}
            
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(self.search_engines['brave']['url'], params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        for item in data.get('web', {}).get('results', []):
                            results.append({
                                'title': item.get('title', ''),
                                'url': item.get('url', ''),
                                'snippet': item.get('description', ''),
                                'type': 'web_result',
                                'source': 'brave'
                            })
                        return {'results': results[:max_results]}
        
        except Exception as e:
            logger.error(f"Brave search failed: {str(e)}")
            return {'results': []}
    
    async def search(self, query: str, search_type: str = "web", max_results: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform web search with multiple engines"""
        try:
            cache_key = self._generate_cache_key(query, search_type, max_results, filters)
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if datetime.now() - cached_result['timestamp'] < self.cache_duration:
                    logger.info(f"Returning cached results for query: {query}")
                    return cached_result['data']
            
            results = await self._perform_search(query, search_type, max_results, filters)
            print(f" here is search result {results} ")
            # Generate a summary of results
            results['summary'] = self._generate_summary(results.get('results', []))
            
            self.cache[cache_key] = {
                'data': results,
                'timestamp': datetime.now()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            return {
                'error': f"Search failed: {str(e)}",
                'query': query,
                'results': [],
                'summary': ''
            }

    async def _perform_search(self, query: str, search_type: str, max_results: int, 
                             filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform actual search across multiple engines"""
        search_results = {
            'query': query,
            'search_type': search_type,
            'timestamp': datetime.utcnow().isoformat(),
            'results': [],
            'total_results': 0,
            'search_engines_used': []
        }
        
        engines_to_try = [
        'duckduckgo',
        'brave' if hasattr(settings, 'BRAVE_API_KEY') else None,
        'google' if hasattr(settings, 'GOOGLE_API_KEY') else None,
        'bing' if hasattr(settings, 'BING_API_KEY') else None
    ]
        
        for engine in engines_to_try:
            try:
                if engine == 'duckduckgo':
                    engine_results = await self._search_duckduckgo(query, max_results)
                elif engine == 'google' and hasattr(settings, 'GOOGLE_API_KEY'):
                    engine_results = await self._search_google(query, max_results, filters)
                elif engine == 'bing' and hasattr(settings, 'BING_API_KEY'):
                    engine_results = await self._search_bing(query, max_results, filters)
                else:
                    continue
                
                if engine_results and engine_results.get('results'):
                    search_results['results'].extend(engine_results['results'])
                    search_results['search_engines_used'].append(engine)
                    
                    if len(search_results['results']) >= max_results:
                        break
                        
            except Exception as e:
                logger.warning(f"Search engine {engine} failed: {str(e)}")
                continue
        
        search_results['results'] = self._deduplicate_results(search_results['results'])[:max_results]
        search_results['total_results'] = len(search_results['results'])
        
        if search_results['results']:
            search_results['results'] = await self._enhance_search_results(search_results['results'])
        
        return search_results

    async def _search_duckduckgo(self, query: str, max_results: int) -> Dict[str, Any]:
        try:
            headers = {'User-Agent': settings.USER_AGENT, 'Accept': 'application/json'}
            params = {'q': query, 'format': 'json', 'no_redirect': 1}
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get('https://api.duckduckgo.com/', params=params) as response:
                    content_type = response.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        data = await response.json()
                    else:
                        text = await response.text()
                        try:
                            data = json.loads(text)
                        except json.JSONDecodeError:
                            raise ValueError(f"Expected JSON but got content type {content_type}. Raw response: {text[:300]}")
                    
                    results = []
                    
                    if data.get('AbstractText'):
                        results.append({
                            'title': data.get('Heading', 'Instant Answer'),
                            'url': data.get('AbstractURL', ''),
                            'snippet': data.get('AbstractText', ''),
                            'type': 'instant_answer',
                            'source': 'duckduckgo'
                        })
                    
                    for topic in data.get('RelatedTopics', []):
                        if 'FirstURL' in topic and 'Text' in topic:
                            results.append({
                                'title': topic['Text'].split(' - ')[0],
                                'url': topic['FirstURL'],
                                'snippet': topic['Text'],
                                'type': 'related_topic',
                                'source': 'duckduckgo'
                            })
                    
                    if not results:
                        return await self._scrape_ddg_html(query, max_results)
                    
                    return {'results': results[:max_results]}
        
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            return await self._scrape_ddg_html(query, max_results)


    async def _scrape_ddg_html(self, query: str, max_results: int) -> Dict[str, Any]:
        try:
            headers = {'User-Agent': settings.USER_AGENT}
            params = {'q': query, 'kl': 'wt-wt', 'ia': 'web'}
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get('https://html.duckduckgo.com/html/', params=params) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    results = []
                    
                    for result in soup.select('.result__body'):
                        if len(results) >= max_results:
                            break
                        try:
                            title_elem = result.select_one('.result__title a')
                            snippet_elem = result.select_one('.result__snippet')
                            
                            if title_elem and snippet_elem:
                                # Extract actual URL from redirect
                                href = title_elem['href']
                                if href.startswith('//duckduckgo.com/l/?uddg='):
                                    url_match = re.search(r'uddg=([^&]+)', href)
                                    if url_match:
                                        url = unquote(url_match.group(1))
                                    else:
                                        url = href
                                else:
                                    url = href
                                
                                results.append({
                                    'title': title_elem.get_text(strip=True),
                                    'url': url,
                                    'snippet': snippet_elem.get_text(strip=True),
                                    'type': 'web_result',
                                    'source': 'duckduckgo'
                                })
                        except Exception:
                            continue
                    
                    return {'results': results}
        
        except Exception as e:
            logger.error(f"DuckDuckGo HTML scrape failed: {str(e)}")
            return {'results': []}
    

    def _generate_summary(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "🔍 No relevant results found. Try different keywords or check your API keys."
        
        summary = f"🌐 Found {len(results)} relevant results:\n"
        for i, result in enumerate(results[:5], 1):
            source = result.get('source', 'unknown').capitalize()
            result_type = result.get('type', 'result').replace('_', ' ')
            
            summary += (
                f"\n{i}. **{result.get('title', 'Untitled')}**\n"
                f"   - Source: {source} ({result_type})\n"
                f"   - {result.get('snippet', 'No description available')[:150]}...\n"
                f"   - [Source URL]({result.get('url', '')})\n"
            )
        
        if len(results) > 5:
            summary += f"\n+ {len(results)-5} more results available..."
        
        return summary

    # ... (Rest of the methods remain unchanged, included for completeness)

    async def _search_google(self, query: str, max_results: int, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Search using Google Custom Search API"""
        try:
            if not hasattr(settings, 'GOOGLE_API_KEY') or not hasattr(settings, 'GOOGLE_CSE_ID'):
                return {'results': []}
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                params = {
                    'key': settings.GOOGLE_API_KEY,
                    'cx': settings.GOOGLE_CSE_ID,
                    'q': query,
                    'num': min(max_results, 10)
                }
                
                if filters:
                    if filters.get('date_range'):
                        params['dateRestrict'] = filters['date_range']
                    if filters.get('site'):
                        params['siteSearch'] = filters['site']
                    if filters.get('file_type'):
                        params['fileType'] = filters['file_type']
                
                async with session.get(self.search_engines['google']['url'], params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        for item in data.get('items', []):
                            results.append({
                                'title': item.get('title', ''),
                                'url': item.get('link', ''),
                                'snippet': item.get('snippet', ''),
                                'type': 'web_result',
                                'source': 'google',
                                'display_link': item.get('displayLink', '')
                            })
                        
                        return {'results': results}
                    
        except Exception as e:
            logger.error(f"Google search failed: {str(e)}")
            
        return {'results': []}

    async def _search_bing(self, query: str, max_results: int, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Search using Bing Search API"""
        try:
            if not hasattr(settings, 'BING_API_KEY'):
                return {'results': []}
            
            headers = {
                **self.headers,
                'Ocp-Apim-Subscription-Key': settings.BING_API_KEY
            }
            
            async with aiohttp.ClientSession(headers=headers) as session:
                params = {
                    'q': query,
                    'count': min(max_results, 50),
                    'responseFilter': 'Webpages'
                }
                
                async with session.get(self.search_engines['bing']['url'], params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        for item in data.get('webPages', {}).get('value', []):
                            results.append({
                                'title': item.get('name', ''),
                                'url': item.get('url', ''),
                                'snippet': item.get('snippet', ''),
                                'type': 'web_result',
                                'source': 'bing',
                                'display_link': item.get('displayUrl', '')
                            })
                        
                        return {'results': results}
                    
        except Exception as e:
            logger.error(f"Bing search failed: {str(e)}")
            
        return {'results': []}

    async def scrape_url(self, url: str, extract_type: str = "content") -> Dict[str, Any]:
        """Scrape content from a URL"""
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        result = {
                            'url': url,
                            'status_code': response.status,
                            'scraped_at': datetime.utcnow().isoformat(),
                            'content_type': response.headers.get('content-type', ''),
                            'title': '',
                            'content': '',
                            'metadata': {}
                        }
                        
                        title_tag = soup.find('title')
                        if title_tag:
                            result['title'] = title_tag.get_text().strip()
                        
                        if extract_type == "content":
                            result['content'] = await self._extract_main_content(soup)
                        elif extract_type == "metadata":
                            result['metadata'] = await self._extract_metadata(soup)
                        elif extract_type == "links":
                            result['links'] = await self._extract_links(soup, url)
                        elif extract_type == "images":
                            result['images'] = await self._extract_images(soup, url)
                        elif extract_type == "all":
                            result['content'] = await self._extract_main_content(soup)
                            result['metadata'] = await self._extract_metadata(soup)
                            result['links'] = await self._extract_links(soup, url)
                            result['images'] = await self._extract_images(soup, url)
                        
                        return result
                    else:
                        return {
                            'error': f"HTTP {response.status}",
                            'url': url,
                            'status_code': response.status
                        }
                        
        except Exception as e:
            logger.error(f"URL scraping failed for {url}: {str(e)}")
            return {
                'error': f"Scraping failed: {str(e)}",
                'url': url
            }

    async def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML"""
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        main_content = ""
        
        content_selectors = [
            'main', 'article', '.content', '#content', '.post', '.entry',
            '.article-body', '.post-content', '.entry-content'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = ' '.join([elem.get_text() for elem in elements])
                break
        
        if not main_content:
            body = soup.find('body')
            if body:
                main_content = body.get_text()
        
        main_content = re.sub(r'\s+', ' ', main_content).strip()
        return main_content[:5000]

    async def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from HTML"""
        metadata = {}
        
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            name = tag.get('name') or tag.get('property') or tag.get('http-equiv')
            content = tag.get('content')
            if name and content:
                metadata[name] = content
        
        og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
        for tag in og_tags:
            property_name = tag.get('property')
            content = tag.get('content')
            if property_name and content:
                metadata[property_name] = content
        
        twitter_tags = soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')})
        for tag in twitter_tags:
            name = tag.get('name')
            content = tag.get('content')
            if name and content:
                metadata[name] = content
        
        return metadata

    async def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract links from HTML"""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text().strip()
            
            if href.startswith('/'):
                href = urljoin(base_url, href)
            elif not href.startswith(('http://', 'https://')):
                continue
            
            links.append({
                'url': href,
                'text': text,
                'title': link.get('title', '')
            })
        
        return links[:50]

    async def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract images from HTML"""
        images = []
        
        for img in soup.find_all('img', src=True):
            src = img['src']
            alt = img.get('alt', '')
            
            if src.startswith('/'):
                src = urljoin(base_url, src)
            elif not src.startswith(('http://', 'https://')):
                continue
            
            images.append({
                'url': src,
                'alt': alt,
                'title': img.get('title', '')
            })
        
        return images[:20]

    async def search_news(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search for news articles"""
        try:
            news_sources = [
                'https://rss.cnn.com/rss/edition.rss',
                'https://feeds.bbci.co.uk/news/rss.xml',
                'https://rss.reuters.com/reuters/topNews'
            ]
            
            all_articles = []
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                for source_url in news_sources:
                    try:
                        async with session.get(source_url) as response:
                            if response.status == 200:
                                content = await response.text()
                                feed = feedparser.parse(content)
                                
                                for entry in feed.entries:
                                    title = entry.get('title', '').lower()
                                    summary = entry.get('summary', '').lower()
                                    
                                    if query.lower() in title or query.lower() in summary:
                                        all_articles.append({
                                            'title': entry.get('title', ''),
                                            'url': entry.get('link', ''),
                                            'snippet': entry.get('summary', ''),
                                            'published': entry.get('published', ''),
                                            'source': feed.feed.get('title', 'Unknown'),
                                            'type': 'news_article'
                                        })
                    except Exception as e:
                        logger.warning(f"Failed to fetch news from {source_url}: {str(e)}")
                        continue
            
            all_articles = sorted(all_articles, key=lambda x: query.lower() in x['title'].lower(), reverse=True)
            
            return {
                'query': query,
                'search_type': 'news',
                'timestamp': datetime.utcnow().isoformat(),
                'results': all_articles[:max_results],
                'total_results': len(all_articles),
                'summary': self._generate_summary(all_articles[:max_results])
            }
            
        except Exception as e:
            logger.error(f"News search failed: {str(e)}")
            return {
                'error': f"News search failed: {str(e)}",
                'query': query,
                'results': [],
                'summary': ''
            }

    async def search_academic(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search for academic papers and research"""
        try:
            academic_query = f"{query} site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov OR site:scholar.google.com"
            
            results = await self.search(academic_query, max_results=max_results)
            
            academic_results = []
            for result in results.get('results', []):
                if any(domain in result.get('url', '') for domain in ['arxiv.org', 'pubmed', 'scholar.google', 'researchgate']):
                    result['type'] = 'academic_paper'
                    academic_results.append(result)
            
            return {
                'query': query,
                'search_type': 'academic',
                'timestamp': datetime.utcnow().isoformat(),
                'results': academic_results[:max_results],
                'total_results': len(academic_results),
                'summary': self._generate_summary(academic_results[:max_results])
            }
            
        except Exception as e:
            logger.error(f"Academic search failed: {str(e)}")
            return {
                'error': f"Academic search failed: {str(e)}",
                'query': query,
                'results': [],
                'summary': ''
            }

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        return unique_results

    async def _enhance_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance search results with additional metadata"""
        enhanced_results = []
        
        for result in results:
            result['relevance_score'] = self._calculate_relevance_score(result)
            
            url = result.get('url', '')
            if url:
                parsed_url = urlparse(url)
                result['domain'] = parsed_url.netloc
            
            enhanced_results.append(result)
        
        enhanced_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return enhanced_results

    def _calculate_relevance_score(self, result: Dict[str, Any]) -> float:
        """Calculate relevance score for search result"""
        score = 0.0
        
        title = result.get('title', '').lower()
        if title:
            score += len(title.split()) * 0.1
        
        snippet = result.get('snippet', '').lower()
        if snippet:
            score += len(snippet.split()) * 0.05
        
        source = result.get('source', '')
        if source in ['google', 'bing']:
            score += 1.0
        elif source == 'duckduckgo':
            score += 0.8
        
        result_type = result.get('type', '')
        if result_type == 'instant_answer':
            score += 2.0
        elif result_type == 'academic_paper':
            score += 1.5
        elif result_type == 'news_article':
            score += 1.2
        
        return score

    def _generate_cache_key(self, query: str, search_type: str, max_results: int, 
                           filters: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for search results"""
        key_data = f"{query}_{search_type}_{max_results}_{str(filters)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def clear_cache(self) -> Dict[str, Any]:
        """Clear search cache"""
        cache_size = len(self.cache)
        self.cache.clear()
        return {
            'status': 'success',
            'cleared_entries': cache_size
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for web search service"""
        return {
            'status': 'healthy',
            'search_engines': list(self.search_engines.keys()),
            'cache_entries': len(self.cache),
            'cache_duration_hours': self.cache_duration.total_seconds() / 3600
        }

# Example usage in a chat service (assuming this is part of another class)
async def process_chat(self, request):  # Hypothetical method in a ChatService class
    combined_context = ""
    search_results = None
    try:
        if request.web_search:
            logger.info("Performing web search")
            search_results = await self.web_search_service.search(
                request.prompt, max_results=10
            )
            if search_results and 'summary' in search_results:
                combined_context += f"\n\nWeb search results:\n{search_results['summary']}"
            else:
                logger.warning("No summary available in search results")
                combined_context += "\n\nWeb search results: No relevant information found."
    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}")
        raise
    return combined_context