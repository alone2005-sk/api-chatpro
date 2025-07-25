"""
LLM Manager - Handles local and remote LLM interactions
Supports Ollama, llamacpp, GPT4All locally and various APIs remotely
"""

import asyncio
import aiohttp
import json
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LLMProvider:
    name: str
    type: str  # "local" or "remote"
    endpoint: str
    api_key: Optional[str] = None
    model: str = None
    available: bool = False

class LLMManager:
    def __init__(self):
        self.providers = {
            # Local providers
            "ollama": LLMProvider("ollama", "local", "http://localhost:11434/api/generate"),
            "llamacpp": LLMProvider("llamacpp", "local", "http://localhost:8080/completion"),
            "gpt4all": LLMProvider("gpt4all", "local", "http://localhost:4891/v1/chat/completions"),
            
            # Remote providers (fallback)
            "openai": LLMProvider("openai", "remote", "https://api.openai.com/v1/chat/completions", 
                                model="gpt-3.5-turbo"),
            "groq": LLMProvider("groq", "remote", "https://api.groq.com/openai/v1/chat/completions",
                              model="llama3-70b-8192"),
            "anthropic": LLMProvider("anthropic", "remote", "https://api.anthropic.com/v1/messages",
                                   model="claude-3-sonnet-20240229")
        }
        
        self.session = None
    
    async def initialize(self):
        """Initialize HTTP session and check provider availability"""
        self.session = aiohttp.ClientSession()
        await self._check_provider_availability()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def _check_provider_availability(self):
        """Check which providers are available"""
        for name, provider in self.providers.items():
            try:
                if provider.type == "local":
                    # Check if local service is running
                    async with self.session.get(provider.endpoint.replace("/api/generate", "/api/tags").replace("/completion", "/health").replace("/v1/chat/completions", "/health"), timeout=5) as response:
                        provider.available = response.status == 200
                else:
                    # For remote providers, assume available if API key is set
                    api_key = self._get_api_key(name)
                    provider.available = api_key is not None
                    provider.api_key = api_key
                
                logger.info(f"Provider {name}: {'✅ Available' if provider.available else '❌ Unavailable'}")
            except Exception as e:
                logger.warning(f"Provider {name} check failed: {str(e)}")
                provider.available = False
    
    def _get_api_key(self, provider_name: str) -> Optional[str]:
        """Get API key from environment"""
        import os
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY"
        }
        return os.getenv(key_mapping.get(provider_name))
    
    async def generate_code(self, prompt: str, language: str, context: Dict = None, prefer_local: bool = True) -> Dict[str, Any]:
        """Generate code using available LLMs"""
        code_prompt = f"""
        Generate clean, well-documented {language} code for the following request:
        
        {prompt}
        
        Requirements:
        - Include proper error handling
        - Add helpful comments
        - Follow best practices for {language}
        - Make the code production-ready
        
        Return only the code without explanations.
        """
        
        if prefer_local:
            # Try local providers first
            for name, provider in self.providers.items():
                if provider.type == "local" and provider.available:
                    try:
                        result = await self._call_local_llm(provider, code_prompt)
                        return {
                            "code": result,
                            "provider": name,
                            "language": language
                        }
                    except Exception as e:
                        logger.warning(f"Local provider {name} failed: {str(e)}")
                        continue
        
        # Fallback to remote providers
        for name, provider in self.providers.items():
            if provider.type == "remote" and provider.available:
                try:
                    result = await self._call_remote_llm(provider, code_prompt)
                    return {
                        "code": result,
                        "provider": name,
                        "language": language
                    }
                except Exception as e:
                    logger.warning(f"Remote provider {name} failed: {str(e)}")
                    continue
        
        raise Exception("No available LLM providers")
    
    async def process_text(self, prompt: str, operation: str = "process", context: Dict = None) -> Dict[str, Any]:
        """Process text using available LLMs"""
        # Try local first, then remote
        for provider_type in ["local", "remote"]:
            for name, provider in self.providers.items():
                if provider.type == provider_type and provider.available:
                    try:
                        if provider.type == "local":
                            result = await self._call_local_llm(provider, prompt)
                        else:
                            result = await self._call_remote_llm(provider, prompt)
                        
                        return {
                            "text": result,
                            "provider": name,
                            "operation": operation
                        }
                    except Exception as e:
                        logger.warning(f"Provider {name} failed: {str(e)}")
                        continue
        
        raise Exception("No available LLM providers")
    
    async def chat(self, prompt: str, context: Dict = None) -> Dict[str, Any]:
        """General chat/query processing"""
        return await self.process_text(prompt, "chat", context)
    
    async def _call_local_llm(self, provider: LLMProvider, prompt: str) -> str:
        """Call local LLM provider"""
        if provider.name == "ollama":
            return await self._call_ollama(prompt)
        elif provider.name == "llamacpp":
            return await self._call_llamacpp(prompt)
        elif provider.name == "gpt4all":
            return await self._call_gpt4all(prompt)
        else:
            raise Exception(f"Unknown local provider: {provider.name}")
    
    async def _call_remote_llm(self, provider: LLMProvider, prompt: str) -> str:
        """Call remote LLM provider"""
        if provider.name == "openai":
            return await self._call_openai(provider, prompt)
        elif provider.name == "groq":
            return await self._call_groq(provider, prompt)
        elif provider.name == "anthropic":
            return await self._call_anthropic(provider, prompt)
        else:
            raise Exception(f"Unknown remote provider: {provider.name}")
    
    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        payload = {
            "model": "llama3.1",  # Default model
            "prompt": prompt,
            "stream": False
        }
        
        async with self.session.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120
        ) as response:
            result = await response.json()
            return result.get("response", "")
    
    async def _call_llamacpp(self, prompt: str) -> str:
        """Call llama.cpp server"""
        payload = {
            "prompt": prompt,
            "n_predict": 512,
            "temperature": 0.7,
            "stop": ["</s>", "Human:", "Assistant:"]
        }
        
        async with self.session.post(
            "http://localhost:8080/completion",
            json=payload,
            timeout=120
        ) as response:
            result = await response.json()
            return result.get("content", "")
    
    async def _call_gpt4all(self, prompt: str) -> str:
        """Call GPT4All API"""
        payload = {
            "model": "gpt4all",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        async with self.session.post(
            "http://localhost:4891/v1/chat/completions",
            json=payload,
            timeout=120
        ) as response:
            result = await response.json()
            return result["choices"][0]["message"]["content"]
    
    async def _call_openai(self, provider: LLMProvider, prompt: str) -> str:
        """Call OpenAI API"""
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": provider.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        async with self.session.post(
            provider.endpoint,
            headers=headers,
            json=payload,
            timeout=60
        ) as response:
            result = await response.json()
            return result["choices"][0]["message"]["content"]
    
    async def _call_groq(self, provider: LLMProvider, prompt: str) -> str:
        """Call Groq API"""
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": provider.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        async with self.session.post(
            provider.endpoint,
            headers=headers,
            json=payload,
            timeout=60
        ) as response:
            result = await response.json()
            return result["choices"][0]["message"]["content"]
    
    async def _call_anthropic(self, provider: LLMProvider, prompt: str) -> str:
        """Call Anthropic API"""
        headers = {
            "x-api-key": provider.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": provider.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000
        }
        
        async with self.session.post(
            provider.endpoint,
            headers=headers,
            json=payload,
            timeout=60
        ) as response:
            result = await response.json()
            return result["content"][0]["text"]
    
    async def get_local_models(self) -> List[str]:
        """Get list of available local models"""
        models = []
        
        # Check Ollama models
        if self.providers["ollama"].available:
            try:
                async with self.session.get("http://localhost:11434/api/tags") as response:
                    data = await response.json()
                    models.extend([model["name"] for model in data.get("models", [])])
            except:
                pass
        
        return models
    
    async def get_remote_apis(self) -> List[str]:
        """Get list of available remote APIs"""
        return [name for name, provider in self.providers.items() 
                if provider.type == "remote" and provider.available]
